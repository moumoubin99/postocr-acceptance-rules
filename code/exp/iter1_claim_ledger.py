#!/usr/bin/env python3
"""Iteration-1 claim ledger (CPU only, frozen candidates).
Recomputes every corrector x dataset result with ONE protocol and corrector-named
outputs, adds the reviewer-requested cheap baselines, paired doc-cluster bootstrap
with Holm correction, and dev-tuned (not test-tuned) corruption-budget operating points.

Protocol (unchanged from the paper): policies fit on dev, thresholds chosen on dev,
evaluated once on held-out test. Micro CER = sum edit distance / sum gold chars.
Outputs: process/refine-logs/iter1/ledger_{corr}_{ds}.json + ledger_summary.md
"""
import os, sys, json, numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJ", os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA

OUT = os.environ.get("OUT") or os.path.join(ROOT, "process", "refine-logs", "iter1"); os.makedirs(OUT, exist_ok=True)
SEED = 20260526; B = 2000
ORIG = [0, 1, 2, 3, 8, 9, 10, 11, 12]          # OCR-only features (no candidate)
ED = 6                                          # edit_density index
CONFIGS = [("bln600", c) for c in ["symspell", "byt5", "byt5base", "byt5base8", "qwen"]] + \
          [("icdar", c) for c in ["symspell", "byt5", "byt5s1", "byt5s2", "byt5s3", "byt5base", "byt5base8"]]
GRID = np.linspace(0.2, 0.8, 61)
INSAMPLE = os.environ.get("INSAMPLE") == "1"   # 1 = legacy protocol (threshold on in-sample dev preds)
from sklearn.model_selection import GroupKFold

def fit_predict(Xd, yd, att, groups, Xt):
    """final model on all attempted dev rows -> test preds; dev preds are 5-fold doc-grouped OOF
    (cross-fitted) unless INSAMPLE, so dev-tuned thresholds/budgets are not optimistic."""
    c = gb(); c.fit(Xd[att], yd[att]); pt = c.predict_proba(Xt)[:, 1]
    if INSAMPLE: return c.predict_proba(Xd)[:, 1], pt
    pd_ = np.zeros(len(Xd))
    for tr, va in GroupKFold(5).split(Xd, yd, groups):
        m = gb(); trm = tr[att[tr]]; m.fit(Xd[trm], yd[trm]); pd_[va] = m.predict_proba(Xd[va])[:, 1]
    return pd_, pt

def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                                l2_regularization=1.0, random_state=SEED)

def build(corr, ds, split):
    p = os.path.join(DATA, f"candidates_{corr}_{ds}_{split}.jsonl")
    R = []
    for r in load_jsonl(p):
        eo, ec = Levenshtein.distance(r["ocr"], r["gold"]), Levenshtein.distance(r["cand"], r["gold"])
        R.append(dict(doc=r["doc_id"], eo=eo, ec=ec, gl=len(r["gold"]), att=r["ocr"] != r["cand"],
                      helped=int(cer(r["cand"], r["gold"]) < cer(r["ocr"], r["gold"]) - 1e-9),
                      feat=features(r["ocr"], r["cand"]), score=r.get("score")))
    return R

def arr(R, k): return np.array([r[k] for r in R])

class Split:
    def __init__(s, R):
        s.R = R; s.eo, s.ec, s.gl = arr(R, "eo"), arr(R, "ec"), arr(R, "gl")
        s.att, s.y = arr(R, "att").astype(bool), arr(R, "helped")
        s.X = np.array([r["feat"] for r in R], dtype=float)
        s.has_score = all(r["score"] is not None for r in R)
        s.score = np.array([r["score"] if r["score"] is not None else np.nan for r in R], dtype=float)
        docs = sorted(set(r["doc"] for r in R)); s.docs = docs
        di = {d: i for i, d in enumerate(docs)}; s.di = np.array([di[r["doc"]] for r in R])
    def cer(s, use): return float(np.where(use, s.ec, s.eo).sum() / max(1, s.gl.sum()))
    def corrupt(s, use): return int((use & (s.ec > s.eo)).sum())

def tune(dv, sd, grid, budget=None):
    """threshold on dev score sd (accept iff att & sd>=t) minimising dev CER, optionally s.t. corruption<=budget"""
    best = None
    for t in grid:
        u = dv.att & (sd >= t)
        if budget is not None and dv.corrupt(u) > budget: continue
        c = dv.cer(u)
        if best is None or c < best[0] - 1e-12: best = (c, t)
    return best[1] if best else np.inf

def run(ds, corr):
    dv, te = Split(build(corr, ds, "dev")), Split(build(corr, ds, "test"))
    P, info = {}, {}
    P["never"] = np.zeros(len(te.R), bool); P["always"] = te.att.copy(); P["oracle"] = te.ec < te.eo
    def seg_model(cols, name):
        pd_, pt = fit_predict(dv.X[:, cols], dv.y, dv.att, dv.di, te.X[:, cols])
        t = tune(dv, pd_, GRID); P[name] = te.att & (pt >= t)
        auc = roc_auc_score(te.y[te.att], pt[te.att]) if len(set(te.y[te.att])) > 1 else float("nan")
        info[name] = dict(thr=float(t), auc_test=float(auc)); return pd_, pt
    rd, rt = seg_model(list(range(13)), "router_text")
    seg_model(ORIG, "errdet_ocr_only")
    # doc-QE: doc-mean features, doc label = always helps the doc
    def agg(S):
        Xd = np.zeros((len(S.docs), 13)); yd = np.zeros(len(S.docs), int)
        for k in range(len(S.docs)):
            m = S.di == k; Xd[k] = S.X[m].mean(0); yd[k] = int(S.ec[m].sum() < S.eo[m].sum())
        return Xd, yd
    Xdd, ydd = agg(dv); Xtd, _ = agg(te)
    if len(set(ydd)) > 1:
        c = gb(); c.fit(Xdd, ydd); pdd, ptd = c.predict_proba(Xdd)[:, 1][dv.di], c.predict_proba(Xtd)[:, 1][te.di]
    else:
        pdd, ptd = np.full(len(dv.R), float(ydd[0])), np.full(len(te.R), float(ydd[0]))
    t = tune(dv, pdd, GRID); P["docqe"] = te.att & (ptd >= t); info["docqe"] = dict(thr=float(t))
    # edit-cap rule: accept iff edit_density <= t  (score = -edit_density)
    qs = np.unique(np.quantile(-dv.X[dv.att, ED], np.linspace(0, 1, 101)))
    t = tune(dv, -dv.X[:, ED], qs); P["edit_cap"] = te.att & (-te.X[:, ED] >= t); info["edit_cap"] = dict(max_edit_density=float(-t))
    if dv.has_score and te.has_score:
        qs = np.unique(np.quantile(dv.score[dv.att], np.linspace(0, 1, 101)))
        t = tune(dv, dv.score, qs); P["conf_only"] = te.att & (te.score >= t); info["conf_only"] = dict(thr=float(t))
        # conf + cap: 2D grid
        best = None
        for a in np.quantile(dv.score[dv.att], np.linspace(0, 0.9, 19)):
            for b in np.quantile(dv.X[dv.att, ED], np.linspace(0.1, 1, 19)):
                u = dv.att & (dv.score >= a) & (dv.X[:, ED] <= b); c_ = dv.cer(u)
                if best is None or c_ < best[0]: best = (c_, a, b)
        P["conf_plus_cap"] = te.att & (te.score >= best[1]) & (te.X[:, ED] <= best[2])
        info["conf_plus_cap"] = dict(conf_thr=float(best[1]), max_edit_density=float(best[2]))
        cols = list(range(13))
        Xd2, Xt2 = np.c_[dv.X, dv.score], np.c_[te.X, te.score]
        cd, ct = fit_predict(Xd2, dv.y, dv.att, dv.di, Xt2)
        t = tune(dv, cd, GRID); P["router_text_conf"] = te.att & (ct >= t)
        info["router_text_conf"] = dict(thr=float(t), auc_test=float(roc_auc_score(te.y[te.att], ct[te.att])))
    else:
        cd = ct = None
    # random policy matched to router acceptance rate among attempted (expected value, closed form)
    rate = P["router_text"][te.att].mean() if te.att.any() else 0.0
    exp_err = te.eo.sum() + rate * (te.ec[te.att] - te.eo[te.att]).sum()
    res = {k: dict(cer=te.cer(u), corrupt=te.corrupt(u), accept_rate_attempted=float(u[te.att].mean()) if te.att.any() else 0.0)
           for k, u in P.items()}
    res["random_matched"] = dict(cer=float(exp_err / te.gl.sum()), accept_rate_attempted=float(rate), note="expected value")
    for k, v in info.items(): res[k].update(v)
    # paired doc-cluster bootstrap: baseline - router_text (positive => router better)
    rng = np.random.default_rng(SEED); nd = len(te.docs)
    def per_doc(u): e = np.where(u, te.ec, te.eo); return np.bincount(te.di, e, nd)
    G = np.bincount(te.di, te.gl, nd); Er = per_doc(P["router_text"])
    comps = [k for k in P if k not in ("router_text", "oracle")]
    Eb = {k: per_doc(P[k]) for k in comps}
    W = rng.multinomial(nd, np.ones(nd) / nd, size=B)          # doc resampling weights
    gl_b = W @ G; er_b = (W @ Er) / gl_b
    boot = {}
    for k in comps:
        d = (W @ Eb[k]) / gl_b - er_b
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        boot[k] = dict(delta=float(res[k]["cer"] - res["router_text"]["cer"]),
                       ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], p_two_sided=p)
    # Holm over comparisons
    order = sorted(boot, key=lambda k: boot[k]["p_two_sided"]); m = len(order); run_max = 0
    for i, k in enumerate(order):
        run_max = max(run_max, min(1.0, (m - i) * boot[k]["p_two_sided"])); boot[k]["p_holm"] = run_max
    # dev-tuned corruption-budget operating points (fix for test-tuned frontier)
    frontier = {}
    a_dev = dv.corrupt(dv.att)
    for frac in [0.10, 0.25, 0.50]:
        bud = int(frac * a_dev); row = {}
        for name, (sd, st) in {"router_text": (rd, rt), "router_text_conf": (cd, ct),
                               "conf_only": (dv.score, te.score) if dv.has_score else (None, None),
                               "edit_cap": (-dv.X[:, ED], -te.X[:, ED])}.items():
            if sd is None: continue
            grid = np.unique(np.quantile(sd[dv.att], np.linspace(0, 1, 201)))
            t = tune(dv, sd, grid, budget=bud); u = te.att & (st >= t)
            row[name] = dict(test_cer=te.cer(u), test_corrupt=te.corrupt(u), test_budget=int(frac * te.corrupt(te.att)))
        frontier[f"{int(frac*100)}pct"] = row
    out = dict(dataset=ds, corrector=corr, n_dev=len(dv.R), n_test=len(te.R), n_test_docs=nd,
               policies=res, bootstrap_vs_router_text=boot, devtuned_budget_frontier=frontier,
               protocol=("in-sample-dev-threshold (legacy) " if INSAMPLE else "cross-fitted 5-fold doc-grouped OOF dev threshold; ") + "fit+threshold on dev, single eval on test; micro CER; paired doc bootstrap B=2000; Holm")
    json.dump(out, open(os.path.join(OUT, f"ledger_{corr}_{ds}{'_insample' if INSAMPLE else ''}.json"), "w"), indent=2)
    return out

if __name__ == "__main__":
    rows = []
    for ds, corr in CONFIGS:
        try: o = run(ds, corr)
        except FileNotFoundError as e: print("skip", ds, corr, e); continue
        p, b = o["policies"], o["bootstrap_vs_router_text"]
        def f(k): return f"{p[k]['cer']:.4f}" if k in p else "—"
        def s(k):
            if k not in b: return "—"
            x = b[k]; sig = "*" if x["p_holm"] < 0.05 else ""
            return f"{x['delta']:+.4f}{sig}"
        rows.append(f"| {ds} | {corr} | {f('never')} | {f('always')} | **{f('router_text')}** | {f('router_text_conf')} | "
                    f"{f('conf_only')} | {f('edit_cap')} | {f('errdet_ocr_only')} | {f('docqe')} | {f('random_matched')} | {f('oracle')} | "
                    f"{s('always')} | {s('conf_only')} | {s('edit_cap')} | {s('errdet_ocr_only')} | {s('docqe')} |")
        print(rows[-1], flush=True)
    hdr = ("| ds | corrector | never | always | router(text) | router(+conf) | conf-only | edit-cap | errdet(OCR) | docQE | random@rate | oracle | "
           "Δalways | Δconf | Δcap | Δerrdet | ΔdocQE |\n|" + "---|" * 17)
    md = ("# Iteration-1 claim ledger (CPU, frozen candidates)\n\nAll policies fit on dev, thresholds on dev, single test eval. "
          "Δ = baseline CER − router(text) CER (positive ⇒ router better); * = Holm-adjusted paired doc-bootstrap p<0.05.\n\n"
          + hdr + "\n" + "\n".join(rows) + "\n")
    open(os.path.join(OUT, "ledger_summary_insample.md" if INSAMPLE else "ledger_summary.md"), "w").write(md)
    print(md)
