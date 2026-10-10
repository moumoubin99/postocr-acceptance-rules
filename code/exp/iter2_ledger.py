#!/usr/bin/env python3
"""Iteration-2 claim ledger (CPU only, frozen candidates from the iter-2 GPU package).

One protocol for every corrector x dataset on the leakage-free splits:
  * every learned policy is cross-fitted (5-fold doc-grouped OOF on dev) before its dev threshold is chosen,
    including doc-QE; final models are fit on all dev and evaluated once on test;
  * thresholds for ALL policies come from the same grid (dev quantiles of the score + "abstain");
  * baselines: edit-cap, mean log-prob, min-token log-prob (calibrated and Kim-style fixed 0.95),
    conf+cap, OCR-only error detector, doc-QE, trained QE encoder (when available);
  * router variants: legacy 13 text features, + aligned token-diff, dictionary-free, + decoder confidence,
    + QE score, expected-character-utility regression target;
  * paired doc-cluster bootstrap (B=10000) with Holm correction; dev-tuned harm budgets;
  * Learn-then-Test control of the harmful-acceptance rate (doc-level Hoeffding-Bentkus, fixed sequence);
  * reassembled-document CER next to segment CER; number / capitalised-token damage proxies.
Outputs: process/refine-logs/iter2/ledger_{corr}_{ds}.json, ledger_summary.md
"""
import os, sys, json, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJ", os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "exp"))
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binom
from rapidfuzz.distance import Levenshtein
from oclib import features, load_jsonl, DATA, in_dict, _pre

OUT = os.environ.get("OUT") or os.path.join(ROOT, "process", "refine-logs", "iter2"); os.makedirs(OUT, exist_ok=True)
SEED = 20260526; B = int(os.environ.get("NBOOT", 10000))

def pval(d):
    """two-sided bootstrap p with the +1 correction, so the smallest attainable value is 2/(B+1)"""
    return float(min(1.0, 2 * (min(int((d <= 0).sum()), int((d >= 0).sum())) + 1) / (len(d) + 1)))

ORIG = [0, 1, 2, 3, 8, 9, 10, 11, 12]          # OCR-only features (no candidate)
DICTF = [3, 4, 12]                              # English-lexicon features among the 13
ED = 6                                          # edit_density index
NA_NODICT = 9                                   # first 9 aligned features use no lexicon
CONFIGS = ([("blnocr", c) for c in ["symspell", "byt5", "byt5base8", "qwen", "qwenctx2"]] +
           [("blnocrw50", "qwen"), ("blnocrw100", "qwen"), ("bln600", "byt5base8x"), ("bln600", "qwenv2")] +
           [("icdarc_EN", c) for c in ["byt5", "byt5base8", "byt5base8m", "qwen"]] +
           [(f"icdarc_{L}", "byt5base8m") for L in ["DE", "FR", "ES", "NL"]] + [("icdarc_multi", "qwen")] +
           [(f"ext_{d}_en", c) for d in ["impresso", "overproof"] for c in ["symspell", "byt5", "byt5base8", "qwen"]])

if os.environ.get("CONFIG_SET") == "iter3":       # held-out replication set: Qwen prompt variants + a second LLM family
    CONFIGS = [(d, c) for c in ["qwenp2", "qwenp3", "llama"] for d in ["blnocr", "icdarc_EN", "icdarc_multi", "ext_impresso_en", "ext_overproof_en"]]

def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                                l2_regularization=1.0, random_state=SEED)
def gbr(): return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=300,
                                                l2_regularization=1.0, random_state=SEED)

def core(t): return _pre.match(t).group(2)

def align_feats(ocr, cand):
    """token-level alignment (Levenshtein opcodes over token lists) -> diff features; first 9 are lexicon-free"""
    to, tc = ocr.split(), cand.split(); nt = max(1, len(to))
    rep = ins = dele = 0; pairs = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(to, tc):
        if tag == "replace":
            k = min(i2 - i1, j2 - j1); pairs += list(zip(to[i1:i1 + k], tc[j1:j1 + k])); rep += k
            dele += (i2 - i1) - k; ins += (j2 - j1) - k
        elif tag == "delete": dele += i2 - i1
        elif tag == "insert": ins += j2 - j1
    npair = max(1, len(pairs))
    ce = [Levenshtein.distance(a, b) / max(1, len(a)) for a, b in pairs]
    al = lambda s: "".join(ch for ch in s if ch.isalnum())
    dig = sum(any(ch.isdigit() for ch in a + b) for a, b in pairs)
    case = sum(a.lower() == b.lower() for a, b in pairs)
    punct = sum(al(a) == al(b) for a, b in pairs)
    first = sum(a[:1].lower() != b[:1].lower() for a, b in pairs)
    def ind(t): c = core(t); return bool(c) and c.isalpha() and in_dict(c)
    o2i = sum((not ind(a)) and ind(b) for a, b in pairs)
    i2d = sum(ind(a) and core(a).lower() != core(b).lower() for a, b in pairs)
    i2o = sum(ind(a) and not ind(b) for a, b in pairs)
    return [rep / nt, ins / nt, dele / nt, max(ce) if ce else 0.0, float(np.mean(ce)) if ce else 0.0,
            dig / npair, case / npair, punct / npair, first / npair, o2i / npair, i2d / npair, i2o / npair]

def conf_feats(r):
    lps = np.array(r.get("lps") or [r["min_lp"]], float); s = np.sort(lps)
    return [r["score"], r["min_lp"], r["n_tok"], float((lps < math.log(0.9)).mean()), float((lps < math.log(0.5)).mean()),
            float(s[1]) if len(s) > 1 else float(s[0]), float(np.argmin(lps) / max(1, len(lps)))]

_num = __import__("re").compile(r"\d+")
def damage(r):
    """evaluation-only proxies: numbers / capitalised tokens that OCR had right and the candidate lost (and v.v.)"""
    def bag(s, kind):
        if kind == "num": return set(_num.findall(s))
        t = s.split(); return {core(x) for x in t[1:] if core(x)[:1].isupper() and len(core(x)) > 2}
    out = {}
    for kind in ("num", "cap"):
        g, o, c = bag(r["gold"], kind), bag(r["ocr"], kind), bag(r["cand"], kind)
        out[kind + "_broken"] = len((g & o) - c); out[kind + "_fixed"] = len((g & c) - o)
    return out

class Split:
    def __init__(s, corr, ds, split):
        rows = load_jsonl(os.path.join(DATA, f"candidates_{corr}_{ds}_{split}.jsonl"))
        s.rows = rows; n = len(rows)
        s.name = (corr, ds, split)
        s.eo = np.array([Levenshtein.distance(r["ocr"], r["gold"]) for r in rows])
        s.ec = np.array([Levenshtein.distance(r["cand"], r["gold"]) for r in rows])
        s.gl = np.array([len(r["gold"]) for r in rows])
        s.att = np.array([r["ocr"] != r["cand"] for r in rows]); s.y = (s.ec < s.eo).astype(int)
        s.X = np.array([features(r["ocr"], r["cand"]) for r in rows], float)
        s.A = np.array([align_feats(r["ocr"], r["cand"]) for r in rows], float)
        s.has_conf = all(r.get("min_lp") is not None and r.get("score") is not None for r in rows)
        s.C = np.array([conf_feats(r) for r in rows], float) if s.has_conf else None
        s.q = None
        qp = os.path.join(DATA, f"qe_{corr}_{ds}_{split}.jsonl")
        if os.path.exists(qp):
            Q = load_jsonl(qp); assert len(Q) == n and all(a["doc_id"] == b["doc_id"] and a["seg_idx"] == b["seg_idx"] for a, b in zip(Q, rows))
            s.q = np.array([x["qe"] for x in Q], float)
        s.docs = sorted({r["doc_id"] for r in rows}); di = {d: i for i, d in enumerate(s.docs)}
        s.di = np.array([di[r["doc_id"]] for r in rows]); s.nd = len(s.docs)
        s.dmg = [damage(r) for r in rows]
        s.G = np.bincount(s.di, s.gl, s.nd); s.nseg = np.bincount(s.di, minlength=s.nd)
    def cer(s, use): return float(np.where(use, s.ec, s.eo).sum() / max(1, s.gl.sum()))
    def corrupt(s, use): return int((use & (s.ec > s.eo)).sum())
    def per_doc(s, use): return np.bincount(s.di, np.where(use, s.ec, s.eo), s.nd)
    def doc_cer(s, use):
        """reassemble documents from chosen segment text + OCR separators; micro CER over documents"""
        hyp, ref = [[] for _ in s.docs], [[] for _ in s.docs]
        order = sorted(range(len(s.rows)), key=lambda i: (s.di[i], s.rows[i].get("seg_idx", i)))
        for i in order:
            r = s.rows[i]
            hyp[s.di[i]].append((r["cand"] if use[i] else r["ocr"]) + r.get("ocr_sep", " "))
            ref[s.di[i]].append(r["gold"] + r.get("gold_sep", " "))
        e = g = 0
        for h, f in zip(hyp, ref):
            h, f = "".join(h).strip(), "".join(f).strip(); e += Levenshtein.distance(h, f); g += len(f)
        return e / max(1, g)
    def dmg_sum(s, use):
        o = {}
        for k in ("num_broken", "num_fixed", "cap_broken", "cap_fixed"): o[k] = int(sum(d[k] for d, u in zip(s.dmg, use) if u))
        return o

def qgrid(sd, att, k=101):
    v = sd[att] if att.any() else sd
    return np.r_[np.unique(np.quantile(v, np.linspace(0, 1, k))), np.inf]

def tune(dv, sd, grid=None, budget=None):
    """threshold on dev score (accept iff att & score>=t) minimising dev CER, optionally s.t. corruption<=budget"""
    grid = qgrid(sd, dv.att) if grid is None else grid
    best = None
    for t in grid:
        u = dv.att & (sd >= t)
        if budget is not None and dv.corrupt(u) > budget: continue
        c = dv.cer(u)
        if best is None or c < best[0] - 1e-12: best = (c, t)
    return best[1] if best else np.inf

def crossfit(make, Xd, yd, mask, groups, Xt, proba=True):
    """final model on dev[mask] -> test scores; dev scores are doc-grouped OOF. Degenerate folds -> constant."""
    def fp(Xtr, ytr, Xap):
        if len(ytr) < 10 or (proba and len(set(ytr)) < 2): return np.full(len(Xap), float(np.mean(ytr)) if len(ytr) else 0.0)
        m = make(); m.fit(Xtr, ytr); return m.predict_proba(Xap)[:, 1] if proba else m.predict(Xap)
    pt = fp(Xd[mask], yd[mask], Xt); pd_ = np.zeros(len(Xd)); ng = len(set(groups))
    if ng < 2: return fp(Xd[mask], yd[mask], Xd), pt
    for tr, va in GroupKFold(min(5, ng)).split(Xd, yd, groups):
        trm = tr[mask[tr]]; pd_[va] = fp(Xd[trm], yd[trm], Xd[va])
    return pd_, pt

def featsets(S):
    nd = [i for i in range(13) if i not in DICTF]
    F = {"router_text": S.X, "router_align": np.c_[S.X, S.A], "router_nodict": np.c_[S.X[:, nd], S.A[:, :NA_NODICT]],
         "errdet_ocr_only": S.X[:, ORIG]}
    if S.has_conf: F["router_conf"] = np.c_[S.X, S.A, S.C]
    if S.has_conf and S.q is not None: F["router_conf_qe"] = np.c_[S.X, S.A, S.C, S.q]
    return F

def doc_agg(S):
    Xd = np.zeros((S.nd, 13)); yd = np.zeros(S.nd, int)
    for k in range(S.nd):
        m = S.di == k; Xd[k] = S.X[m].mean(0); yd[k] = int(S.ec[m].sum() < S.eo[m].sum())
    return Xd, yd

def hb_p(r, n, a):
    """Hoeffding-Bentkus p-value for H0: risk > a, given empirical risk r over n bounded losses"""
    if r >= a: return 1.0
    def h(p, q): return (p * math.log(p / q) if p > 0 else 0.0) + ((1 - p) * math.log((1 - p) / (1 - q)) if p < 1 else 0.0)
    return float(min(1.0, math.exp(-n * h(r, a)), math.e * binom.cdf(math.ceil(n * r), n, a)))

def ltt(sd, harm, att, di, nd, nseg, cal_docs, alpha, delta=0.10):
    """fixed-sequence test from most conservative threshold; loss per doc = harmful accepted segs / segs"""
    grid = np.unique(np.quantile(sd[att], np.linspace(0, 1, 101)))[::-1] if att.any() else np.array([])
    n = len(cal_docs); t_ok = np.inf
    for t in grid:
        u = att & (sd >= t) & harm
        L = (np.bincount(di, u, nd) / np.maximum(1, nseg))[cal_docs]
        if hb_p(float(L.mean()), n, alpha) > delta: break
        t_ok = t
    return t_ok

def scores(dv, te):
    """all policy scores (dev OOF / test) with accept rule score>=t"""
    S = {}; Fd, Ft = featsets(dv), featsets(te)
    for k in Fd:
        if k in Ft: S[k] = crossfit(gb, Fd[k], dv.y, dv.att, dv.di, Ft[k])
    full = "router_conf" if dv.has_conf else "router_align"
    S["router_util"] = crossfit(gbr, Fd[full], (dv.eo - dv.ec) / np.maximum(1, dv.gl), dv.att, dv.di, Ft[full], proba=False)
    Xdd, ydd = doc_agg(dv); Xtd, _ = doc_agg(te)            # doc-QE, cross-fitted over documents
    a, b = crossfit(gb, Xdd, ydd, np.ones(dv.nd, bool), np.arange(dv.nd), Xtd); S["docqe"] = (a[dv.di], b[te.di])
    S["edit_cap"] = (-dv.X[:, ED], -te.X[:, ED])
    if dv.has_conf: S["conf_mean"] = (dv.C[:, 0], te.C[:, 0]); S["conf_min"] = (dv.C[:, 1], te.C[:, 1])
    if dv.q is not None and te.q is not None: S["qe_encoder"] = (dv.q, te.q)
    return S, full

def run(ds, corr):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test")
    S, full = scores(dv, te); P, info = {}, {}
    P["never"] = np.zeros(len(te.rows), bool); P["always"] = te.att.copy(); P["oracle"] = te.ec < te.eo
    for k, (sd, st) in S.items():
        t = tune(dv, sd); P[k] = te.att & (st >= t); info[k] = dict(thr=float(t))
        if len(set(te.y[te.att])) > 1: info[k]["auc_test"] = float(roc_auc_score(te.y[te.att], st[te.att]))
    if dv.has_conf:
        P["kim_min095"] = te.att & (te.C[:, 1] >= math.log(0.95)); info["kim_min095"] = dict(thr=math.log(0.95), note="fixed, untuned")
        best = None
        for a in np.r_[np.quantile(dv.C[dv.att, 1], np.linspace(0, 0.95, 20)), np.inf]:
            for b in np.quantile(dv.X[dv.att, ED], np.linspace(0.05, 1, 20)):
                u = dv.att & (dv.C[:, 1] >= a) & (dv.X[:, ED] <= b); c_ = dv.cer(u)
                if best is None or c_ < best[0] - 1e-12: best = (c_, a, b)
        P["conf_plus_cap"] = te.att & (te.C[:, 1] >= best[1]) & (te.X[:, ED] <= best[2])
        info["conf_plus_cap"] = dict(min_lp_thr=float(best[1]), max_edit_density=float(best[2]))
    natt = max(1, int(te.att.sum()))
    res = {k: dict(cer=te.cer(u), doc_cer=te.doc_cer(u), corrupt=te.corrupt(u), harm_rate=te.corrupt(u) / len(te.rows),
                   accept_rate_attempted=float(u[te.att].sum() / natt), **te.dmg_sum(u)) for k, u in P.items()}
    rate = res[full]["accept_rate_attempted"]
    res["random_matched"] = dict(cer=float((te.eo.sum() + rate * (te.ec[te.att] - te.eo[te.att]).sum()) / te.gl.sum()),
                                 accept_rate_attempted=rate, note=f"expected value at {full} acceptance rate")
    for k, v in info.items(): res[k].update(v)
    # paired doc-cluster bootstrap vs two pre-specified references
    rng = np.random.default_rng(SEED); W = rng.multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gl_b = W @ te.G
    E = {k: te.per_doc(u) for k, u in P.items()}
    def boot_vs(ref, comps, Emap, cers):
        out = {}; er = (W @ Emap[ref]) / gl_b
        for k in comps:
            d = (W @ Emap[k]) / gl_b - er
            out[k] = dict(delta=float(cers[k] - cers[ref]), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                          p_two_sided=pval(d))
        order = sorted(out, key=lambda k: out[k]["p_two_sided"]); m = len(order); mx = 0
        for i, k in enumerate(order): mx = max(mx, min(1.0, (m - i) * out[k]["p_two_sided"])); out[k]["p_holm"] = mx
        return out
    cers = {k: res[k]["cer"] for k in P}
    boot = {ref: boot_vs(ref, [k for k in P if k not in (ref, "oracle")], E, cers) for ref in dict.fromkeys(["router_text", full])}
    # dev-tuned harm budgets (fractions of always-accept corruption on dev), with paired bootstrap vs the full router
    frontier = {}; a_dev = dv.corrupt(dv.att)
    for frac in [0.10, 0.25, 0.50]:
        bud = int(frac * a_dev); U = {}
        for k, (sd, st) in S.items():
            if k in ("docqe", "errdet_ocr_only"): continue
            U[k] = te.att & (st >= tune(dv, sd, qgrid(sd, dv.att, 201), budget=bud))
        Eb = {k: te.per_doc(u) for k, u in U.items()}; cb = {k: te.cer(u) for k, u in U.items()}
        bt = boot_vs(full, [k for k in U if k != full], Eb, cb)
        frontier[f"{int(frac*100)}pct"] = {k: dict(test_cer=cb[k], test_corrupt=te.corrupt(u), test_budget=int(frac * te.corrupt(te.att)), accept_rate_attempted=float(u[te.att].sum() / natt),
                                                   **({"vs_full": bt[k]} if k in bt else {})) for k, u in U.items()}
    # threshold-free ranking quality on test (descriptive): mean CER along the harm axis, accepting in score order
    rank = {}
    for k, (sd, st) in S.items():
        o = np.argsort(-st[te.att], kind="stable"); de = (te.ec - te.eo)[te.att][o]; hm = (de > 0).cumsum()
        c = (te.eo.sum() + np.r_[0, de.cumsum()]) / te.gl.sum(); h = np.r_[0, hm]; tot = max(1, h[-1])
        rank[k] = float(np.mean([c[h <= b].min() for b in range(tot + 1)]))
    # Learn-then-Test: control doc-macro harmful-acceptance rate at alpha with prob 1-delta; 20 random fit/cal doc splits
    risk = {}; harm_d, harm_t = dv.ec > dv.eo, te.ec > te.eo
    def test_risk(u): return float((np.bincount(te.di, u & harm_t, te.nd) / np.maximum(1, te.nseg)).mean())
    risk["always_test_risk"] = test_risk(te.att)
    Fd, Ft = featsets(dv), featsets(te)
    for alpha in [0.02, 0.05, 0.10]:
        acc = {}
        for rep in range(20):
            r2 = np.random.default_rng(SEED + rep); perm = r2.permutation(dv.nd); fit_docs, cal_docs = perm[: dv.nd // 2], perm[dv.nd // 2:]
            fm = np.isin(dv.di, fit_docs) & dv.att
            cand = {"edit_cap": S["edit_cap"]}
            if dv.has_conf: cand["conf_min"] = S["conf_min"]
            if len(set(dv.y[fm])) > 1 and fm.sum() >= 10:
                m = gb(); m.fit(Fd[full][fm], dv.y[fm]); cand[full] = (m.predict_proba(Fd[full])[:, 1], m.predict_proba(Ft[full])[:, 1])
            for k, (sd, st) in cand.items():
                t = ltt(sd, harm_d, dv.att, dv.di, dv.nd, dv.nseg, cal_docs, alpha); u = te.att & (st >= t)
                acc.setdefault(k, []).append((test_risk(u), te.cer(u), float(u[te.att].sum() / natt)))
        risk[f"alpha{alpha}"] = {k: dict(test_risk_mean=float(np.mean([x[0] for x in v])), violation_frac=float(np.mean([x[0] > alpha for x in v])),
                                         test_cer_mean=float(np.mean([x[1] for x in v])), accept_rate_mean=float(np.mean([x[2] for x in v])),
                                         n_cal_docs=int(dv.nd - dv.nd // 2)) for k, v in acc.items()}
    out = dict(dataset=ds, corrector=corr, n_dev=len(dv.rows), n_test=len(te.rows), n_dev_docs=dv.nd, n_test_docs=te.nd,
               attempted_test=int(te.att.sum()), full_router=full, policies=res, bootstrap=boot, devtuned_budget_frontier=frontier,
               rank_mean_cer_over_harm=rank, risk_control=risk,
               protocol="learned policies cross-fitted (5-fold doc-grouped OOF) on dev; thresholds on dev quantile grid incl. abstain; "
                        "single test eval; micro CER; paired doc bootstrap B=%d, p=2(k+1)/(B+1); Holm; LTT with doc-level Hoeffding-Bentkus, delta=0.10" % B)
    json.dump(out, open(os.path.join(OUT, f"ledger_{corr}_{ds}.json"), "w"), indent=1)
    return out

COLS = ["never", "always", "router_text", "router_align", "router_nodict", "router_conf", "router_conf_qe", "router_util",
        "conf_mean", "conf_min", "kim_min095", "conf_plus_cap", "edit_cap", "errdet_ocr_only", "docqe", "qe_encoder", "oracle"]

if __name__ == "__main__":
    only = sys.argv[1:]; rows = []
    for ds, corr in CONFIGS:
        if only and f"{corr}_{ds}" not in only: continue
        if not os.path.exists(os.path.join(DATA, f"candidates_{corr}_{ds}_dev.jsonl")): print("skip", ds, corr); continue
        o = run(ds, corr); p = o["policies"]
        print(f"| {ds} | {corr} | " + " | ".join(f"{p[k]['cer']:.4f}" if k in p else "—" for k in COLS) + " |", flush=True)
