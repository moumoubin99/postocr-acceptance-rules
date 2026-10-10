#!/usr/bin/env python3
"""B6: second-pass LLM verifier as an acceptance rule (CPU analysis of verif_* files; iter-2 ledger protocol).

Policies: `verifier` (accept iff v >= dev-calibrated threshold), `verifier_sign` (fixed, untuned: accept iff v > 0),
`router+ver` (full router features + v, cross-fitted), next to the segment-level full router, edit-cap and conf-min.
Outputs: process/refine-logs/iter2/verifier_{corr}_{ds}.json, verifier_summary.md
"""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.metrics import roc_auc_score
from oclib import DATA, load_jsonl
from iter2_ledger import OUT, SEED, B, pval, Split, scores, featsets, crossfit, gb, tune, qgrid

def vload(S, corr, ds, split):
    V = load_jsonl(os.path.join(DATA, f"verif_{corr}_{ds}_{split}.jsonl"))
    assert len(V) == len(S.rows) and all(a["doc_id"] == r["doc_id"] and a["seg_idx"] == r["seg_idx"] and bool(a["att"]) == bool(t)
                                         for a, r, t in zip(V, S.rows, S.att))
    v = np.array([x["v"] if x["att"] else 0.0 for x in V], float)
    return v, int(sum(1 for x in V if x["att"] and x.get("missing")))

def run(corr, ds):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test"); (vd, md), (vt, mt) = vload(dv, corr, ds, "dev"), vload(te, corr, ds, "test")
    S, full = scores(dv, te); Fd, Ft = featsets(dv), featsets(te)
    S = {k: S[k] for k in (full, "edit_cap", "conf_min") if k in S}
    S["verifier"] = (vd, vt); S["router+ver"] = crossfit(gb, np.c_[Fd[full], vd], dv.y, dv.att, dv.di, np.c_[Ft[full], vt])
    P = {"never": np.zeros(len(te.rows), bool), "always": te.att.copy(), "oracle": te.ec < te.eo, "verifier_sign": te.att & (vt > 0)}
    info = {}
    for k, (sd, st) in S.items():
        t = tune(dv, sd); P[k] = te.att & (st >= t); info[k] = dict(thr=float(t))
        if len(set(te.y[te.att])) > 1: info[k]["auc_test"] = float(roc_auc_score(te.y[te.att], st[te.att]))
    natt = max(1, int(te.att.sum()))
    res = {k: dict(cer=te.cer(u), corrupt=te.corrupt(u), accept_rate_attempted=float(u[te.att].sum() / natt), **info.get(k, {})) for k, u in P.items()}
    rng = np.random.default_rng(SEED); W = rng.multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gl_b = W @ te.G
    def boot_vs(ref, comps, U):
        out = {}; er = (W @ te.per_doc(U[ref])) / gl_b
        for k in comps:
            d = (W @ te.per_doc(U[k])) / gl_b - er
            out[k] = dict(delta=float(te.cer(U[k]) - te.cer(U[ref])), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                          p_two_sided=pval(d))
        order = sorted(out, key=lambda k: out[k]["p_two_sided"]); m = len(order); mx = 0
        for i, k in enumerate(order): mx = max(mx, min(1.0, (m - i) * out[k]["p_two_sided"])); out[k]["p_holm"] = mx
        return out
    comps = [k for k in P if k not in ("oracle",)]
    boot = {r: boot_vs(r, [k for k in comps if k != r], P) for r in ("verifier", full)}
    frontier = {}; a_dev = dv.corrupt(dv.att)
    for frac in [0.10, 0.25, 0.50]:
        bud = int(frac * a_dev); U = {k: te.att & (st >= tune(dv, sd, qgrid(sd, dv.att, 201), budget=bud)) for k, (sd, st) in S.items()}
        bt = boot_vs("verifier", [k for k in U if k != "verifier"], U)
        frontier[f"{int(frac*100)}pct"] = {k: dict(test_cer=te.cer(u), test_corrupt=te.corrupt(u), **({"vs_verifier": bt[k]} if k in bt else {})) for k, u in U.items()}
    lat = {}
    mp = os.path.join(DATA, f"verif_{corr}_{ds}_metrics.json")
    if os.path.exists(mp): lat = json.load(open(mp))
    out = dict(dataset=ds, corrector=corr, full_router=full, n_test=len(te.rows), attempted_test=int(te.att.sum()), missing_dev=md, missing_test=mt,
               policies=res, bootstrap=boot, devtuned_budget_frontier=frontier, verifier_metrics=lat)
    json.dump(out, open(os.path.join(OUT, f"verifier_{corr}_{ds}.json"), "w"), indent=1); return out

def tally(R, get):
    w = l = n = 0
    for o in R:
        b = get(o)
        if b is None: continue
        if b["p_holm"] < 0.05 and b["delta"] < 0: w += 1
        elif b["p_holm"] < 0.05 and b["delta"] > 0: l += 1
        else: n += 1
    return f"{w} / {l} / {n}"

def summary(R):
    L = ["# B6 — second-pass Qwen2.5-7B verifier as an acceptance rule", "",
         "Test micro CER. `verifier`: accept iff v ≥ dev-calibrated threshold (v = two-order averaged log-odds that the candidate is the better transcription); "
         "`sign`: untuned v > 0; `router+ver`: full router with v as an extra feature.", "",
         "| dataset | corrector | never | always | router | edit-cap | conf-min | verifier | sign | router+ver | oracle | AUC router | AUC verifier | AUC router+ver | ms / attempted seg |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in R:
        p = o["policies"]; f = lambda k: f"{p[k]['cer']:.4f}" if k in p else "—"; a = lambda k: f"{p[k]['auc_test']:.3f}" if k in p and "auc_test" in p[k] else "—"
        ms = (o["verifier_metrics"].get("test") or o["verifier_metrics"]).get("ms_per_attempted_segment"); ms = f"{ms:.1f}" if isinstance(ms, (int, float)) else "—"
        L.append(f"| {o['dataset']} | {o['corrector']} | " + " | ".join(f(k) for k in ["never", "always", o["full_router"], "edit_cap", "conf_min", "verifier", "verifier_sign", "router+ver", "oracle"])
                 + f" | {a(o['full_router'])} | {a('verifier')} | {a('router+ver')} | {ms} |")
    L += ["", "## Significant wins / losses / n.s. (Holm p<0.05, paired doc bootstrap)", "", "| comparison | CER-optimal | 10% budget | 25% budget | 50% budget |", "|---|---|---|---|---|"]
    def row(lab, a, b):   # a vs b, delta = a - b
        def g0(o):
            ka, kb = (o["full_router"] if x == "full" else x for x in (a, b))
            if kb in o["bootstrap"]: return o["bootstrap"][kb].get(ka)
            x = o["bootstrap"].get(ka, {}).get(kb); return None if x is None else dict(p_holm=x["p_holm"], delta=-x["delta"])
        cells = [tally(R, g0)]
        for fz in ["10pct", "25pct", "50pct"]:
            def g(o):
                ka = o["full_router"] if a == "full" else a
                if b != "verifier": return None
                return o["devtuned_budget_frontier"][fz].get(ka, {}).get("vs_verifier")
            cells.append(tally(R, g) if b == "verifier" else "—")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    row("router vs verifier", "full", "verifier"); row("edit-cap vs verifier", "edit_cap", "verifier"); row("conf-min vs verifier", "conf_min", "verifier")
    row("router+ver vs verifier", "router+ver", "verifier"); row("router+ver vs router", "router+ver", "full"); row("verifier sign vs router", "verifier_sign", "full")
    L += ["", f"- rows with a missing A/B label in the top-20 log-probs (attempted, dev+test): {sum(o['missing_dev'] + o['missing_test'] for o in R)}"]
    open(os.path.join(OUT, "verifier_summary.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    R = []
    for p in sorted(glob.glob(os.path.join(DATA, "verif_*_dev.jsonl"))):
        name = os.path.basename(p)[len("verif_"):-len("_dev.jsonl")]; corr, ds = name.split("_", 1)
        if not os.path.exists(os.path.join(DATA, f"verif_{name}_test.jsonl")): continue
        o = run(corr, ds); R.append(o); p_ = o["policies"]; print(ds, corr, " ".join(f"{k}={v['cer']:.4f}" for k, v in p_.items()), flush=True)
    summary(R)
