#!/usr/bin/env python3
"""Iteration-5 extras (CPU only, frozen candidates).
  lc   <corr>_<ds> : dev-document learning curve at the CER-optimal point and at the 25% harm budget
  rank <corr>_<ds> : threshold-reachable CER-vs-harm summary (ties grouped) with a paired document bootstrap
Output: $OUT/extra5/{lc,rank}_<corr>_<ds>.json"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]; sys.path.insert(0, HERE)
from iter2_ledger import Split, scores, tune, qgrid, featsets, crossfit, gb, pval, CONFIGS, OUT, SEED, B
import copy
def subset(S, docs):
    """view of a Split restricted to a set of doc indices (re-indexed); same as iter2_transfer.subset"""
    m = np.isin(S.di, docs); T = copy.copy(S); idx = np.where(m)[0]
    for a in ("eo", "ec", "gl", "att", "y", "X", "A", "C", "q"):
        v = getattr(S, a); setattr(T, a, None if v is None else v[idx])
    T.rows = [S.rows[i] for i in idx]; T.dmg = [S.dmg[i] for i in idx]
    remap = {d: i for i, d in enumerate(sorted(docs))}; T.di = np.array([remap[d] for d in S.di[idx]]); T.nd = len(remap)
    T.docs = [S.docs[d] for d in sorted(docs)]; T.G = np.bincount(T.di, T.gl, T.nd); T.nseg = np.bincount(T.di, minlength=T.nd)
    return T
OUT5 = os.path.join(OUT, "extra5"); os.makedirs(OUT5, exist_ok=True)
REPS = int(os.environ.get("LC_REPS", 20)); SIZES = [5, 10, 20, 40, 80]

def light_scores(dv, te):
    Fd, Ft = featsets(dv), featsets(te); full = "router_conf" if dv.has_conf else "router_align"
    S = {"full": crossfit(gb, Fd[full], dv.y, dv.att, dv.di, Ft[full]),
         "router_text": crossfit(gb, Fd["router_text"], dv.y, dv.att, dv.di, Ft["router_text"]),
         "edit_cap": (-dv.X[:, 6], -te.X[:, 6])}
    if dv.has_conf: S["conf_min"] = (dv.C[:, 1], te.C[:, 1])
    return S

def lc_eval(dv, te):
    S = light_scores(dv, te); bud = int(0.25 * dv.corrupt(dv.att)); out = {}
    for k, (sd, st) in S.items():
        u = te.att & (st >= tune(dv, sd)); ub = te.att & (st >= tune(dv, sd, qgrid(sd, dv.att, 201), budget=bud))
        out[k] = dict(opt_cer=te.cer(u), opt_harm=te.corrupt(u), b25_cer=te.cer(ub), b25_harm=te.corrupt(ub))
    return out

def run_lc(corr, ds):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test"); res = []
    base = dict(never=te.cer(np.zeros(len(te.rows), bool)), always=te.cer(te.att), always_harm=te.corrupt(te.att))
    for k in [s for s in SIZES if s < dv.nd] + [dv.nd]:
        runs = []
        for rep in range(1 if k == dv.nd else REPS):
            docs = np.random.default_rng(SEED + rep).choice(dv.nd, k, replace=False)
            try: runs.append(lc_eval(subset(dv, docs), te))
            except Exception as e: print("skip", corr, ds, k, rep, repr(e)[:120], flush=True)
        if not runs: continue
        agg = {p: {m: dict(mean=float(np.mean([x[p][m] for x in runs])), sd=float(np.std([x[p][m] for x in runs])))
                   for m in runs[0][p]} for p in runs[0] if all(p in x for x in runs)}
        res.append(dict(dev_docs=int(k), reps=len(runs), **agg))
        print("lc", ds, corr, k, len(runs), {p: (round(v["opt_cer"]["mean"], 4), round(v["b25_cer"]["mean"], 4)) for p, v in agg.items()}, flush=True)
    json.dump(dict(dataset=ds, corrector=corr, n_dev_docs=int(dv.nd), **base, curve=res),
              open(os.path.join(OUT5, f"lc_{corr}_{ds}.json"), "w"), indent=1)

def run_rank(corr, ds):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test"); S, full = scores(dv, te)
    rules = {"router": full, "router_text": "router_text", "edit_cap": "edit_cap"}
    if "conf_min" in S: rules["conf_min"] = "conf_min"
    att = te.att; de = (te.ec - te.eo)[att].astype(float); di = te.di[att]; P = {}
    for name, key in rules.items():
        st = S[key][1][att]; o = np.argsort(-st, kind="stable"); s_o = st[o]
        P[name] = (o, np.r_[s_o[1:] != s_o[:-1], True])       # order, end-of-tie-group mask
    def stat(w):                                               # w: per-document weights
        ws = w[di]; num0 = float((w[te.di] * te.eo).sum()); den = float((w[te.di] * te.gl).sum()); out = {}
        for name, (o, ge) in P.items():
            d = (ws * de)[o]; c = np.r_[num0, num0 + d.cumsum()[ge]] / den
            h = np.r_[0, (ws[o] * (de[o] > 0)).cumsum()[ge]]; tot = max(1.0, h[-1])
            out[name] = float((np.minimum.accumulate(c) * np.diff(np.r_[h, tot + 1])).sum() / (tot + 1))
        return out
    point = stat(np.ones(te.nd)); rng = np.random.default_rng(SEED)
    W = rng.multinomial(te.nd, np.full(te.nd, 1.0 / te.nd), size=B).astype(float)
    bs = [stat(w) for w in W]; res = dict(dataset=ds, corrector=corr, B=B, point=point, vs={})
    for a, b in [("router", "edit_cap"), ("router", "conf_min"), ("router_text", "edit_cap")]:
        if a not in point or b not in point: continue
        d = np.array([x[a] - x[b] for x in bs])
        res["vs"][f"{a}|{b}"] = dict(delta=point[a] - point[b], ci=[float(np.quantile(d, .025)), float(np.quantile(d, .975))], p_two_sided=pval(d))
    print("rank", ds, corr, {k: round(v, 4) for k, v in point.items()}, {k: (round(v["delta"], 4), v["p_two_sided"]) for k, v in res["vs"].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT5, f"rank_{corr}_{ds}.json"), "w"), indent=1)

if __name__ == "__main__":
    mode, name = ARGS
    for ds, corr in CONFIGS:
        if f"{corr}_{ds}" == name: (run_lc if mode == "lc" else run_rank)(corr, ds)
