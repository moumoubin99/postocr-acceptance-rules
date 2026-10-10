#!/usr/bin/env python3
"""Iteration-2 A9: (a) cross-dataset policy transfer — fit + threshold on dataset A dev, single eval on dataset B test
(same corrector family, zero target-domain labels); (b) dev-document learning curve (how many labelled dev
documents each policy needs). CPU only, frozen candidates. Output: process/refine-logs/iter2/transfer.json, transfer.md"""
import os, sys, json, copy, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:1]; sys.path.insert(0, HERE)
from iter2_ledger import Split, scores, tune, OUT, SEED

POL = ["full", "router_text", "router_nodict", "edit_cap", "conf_min"]
PAIRS = [("blnocr", "icdarc_EN"), ("icdarc_EN", "blnocr"), ("blnocr", "ext_impresso_en"), ("blnocr", "ext_overproof_en"),
         ("icdarc_EN", "ext_overproof_en")]
CORRS = ["byt5", "byt5base8", "qwen"]
_cache = {}
def split(corr, ds, sp):
    k = (corr, ds, sp)
    if k not in _cache: _cache[k] = Split(corr, ds, sp)
    return _cache[k]

def subset(S, docs):
    """view of a Split restricted to a set of doc indices (re-indexed)"""
    m = np.isin(S.di, docs); T = copy.copy(S); idx = np.where(m)[0]
    for a in ("eo", "ec", "gl", "att", "y", "X", "A", "C", "q"):
        v = getattr(S, a); setattr(T, a, None if v is None else v[idx])
    T.rows = [S.rows[i] for i in idx]; T.dmg = [S.dmg[i] for i in idx]
    remap = {d: i for i, d in enumerate(sorted(docs))}; T.di = np.array([remap[d] for d in S.di[idx]]); T.nd = len(remap)
    T.docs = [S.docs[d] for d in sorted(docs)]; T.G = np.bincount(T.di, T.gl, T.nd); T.nseg = np.bincount(T.di, minlength=T.nd)
    return T

def evaluate(dv, te):
    S, full = scores(dv, te); out = {"never": te.cer(np.zeros(len(te.rows), bool)), "always": te.cer(te.att), "oracle": te.cer(te.ec < te.eo)}
    for k in POL:
        key = full if k == "full" else k
        if key not in S: continue
        sd, st = S[key]; u = te.att & (st >= tune(dv, sd)); out[k] = te.cer(u); out[k + "_corrupt"] = te.corrupt(u)
    return out

res = {"transfer": [], "learning_curve": []}
for corr in CORRS:
    for a, b in PAIRS:
        try: dv, dvb, te = split(corr, a, "dev"), split(corr, b, "dev"), split(corr, b, "test")
        except FileNotFoundError: continue
        r = dict(corrector=corr, source=a, target=b, transfer=evaluate(dv, te), in_domain=evaluate(dvb, te))
        res["transfer"].append(r); print("transfer", corr, a, "->", b, {k: round(v, 4) for k, v in r["transfer"].items() if not k.endswith("corrupt")}, flush=True)
for ds, corr in [("blnocr", "byt5base8"), ("blnocr", "qwen"), ("icdarc_DE", "byt5base8m"), ("icdarc_multi", "qwen")]:
    dv, te = split(corr, ds, "dev"), split(corr, ds, "test")
    for k in [5, 10, 20, 40, dv.nd]:
        if k > dv.nd: continue
        runs = []
        for rep in range(1 if k == dv.nd else 10):
            docs = np.random.default_rng(SEED + rep).choice(dv.nd, k, replace=False); runs.append(evaluate(subset(dv, docs), te))
        agg = {p: dict(mean=float(np.mean([x[p] for x in runs])), sd=float(np.std([x[p] for x in runs]))) for p in POL if p in runs[0]}
        res["learning_curve"].append(dict(dataset=ds, corrector=corr, dev_docs=int(k), reps=len(runs), never=runs[0]["never"], always=runs[0]["always"], **agg))
        print("lc", ds, corr, k, {p: round(v["mean"], 4) for p, v in agg.items()}, flush=True)
json.dump(res, open(os.path.join(OUT, "transfer.json"), "w"), indent=1)
f = lambda x: "—" if x is None else f"{x:.4f}"
md = ["# Iteration-2 A9: policy transfer and dev-size learning curve\n", "## Transfer (test CER on target; fit+threshold on source dev | in-domain dev in parentheses)\n",
      "| corr | source → target | never | always | " + " | ".join(POL) + " | oracle |", "|" + "---|" * (5 + len(POL))]
for r in res["transfer"]:
    t, i = r["transfer"], r["in_domain"]
    md.append(f"| {r['corrector']} | {r['source']} → {r['target']} | {f(t['never'])} | {f(t['always'])} | " +
              " | ".join(f"{f(t.get(p))} ({f(i.get(p))})" for p in POL) + f" | {f(t['oracle'])} |")
md += ["\n## Learning curve (test CER, mean ± sd over 10 random dev-document subsets)\n",
       "| ds | corr | dev docs | never | always | " + " | ".join(POL) + " |", "|" + "---|" * (5 + len(POL))]
for r in res["learning_curve"]:
    md.append(f"| {r['dataset']} | {r['corrector']} | {r['dev_docs']} | {f(r['never'])} | {f(r['always'])} | " +
              " | ".join(f"{r[p]['mean']:.4f}±{r[p]['sd']:.4f}" if p in r else "—" for p in POL) + " |")
open(os.path.join(OUT, "transfer.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
