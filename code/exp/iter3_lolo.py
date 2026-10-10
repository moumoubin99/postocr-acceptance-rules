#!/usr/bin/env python3
"""Iteration-3: leave-one-language-out transfer on clean ICDAR (EN, DE, FR, ES, NL).
For each held-out language L: every policy is fitted and its threshold chosen on the pooled dev sets of the other four
languages (zero labels in L), then evaluated once on L's test set; the in-language protocol (fit + threshold on L dev)
is the reference. Correctors: byt5base8m (per-language candidate files) and qwen (icdarc_multi split by `lang`).
Paired doc bootstrap (B=10000) on the target test set; Holm across the 10 corrector x language cells per comparison.
CPU only, frozen candidates. Output: process/refine-logs/iter3/lolo.json, lolo.md"""
import os, sys, json, copy, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:1]; sys.path.insert(0, HERE)
from iter2_ledger import Split, scores, tune, qgrid, ROOT, SEED, B, pval
OUT = os.path.join(ROOT, "process", "refine-logs", "iter3"); os.makedirs(OUT, exist_ok=True)
LANGS = ["EN", "DE", "FR", "ES", "NL"]; POL = ["full", "router_nodict", "edit_cap", "conf_min"]
ARR = ("eo", "ec", "gl", "att", "y", "X", "A", "C", "q")

def view(S, idx, docs):
    T = copy.copy(S)
    for a in ARR:
        v = getattr(S, a); setattr(T, a, None if v is None else v[idx])
    T.rows = [S.rows[i] for i in idx]; T.dmg = [S.dmg[i] for i in idx]
    remap = {d: i for i, d in enumerate(docs)}; T.di = np.array([remap[S.rows[i]["doc_id"]] for i in idx]); T.nd = len(docs)
    T.docs = docs; T.G = np.bincount(T.di, T.gl, T.nd); T.nseg = np.bincount(T.di, minlength=T.nd)
    return T

def by_lang(S):
    out = {}
    for L in LANGS:
        idx = np.array([i for i, r in enumerate(S.rows) if r.get("lang") == L]); out[L] = view(S, idx, sorted({S.rows[i]["doc_id"] for i in idx}))
    return out

def pool(parts):
    T = copy.copy(parts[0])
    for a in ARR:
        vs = [getattr(p, a) for p in parts]; setattr(T, a, None if any(v is None for v in vs) else np.concatenate(vs))
    T.rows = sum((p.rows for p in parts), []); T.dmg = sum((p.dmg for p in parts), [])
    off = np.cumsum([0] + [p.nd for p in parts]); T.di = np.concatenate([p.di + o for p, o in zip(parts, off)]); T.nd = int(off[-1])
    T.docs = [f"{k}:{d}" for k, p in enumerate(parts) for d in p.docs]; T.has_conf = all(p.has_conf for p in parts)
    T.G = np.bincount(T.di, T.gl, T.nd); T.nseg = np.bincount(T.di, minlength=T.nd)
    return T

def policies(dv, te):
    S, full = scores(dv, te); U = {"never": np.zeros(len(te.rows), bool), "always": te.att.copy(), "oracle": te.ec < te.eo}
    a_dev = dv.corrupt(dv.att)
    for k in POL:
        key = full if k == "full" else k
        if key not in S: continue
        sd, st = S[key]; U[k] = te.att & (st >= tune(dv, sd))
        U[k + "@25pct"] = te.att & (st >= tune(dv, sd, qgrid(sd, dv.att, 201), budget=int(0.25 * a_dev)))
    return U

def load(corr):
    if corr == "qwen": return by_lang(Split("qwen", "icdarc_multi", "dev")), by_lang(Split("qwen", "icdarc_multi", "test"))
    return ({L: Split(corr, f"icdarc_{L}", "dev") for L in LANGS}, {L: Split(corr, f"icdarc_{L}", "test") for L in LANGS})

COMP = [("full", "never"), ("full", "always"), ("full", "edit_cap"), ("full", "conf_min"), ("full@25pct", "edit_cap@25pct"), ("full", "in:full"), ("router_nodict", "in:router_nodict")]
res = []
for corr in ["byt5base8m", "qwen"]:
    dev, test = load(corr)
    for L in LANGS:
        te = test[L]; src = pool([dev[x] for x in LANGS if x != L])
        U = policies(src, te); U.update({"in:" + k: v for k, v in policies(dev[L], te).items() if k not in ("never", "always", "oracle")})
        W = np.random.default_rng(SEED).multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gb_ = W @ te.G
        E = {k: te.per_doc(u) for k, u in U.items()}
        r = dict(corrector=corr, lang=L, n_test=len(te.rows), n_test_docs=te.nd, n_src_dev=len(src.rows), n_in_dev=len(dev[L].rows),
                 cer={k: te.cer(u) for k, u in U.items()}, corrupt={k: te.corrupt(u) for k, u in U.items()}, comp={})
        for a, b in COMP:
            if a not in U or b not in U: continue
            d = (W @ (E[a] - E[b])) / gb_
            r["comp"][f"{a} vs {b}"] = dict(delta=r["cer"][a] - r["cer"][b], ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                                            p=pval(d))
        res.append(r); print(corr, L, {k: round(v, 4) for k, v in r["cer"].items()}, flush=True)
for a, b in COMP:   # Holm across cells
    key = f"{a} vs {b}"; cells = [r for r in res if key in r["comp"]]; order = sorted(cells, key=lambda r: r["comp"][key]["p"]); mx = 0
    for i, r in enumerate(order): mx = max(mx, min(1.0, (len(order) - i) * r["comp"][key]["p"])); r["comp"][key]["p_holm"] = mx
def tally(key):
    w = l = t = 0
    for r in res:
        c = r["comp"].get(key)
        if c is None: continue
        if c["p_holm"] < 0.05 and c["delta"] < 0: w += 1
        elif c["p_holm"] < 0.05 and c["delta"] > 0: l += 1
        else: t += 1
    return f"{w}/{l}/{t}"
json.dump(dict(cells=res, tally={f"{a} vs {b}": tally(f"{a} vs {b}") for a, b in COMP},
               protocol="leave-one-language-out: fit + dev threshold on the pooled dev of the other four languages; in:* = fit + threshold on the "
                        "held-out language's own dev; single test eval; paired doc bootstrap B=10000; Holm across cells per comparison"),
          open(os.path.join(OUT, "lolo.json"), "w"), indent=1)
f = lambda x: "—" if x is None else f"{x:.4f}"
cols = ["never", "always", "full", "in:full", "router_nodict", "in:router_nodict", "edit_cap", "in:edit_cap", "conf_min", "in:conf_min", "oracle"]
md = ["# Iteration-3: leave-one-language-out transfer (clean ICDAR)\n",
      "Test CER on the held-out language. `full`/`router_nodict`/`edit_cap`/`conf_min`: fitted and thresholded on the other four languages' dev. `in:*`: same policy with the held-out language's own dev.\n",
      "| corr | held-out | test docs | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
for r in res: md.append(f"| {r['corrector']} | {r['lang']} | {r['n_test_docs']} | " + " | ".join(f(r["cer"].get(c)) for c in cols) + " |")
md += ["\n## Tallies (lower CER, Holm < 0.05 across the 10 cells: first better / second better / n.s.)\n", "| comparison | tally |", "|---|---|"]
md += [f"| {a} vs {b} | {tally(f'{a} vs {b}')} |" for a, b in COMP]
md += ["\n## Harmful acceptances at the 25% dev budget (test corruptions; always-accept in parentheses)\n", "| corr | held-out | full@25pct | edit_cap@25pct | in:full@25pct | always |", "|---|---|---|---|---|---|"]
for r in res: md.append(f"| {r['corrector']} | {r['lang']} | {r['corrupt'].get('full@25pct')} | {r['corrupt'].get('edit_cap@25pct')} | {r['corrupt'].get('in:full@25pct')} | {r['corrupt']['always']} |")
open(os.path.join(OUT, "lolo.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
