#!/usr/bin/env python3
"""B7: corrector-seed robustness. Re-runs the iter-2 ledger and the edit-level analysis on candidates from
ByT5-base correctors re-trained with seeds 1 and 2 (seed 42 = the original run) and tabulates, per dataset,
whether the policy ordering is stable across corrector seeds.
Outputs: process/refine-logs/iter2/seeds/*.json, seeds_summary.md
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import iter2_ledger as L, iter2_editlevel as E
from oclib import DATA
FAM = {"byt5base8": ["blnocr", "icdarc_EN", "ext_impresso_en", "ext_overproof_en"], "byt5base8m": ["icdarc_EN", "icdarc_DE", "icdarc_FR", "icdarc_ES", "icdarc_NL"]}
SD = os.path.join(L.OUT, "seeds"); os.makedirs(SD, exist_ok=True)
names = lambda fam: {42: fam, 1: fam + "s1", 2: fam + "s2"}

def one(arg):
    corr, ds = arg
    if not os.path.exists(os.path.join(DATA, f"candidates_{corr}_{ds}_dev.jsonl")): return None
    L.OUT = E.OUT = SD
    a, b = L.run(ds, corr), E.run(ds, corr); return corr, ds

def summary():
    out = ["# B7 — corrector-seed robustness (ByT5-base, seeds 42 / 1 / 2)", "",
           "Test micro CER per corrector seed; Δ columns are paired doc-bootstrap differences (negative = first policy better; * = Holm p<0.05 within that seed).", "",
           "| dataset | family | seed | always | router | edit-cap | conf-min | edit router | Δ router−edit-cap | Δ router−conf-min | Δ edit router−seg router | 25% budget: Δ edit-cap−router |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for fam, dss in FAM.items():
        for ds in dss:
            for seed, corr in names(fam).items():
                pl, pe = os.path.join(SD, f"ledger_{corr}_{ds}.json"), os.path.join(SD, f"editlevel_{corr}_{ds}.json")
                if not (os.path.exists(pl) and os.path.exists(pe)): continue
                a, b = json.load(open(pl)), json.load(open(pe)); p = a["policies"]; fr = a["full_router"]
                def d(x, neg=False):
                    if x is None: return "—", None
                    v = -x["delta"] if neg else x["delta"]; return f"{v:+.4f}" + ("*" if x["p_holm"] < 0.05 else ""), (v, x["p_holm"] < 0.05)
                c1 = d(a["bootstrap"][fr].get("edit_cap"), neg=True); c2 = d(a["bootstrap"][fr].get("conf_min"), neg=True)
                c3 = d(b["bootstrap"]["seg_" + fr].get("edit_router")); c4 = d(a["devtuned_budget_frontier"]["25pct"].get("edit_cap", {}).get("vs_full"))
                out.append(f"| {ds} | {fam} | {seed} | {p['always']['cer']:.4f} | {p[fr]['cer']:.4f} | {p['edit_cap']['cer']:.4f} | {p['conf_min']['cer']:.4f} | "
                           f"{b['policies']['edit_router']['cer']:.4f} | {c1[0]} | {c2[0]} | {c3[0]} | {c4[0]} |")
                for k, c in zip(["router−edit-cap", "router−conf-min", "edit router−seg router", "25% budget edit-cap−router"], [c1, c2, c3, c4]):
                    if c[1]: agg.setdefault(k, {}).setdefault((ds, fam), []).append(c[1])
                agg.setdefault("always", {}).setdefault((ds, fam), []).append(p["always"]["cer"]); agg.setdefault("router", {}).setdefault((ds, fam), []).append(p[fr]["cer"])
    out += ["", "## Stability across corrector seeds (dataset×family cells with all 3 seeds)", ""]
    for k in ["router−edit-cap", "router−conf-min", "edit router−seg router", "25% budget edit-cap−router"]:
        cells = {c: v for c, v in agg.get(k, {}).items() if len(v) == 3}
        same = sum(len({np.sign(x[0]) for x in v}) == 1 for v in cells.values())
        flip = sum(any(x[1] and x[0] < 0 for x in v) and any(x[1] and x[0] > 0 for x in v) for v in cells.values())
        out.append(f"- {k}: same sign in all 3 seeds in {same}/{len(cells)} cells; significant in opposite directions across seeds in {flip}/{len(cells)} cells")
    out += ["", "## Harm-budget stability (baseline − router at the dev-tuned budget; positive = router better)", "",
            "| baseline | budget | same sign in all 3 seeds | router better in all 3 | router sig. better in all 3 | baseline better in all 3 | sig. in opposite directions |", "|---|---|---|---|---|---|---|"]
    for base in ["edit_cap", "conf_min"]:
        for fz in ["10pct", "25pct", "50pct"]:
            cells = {}
            for fam, dss in FAM.items():
                for ds in dss:
                    for seed, corr in names(fam).items():
                        pl = os.path.join(SD, f"ledger_{corr}_{ds}.json")
                        if not os.path.exists(pl): continue
                        x = json.load(open(pl))["devtuned_budget_frontier"][fz].get(base, {}).get("vs_full")
                        if x: cells.setdefault((ds, fam), []).append((x["delta"], x["p_holm"] < 0.05))
            cells = {c: v for c, v in cells.items() if len(v) == 3}; n = len(cells)
            sg = lambda v: {np.sign(x[0]) for x in v}
            out.append(f"| {base} | {fz} | {sum(len(sg(v)) == 1 for v in cells.values())}/{n} | {sum(all(x[0] > 0 for x in v) for v in cells.values())}/{n} | "
                       f"{sum(all(x[0] > 0 and x[1] for x in v) for v in cells.values())}/{n} | {sum(all(x[0] < 0 for x in v) for v in cells.values())}/{n} | "
                       f"{sum(any(x[1] and x[0] < 0 for x in v) and any(x[1] and x[0] > 0 for x in v) for v in cells.values())}/{n} |")
    out.append("")
    for k in ["always", "router"]:
        cells = [v for v in agg.get(k, {}).values() if len(v) == 3]
        if cells: out.append(f"- {k} CER: mean across-seed std {np.mean([np.std(v, ddof=1) for v in cells]):.4f} (max {np.max([np.std(v, ddof=1) for v in cells]):.4f})")
    open(os.path.join(L.OUT, "seeds_summary.md"), "w").write("\n".join(out) + "\n")

if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]: summary(); sys.exit()
    from multiprocessing import Pool
    jobs = [(c, ds) for fam, dss in FAM.items() for ds in dss for c in names(fam).values()]
    with Pool(9) as pool:
        for r in pool.imap_unordered(one, jobs): print(r, flush=True)
    L.OUT = os.path.dirname(SD); summary()
