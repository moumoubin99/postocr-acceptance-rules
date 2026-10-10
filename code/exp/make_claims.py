#!/usr/bin/env python3
"""A3: build CLAIM_MANIFEST.json from the iter-2 ledgers. Every number a paper claim may cite is derived here from
process/refine-logs/iter2/ledger_*.json (+ transfer.json); nothing is typed by hand. Each claim carries its source files."""
import os, sys, json, glob, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:1]; sys.path.insert(0, HERE)
from iter2_ledger import CONFIGS, OUT
L = {}
for ds, corr in CONFIGS:
    p = os.path.join(OUT, f"ledger_{corr}_{ds}.json")
    if os.path.exists(p): L[f"{corr}@{ds}"] = json.load(open(p))
NEURAL = [k for k, o in L.items() if o["full_router"] == "router_conf"]
def tally(pairs):
    """pairs: {cfg: (delta, p_holm)} with delta = baseline - router; -> counts"""
    w = [c for c, (d, p) in pairs.items() if p < 0.05 and d > 0]; l = [c for c, (d, p) in pairs.items() if p < 0.05 and d < 0]
    return dict(n=len(pairs), router_sig_better=len(w), router_sig_worse=len(l), not_significant=len(pairs) - len(w) - len(l),
                better_cfgs=w, worse_cfgs=l, mean_delta=sum(d for d, _ in pairs.values()) / max(1, len(pairs)))
C = {}
C["always_accept_vs_never"] = {k: dict(never=o["policies"]["never"]["cer"], always=o["policies"]["always"]["cer"],
                                       always_hurts=o["policies"]["always"]["cer"] > o["policies"]["never"]["cer"]) for k, o in L.items()}
for base in ["edit_cap", "conf_min", "conf_plus_cap", "conf_mean", "kim_min095", "always", "never", "router_text", "router_nodict", "qe_encoder"]:
    C[f"ceropt_full_router_vs_{base}"] = tally({k: (o["bootstrap"][o["full_router"]][base]["delta"], o["bootstrap"][o["full_router"]][base]["p_holm"])
                                                for k, o in L.items() if base in o["bootstrap"][o["full_router"]]})
for b in ["10pct", "25pct", "50pct"]:
    for base in ["edit_cap", "conf_min", "router_text", "router_nodict"]:
        C[f"budget{b}_full_router_vs_{base}"] = tally({k: (o["devtuned_budget_frontier"][b][base]["vs_full"]["delta"], o["devtuned_budget_frontier"][b][base]["vs_full"]["p_holm"])
                                                       for k, o in L.items() if "vs_full" in o["devtuned_budget_frontier"][b].get(base, {})})
    C[f"budget{b}_realised_over_budget"] = {k: {p: v["test_corrupt"] - v["test_budget"] for p, v in o["devtuned_budget_frontier"][b].items() if v["test_corrupt"] > v["test_budget"]} for k, o in L.items()}
for base in ["edit_cap", "conf_min"]:
    r = {k: (o["rank_mean_cer_over_harm"][base] - o["rank_mean_cer_over_harm"][o["full_router"]]) for k, o in L.items() if base in o["rank_mean_cer_over_harm"]}
    C[f"rank_full_router_vs_{base}"] = dict(n=len(r), router_better=sum(v > 1e-9 for v in r.values()), router_worse=sum(v < -1e-9 for v in r.values()), deltas=r)
ltt = {}
for a in ["alpha0.02", "alpha0.05", "alpha0.1"]:
    for pol in ["full", "conf_min", "edit_cap"]:
        v = [o["risk_control"][a].get(o["full_router"] if pol == "full" else pol) for o in L.values()]; v = [x for x in v if x]
        ltt[f"{a}_{pol}"] = dict(n_cfg=len(v), max_violation_frac=max(x["violation_frac"] for x in v), mean_test_risk=sum(x["test_risk_mean"] for x in v) / len(v),
                                 cfgs_abstaining_entirely=sum(x["accept_rate_mean"] == 0 for x in v))
C["ltt_risk_control"] = ltt
C["doc_vs_segment_cer"] = {k: {p: dict(seg=o["policies"][p]["cer"], doc=o["policies"][p]["doc_cer"]) for p in ["never", "always", o["full_router"], "edit_cap"] if p in o["policies"]} for k, o in L.items()}
C["damage_proxies"] = {k: {p: {m: o["policies"][p][m] for m in ["num_broken", "num_fixed", "cap_broken", "cap_fixed"]} for p in ["always", o["full_router"], "edit_cap", "conf_min"] if p in o["policies"]} for k, o in L.items()}
tp = os.path.join(OUT, "transfer.json")
if os.path.exists(tp):
    T = json.load(open(tp)); C["transfer"] = T["transfer"]; C["learning_curve"] = T["learning_curve"]
# iter-2 follow-ups: B8 edit-level acceptance, B6 LLM verifier, B7 corrector seeds
def wl(bs):
    bs = [b for b in bs if b]
    return dict(n=len(bs), sig_better=sum(b["p_holm"] < 0.05 and b["delta"] < 0 for b in bs), sig_worse=sum(b["p_holm"] < 0.05 and b["delta"] > 0 for b in bs),
                not_significant=sum(b["p_holm"] >= 0.05 for b in bs), mean_delta=sum(b["delta"] for b in bs) / max(1, len(bs)))
EL = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "editlevel_*.json")))]
if EL:
    for pol in ["edit_router", "cascade", "edit_size", "edit_lp"]:
        for ref in ["router", "edit_cap"]:
            rk = lambda o: "seg_" + (o["full_router"] if ref == "router" else "edit_cap")
            C[f"B8_{pol}_vs_seg_{ref}"] = wl([o["bootstrap"][rk(o)].get(pol) for o in EL])
            if pol != "cascade":
                for b in ["10pct", "25pct", "50pct"]:
                    C[f"B8_budget{b}_{pol}_vs_seg_{ref}"] = wl([o["devtuned_budget_frontier"][b].get(pol, {}).get("vs_" + rk(o)) for o in EL])
    mean = lambda k: sum(o["policies"][k(o)]["cer"] for o in EL) / len(EL)
    C["B8_mean_cer"] = dict(seg_router=mean(lambda o: "seg_" + o["full_router"]), seg_edit_cap=mean(lambda o: "seg_edit_cap"), edit_router=mean(lambda o: "edit_router"),
                            cascade=mean(lambda o: "cascade"), oracle_seg=mean(lambda o: "oracle_seg"), oracle_edit=mean(lambda o: "oracle_edit"),
                            mixed_segments=sum(o["diagnostics"]["mixed_segments"] for o in EL) / len(EL))
VL = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "verifier_*.json")))]
if VL:
    neg = lambda b: None if b is None else dict(p_holm=b["p_holm"], delta=-b["delta"])
    C["B6_router_vs_verifier"] = wl([o["bootstrap"]["verifier"].get(o["full_router"]) for o in VL])
    C["B6_router_plus_verifier_vs_router"] = wl([o["bootstrap"][o["full_router"]].get("router+ver") for o in VL])
    C["B6_edit_cap_vs_verifier"] = wl([o["bootstrap"]["verifier"].get("edit_cap") for o in VL])
    for b in ["10pct", "25pct", "50pct"]:
        C[f"B6_budget{b}_router_vs_verifier"] = wl([o["devtuned_budget_frontier"][b].get(o["full_router"], {}).get("vs_verifier") for o in VL])
        C[f"B6_budget{b}_edit_cap_vs_verifier"] = wl([o["devtuned_budget_frontier"][b].get("edit_cap", {}).get("vs_verifier") for o in VL])
    C["B6_verifier_stats"] = {f"{o['corrector']}_{o['dataset']}": dict(auc_verifier=o["policies"]["verifier"].get("auc_test"), auc_router=o["policies"][o["full_router"]].get("auc_test"),
                              ms_per_attempted_segment=(o["verifier_metrics"].get("test") or {}).get("ms_per_attempted_segment"),
                              order_disagree_frac=(lambda m: m.get("order_disagree", 0) / max(1, m.get("attempted", 1)))(o["verifier_metrics"].get("test") or {}),
                              missing=o["missing_dev"] + o["missing_test"]) for o in VL}
SDIR = os.path.join(OUT, "seeds")
if os.path.isdir(SDIR):
    cells = {}
    for p in sorted(glob.glob(os.path.join(SDIR, "ledger_*.json"))):
        a = json.load(open(p)); e = json.load(open(p.replace("ledger_", "editlevel_"))); fr = a["full_router"]; c = a["corrector"]
        fam = c[:-2] if c[-2:] in ("s1", "s2") else c; seed = int(c[-1]) if c != fam else 42
        cells.setdefault(f"{fam}_{a['dataset']}", {})[seed] = dict(
            always=a["policies"]["always"]["cer"], router=a["policies"][fr]["cer"], edit_cap=a["policies"]["edit_cap"]["cer"], edit_router=e["policies"]["edit_router"]["cer"],
            router_minus_edit_cap=neg(a["bootstrap"][fr].get("edit_cap")), router_minus_conf_min=neg(a["bootstrap"][fr].get("conf_min")),
            edit_router_minus_seg_router=e["bootstrap"]["seg_" + fr].get("edit_router"),
            budget25_edit_cap_minus_router=a["devtuned_budget_frontier"]["25pct"].get("edit_cap", {}).get("vs_full"))
    cells = {k: v for k, v in cells.items() if len(v) == 3}
    def stab(key):
        sg = lambda v: [(-1 if x[key]["delta"] < 0 else 1) for x in v.values() if x[key]]
        return dict(cells=len(cells), same_sign_all_seeds=sum(len(set(sg(v))) == 1 for v in cells.values()),
                    sig_opposite=sum(any(x[key]["p_holm"] < 0.05 and x[key]["delta"] < 0 for x in v.values()) and any(x[key]["p_holm"] < 0.05 and x[key]["delta"] > 0 for x in v.values()) for v in cells.values()),
                    sig_same_direction_all_seeds=sum(all(x[key]["p_holm"] < 0.05 for x in v.values()) and len(set(sg(v))) == 1 for v in cells.values()))
    C["B7_seed_stability"] = {k: stab(k) for k in ["router_minus_edit_cap", "router_minus_conf_min", "edit_router_minus_seg_router", "budget25_edit_cap_minus_router"]}
    C["B7_cells"] = cells
RS = os.path.join(OUT, "robust_summary.json")
if os.path.exists(RS):   # second re-review: Holm across configs, corpus pooling, realised harm, matched realised harm, no-lp edit router, LTT minimum n
    r = json.load(open(RS)); r.pop("transfer", None); C["R2_robust"] = r
    tr = os.path.join(OUT, "robust", "transfer_significance.json")
    if os.path.exists(tr): C["R2_transfer_significance"] = json.load(open(tr))
src = sorted(glob.glob(os.path.join(OUT, "ledger_*.json")) + glob.glob(os.path.join(OUT, "editlevel_*.json")) + glob.glob(os.path.join(OUT, "verifier_*.json")) + glob.glob(os.path.join(OUT, "seeds", "*.json")) + glob.glob(os.path.join(OUT, "robust", "*.json")) + glob.glob(os.path.join(OUT, "robust_summary.json"))) + ([tp] if os.path.exists(tp) else [])
man = dict(generated_by="exp/make_claims.py", neural_configs=NEURAL, n_configs=len(L),
           sources={os.path.relpath(p, os.path.dirname(HERE)): hashlib.md5(open(p, "rb").read()).hexdigest() for p in src}, claims=C)
json.dump(man, open(os.path.join(OUT, "CLAIM_MANIFEST.json"), "w"), indent=1)
for k, v in C.items():
    if "router_sig_better" in v: print(f"{k:45s} n={v['n']:2d} better={v['router_sig_better']:2d} worse={v['router_sig_worse']:2d} ns={v['not_significant']:2d} meanΔ={v['mean_delta']:+.4f} worse={v['worse_cfgs']}")
    elif "router_better" in v: print(f"{k:45s} n={v['n']} better={v['router_better']} worse={v['router_worse']}")
for k, v in ltt.items(): print(k, v)
for k, v in C.items():
    if k.startswith(("B8_", "B6_")) and "sig_better" in v: print(f"{k:50s} n={v['n']:2d} better={v['sig_better']:2d} worse={v['sig_worse']:2d} ns={v['not_significant']:2d} meanΔ={v['mean_delta']:+.4f}")
if "B7_seed_stability" in C:
    for k, v in C["B7_seed_stability"].items(): print("B7", k, v)
