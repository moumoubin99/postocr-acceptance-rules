#!/usr/bin/env python3
"""Render process/refine-logs/iter2/ledger_*.json into ledger_summary.md (no numbers typed by hand)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("PROJ", os.path.dirname(HERE))
sys.argv = sys.argv[:1]; sys.path.insert(0, HERE)
from iter2_ledger import CONFIGS, COLS, OUT
L = []
for ds, corr in CONFIGS:
    p = os.path.join(OUT, f"ledger_{corr}_{ds}.json")
    if os.path.exists(p): L.append(json.load(open(p)))
def f(x): return "—" if x is None else f"{x:.4f}"
def tab(hdr, rows): return "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"
md = ["# Iteration-2 claim ledger (CPU, frozen iter-2 candidates, leakage-free splits)\n",
      "Protocol: " + L[0]["protocol"] + ". `full` = router_conf when decoder log-probs exist, else router_align.\n"]
md.append("## 1. Segment micro-CER on test (dev-tuned CER-optimal thresholds)\n")
md.append(tab(["ds", "corr"] + COLS, [[o["dataset"], o["corrector"]] + [f(o["policies"].get(k, {}).get("cer")) for k in COLS] for o in L]))
md.append("## 2. Paired doc-bootstrap: baseline CER − full-router CER (positive ⇒ router better; * Holm p<0.05)\n")
C2 = ["never", "always", "router_text", "router_nodict", "conf_min", "conf_mean", "kim_min095", "conf_plus_cap", "edit_cap", "qe_encoder", "router_util"]
def d(o, k):
    b = o["bootstrap"][o["full_router"]].get(k)
    return "—" if not b else f"{b['delta']:+.4f}{'*' if b['p_holm'] < 0.05 else ''}"
md.append(tab(["ds", "corr", "full"] + C2, [[o["dataset"], o["corrector"], o["full_router"]] + [d(o, k) for k in C2] for o in L]))
md.append("## 3. Matched harm: dev-tuned corruption budgets (test CER / realised test corruptions; budget = fraction of always-accept corruptions)\n")
C3 = ["router_text", "router_nodict", "router_conf", "router_align", "router_util", "conf_min", "conf_mean", "edit_cap", "qe_encoder"]
for b in ["10pct", "25pct", "50pct"]:
    rows = []
    for o in L:
        fr = o["devtuned_budget_frontier"][b]; r = [o["dataset"], o["corrector"], str(next(iter(fr.values()))["test_budget"])]
        for k in C3:
            if k not in fr: r.append("—"); continue
            v = fr[k]; s = ""
            if "vs_full" in v and v["vs_full"]["p_holm"] < 0.05: s = "↑" if v["vs_full"]["delta"] > 0 else "↓"
            r.append(f"{v['test_cer']:.4f}/{v['test_corrupt']}{s}")
        rows.append(r)
    md.append(f"### budget {b} (↑/↓: significantly worse/better than the full router, Holm p<0.05)\n"); md.append(tab(["ds", "corr", "test budget"] + C3, rows))
md.append("## 4. Threshold-free ranking: mean test CER along the harm axis (lower = better ordering; descriptive)\n")
C4 = ["router_text", "router_nodict", "router_align", "router_conf", "router_conf_qe", "router_util", "conf_min", "conf_mean", "edit_cap", "qe_encoder"]
md.append(tab(["ds", "corr"] + C4, [[o["dataset"], o["corrector"]] + [f(o["rank_mean_cer_over_harm"].get(k)) for k in C4] for o in L]))
md.append("## 5. Reassembled-document CER (test)\n")
C5 = ["never", "always", "router_text", "router_conf", "router_align", "conf_min", "edit_cap", "oracle"]
md.append(tab(["ds", "corr"] + C5, [[o["dataset"], o["corrector"]] + [f(o["policies"].get(k, {}).get("doc_cer")) for k in C5] for o in L]))
md.append("## 6. Learn-then-Test control of the doc-macro harmful-acceptance rate (δ=0.10; 20 random fit/calibration doc splits)\n"
          "cell = mean test risk / fraction of splits violating α / mean test CER\n")
rows = []
for o in L:
    for a in ["alpha0.02", "alpha0.05", "alpha0.1"]:
        rk = o["risk_control"][a]; r = [o["dataset"], o["corrector"], a[5:], f(o["risk_control"]["always_test_risk"]), str(next(iter(rk.values()))["n_cal_docs"])]
        for k in [o["full_router"], "conf_min", "edit_cap"]:
            v = rk.get(k); r.append("—" if not v else f"{v['test_risk_mean']:.3f} / {v['violation_frac']:.2f} / {v['test_cer_mean']:.4f}")
        rows.append(r)
md.append(tab(["ds", "corr", "α", "always risk", "cal docs", "full router", "conf_min", "edit_cap"], rows))
md.append("## 7. Damage proxies on accepted segments (numbers / capitalised tokens OCR had right and the output lost : repaired)\n")
C7 = ["always", "router_text", "router_conf", "router_align", "conf_min", "edit_cap"]
def dm(v): return "—" if not v else f"{v['num_broken']}:{v['num_fixed']} / {v['cap_broken']}:{v['cap_fixed']}"
md.append(tab(["ds", "corr"] + C7, [[o["dataset"], o["corrector"]] + [dm(o["policies"].get(k)) for k in C7] for o in L]))
open(os.path.join(OUT, "ledger_summary.md"), "w").write("\n".join(md)); print("\n".join(md))
