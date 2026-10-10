#!/usr/bin/env python3
"""Summary figure (post-ARS, 3 correctors): (A) router CER by corrector on both
datasets; (B) reframed M4 — router gain vs oracle headroom (risk) across all
corrector×dataset points."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL = os.path.join(ROOT, "refine-logs")
bln = json.load(open(os.path.join(RL, "stage2_results_bln600.json")))
icd = json.load(open(os.path.join(RL, "stage2_results_icdar.json")))
correctors = [c for c in ["symspell", "byt5", "byt5base", "byt5base8"] if c in bln]
cnames = {"symspell": "SymSpell", "byt5": "ByT5-small", "byt5base": "ByT5-base(3ep)", "byt5base8": "ByT5-base(8ep)"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: router CER (+ never line) per corrector, both datasets
x = np.arange(len(correctors)); w = 0.35
for k, (ds, dd, col) in enumerate([("BLN600", bln, "#4c72b0"), ("ICDAR-EN", icd, "#dd8452")]):
    ax1.bar(x + (k-0.5)*w, [dd[c]["router"] for c in correctors], w, label=f"{ds} router", color=col)
ax1.axhline(bln["symspell"]["never"], ls="--", c="#4c72b0", lw=1, alpha=.7)
ax1.axhline(icd["symspell"]["never"], ls="--", c="#dd8452", lw=1, alpha=.7)
ax1.text(2.1, bln["symspell"]["never"]+.003, "BLN600 never", fontsize=7, c="#4c72b0")
ax1.text(2.1, icd["symspell"]["never"]+.006, "ICDAR never", fontsize=7, c="#dd8452")
ax1.set_xticks(x); ax1.set_xticklabels([cnames[c] for c in correctors], fontsize=8)
ax1.set_ylabel("router test CER"); ax1.set_title("(A) Router CER by corrector (dashed = never-correct)")
ax1.legend(fontsize=8)

# Panel B: router MARGINAL gain over blind always-correct (always - router), per corrector
xb = np.arange(len(correctors))
for k, (ds, dd, col) in enumerate([("BLN600", bln, "#4c72b0"), ("ICDAR-EN", icd, "#dd8452")]):
    gains = [dd[c]["always"] - dd[c]["router"] for c in correctors]
    ax2.bar(xb + (k-0.5)*w, gains, w, label=ds, color=col)
ax2.axhline(0, c="0.6", lw=.8)
ax2.set_xticks(xb); ax2.set_xticklabels([cnames[c].replace("ByT5-","B5-") for c in correctors], fontsize=7.5, rotation=12)
ax2.set_ylabel("router CER gain over always-correct")
ax2.set_title("(B) Routing helps on top of EVERY corrector (always − router)")
ax2.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(ROOT, "fig_router_summary.png")
plt.savefig(out, dpi=150); print("saved ->", out)
