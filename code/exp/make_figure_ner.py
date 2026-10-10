#!/usr/bin/env python3
"""Downstream NER figure: never / always / router NER detection F1 (CLEF-HIPE test),
with the objective-divergence message."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "refine-logs", "ner_routing_test.json")))
fig, ax = plt.subplots(figsize=(6.2, 4.3))
labels = ["never\n(no correction)", "always\n(blind correct)", "NER-aware\nrouter"]
vals = [d["never_f"], d["always_f"], d["router_f"]]
cols = ["#bbbbbb", "#dd8452", "#4c72b0"]
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}", ha="center", fontsize=10)
ax.annotate("blind correction\nHURTS NER\n(−0.083, sig.)", xy=(1, d["always_f"]), xytext=(1.35, d["always_f"]-0.06),
            fontsize=8, color="#a33", arrowprops=dict(arrowstyle="->", color="#a33"))
ax.annotate("router recovers\n(+0.084 vs always, sig.)", xy=(2, d["router_f"]), xytext=(1.45, d["router_f"]+0.03),
            fontsize=8, color="#247", arrowprops=dict(arrowstyle="->", color="#247"))
ax.set_ylabel("NER detection F1 (CLEF-HIPE hipe2020-en test)")
ax.set_ylim(0, 0.72)
ax.set_title("Objective divergence: for downstream NER, the optimal policy is\nto REJECT corrections (opposite of the CER objective)")
plt.tight_layout()
out = os.path.join(ROOT, "fig_ner.png")
plt.savefig(out, dpi=150); print("saved ->", out)
