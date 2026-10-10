#!/usr/bin/env python3
"""CER-corruption frontier figure: text-only vs +confidence acceptance router."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, ds, title in [(axes[0], "bln600", "BLN600"), (axes[1], "icdar", "ICDAR-EN")]:
    d = json.load(open(os.path.join(ROOT, "refine-logs", f"frontier_conf_{ds}.json")))
    ft, fc = d["frontier_text"], d["frontier_conf"]
    # frontier = list of (cer, corruption_count); plot lower envelope sorted by corruption
    def env(front):
        pts = sorted(set((k, c) for c, k in front))  # (corr, cer)
        best, out = 1e9, []
        for k, c in pts:
            best = min(best, c); out.append((k, best))
        return zip(*out)
    kt, ct = env(ft); kc, cc = env(fc)
    ax.plot(kt, ct, "-o", ms=2, label="text-only router", color="#dd8452")
    ax.plot(kc, cc, "-o", ms=2, label="+confidence router", color="#4c72b0")
    ax.axhline(d["never"], ls="--", c="0.6", lw=1, label="never-correct")
    ax.set_xlabel("induced corruptions (segments made worse)")
    ax.set_ylabel("net CER"); ax.set_title(f"{title}: CER–corruption frontier")
    ax.set_xlim(0, d["always_corr"]*1.05); ax.legend(fontsize=8)
plt.suptitle("Confidence-aware router dominates in the corruption-constrained (high-precision) regime", fontsize=11)
plt.tight_layout()
out = os.path.join(ROOT, "fig_frontier.png")
plt.savefig(out, dpi=150); print("saved ->", out)
