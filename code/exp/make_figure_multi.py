#!/usr/bin/env python3
"""Multilingual generalization figure: per-language never/always/router/oracle CER,
one multilingual ByT5-base-8ep corrector + one router."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = json.load(open(os.path.join(ROOT, "refine-logs", "multilingual_results.json")))["per_lang"]
boot = json.load(open(os.path.join(ROOT, "refine-logs", "multilingual_bootstrap.json")))
LANGS = ["EN", "DE", "FR", "ES", "NL"]
fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(len(LANGS)); w = 0.2
for k, (key, lab, col) in enumerate([("never","never","#bbbbbb"),("always","always-correct","#dd8452"),
                                     ("router","router (ours)","#4c72b0"),("oracle","oracle","#55a868")]):
    ax.bar(x+(k-1.5)*w, [res[L][key] for L in LANGS], w, label=lab, color=col)
# mark where router significantly beats always
for i, L in enumerate(LANGS):
    if boot[L]["sig_always"] == "sig":
        ax.text(x[i]+0.5*w, res[L]["router"]-0.012, "*", ha="center", fontsize=13, color="#4c72b0")
ax.set_xticks(x); ax.set_xticklabels([f"{L}\n(↓{res[L]['rel_gain']*100:.0f}%)" for L in LANGS])
ax.set_ylabel("Character Error Rate (held-out test)")
ax.set_title("Multilingual generalization: 1 corrector + 1 router across 5 languages\n(* = router significantly beats always-correct; router beats never in ALL)")
ax.legend(fontsize=8, ncol=4, loc="upper center")
ax.set_ylim(0, 0.45)
plt.tight_layout()
out = os.path.join(ROOT, "fig_multilingual.png")
plt.savefig(out, dpi=150); print("saved ->", out)
