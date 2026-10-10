#!/usr/bin/env python3
"""Data figures of the manuscript, drawn from JSON only (paper/figures/data/*.json, robust_summary.json).
fig2_frontier.pdf  test CER vs realised harm when the threshold is swept, with dev-chosen operating points
fig3_forest.pdf    corpus-pooled CER difference router - edit cap, CER-optimal point and 25% harm budget
fig4_seeds.pdf     corrector-seed stability
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); FD = os.path.join(ROOT, "generated/figures"); DD = os.path.join(ROOT, "figure_data")
I2 = os.path.join(ROOT, "results/iter2"); os.makedirs(FD, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
COL = dict(router="#0072B2", edit_cap="#D55E00", conf_min="#009E73", router_nodict="#56B4E9")   # Okabe-Ito
LAB = dict(router="Learned router", edit_cap="Edit-density cap", conf_min="Min-token confidence")
CORR = dict(symspell="SymSpell", byt5="ByT5-small", byt5base8="ByT5-base", byt5base8m="ByT5-base-ML", qwen="Qwen2.5-7B")
DS = dict(blnocr="BLN600", icdarc_EN="ICDAR-EN", icdarc_DE="ICDAR-DE", icdarc_FR="ICDAR-FR", icdarc_ES="ICDAR-ES", icdarc_NL="ICDAR-NL", icdarc_multi="ICDAR-multi",
          ext_impresso_en="impresso", ext_overproof_en="Overproof")
DS["blnocr+blnocrw50+blnocrw100+bln600"] = "BLN600"


def frontier():
    cfgs = ["byt5base8_blnocr", "qwen_blnocr", "byt5base8m_icdarc_DE", "byt5base8_icdarc_EN", "qwen_icdarc_multi", "byt5base8_ext_overproof_en"]
    cfgs = [c for c in cfgs if os.path.exists(os.path.join(DD, "frontier_%s.json" % c))]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.5)); axs = axs.ravel()
    for i, (ax, c) in enumerate(zip(axs, cfgs)):
        o = json.load(open(os.path.join(DD, "frontier_%s.json" % c))); H = max(o["always"]["harm"], 1)
        for r in ["edit_cap", "conf_min", "router"]:
            if r not in o["curves"]: continue
            cur = np.array(o["curves"][r], float)
            ax.plot(100 * cur[:, 0] / H, cur[:, 1], color=COL[r], lw=1.2, label=LAB[r], zorder=3 if r == "router" else 2)
            for k, m in [("opt", "o"), ("25pct", "s")]:
                p = o["points"][r][k]
                ax.plot(100 * p["harm"] / H, p["cer"], m, color=COL[r], ms=6.5, mec="k", mew=0.6, zorder=5)
        ax.axhline(o["never"]["cer"], color="0.45", lw=0.7, ls=":", zorder=1); ax.axhline(o["oracle"]["cer"], color="0.45", lw=0.7, ls="--", zorder=1)
        ax.plot(100, o["always"]["cer"], "x", color="k", ms=5, mew=1.1, zorder=6)
        ax.set_xlim(-3, 104); corr, ds = c.split("_", 1)
        ax.set_title("(%s) %s, %s" % ("abcdef"[i], DS[ds], CORR[corr]), fontsize=8, loc="left")
        if i % 3 == 0: ax.set_ylabel("Test CER")
        lo = min(o["oracle"]["cer"], min(np.array(v)[:, 1].min() for v in o["curves"].values())); hi = max(o["never"]["cer"], o["always"]["cer"])
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.08 * (hi - lo))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[r], lw=1.4, label=LAB[r]) for r in ["router", "edit_cap", "conf_min"]]
    h += [Line2D([], [], color="0.3", marker="o", ls="", ms=6, label="dev-chosen: CER-optimal"), Line2D([], [], color="0.3", marker="s", ls="", ms=6, label="dev-chosen: 25% harm budget"),
          Line2D([], [], color="k", marker="x", ls="", ms=5, label="always accept"), Line2D([], [], color="0.45", ls=":", lw=0.8, label="no correction"),
          Line2D([], [], color="0.45", ls="--", lw=0.8, label="segment oracle")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, handlelength=1.8, columnspacing=1.2)
    fig.supxlabel("Segments made worse on test (% of the number under always-accept)", fontsize=8, y=0.125)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=0.8, h_pad=0.9); fig.savefig(os.path.join(FD, "fig2_frontier.pdf")); plt.close(fig); print("fig2", len(cfgs), "panels")


def forest():
    R = json.load(open(os.path.join(I2, "robust_summary.json")))["comparisons"]
    panels = [("seg_router|seg_edit_cap", "opt", "Router − edit cap, CER-optimal"), ("seg_router|seg_edit_cap", "25pct", "Router − edit cap, 25% harm budget"),
              ("edit_router|seg_router", "opt", "Edit router − router, CER-optimal")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (name, op, title) in zip(axs, panels):
        pc = R[name][op]["per_corpus"]; ks = list(pc)[::-1]
        for y, k in enumerate(ks):
            v = pc[k]; sig = v["p_holm"] < 0.05; col = ("#0072B2" if v["delta"] < 0 else "#D55E00") if sig else "0.55"
            ax.plot(v["ci95"], [y, y], color=col, lw=1.3); ax.plot(v["delta"], y, "o", color=col, ms=4, mfc=col if sig else "white", mew=1)
        ax.axvline(0, color="k", lw=0.6); ax.set_xscale("symlog", linthresh=0.002, linscale=0.6)
        ax.set_xticks([-0.05, -0.01, -0.002, 0, 0.002, 0.01]); ax.set_xticklabels(["−0.05", "−0.01", "−0.002", "0", "0.002", "0.01"], fontsize=6.5, rotation=40, ha="right"); ax.grid(axis="x", lw=0.3, color="0.85")
        ax.set_yticks(range(len(ks))); ax.set_yticklabels(["%s (%d)" % (DS.get(k, k), pc[k]["n_configs"]) for k in ks]); ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("ΔCER (negative = first rule better)", fontsize=7); ax.set_xlim(-0.12, 0.05)
    fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(FD, "fig3_forest.pdf")); plt.close(fig); print("fig3")


def seeds():
    p = os.path.join(DD, "seeds.json")
    if not os.path.exists(p): return
    S = json.load(open(p)); cells = {}
    for k, v in S.items():
        ds, fam, corr = k.split("|"); cells.setdefault((ds, fam), []).append(v)
    cells = {k: v for k, v in cells.items() if len(v) == 3}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, mode, title in [(axs[0], "opt", "CER-optimal point"), (axs[1], "b25", "25% harm budget")]:
        for i, ((ds, fam), vs) in enumerate(cells.items()):
            for r, key, off in [("edit_cap", "edit_cap", -0.2), ("conf_min", "conf_min", 0.0), ("router", "router", 0.2)]:
                if mode == "opt": ys = [v[key] / v["never"] for v in vs]
                else: ys = [v["b25"].get(v["fr"] if r == "router" else key) for v in vs]; ys = [y / v["never"] if y is not None else np.nan for y, v in zip(ys, vs)]
                ax.plot([i + off] * 3, ys, "o", color=COL[r], ms=3.2, alpha=0.9, mec="none", label=LAB[r] if i == 0 else None)
                ax.plot([i + off - 0.08, i + off + 0.08], [np.nanmean(ys)] * 2, color=COL[r], lw=1.2)
        ax.axhline(1, color="0.45", lw=0.7, ls=":"); ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(["%s%s" % (DS[ds], "†" if fam.endswith("m") else "") for ds, fam in cells], fontsize=6.5, rotation=35, ha="right"); ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("Test CER / OCR CER"); axs[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(FD, "fig4_seeds.pdf")); plt.close(fig); print("fig4")


CORRX = dict(CORR, byt5base8x="ByT5-base", qwenctx2="Qwen2.5-7B+ctx", qwenv2="Qwen2.5-7B")
DSX = dict(blnocr="BLN600", blnocrw50="BLN600-w50", blnocrw100="BLN600-w100", bln600="BLN600-legacy", icdarc_EN="ICDAR-EN", icdarc_DE="ICDAR-DE", icdarc_FR="ICDAR-FR",
           icdarc_ES="ICDAR-ES", icdarc_NL="ICDAR-NL", icdarc_multi="ICDAR-multi", ext_impresso_en="impresso", ext_overproof_en="Overproof")
CORD = ["symspell", "byt5", "byt5base8", "byt5base8m", "byt5base8x", "qwen", "qwenctx2", "qwenv2"]


def ledgers():
    import glob
    L = [json.load(open(f)) for f in glob.glob(os.path.join(I2, "ledger_*.json"))]
    return sorted(L, key=lambda o: (list(DSX).index(o["dataset"]), CORD.index(o["corrector"])))


def overview():
    """fig5: every primary configuration at the CER-optimal point, CER relative to the uncorrected OCR"""
    L = ledgers()[::-1]; fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for y, o in enumerate(L):
        P = o["policies"]; nv = P["never"]["cer"]; r = lambda k: P[k]["cer"] / nv
        ax.plot([r("oracle"), max(r("always"), 1)], [y, y], color="0.85", lw=0.8, zorder=1)
        ax.plot(r("oracle"), y, "|", color="0.35", ms=7, mew=1.2, zorder=2, label="Segment oracle" if y == 0 else None)
        ax.plot(r("always"), y, "x", color="k", ms=4.5, mew=1.1, zorder=3, label="Always accept" if y == 0 else None)
        ax.plot(r("edit_cap"), y, "s", color=COL["edit_cap"], ms=4.2, mec="k", mew=0.4, zorder=4, label=LAB["edit_cap"] if y == 0 else None)
        ax.plot(r(o["full_router"]), y, "o", color=COL["router"], ms=4.2, mec="k", mew=0.4, zorder=5, label=LAB["router"] if y == 0 else None)
    ax.axvline(1, color="0.3", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_xticks([0.5, 0.7, 1, 1.5, 2, 3, 4]); ax.set_xticklabels(["0.5", "0.7", "1", "1.5", "2", "3", "4"])
    ax.minorticks_off(); ax.set_yticks(range(len(L))); ax.set_yticklabels(["%s, %s" % (DSX[o["dataset"]], CORRX[o["corrector"]]) for o in L], fontsize=6.5)
    ax.set_ylim(-0.7, len(L) - 0.3); ax.set_xlabel("Test CER relative to the uncorrected OCR (log scale; below 1 = better than no correction)")
    ax.legend(fontsize=7, loc="lower right", ncol=2); ax.grid(axis="x", lw=0.3, color="0.9")
    fig.tight_layout(); fig.savefig(os.path.join(FD, "fig5_overview.pdf")); plt.close(fig); print("fig5")


def budget25():
    """fig6: the 25% harm budget per configuration: CER difference router - cap with 95% CI, and share of attempted segments accepted"""
    L = ledgers()[::-1]; fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.9), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
    ps = np.array([o["devtuned_budget_frontier"]["25pct"]["edit_cap"]["vs_full"]["p_two_sided"] for o in L]); od = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(od): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)   # Holm across configurations
    for y, o in enumerate(L):
        f = o["devtuned_budget_frontier"]["25pct"]; v = f["edit_cap"]["vs_full"]; d, ci = -v["delta"], [-v["ci95"][1], -v["ci95"][0]]
        sig = adj[y] < 0.05; col = ("#0072B2" if d < 0 else "#D55E00") if sig else "0.55"
        axs[0].plot(ci, [y, y], color=col, lw=1.2); axs[0].plot(d, y, "o", color=col, ms=3.8, mfc=col if sig else "white", mew=0.9)
        a, b = 100 * f["edit_cap"]["accept_rate_attempted"], 100 * f[o["full_router"]]["accept_rate_attempted"]
        axs[1].plot([a, b], [y, y], color="0.8", lw=0.9, zorder=1)
        axs[1].plot(a, y, "s", color=COL["edit_cap"], ms=4, mec="k", mew=0.4, label=LAB["edit_cap"] if y == 0 else None)
        axs[1].plot(b, y, "o", color=COL["router"], ms=4, mec="k", mew=0.4, label=LAB["router"] if y == 0 else None)
        if "conf_min" in f: axs[1].plot(100 * f["conf_min"]["accept_rate_attempted"], y, "D", color=COL["conf_min"], ms=3, mec="k", mew=0.3, label=LAB["conf_min"] if y == len(L) - 2 else None)
    axs[0].axvline(0, color="k", lw=0.6); axs[0].set_xscale("symlog", linthresh=0.002, linscale=0.6); axs[0].set_xlim(-0.12, 0.01)
    axs[0].set_xticks([-0.1, -0.03, -0.01, -0.002, 0, 0.002]); axs[0].set_xticklabels(["−0.1", "−0.03", "−0.01", "−0.002", "0", "0.002"], fontsize=6.5)
    axs[0].grid(axis="x", lw=0.3, color="0.9"); axs[0].set_xlabel("ΔCER, router − edit cap (negative = router better)", fontsize=7); axs[0].set_title("(a) CER difference, 95% interval", fontsize=8, loc="left")
    axs[0].set_yticks(range(len(L))); axs[0].set_yticklabels(["%s, %s" % (DSX[o["dataset"]], CORRX[o["corrector"]]) for o in L], fontsize=6.5); axs[0].set_ylim(-0.7, len(L) - 0.3)
    axs[1].set_xlabel("Attempted segments accepted (%)", fontsize=7); axs[1].set_xlim(-3, 103); axs[1].grid(axis="x", lw=0.3, color="0.9")
    axs[1].set_title("(b) coverage", fontsize=8, loc="left"); h, l = axs[1].get_legend_handles_labels(); o_ = [l.index(LAB[k]) for k in ["edit_cap", "conf_min", "router"] if LAB[k] in l]
    fig.legend([h[i] for i in o_], [l[i] for i in o_], fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.6, -0.01))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.035, 1, 1)); fig.savefig(os.path.join(FD, "fig6_budget25.pdf")); plt.close(fig); print("fig6")


def winloss():
    """fig7: counts better / not significant / worse (Holm across configurations) for the main pairwise comparisons at the four operating points"""
    R = json.load(open(os.path.join(I2, "robust_summary.json")))["comparisons"]
    rows = [("seg_router|seg_edit_cap", "Router vs edit cap"), ("seg_router_text|seg_edit_cap", "Text-only router vs edit cap"), ("seg_router|seg_conf_min", "Router vs min-confidence"),
            ("seg_router|seg_conf_plus_cap", "Router vs confidence + cap"), ("seg_router_text|seg_conf_plus_cap", "Text-only router vs confidence + cap"),
            ("edit_router|seg_router", "Edit router vs router"), ("edit_router|edit_size", "Edit router vs edit-size rule")]
    ops = [("opt", "CER-optimal"), ("10pct", "10% budget"), ("25pct", "25% budget"), ("50pct", "50% budget")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    for ax, (op, title) in zip(axs, ops):
        for y, (k, lab) in enumerate(rows[::-1]):
            w, l, t = [int(x) for x in R[k][op]["holm_all"].split("/")]; n = w + l + t; x = 0
            for v, c, tc in [(w, "#0072B2", "w"), (t, "0.86", "k"), (l, "#D55E00", "w")]:
                if v: ax.barh(y, v / n, left=x, color=c, height=0.72, edgecolor="w", lw=0.4); ax.text(x + v / n / 2, y, str(v), ha="center", va="center", fontsize=6.3, color=tc)
                x += v / n
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=6.5); ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab in rows[::-1]], fontsize=7); ax.tick_params(axis="y", length=0)
        for sp in ["left", "bottom"]: ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend([Patch(color="#0072B2"), Patch(color="0.86"), Patch(color="#D55E00")], ["first rule significantly better", "not significant", "first rule significantly worse"],
               ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.6, -0.02))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FD, "fig7_winloss.pdf")); plt.close(fig); print("fig7")


def lcurve():
    """fig8: test CER at the 25% harm budget and at the CER-optimal point against the number of development documents (mean and sd over random subsets)"""
    import glob
    X = [json.load(open(f)) for f in glob.glob(os.path.join(I2, "extra5", "lc_*.json"))]
    if not X: print("skip fig8"); return
    X = sorted(X, key=lambda o: (list(DSX).index(o["dataset"]), CORD.index(o["corrector"])))
    for mode, tag, fn in [("b25_cer", "25% harm budget", "fig8_lcurve.pdf"), ("opt_cer", "CER-optimal point", "fig9_lcurve_opt.pdf")]:
        nc = 5; nr = int(np.ceil(len(X) / nc)); fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.75 * nr + 0.35)); axs = np.atleast_1d(axs).ravel()
        for i, (ax, o) in enumerate(zip(axs, X)):
            ks = [c["dev_docs"] for c in o["curve"]]
            for r, key in [("edit_cap", "edit_cap"), ("conf_min", "conf_min"), ("router", "full")]:
                if key not in o["curve"][0]: continue
                m = np.array([c[key][mode]["mean"] for c in o["curve"]]); sd = np.array([c[key][mode]["sd"] for c in o["curve"]])
                ax.fill_between(ks, m - sd, m + sd, color=COL[r], alpha=0.15, lw=0); ax.plot(ks, m, "-o", color=COL[r], lw=1.1, ms=2.6, label=LAB[r])
            ax.axhline(o["never"], color="0.45", lw=0.7, ls=":"); ax.set_xscale("log"); tk = [k for k in ks if k == ks[-1] or ks[-1] / k > 1.35]; ax.set_xticks(tk); ax.set_xticklabels([str(k) for k in tk], fontsize=5.8); ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=6); ax.set_title("%s, %s" % (DSX[o["dataset"]], CORRX[o["corrector"]]), fontsize=6.6)
            if i % nc == 0: ax.set_ylabel("Test CER", fontsize=7)
            if i >= len(X) - nc: ax.set_xlabel("Development documents", fontsize=6.5)
        for ax in axs[len(X):]: ax.axis("off")
        h, l = axs[0].get_legend_handles_labels()
        for ax in axs[:len(X)]:
            hh, ll = ax.get_legend_handles_labels()
            if len(ll) > len(l): h, l = hh, ll
        fig.legend(h, l, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(w_pad=0.4, h_pad=0.6, rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FD, fn)); plt.close(fig); print(fn)


if __name__ == "__main__":
    frontier(); forest(); seeds(); overview(); budget25(); winloss(); lcurve()
