#!/usr/bin/env python3
"""Iter-2 robustness analyses requested by the second re-review (CPU only, frozen candidates, same protocol as
iter2_ledger / iter2_editlevel; nothing is re-tuned on test except where a row is labelled "descriptive").

Per configuration (robust/robust_{corr}_{ds}.json + .npz with the bootstrap replicates):
  * policies at the CER-optimal dev threshold and at the 10/25/50% dev harm budgets: segment router, segment edit-cap,
    segment conf-min, edit router, edit-size rule, edit router without the byte-mapped log-prob features;
  * micro CER, reassembled-document CER, realised test corruptions and the test budget;
  * paired doc bootstrap (B=10000, raw p) for: edit router vs seg router, edit router vs edit size, edit router
    (no lp features) vs seg router, seg router vs edit-cap, seg router vs conf-min — on both CER definitions;
  * matched realised harm (descriptive, threshold picked on test): router at the edit-cap's realised corruptions
    and edit-cap at the router's;
  * icdarc_FR: sensitivity to dropping test segments whose gold text also occurs in train/dev.
Summary (robust_summary.md): Holm across all configurations, corpus-level pooling (configurations that share the
same test documents are averaged inside each bootstrap replicate, Holm across corpora), realised-harm tables,
OCR-only-segmentation subset, minimum LTT calibration sizes, transfer significance.
"""
import os, sys, json, glob, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rapidfuzz.distance import Levenshtein
from oclib import DATA, load_jsonl
import iter2_ledger as LG
from iter2_ledger import CONFIGS, SEED, B, pval, ED, Split, scores, crossfit, gb, tune, qgrid, hb_p
from iter2_editlevel import ESplit, etune, apply

OUT = os.path.join(LG.OUT, "robust"); os.makedirs(OUT, exist_ok=True)
OPS = ["opt", "10pct", "25pct", "50pct"]; FRAC = {"10pct": 0.10, "25pct": 0.25, "50pct": 0.50}
LEGACY_SEG = {"bln600"}                                   # configurations that keep the old (gold-aware) segmentation
COMPS = [("edit_router", "seg_router"), ("edit_router", "edit_size"), ("edit_router_nolp", "seg_router"),
         ("seg_router", "seg_edit_cap"), ("seg_router", "seg_conf_min"), ("edit_size", "seg_edit_cap"),
         ("seg_router_text", "seg_edit_cap"), ("seg_router", "seg_conf_plus_cap"), ("seg_router_text", "seg_conf_plus_cap")]

class Docs:
    """reassembled-document errors (same reassembly as Split.doc_cer)"""
    def __init__(s, te):
        s.te = te; s.order = sorted(range(len(te.rows)), key=lambda i: (te.di[i], te.rows[i].get("seg_idx", i)))
        ref = [[] for _ in te.docs]
        for i in s.order: ref[te.di[i]].append(te.rows[i]["gold"] + te.rows[i].get("gold_sep", " "))
        s.ref = ["".join(x).strip() for x in ref]; s.G = np.array([len(x) for x in s.ref], float)
    def err(s, texts):
        hyp = [[] for _ in s.te.docs]
        for i in s.order: hyp[s.te.di[i]].append(texts[i] + s.te.rows[i].get("ocr_sep", " "))
        return np.array([Levenshtein.distance("".join(h).strip(), f) for h, f in zip(hyp, s.ref)], float)

def run(ds, corr):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test"); Ed, Et = ESplit(dv), ESplit(te)
    Sseg, full = scores(dv, te); D = Docs(te)
    ones = np.ones(len(Ed.y), bool); grp = dv.di[Ed.row]
    ES = {"edit_router": crossfit(gb, Ed.F, Ed.y, ones, grp, Et.F), "edit_size": (-Ed.F[:, 5], -Et.F[:, 5])}
    has_lp = bool(Ed.has_lp and Et.has_lp)
    if has_lp:
        keep = [j for j in range(Ed.F.shape[1]) if j not in (17, 18)]
        ES["edit_router_nolp"] = crossfit(gb, Ed.F[:, keep], Ed.y, ones, grp, Et.F[:, keep])
    Dd = {k: Ed.deltas(v[0]) for k, v in ES.items()}; Dt = {k: Et.deltas(v[1]) for k, v in ES.items()}
    SG = {"seg_router": Sseg[full], "seg_edit_cap": Sseg["edit_cap"], "seg_router_text": Sseg["router_text"]}
    if "conf_min" in Sseg: SG["seg_conf_min"] = Sseg["conf_min"]
    ocr = [r["ocr"] for r in te.rows]; cand = [r["cand"] for r in te.rows]
    def seg_pol(mask): return np.where(mask, te.ec, te.eo).astype(float), [c if m else o for o, c, m in zip(ocr, cand, mask)]
    # edit acceptance -> texts (rows are contiguous in Et.row order)
    starts = {};
    for j, i in enumerate(Et.row): starts.setdefault(int(i), j)
    def edit_texts(k, t):
        st = ES[k][1]; tx = list(ocr)
        for i, eds in Et.eds.items():
            j0 = starts[int(i)]; keep = {m for m in range(len(eds)) if st[j0 + m] >= t}
            tx[i] = cand[i] if len(keep) == len(eds) else (ocr[i] if not keep else apply(ocr[i], cand[i], eds, keep))
        return tx
    a_dev = dv.corrupt(dv.att); a_te = te.corrupt(te.att)
    def conf_plus_cap(bud):
        """two-threshold rule (min-token confidence >= a and edit density <= b), same dev grid as iter2_ledger; abstain is on the grid"""
        best = None
        for a in np.r_[np.quantile(dv.C[dv.att, 1], np.linspace(0, 0.95, 20)), np.inf]:
            for b in np.quantile(dv.X[dv.att, ED], np.linspace(0.05, 1, 20)):
                u = dv.att & (dv.C[:, 1] >= a) & (dv.X[:, ED] <= b)
                if bud is not None and dv.corrupt(u) > bud: continue
                c_ = dv.cer(u)
                if best is None or c_ < best[0] - 1e-12: best = (c_, a, b)
        return te.att & (te.C[:, 1] >= best[1]) & (te.X[:, ED] <= best[2])
    P = {op: {} for op in OPS}                             # op -> policy -> (row errors, texts)
    for op in OPS:
        bud = None if op == "opt" else int(FRAC[op] * a_dev)
        for k, (sd, st) in SG.items():
            t = tune(dv, sd) if bud is None else tune(dv, sd, qgrid(sd, dv.att, 201), budget=bud)
            P[op][k] = seg_pol(te.att & (st >= t))
        if dv.has_conf: P[op]["seg_conf_plus_cap"] = seg_pol(conf_plus_cap(bud))
        for k, (sd, st) in ES.items():
            t = etune(Ed, sd, Dd[k]) if bud is None else etune(Ed, sd, Dd[k], 201, budget=bud)
            P[op][k] = (Et.errs(st, Dt[k], t), edit_texts(k, t))
    # matched realised harm (descriptive: threshold chosen on test so that realised corruptions <= the other policy's)
    def at_harm(st, h):
        best = np.inf
        for t in np.unique(st[te.att])[::-1]:
            if te.corrupt(te.att & (st >= t)) > h: break
            best = t
        return te.att & (st >= best)
    for op in OPS[1:]:
        hr = int((P[op]["seg_router"][0] > te.eo).sum()); hc = int((P[op]["seg_edit_cap"][0] > te.eo).sum())
        P[op]["seg_router@cap_harm"] = seg_pol(at_harm(SG["seg_router"][1], hc))
        P[op]["seg_edit_cap@router_harm"] = seg_pol(at_harm(SG["seg_edit_cap"][1], hr))
    rng = np.random.default_rng(SEED); W = rng.multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gl_b = W @ te.G; gd_b = W @ D.G
    DE = {op: {k: D.err(tx) for k, (e, tx) in P[op].items()} for op in OPS}
    pol = {op: {k: dict(cer=float(e.sum() / te.gl.sum()), doc_cer=float(DE[op][k].sum() / D.G.sum()), corrupt=int((e > te.eo).sum()),
                        improved=int((e < te.eo).sum())) for k, (e, tx) in P[op].items()} for op in OPS}
    comp, vec = {}, {}
    def cmp(a, b, op, name=None):
        if a not in P[op] or b not in P[op]: return
        ea, eb = P[op][a][0], P[op][b][0]
        d = (W @ np.bincount(te.di, ea - eb, minlength=te.nd)) / gl_b; dd = (W @ (DE[op][a] - DE[op][b])) / gd_b
        name = name or f"{a}|{b}"
        comp.setdefault(name, {})[op] = dict(delta=float((ea - eb).sum() / te.gl.sum()), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], p=pval(d),
                                             delta_doc=float((DE[op][a] - DE[op][b]).sum() / D.G.sum()), p_doc=pval(dd))
        vec[f"{name}|{op}"] = d.astype(np.float32)
    for op in OPS:
        for a, b in COMPS: cmp(a, b, op)
        if op != "opt":
            cmp("seg_router@cap_harm", "seg_edit_cap", op); cmp("seg_router", "seg_edit_cap@router_harm", op)
    out = dict(dataset=ds, corrector=corr, full_router=full, has_edit_lp=has_lp, n_test_docs=te.nd,
               test_docs_md5=hashlib.md5("\n".join(te.docs).encode()).hexdigest(), legacy_segmentation=ds in LEGACY_SEG,
               always_corrupt_dev=int(a_dev), always_corrupt_test=int(a_te),
               budgets={op: dict(dev=int(FRAC[op] * a_dev), test=int(FRAC[op] * a_te)) for op in OPS[1:]}, policies=pol, comparisons=comp)
    if ds == "icdarc_FR":                                  # overlap sensitivity: drop test segments whose gold also occurs in train/dev
        seen = set()
        for sp in ("train", "dev"):
            p = os.path.join(DATA, f"segments_{ds}_{sp}.jsonl")
            if os.path.exists(p): seen |= {r["gold"] for r in load_jsonl(p)}
        m = np.array([r["gold"] not in seen for r in te.rows])
        out["overlap_sensitivity"] = dict(dropped_segments=int((~m).sum()), n_test=len(te.rows),
            cer={k: dict(all=float(e.sum() / te.gl.sum()), no_overlap=float(e[m].sum() / te.gl[m].sum()))
                 for k, (e, tx) in {**P["opt"], "never": (te.eo.astype(float), None), "always": (np.where(te.att, te.ec, te.eo).astype(float), None)}.items()})
    json.dump(out, open(os.path.join(OUT, f"robust_{corr}_{ds}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"robust_{corr}_{ds}.npz"), **vec)
    return out

# ---------------------------------------------------------------- transfer significance
def transfer():
    PAIRS = [("blnocr", "icdarc_EN"), ("icdarc_EN", "blnocr"), ("blnocr", "ext_impresso_en"), ("blnocr", "ext_overproof_en"), ("icdarc_EN", "ext_overproof_en")]
    res = []
    for corr in ["byt5", "byt5base8", "qwen"]:
        for a, b in PAIRS:
            try: dv, dvb, te = Split(corr, a, "dev"), Split(corr, b, "dev"), Split(corr, b, "test")
            except FileNotFoundError: continue
            S, full = scores(dv, te); Sb, _ = scores(dvb, te)
            U = {"never": np.zeros(len(te.rows), bool), "always": te.att.copy(), "router_transfer": te.att & (S[full][1] >= tune(dv, S[full][0])),
                 "cap_transfer": te.att & (S["edit_cap"][1] >= tune(dv, S["edit_cap"][0])), "router_in_domain": te.att & (Sb[full][1] >= tune(dvb, Sb[full][0]))}
            rng = np.random.default_rng(SEED); W = rng.multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gl_b = W @ te.G
            E = {k: te.per_doc(u) for k, u in U.items()}; r = dict(corrector=corr, source=a, target=b, cer={k: te.cer(u) for k, u in U.items()}, vs={})
            for k in ["never", "always", "cap_transfer", "router_in_domain"]:
                d = (W @ (E["router_transfer"] - E[k])) / gl_b
                r["vs"][k] = dict(delta=float(te.cer(U["router_transfer"]) - te.cer(U[k])), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], p=pval(d))
            ps = sorted(r["vs"], key=lambda k: r["vs"][k]["p"]); mx = 0
            for i, k in enumerate(ps): mx = max(mx, min(1.0, (len(ps) - i) * r["vs"][k]["p"])); r["vs"][k]["p_holm"] = mx
            res.append(r); print("transfer", corr, a, b, {k: round(v, 4) for k, v in r["cer"].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "transfer_significance.json"), "w"), indent=1)

# ---------------------------------------------------------------- summary
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty(len(ps)); mx = 0
    for i, j in enumerate(o): mx = max(mx, min(1.0, (len(ps) - i) * ps[j])); out[j] = mx
    return out

def summary():
    R = []
    for ds, corr in CONFIGS:
        p = os.path.join(OUT, f"robust_{corr}_{ds}.json")
        if os.path.exists(p): o = json.load(open(p)); o["vec"] = np.load(p[:-5] + ".npz"); R.append(o)
    groups = {}
    for o in R: groups.setdefault(o["test_docs_md5"], []).append(o)
    gname = {k: "+".join(dict.fromkeys(o["dataset"] for o in v)) for k, v in groups.items()}
    S = dict(n_configs=len(R), corpora={gname[k]: [f"{o['corrector']}@{o['dataset']}" for o in v] for k, v in groups.items()}, comparisons={})
    wl = lambda d, p: f"{int(((p < .05) & (d < 0)).sum())} / {int(((p < .05) & (d > 0)).sum())} / {int((p >= .05).sum())}"
    L = ["# Iter-2 robustness analyses (second re-review)", "",
         f"{len(R)} configurations on {len(groups)} distinct test-document sets (corpora): " + "; ".join(f"**{gname[k]}** ×{len(v)}" for k, v in groups.items()) + ".",
         "", "Cells are `first significantly better / first significantly worse / n.s.` (paired document bootstrap, B=10000, p<0.05 after the stated correction; Δ = first − second, negative = first better).",
         "*Holm-26* corrects across all configurations of one comparison at one operating point. *Corpus-pooled* averages the configurations that share test documents inside each bootstrap replicate (one test per corpus) and applies Holm across corpora.", ""]
    names = [f"{a}|{b}" for a, b in COMPS]
    for name in names:
        L += [f"## {name.replace('|', ' vs ')}", "", "| operating point | n | raw p<.05 | Holm across configs | Holm, OCR-only segmentation (excl. bln600) | Holm, reassembled-document CER | corpus-pooled (Holm across corpora) | mean Δ | corpora with pooled win | corpora with pooled loss |", "|---|---|---|---|---|---|---|---|---|---|"]
        S["comparisons"][name] = {}
        for op in OPS:
            rs = [o for o in R if op in o["comparisons"].get(name, {})]
            if not rs: continue
            d = np.array([o["comparisons"][name][op]["delta"] for o in rs]); p = np.array([o["comparisons"][name][op]["p"] for o in rs])
            dd = np.array([o["comparisons"][name][op]["delta_doc"] for o in rs]); pd_ = np.array([o["comparisons"][name][op]["p_doc"] for o in rs])
            ok = np.array([not o["legacy_segmentation"] for o in rs]); ph = holm(p)
            gk = [k for k in groups if any(o in rs for o in groups[k])]; gd, gp, gci = [], [], []
            for k in gk:
                m = [o for o in groups[k] if o in rs]; v = np.mean([o["vec"][f"{name}|{op}"] for o in m], 0)
                gd.append(float(np.mean([o["comparisons"][name][op]["delta"] for o in m]))); gp.append(pval(v)); gci.append([float(x) for x in np.percentile(v, [2.5, 97.5])])
            gd, gp = np.array(gd), holm(gp)
            win = [gname[k] for k, a, b in zip(gk, gd, gp) if b < .05 and a < 0]; los = [gname[k] for k, a, b in zip(gk, gd, gp) if b < .05 and a > 0]
            S["comparisons"][name][op] = dict(n=len(rs), raw=wl(d, p), holm_all=wl(d, ph), holm_ocr_only=wl(d[ok], holm(p[ok])), holm_doc_cer=wl(dd, holm(pd_)),
                                              corpus_pooled=wl(gd, gp), n_corpora=len(gk), mean_delta=float(d.mean()), corpora_win=win, corpora_loss=los,
                                              per_corpus={gname[k]: dict(delta=float(a), p_holm=float(b), ci95=c, n_configs=len([o for o in groups[k] if o in rs])) for k, a, b, c in zip(gk, gd, gp, gci)},
                                              sig_configs_holm_all={f"{o['corrector']}@{o['dataset']}": float(x) for o, x, y in zip(rs, d, ph) if y < .05})
            L.append(f"| {op} | {len(rs)} | {wl(d, p)} | {wl(d, ph)} | {wl(d[ok], holm(p[ok]))} | {wl(dd, holm(pd_))} | {wl(gd, gp)} (of {len(gk)}) | {d.mean():+.4f} | {', '.join(win) or '—'} | {', '.join(los) or '—'} |")
        L.append("")
    # realised harm
    L += ["## Realised test corruptions at dev-tuned harm budgets", "",
          "Test budget = the same fraction of always-accept corruptions on test. `over` = configurations whose realised test corruptions exceed it; ratio = realised / test budget (median over configurations with a non-zero budget).", "",
          "| budget | policy | n | over budget | median realised/budget | max realised/budget |", "|---|---|---|---|---|---|"]
    S["realised_harm"] = {}
    for op in OPS[1:]:
        for k in ["seg_router", "seg_edit_cap", "seg_conf_min", "edit_router", "edit_size"]:
            rs = [o for o in R if k in o["policies"][op]]; c = np.array([o["policies"][op][k]["corrupt"] for o in rs]); b = np.array([o["budgets"][op]["test"] for o in rs])
            ratio = c[b > 0] / b[b > 0]
            S["realised_harm"][f"{op}|{k}"] = dict(n=len(rs), over=int((c > b).sum()), median_ratio=float(np.median(ratio)), max_ratio=float(ratio.max()))
            L.append(f"| {op} | {k} | {len(rs)} | {int((c > b).sum())} | {np.median(ratio):.2f} | {ratio.max():.2f} |")
    L += ["", "## Router vs edit-cap at matched *realised* harm (descriptive: the matching threshold is chosen on test)", "",
          "`router@cap` = router thresholded so that its realised test corruptions do not exceed the edit-cap's at that budget, compared with the edit-cap; `cap@router` = the reverse. "
          "`router harm > cap harm` counts configurations where the dev-budget router realises more corruptions than the dev-budget edit-cap.", "",
          "| budget | router harm > cap harm | …among Holm-26 router wins | router@cap vs cap: Holm-26 W/L/ns | sign (better/worse/equal) | mean Δ | router vs cap@router: Holm-26 W/L/ns | sign | mean Δ |", "|---|---|---|---|---|---|---|---|---|"]
    S["matched_realised"] = {}
    for op in OPS[1:]:
        more = np.array([o["policies"][op]["seg_router"]["corrupt"] > o["policies"][op]["seg_edit_cap"]["corrupt"] for o in R])
        d0 = np.array([o["comparisons"]["seg_router|seg_edit_cap"][op]["delta"] for o in R]); p0 = holm([o["comparisons"]["seg_router|seg_edit_cap"][op]["p"] for o in R])
        won = (p0 < .05) & (d0 < 0); cells = []
        for name in ["seg_router@cap_harm|seg_edit_cap", "seg_router|seg_edit_cap@router_harm"]:
            d = np.array([o["comparisons"][name][op]["delta"] for o in R]); p = holm([o["comparisons"][name][op]["p"] for o in R])
            cells += [wl(d, p), f"{int((d < -1e-12).sum())} / {int((d > 1e-12).sum())} / {int((abs(d) <= 1e-12).sum())}", f"{d.mean():+.4f}"]
            S["matched_realised"][f"{op}|{name}"] = dict(holm=wl(d, p), better=int((d < -1e-12).sum()), worse=int((d > 1e-12).sum()), mean_delta=float(d.mean()))
        S["matched_realised"][f"{op}|router_more_harm"] = dict(n=int(more.sum()), among_wins=int((more & won).sum()), wins=int(won.sum()))
        L.append(f"| {op} | {int(more.sum())} / {len(R)} | {int((more & won).sum())} / {int(won.sum())} | " + " | ".join(cells) + " |")
    # per-config table of the edit-level comparison
    L += ["", "## Edit router vs segment router and vs edit-size rule, per configuration (CER-optimal)", "",
          "| corpus | dataset | corrector | seg router | edit router | edit size | edit router (no lp feats) | Δ edit−seg (p, Holm-26) | Δ edit−size (p, Holm-26) | seg router abstains |", "|---|---|---|---|---|---|---|---|---|---|"]
    h1 = holm([o["comparisons"]["edit_router|seg_router"]["opt"]["p"] for o in R]); h2 = holm([o["comparisons"]["edit_router|edit_size"]["opt"]["p"] for o in R])
    for o, a, b in zip(R, h1, h2):
        p = o["policies"]["opt"]; c1, c2 = o["comparisons"]["edit_router|seg_router"]["opt"], o["comparisons"]["edit_router|edit_size"]["opt"]
        L.append(f"| {gname[o['test_docs_md5']]} | {o['dataset']} | {o['corrector']} | {p['seg_router']['cer']:.4f} | {p['edit_router']['cer']:.4f} | {p['edit_size']['cer']:.4f} | "
                 + (f"{p['edit_router_nolp']['cer']:.4f}" if "edit_router_nolp" in p else "—") + f" | {c1['delta']:+.4f} ({a:.3f}) | {c2['delta']:+.4f} ({b:.3f}) | "
                 + ("yes" if p["seg_router"]["corrupt"] == 0 and p["seg_router"]["improved"] == 0 else "") + " |")
    lp = [o for o in R if o["has_edit_lp"]]
    if lp:
        dl = np.array([o["policies"]["opt"]["edit_router_nolp"]["cer"] - o["policies"]["opt"]["edit_router"]["cer"] for o in lp])
        S["nolp"] = dict(n=len(lp), mean_cer_change=float(dl.mean()), max_abs_change=float(abs(dl).max()))
        L += ["", f"Removing the two byte-mapped log-prob features from the edit router ({len(lp)} ByT5 configurations): mean CER change {dl.mean():+.5f}, max |change| {abs(dl).max():.5f}."]
    mean = lambda k: float(np.mean([o["policies"]["opt"][k]["cer"] for o in R]))
    S["mean_cer_opt"] = {k: mean(k) for k in ["seg_router", "seg_edit_cap", "edit_router", "edit_size"]}
    S["mean_cer_opt_corpus_balanced"] = {k: float(np.mean([np.mean([o["policies"]["opt"][k]["cer"] for o in v]) for v in groups.values()])) for k in ["seg_router", "seg_edit_cap", "edit_router", "edit_size"]}
    L += ["", "Mean CER over configurations: " + ", ".join(f"{k} {v:.4f}" for k, v in S["mean_cer_opt"].items()) + ". Corpus-balanced (mean of corpus means): " + ", ".join(f"{k} {v:.4f}" for k, v in S["mean_cer_opt_corpus_balanced"].items()) + "."]
    fr = [o for o in R if "overlap_sensitivity" in o]
    for o in fr:
        s = o["overlap_sensitivity"]; S["fr_overlap"] = s
        L += ["", f"## Overlap sensitivity ({o['corrector']}@{o['dataset']})", "", f"{s['dropped_segments']} of {s['n_test']} test segments have a gold text that also occurs in train/dev. CER all → without them: "
              + "; ".join(f"{k} {v['all']:.4f} → {v['no_overlap']:.4f}" for k, v in s["cer"].items() if "@" not in k) + "."]
    # LTT: smallest calibration size that can certify anything (zero empirical loss)
    S["ltt_min_cal_docs"] = {}
    for a in [0.02, 0.05, 0.10]:
        n = next(n for n in range(1, 2000) if hb_p(0.0, n, a) <= 0.10); S["ltt_min_cal_docs"][str(a)] = n
    L += ["", "## Learn-then-Test: minimum calibration documents", "", "With zero observed harmful acceptances on the calibration documents, the Hoeffding–Bentkus p-value reaches δ=0.10 at "
          + ", ".join(f"n={n} for α={a}" for a, n in S["ltt_min_cal_docs"].items()) + ". Smaller calibration sets must abstain; this is a property of the bound, not of the policies."]
    tp = os.path.join(OUT, "transfer_significance.json")
    if os.path.exists(tp):
        T = json.load(open(tp)); S["transfer"] = T
        L += ["", "## Transfer significance (router fit + thresholded on source dev, evaluated on target test)", "",
              "Δ = transferred router − reference; * = Holm p<0.05 within the row.", "",
              "| corrector | source → target | never | always | router (transfer) | edit-cap (transfer) | router (in-domain) | Δ vs never | Δ vs always | Δ vs cap (transfer) | Δ vs router (in-domain) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: f"{v['delta']:+.4f}" + ("*" if v["p_holm"] < .05 else "")
        for r in T:
            c = r["cer"]; L.append(f"| {r['corrector']} | {r['source']} → {r['target']} | {c['never']:.4f} | {c['always']:.4f} | {c['router_transfer']:.4f} | {c['cap_transfer']:.4f} | {c['router_in_domain']:.4f} | "
                                   + " | ".join(f(r["vs"][k]) for k in ["never", "always", "cap_transfer", "router_in_domain"]) + " |")
    open(os.path.join(LG.OUT, "robust_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(S, open(os.path.join(LG.OUT, "robust_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    only = sys.argv[1:]
    if only == ["summary"]: summary(); sys.exit()
    if only == ["transfer"]: transfer(); sys.exit()
    for ds, corr in CONFIGS:
        if only and f"{corr}_{ds}" not in only: continue
        if not os.path.exists(os.path.join(DATA, f"candidates_{corr}_{ds}_dev.jsonl")): print("skip", ds, corr); continue
        o = run(ds, corr); p = o["policies"]["opt"]
        print(ds, corr, " ".join(f"{k}={v['cer']:.4f}/{v['doc_cer']:.4f}" for k, v in p.items()), flush=True)
