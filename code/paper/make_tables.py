#!/usr/bin/env python3
"""Generate every table and every in-text number of the manuscript from the result JSON files.
Outputs: paper/tables/*.tex, paper/numbers.tex (\\R{key} macros), paper/numbers.json (key -> value, source).
No number in the manuscript is typed by hand.
"""
import os, json, glob, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
I2 = os.path.join(ROOT, "results/iter2"); I3 = os.path.join(ROOT, "results/iter3")
NEW = os.path.join(I3, "newcfg"); TB = os.path.join(ROOT, "generated/tables"); os.makedirs(TB, exist_ok=True)
J = lambda p: json.load(open(p))
NUM = {}

def N(key, val, src):
    assert re.fullmatch(r"[A-Za-z0-9.]+", key), key
    NUM[key] = dict(value=str(val), source=os.path.relpath(src, ROOT) if os.path.isabs(src) else src)

CORR = dict(symspell="SymSpell", byt5="ByT5-small", byt5base8="ByT5-base", byt5base8m="ByT5-base-ML", byt5base8x="ByT5-base", qwen="Qwen2.5-7B",
            qwenctx2="Qwen2.5-7B+ctx", qwenv2="Qwen2.5-7B", qwenp2="Qwen2.5-7B (prompt B)", qwenp3="Qwen2.5-7B (prompt C)", llama="Llama-3.1-8B")
DS = dict(blnocr="BLN600", blnocrw50="BLN600-w50", blnocrw100="BLN600-w100", bln600="BLN600-legacy", icdarc_EN="ICDAR-EN", icdarc_DE="ICDAR-DE",
          icdarc_FR="ICDAR-FR", icdarc_ES="ICDAR-ES", icdarc_NL="ICDAR-NL", icdarc_multi="ICDAR-multi", ext_impresso_en="impresso", ext_overproof_en="Overproof")
DSORD = list(DS); CORD = list(CORR)
CORPUS = {"blnocr+blnocrw50+blnocrw100+bln600": "BLN600"}
cname = lambda c: CORPUS.get(c, DS.get(c, c))
f4 = lambda x: "%.4f" % x
f3 = lambda x: "%.3f" % x
sg = lambda x: ("$-$" if x < 0 else "$+$") + "%.4f" % abs(x)
wl = lambda s: s.replace(" / ", "/")


def load(dirn, kind):
    R = {}
    for p in glob.glob(os.path.join(dirn, kind + "_*.json")):
        o = J(p); R[(o["corrector"], o["dataset"])] = o
    return dict(sorted(R.items(), key=lambda kv: (DSORD.index(kv[0][1]), CORD.index(kv[0][0]))))


def write(name, body):
    open(os.path.join(TB, name + ".tex"), "w").write(body + "\n"); print("table", name)


def bold_min(vals, fmt=f4):
    m = min(v for v in vals if v is not None)
    return [("—" if v is None else ("\\textbf{%s}" % fmt(v) if fmt(v) == fmt(m) else fmt(v))) for v in vals]   # ties at the printed precision are all bold


# ---------------------------------------------------------------- main ledger
LED, EDL = load(I2, "ledger"), load(I2, "editlevel")
RS = J(os.path.join(I2, "robust_summary.json")); RSP = os.path.join(I2, "robust_summary.json")
N("n.cfg", len(LED), I2 + "/ledger_*.json"); N("n.corpora", len(RS["corpora"]), RSP)
worse = [k for k, o in LED.items() if o["policies"]["always"]["cer"] > o["policies"]["never"]["cer"]]
N("always.worse", len(worse), I2 + "/ledger_*.json")
_B = {int(re.search(r"B=(\d+)", o["protocol"]).group(1)) for o in LED.values()}; assert len(_B) == 1, _B
N("boot.B", "{:,}".format(_B.pop()).replace(",", "{,}"), I2 + "/ledger_*.json")
sigw = [k for k, o in LED.items() if (b := o["bootstrap"][o["full_router"]]) and False]


def t_data():
    rows, seen = [], set()
    L = ["\\begin{tabular}{llrrrrrl}", "\\toprule",
         "Test set & Lang. & \\multicolumn{2}{c}{Dev} & \\multicolumn{2}{c}{Test} & OCR CER & Correctors \\\\",
         " & & docs & segs & docs & segs & (test) & \\\\", "\\midrule"]
    lang = dict(blnocr="en", blnocrw50="en", blnocrw100="en", bln600="en", icdarc_EN="en", icdarc_DE="de", icdarc_FR="fr", icdarc_ES="es", icdarc_NL="nl",
                icdarc_multi="en,de,fr,es,nl", ext_impresso_en="en", ext_overproof_en="en")
    for d in DSORD:
        cs = [(c, o) for (c, dd), o in LED.items() if dd == d]
        if not cs: continue
        o = cs[0][1]
        L.append("%s & %s & %d & %d & %d & %d & %s & %s \\\\" % (DS[d], lang[d], o["n_dev_docs"], o["n_dev"], o["n_test_docs"], o["n_test"],
                                                                f4(o["policies"]["never"]["cer"]), ", ".join(dict.fromkeys(CORR[c] for c, _ in cs))))
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_data", "\n".join(L))


def t_cer():
    cols = [("never", "No corr."), ("always", "Always"), ("edit_cap", "Edit cap"), ("conf_min", "Min-conf."), ("kim_min095", "Conf.$\\ge$0.95"), ("FR", "Router"),
            ("E:edit_size", "Edit size"), ("E:edit_router", "Edit router"), ("E:cascade", "Cascade"), ("oracle", "Seg.\\ oracle"), ("E:oracle_edit", "Edit oracle")]
    L = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Test set & Corrector & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    prev = None; M = {k: [] for k, _ in cols}
    for (c, d), o in LED.items():
        e = EDL.get((c, d)); vals = []
        for k, _ in cols:
            if k == "FR": v = o["policies"][o["full_router"]]["cer"]
            elif k.startswith("E:"): v = e["policies"].get(k[2:], {}).get("cer") if e else None
            else: v = o["policies"].get(k, {}).get("cer")
            vals.append(v); M[k].append(v)
        nb = len(cols) - 2; cells = bold_min(vals[:nb]) + [("—" if v is None else f4(v)) for v in vals[nb:]]
        if prev and prev != DS[d].split("-")[0] and not (prev == "BLN600" and d.startswith("bln")): L.append("\\addlinespace[2pt]")
        prev = DS[d].split("-")[0] if d.startswith("bln") else DS[d]
        L.append("%s & %s & " % (DS[d], CORR[c]) + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{2}{l}{Mean (configurations with a value)} & " + " & ".join(f4(np.mean([v for v in M[k] if v is not None])) for k, _ in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_cer", "\n".join(L))
    for k, key in [("never", "never"), ("always", "always"), ("edit_cap", "cap"), ("FR", "router"), ("E:edit_router", "editrouter"), ("E:edit_size", "editsize"),
                   ("E:cascade", "cascade"), ("oracle", "oracleseg"), ("E:oracle_edit", "oracleedit"), ("conf_min", "confmin")]:
        N("meancer." + key, f4(np.mean([v for v in M[k] if v is not None])), I2 + "/ledger_*.json, editlevel_*.json")
    for (c, d), key in [(("byt5", "icdarc_EN"), "ex.byt5en"), (("byt5base8", "blnocr"), "ex.b8bln"), (("byt5base8m", "icdarc_DE"), "ex.b8de")]:
        o = LED[(c, d)]["policies"]; N(key + ".never", f4(o["never"]["cer"]), I2); N(key + ".always", f4(o["always"]["cer"]), I2)


OPS = [("opt", "CER-optimal"), ("10pct", "10\\% budget"), ("25pct", "25\\% budget"), ("50pct", "50\\% budget")]
CMP = [("seg_router|seg_edit_cap", "rc", "Router vs.\\ edit cap"), ("seg_router|seg_conf_min", "rm", "Router vs.\\ min-confidence"),
       ("seg_router_text|seg_edit_cap", "tc", "Text-only router vs.\\ edit cap"), ("seg_router|seg_conf_plus_cap", "rp", "Router vs.\\ confidence + cap"),
       ("seg_router_text|seg_conf_plus_cap", "tp", "Text-only router vs.\\ confidence + cap"),
       ("edit_router|seg_router", "es", "Edit router vs.\\ router"), ("edit_router|edit_size", "ez", "Edit router vs.\\ edit size"),
       ("edit_size|seg_edit_cap", "zc", "Edit size vs.\\ edit cap"), ("edit_router_nolp|seg_router", "en", "Edit router w/o per-edit conf.\\ vs.\\ router")]


def t_same23():
    """edit router vs segment router restricted to the configurations with per-edit confidence (same population as the ablated edit router)"""
    import glob
    rb = [J(p) for p in sorted(glob.glob(os.path.join(I2, "robust", "robust_*.json")))]
    rb = [o for o in rb if o.get("has_edit_lp") and "edit_router_nolp|seg_router" in o["comparisons"]]
    for op, _ in OPS:
        o_ = "opt" if op == "opt" else "b" + op[:2]
        xs = [o["comparisons"]["edit_router|seg_router"][op] for o in rb if op in o["comparisons"]["edit_router|seg_router"] and op in o["comparisons"]["edit_router_nolp|seg_router"]]
        a = holm([x["p"] for x in xs]); w = sum(x["delta"] < 0 and p < .05 for x, p in zip(xs, a)); l = sum(x["delta"] > 0 and p < .05 for x, p in zip(xs, a))
        N("es23.%s" % o_, "%d / %d / %d" % (w, l, len(xs) - w - l), I2 + "/robust/robust_*.json"); N("es23.%s.n" % o_, len(xs), I2 + "/robust/robust_*.json")


def t_wl():
    L = ["\\begin{tabular}{llrcccccr}", "\\toprule",
         "Comparison & Operating point & $n$ & Raw & Holm & Holm, & Holm, & Corpus- & Mean $\\Delta$CER \\\\",
         " & & & & (all) & OCR-only seg. & doc.\\ CER & pooled & \\\\", "\\midrule"]
    for name, al, lab in CMP:
        if name not in RS["comparisons"]: continue
        for i, (op, oplab) in enumerate(OPS):
            x = RS["comparisons"][name].get(op)
            if not x: continue
            L.append("%s & %s & %d & %s & %s & %s & %s & %s & %s \\\\" % (lab if i == 0 else "", oplab, x["n"], wl(x["raw"]), wl(x["holm_all"]), wl(x["holm_ocr_only"]),
                                                                        wl(x["holm_doc_cer"]), wl(x["corpus_pooled"]), sg(x["mean_delta"])))
            o = "opt" if op == "opt" else "b" + op[:2]
            for f, k in [("raw", "raw"), ("holm_all", "holm"), ("holm_ocr_only", "ocr"), ("holm_doc_cer", "doc"), ("corpus_pooled", "pool")]:
                N("%s.%s.%s" % (al, o, k), x[f], RSP)
                w, l, t = [s.strip() for s in x[f].split("/")]
                N("%s.%s.%s.w" % (al, o, k), w, RSP); N("%s.%s.%s.l" % (al, o, k), l, RSP); N("%s.%s.%s.t" % (al, o, k), t, RSP)
            N("%s.%s.n" % (al, o), x["n"], RSP); N("%s.%s.delta" % (al, o), sg(x["mean_delta"]), RSP)
            pc = {k: v for k, v in x["per_corpus"].items() if k != "icdarc_multi"}   # disjoint collections (ICDAR-multi is the union of the language sets)
            N("%s.%s.dis.w" % (al, o), sum(v["delta"] < 0 and v["p_holm"] < .05 for v in pc.values()), RSP); N("%s.%s.dis.l" % (al, o), sum(v["delta"] > 0 and v["p_holm"] < .05 for v in pc.values()), RSP)
            N("n.disjoint", len(pc), RSP)
            N("%s.%s.win" % (al, o), ", ".join(cname(c) for c in x["corpora_win"]) or "none", RSP)
            N("%s.%s.loss" % (al, o), ", ".join(cname(c) for c in x["corpora_loss"]) or "none", RSP)
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]; write("tab_winloss", "\n".join(L))
    for k, v in RS["mean_cer_opt_corpus_balanced"].items(): N("balcer." + k.replace("_", ""), f4(v), RSP)
    for k, v in RS["ltt_min_cal_docs"].items(): N("ltt.mindocs." + k.replace("0.", "a"), v, RSP)
    N("nolp.n", RS["nolp"]["n"], RSP); N("nolp.meanchange", sg(RS["nolp"]["mean_cer_change"]), RSP); N("nolp.maxabs", f4(RS["nolp"]["max_abs_change"]), RSP)
    N("fr.dropped", RS["fr_overlap"]["dropped_segments"], RSP); N("fr.ntest", RS["fr_overlap"]["n_test"], RSP)


def t_harm():
    L = ["\\begin{tabular}{llrrrrcc}", "\\toprule",
         "Budget & Rule & $n$ & Over budget & Median ratio & Max ratio & Router at cap's harm & Router vs.\\ cap at router's harm \\\\", "\\midrule"]
    lab = dict(seg_router="Router", seg_edit_cap="Edit cap", seg_conf_min="Min-confidence", edit_router="Edit router", edit_size="Edit size")
    for op in ["10pct", "25pct", "50pct"]:
        first = True
        for r in lab:
            x = RS["realised_harm"].get("%s|%s" % (op, r))
            if not x: continue
            m1 = RS["matched_realised"].get("%s|seg_router@cap_harm|seg_edit_cap" % op, {}).get("holm", ""); m2 = RS["matched_realised"].get("%s|seg_router|seg_edit_cap@router_harm" % op, {}).get("holm", "")
            L.append("%s & %s & %d & %d & %.2f & %.2f & %s & %s \\\\" % (op[:2] + "\\%" if first else "", lab[r], x["n"], x["over"], x["median_ratio"], x["max_ratio"],
                                                                      wl(m1) if first else "", wl(m2) if first else ""))
            o = "b" + op[:2]; rk = r.replace("seg_", "").replace("_", "")
            N("harm.%s.%s.over" % (o, rk), x["over"], RSP); N("harm.%s.%s.n" % (o, rk), x["n"], RSP); N("harm.%s.%s.med" % (o, rk), "%.2f" % x["median_ratio"], RSP)
            N("harm.%s.%s.max" % (o, rk), "%.1f" % x["max_ratio"], RSP)
            if first:
                N("match.%s.capharm" % o, m1, RSP); N("match.%s.routerharm" % o, m2, RSP)
                N("match.%s.capharm.delta" % o, sg(RS["matched_realised"]["%s|seg_router@cap_harm|seg_edit_cap" % op]["mean_delta"]), RSP)
            first = False
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]; write("tab_harm", "\n".join(L))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj


def tally(deltas, ps):
    """first-vs-second W/L/T with Holm across the family; delta<0 = first better"""
    a = holm(ps); w = sum(d < 0 and p < .05 for d, p in zip(deltas, a)); l = sum(d > 0 and p < .05 for d, p in zip(deltas, a))
    return "%d / %d / %d" % (w, l, len(deltas) - w - l)


def t_ablation():
    """router feature variants vs the full router (ledger bootstrap: delta = other - full, so it is negated to make negative = full better)"""
    var = [("router_text", "Text features only (13)"), ("router_align", "+ aligned token-diff features"), ("router_nodict", "Dictionary-free"),
           ("router_conf_qe", "Full + QE-encoder score"), ("router_util", "Utility-regression target"), ("errdet_ocr_only", "OCR-only error detector"),
           ("docqe", "Document-level QE"), ("qe_encoder", "Trained QE encoder"), ("conf_mean", "Mean-token confidence"), ("conf_plus_cap", "Confidence + cap"),
           ("kim_min095", "Min-confidence $\\ge 0.95$ (fixed)"), ("random_matched", "Random, matched accept rate")]
    L = ["\\begin{tabular}{lrrcccc}", "\\toprule",
         "Alternative rule & $n$ & Mean CER & \\multicolumn{4}{c}{Full router better / worse / n.s.\\ (Holm across configurations)} \\\\",
         " & & & CER-opt. & 10\\% & 25\\% & 50\\% \\\\", "\\midrule"]
    for k, lab in var:
        ds, ps, cer = [], [], []; B = {b: ([], []) for b in ["10pct", "25pct", "50pct"]}
        for o in LED.values():
            fr = o["full_router"]
            if k == fr or k not in o["policies"]: continue
            x = o["bootstrap"][fr].get(k)
            if x is None: continue
            ds.append(-x["delta"]); ps.append(x["p_two_sided"]); cer.append(o["policies"][k]["cer"])
            for b in B:
                y = o["devtuned_budget_frontier"][b].get(k, {}).get("vs_full")
                if y: B[b][0].append(-y["delta"]); B[b][1].append(y["p_two_sided"])
        if not ds: continue
        cells = [wl(tally(ds, ps))] + [wl(tally(*B[b])) if B[b][0] else "—" for b in B]
        L.append("%s & %d & %s & %s \\\\" % (lab, len(ds), f4(np.mean(cer)), " & ".join(cells)))
        kk = k.replace("_", "")
        N("abl.%s.n" % kk, len(ds), I2 + "/ledger_*.json"); N("abl.%s.opt" % kk, tally(ds, ps), I2 + "/ledger_*.json")
        for b in B:
            if B[b][0]: N("abl.%s.b%s" % (kk, b[:2]), tally(*B[b]), I2 + "/ledger_*.json")
        if k == "kim_min095": N("abl.kim.routerbetter", sum(d < 0 and p < .05 for d, p in zip(ds, holm(ps))), I2)
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_ablation", "\n".join(L))


def t_edit():
    L = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Test set & Corrector & Edits/seg. & Helpful & Harmful & Mixed seg. & Edit AUC & $\\Delta$ edit$-$seg.\\ router \\\\", "\\midrule"]
    mixed = []
    kx = [k for k, e in EDL.items() if e["bootstrap"].get("seg_" + e["full_router"], {}).get("edit_router")]
    adj = dict(zip(kx, holm([EDL[k]["bootstrap"]["seg_" + EDL[k]["full_router"]]["edit_router"]["p_two_sided"] for k in kx])))   # Holm across configurations
    for (c, d), e in EDL.items():
        g = e["diagnostics"]; b = e["bootstrap"]; fr = "seg_" + e["full_router"]
        x = b.get(fr, {}).get("edit_router"); y = b.get("edit_router", {}).get("edit_size")
        st = lambda z, neg=False: "—" if z is None else sg(-z["delta"] if neg else z["delta"]) + ("$^{*}$" if adj[(c, d)] < .05 else "")
        L.append("%s & %s & %.2f & %.1f & %.1f & %.1f & %s & %s \\\\" % (DS[d], CORR[c], g["edits_per_attempted_seg"], 100 * g["edit_helpful"], 100 * g["edit_harmful"],
                                                                         100 * g["mixed_segments"], f3(g["edit_router_auc"]) if g.get("edit_router_auc") else "—", st(x)))
        mixed.append(g["mixed_segments"])
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_edit", "\n".join(L))
    N("edit.mixed.mean", "%.1f" % (100 * np.mean(mixed)), I2 + "/editlevel_*.json"); N("edit.mixed.min", "%.1f" % (100 * min(mixed)), I2); N("edit.mixed.max", "%.1f" % (100 * max(mixed)), I2)
    ex = [e["diagnostics"].get("oracle_exhaustive_rows") for e in EDL.values() if e["diagnostics"].get("oracle_exhaustive_rows") is not None]
    if ex: N("edit.oracle.exhaustive.min", "%.1f" % (100 * min(ex)), I2 + "/editlevel_*.json")
    N("edit.haslp.n", sum(bool(e["diagnostics"].get("has_edit_lp")) for e in EDL.values()), I2 + "/editlevel_*.json")


def t_verifier():
    V = load(I2, "verifier")
    L = ["\\begin{tabular}{llrrrrrrc}", "\\toprule",
         "Test set & Corrector & \\multicolumn{2}{c}{AUC} & \\multicolumn{3}{c}{CER} & $\\Delta$ router$-$verifier & Order \\\\",
         " & & router & verifier & router & verifier & router+ver. & & disagreement \\\\", "\\midrule"]
    ds, ps, auc_w, plus = [], [], 0, 0; dis = []; ms = []; rauc = []; vauc = []
    adj = dict(zip(V, holm([o["bootstrap"][o["full_router"]]["verifier"]["p_two_sided"] for o in V.values()])))   # Holm across configurations
    yk = [k for k, o in V.items() if o["bootstrap"][o["full_router"]].get("router+ver")]
    adjp = dict(zip(yk, holm([V[k]["bootstrap"][V[k]["full_router"]]["router+ver"]["p_two_sided"] for k in yk])))
    for (c, d), o in V.items():
        p = o["policies"]; fr = o["full_router"]; x = o["bootstrap"][fr]["verifier"]; ds.append(-x["delta"]); ps.append(x["p_two_sided"])
        auc_w += p[fr]["auc_test"] > p["verifier"]["auc_test"]; rauc.append(p[fr]["auc_test"]); vauc.append(p["verifier"]["auc_test"])
        y = o["bootstrap"][fr].get("router+ver"); plus += bool(y and y["delta"] < 0 and adjp[(c, d)] < .05)
        t = o.get("verifier_metrics", {}).get("test", {}) if isinstance(o.get("verifier_metrics"), dict) else {}
        dg = (t["order_disagree"] / t["attempted"]) if t.get("attempted") else None; dis.append(dg); ms.append(t.get("ms_per_attempted_segment"))
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (DS[d], CORR[c], f3(p[fr]["auc_test"]), f3(p["verifier"]["auc_test"]), f4(p[fr]["cer"]), f4(p["verifier"]["cer"]),
                                                                  f4(p["router+ver"]["cer"]), sg(-x["delta"]) + ("$^{*}$" if adj[(c, d)] < .05 else ""), "—" if dg is None else "%.0f\\%%" % (100 * dg)))
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_verifier", "\n".join(L))
    src = I2 + "/verifier_*.json"
    N("ver.n", len(V), src); N("ver.aucwins", auc_w, src); N("ver.cer", tally(ds, ps), src); N("ver.plushelps", plus, src)
    N("ver.rauc.min", "%.2f" % min(rauc), src); N("ver.rauc.max", "%.2f" % max(rauc), src); N("ver.vauc.min", "%.2f" % min(vauc), src); N("ver.vauc.max", "%.2f" % max(vauc), src)
    d2 = [x for x in dis if x is not None]; m2 = [x for x in ms if x is not None]
    if d2: N("ver.dis.min", "%.0f" % (100 * min(d2)), src); N("ver.dis.max", "%.0f" % (100 * max(d2)), src)
    if m2: N("ver.ms.min", "%.0f" % min(m2), src); N("ver.ms.max", "%.0f" % max(m2), src)
    for b in ["10pct", "25pct", "50pct"]:
        dd, pp = [], []
        for o in V.values():
            y = o["devtuned_budget_frontier"][b].get(o["full_router"], {}).get("vs_verifier")
            if y: dd.append(y["delta"]); pp.append(y["p_two_sided"])
        if dd: N("ver.b%s" % b[:2], tally(dd, pp), src)


def t_ltt():
    L = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "$\\alpha$ & Min.\\ cal.\\ docs & \\multicolumn{2}{c}{Configurations abstaining} & \\multicolumn{2}{c}{Mean accept rate} & \\multicolumn{2}{c}{Splits exceeding $\\alpha$ on test} \\\\",
         " & & router & edit cap & router & edit cap & router or cap & any rule \\\\", "\\midrule"]
    for a in ["0.1", "0.05", "0.02"]:
        ab = {"R": 0, "edit_cap": 0}; ar = {"R": [], "edit_cap": []}; viol = 0; vrc = 0; nsp = 0
        for o in LED.values():
            rc = o["risk_control"]["alpha" + a]; fr = o["full_router"]
            for k, kk in [("R", fr), ("edit_cap", "edit_cap")]:
                ab[k] += rc[kk]["accept_rate_mean"] == 0; ar[k].append(rc[kk]["accept_rate_mean"])
            viol += int(round(20 * sum(v["violation_frac"] for v in rc.values())))   # events over the 20 calibration splits
            vrc += int(round(20 * (rc[fr]["violation_frac"] + rc["edit_cap"]["violation_frac"]))); nsp += 20
        L.append("%s & %d & %d & %d & %.3f & %.3f & %d & %d \\\\" % ("%.2f" % float(a), RS["ltt_min_cal_docs"][a], ab["R"], ab["edit_cap"], np.mean(ar["R"]), np.mean(ar["edit_cap"]), vrc, viol))
        ncal = [o["risk_control"]["alpha" + a]["edit_cap"]["n_cal_docs"] for o in LED.values()]
        N("ltt.toofew.%s" % a.replace("0.", "a"), sum(n < RS["ltt_min_cal_docs"][a] for n in ncal), I2 + "/ledger_*.json"); N("ltt.ncal.min", min(ncal), I2); N("ltt.ncal.max", max(ncal), I2)
        ak = a.replace("0.", "a"); N("ltt.abstain.router.%s" % ak, ab["R"], I2 + "/ledger_*.json"); N("ltt.abstain.cap.%s" % ak, ab["edit_cap"], I2); N("ltt.viol.%s" % ak, viol, I2); N("ltt.violrc.%s" % ak, vrc, I2); N("ltt.splits", nsp, I2)
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_ltt", "\n".join(L))


def t_transfer():
    T = RS["transfer"]
    L = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
         "Source $\\rightarrow$ target & Corrector & & No corr. & Always & Cap (transf.) & Router (transf.) & Router (in-domain) & Oracle \\\\", "\\midrule"]
    TJ = {(t["corrector"], t["source"], t["target"]): t for t in J(os.path.join(I2, "transfer.json"))["transfer"]}
    ext_ok = ext_n = indom = 0
    for k in ["never", "always"]:   # Holm across the rows of the table, per comparison
        tt = [t for t in T if t["vs"].get(k)]
        for t, a in zip(tt, holm([t["vs"][k]["p"] for t in tt])): t["vs"][k]["p_adj"] = a
    for t in T:
        c = t["cer"]; v = t["vs"]; star = lambda k: "$^{*}$" if v.get(k) and v[k]["delta"] < 0 and v[k]["p_adj"] < .05 else ""
        orc = TJ.get((t["corrector"], t["source"], t["target"]), {}).get("transfer", {}).get("oracle")
        L.append("%s $\\rightarrow$ %s & %s & & %s & %s & %s & %s & %s & %s \\\\" % (DS[t["source"]], DS[t["target"]], CORR[t["corrector"]], f4(c["never"]) + star("never"), f4(c["always"]) + star("always"),
                                                                              f4(c["cap_transfer"]), f4(c["router_transfer"]), f4(c["router_in_domain"]), f4(orc) if orc else "—"))
        if t["target"].startswith("ext_") and t["corrector"] != "symspell":
            ext_n += 1; ext_ok += all(v[k]["delta"] < 0 and v[k]["p_adj"] < .05 for k in ["never", "always"]); indom += c["router_transfer"] < c["router_in_domain"]
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_transfer", "\n".join(L))
    N("tr.ext.n", ext_n, RSP); N("tr.ext.ok", ext_ok, RSP); N("tr.ext.beatsindomain", indom, RSP)
    be = [t for t in T if t["source"] == "blnocr" and t["target"] == "icdarc_EN"]
    N("tr.blnen.n", len(be), RSP); N("tr.blnen.worse", sum(t["cer"]["router_transfer"] > t["cer"]["never"] for t in be), RSP)


def t_seeds():
    SD = os.path.join(I2, "seeds"); FAM = {"byt5base8": ["blnocr", "icdarc_EN", "ext_impresso_en", "ext_overproof_en"], "byt5base8m": ["icdarc_EN", "icdarc_DE", "icdarc_FR", "icdarc_ES", "icdarc_NL"]}
    cells = {}; out = {}
    for fam, dss in FAM.items():
        for ds in dss:
            for corr in [fam, fam + "s1", fam + "s2"]:
                pl, pe = os.path.join(SD, "ledger_%s_%s.json" % (corr, ds)), os.path.join(SD, "editlevel_%s_%s.json" % (corr, ds))
                if not (os.path.exists(pl) and os.path.exists(pe)): continue
                a, b = J(pl), J(pe); fr = a["full_router"]
                for b_ in ["10pct", "25pct", "50pct"]:
                    for base in ["edit_cap", "conf_min"]:
                        x = a["devtuned_budget_frontier"][b_].get(base, {}).get("vs_full")
                        if x: cells.setdefault((base, b_), {}).setdefault((ds, fam), []).append((x["delta"], x["p_holm"] < .05))
                for base in ["edit_cap", "conf_min"]:
                    x = a["bootstrap"][fr].get(base)
                    if x: cells.setdefault((base, "opt"), {}).setdefault((ds, fam), []).append((x["delta"], x["p_holm"] < .05))   # other - full: >0 = router better
                x = b["bootstrap"]["seg_" + fr].get("edit_router")
                if x: cells.setdefault(("edit_router", "opt"), {}).setdefault((ds, fam), []).append((-x["delta"], x["p_holm"] < .05))
                out[(ds, fam, corr)] = (a, b)
    L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         "Comparison & Operating point & Cells & Same sign & First better & First sig.\\ better & Sig.\\ in opposite \\\\", " & & & in all 3 & in all 3 & in all 3 & directions \\\\", "\\midrule"]
    lab = dict(edit_cap="Router vs.\\ edit cap", conf_min="Router vs.\\ min-confidence", edit_router="Edit router vs.\\ router")
    for (base, op), cc in cells.items():
        cc = {k: v for k, v in cc.items() if len(v) == 3}; n = len(cc)
        if not n: continue
        same = sum(len({np.sign(x[0]) for x in v}) == 1 for v in cc.values()); bet = sum(all(x[0] > 0 for x in v) for v in cc.values())
        sig = sum(all(x[0] > 0 and x[1] for x in v) for v in cc.values()); flip = sum(any(x[1] and x[0] < 0 for x in v) and any(x[1] and x[0] > 0 for x in v) for v in cc.values())
        cells[(base, op)] = dict(n=n, same=same, bet=bet, sig=sig, flip=flip)
    for base in ["edit_cap", "conf_min", "edit_router"]:
        for op in ["opt", "10pct", "25pct", "50pct"]:
            s = cells.get((base, op))
            if not isinstance(s, dict) or "n" not in s: continue
            L.append("%s & %s & %d & %d & %d & %d & %d \\\\" % (lab[base], dict(OPS)[op], s["n"], s["same"], s["bet"], s["sig"], s["flip"]))
            o = "opt" if op == "opt" else "b" + op[:2]; bk = base.replace("_", "")
            for k in ["n", "same", "bet", "sig", "flip"]: N("seed.%s.%s.%s" % (bk, o, k), s[k], I2 + "/seeds/*.json")
        L.append("\\addlinespace[2pt]")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]; write("tab_seeds", "\n".join(L))
    json.dump({"%s|%s|%s" % k: dict(always=a["policies"]["always"]["cer"], router=a["policies"][a["full_router"]]["cer"], edit_cap=a["policies"]["edit_cap"]["cer"],
                                    conf_min=a["policies"]["conf_min"]["cer"], edit_router=b["policies"]["edit_router"]["cer"], never=a["policies"]["never"]["cer"],
                                    b25={r: a["devtuned_budget_frontier"]["25pct"].get(r, {}).get("test_cer") for r in [a["full_router"], "edit_cap", "conf_min"]}, fr=a["full_router"])
               for k, (a, b) in out.items()}, open(os.path.join(ROOT, "generated/figures/data/seeds.json"), "w"), indent=1)


def t_newcfg():
    """held-out replication: Qwen prompt variants and Llama-3.1-8B (iter-3); analysis frozen before these candidates existed"""
    if not glob.glob(os.path.join(NEW, "ledger_*.json")): print("skip newcfg"); return
    Ln, En = load(NEW, "ledger"), load(NEW, "editlevel")
    L = ["\\begin{tabular}{llrrrrrrcc}", "\\toprule",
         "Test set & Corrector & No corr. & Always & Edit cap & Router & Edit router & Seg.\\ oracle & \\multicolumn{2}{c}{$\\Delta$ router$-$cap} \\\\",
         " & & & & & & & & CER-opt. & 25\\% budget \\\\", "\\midrule"]
    T = {op: ([], []) for op in ["opt", "10pct", "25pct", "50pct"]}; TE = ([], []); aw = 0; rows = []
    for (c, d), o in sorted(Ln.items(), key=lambda kv: (CORD.index(kv[0][0]), DSORD.index(kv[0][1]))):
        p = o["policies"]; fr = o["full_router"]; x = o["bootstrap"][fr]["edit_cap"]; T["opt"][0].append(-x["delta"]); T["opt"][1].append(x["p_two_sided"])
        cell = {}
        for b in ["10pct", "25pct", "50pct"]:
            y = o["devtuned_budget_frontier"][b].get("edit_cap", {}).get("vs_full")
            if y: T[b][0].append(-y["delta"]); T[b][1].append(y["p_two_sided"]); cell[b] = y
        e = En.get((c, d)); er = e["policies"]["edit_router"]["cer"] if e else None
        if e:
            z = e["bootstrap"]["seg_" + fr].get("edit_router")
            if z: TE[0].append(z["delta"]); TE[1].append(z["p_two_sided"])
        aw += p["always"]["cer"] > p["never"]["cer"]
        y = cell.get("25pct"); vals = bold_min([p["never"]["cer"], p["always"]["cer"], p["edit_cap"]["cer"], p[fr]["cer"], er])
        rows.append([(c, d), "%s & %s & %s & %s" % (DS[d], CORR[c], " & ".join(vals), f4(p["oracle"]["cer"])), x, y])
    a1 = holm([r[2]["p_two_sided"] for r in rows]); r2 = [r for r in rows if r[3]]; a2 = dict(zip([r[0] for r in r2], holm([r[3]["p_two_sided"] for r in r2])))   # Holm across configurations
    for (k, head, x, y), ax in zip(rows, a1):
        L.append("%s & %s & %s \\\\" % (head, sg(-x["delta"]) + ("$^{*}$" if ax < .05 else ""), "—" if not y else sg(-y["delta"]) + ("$^{*}$" if a2[k] < .05 else "")))
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_newcfg", "\n".join(L))
    src = "results/iter3/newcfg/*.json"
    N("new.n", len(Ln), src); N("new.always.worse", aw, src)
    for op, (d, p) in T.items():
        if d: N("new.rc.%s" % ("opt" if op == "opt" else "b" + op[:2]), tally(d, p), src)
    if TE[0]: N("new.es.opt", tally(*TE), src)
    rs = os.path.join(NEW, "robust_summary.json")
    if os.path.exists(rs):
        R = J(rs)
        for name, al, _ in CMP:
            for op, _ in OPS:
                x = R["comparisons"].get(name, {}).get(op)
                if x:
                    o = "opt" if op == "opt" else "b" + op[:2]
                    N("new.%s.%s.holm" % (al, o), x["holm_all"], rs); N("new.%s.%s.pool" % (al, o), x["corpus_pooled"], rs); N("new.%s.%s.delta" % (al, o), sg(x["mean_delta"]), rs)
                    N("new.%s.%s.n" % (al, o), x["n"], rs)


def t_lolo():
    """leave-one-language-out transfer on ICDAR-2019 (iter-3)"""
    p = os.path.join(I3, "lolo.json")
    if not os.path.exists(p): print("skip lolo"); return
    X = J(p); json.dump(X, open(os.path.join(ROOT, "generated/figures/data/lolo.json"), "w"))
    L = ["\\begin{tabular}{llrrrrrrrr}", "\\toprule",
         "Corrector & Held-out & No corr. & Always & Cap & Router & Router & Dict.-free & Dict.-free & Oracle \\\\",
         " & language & & & (transf.) & (transf.) & (in-lang.) & (transf.) & (in-lang.) & \\\\", "\\midrule"]
    st = lambda r, k: "$^{*}$" if r["comp"].get(k, {}).get("p_holm", 1) < .05 else ""
    for r in X["cells"]:
        c = r["cer"]
        L.append("%s & %s & %s%s & %s%s & %s%s & %s & %s%s & %s & %s%s & %s \\\\" % (
            CORR[r["corrector"]], r["lang"], f4(c["never"]), st(r, "full vs never"), f4(c["always"]), st(r, "full vs always"),
            f4(c["edit_cap"]), st(r, "full vs edit_cap"), f4(c["full"]), f4(c["in:full"]), st(r, "full vs in:full"),
            f4(c["router_nodict"]), f4(c["in:router_nodict"]), st(r, "router_nodict vs in:router_nodict"), f4(c["oracle"])))
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_lolo", "\n".join(L))
    src = "results/iter3/lolo.json"; N("lolo.n", len(X["cells"]), src)
    for k, a in [("full vs never", "never"), ("full vs always", "always"), ("full vs edit_cap", "cap"), ("full vs conf_min", "conf"),
                 ("full@25pct vs edit_cap@25pct", "cap25"), ("full vs in:full", "indomain"), ("router_nodict vs in:router_nodict", "nodict.indomain")]:
        if k in X["tally"]: N("lolo." + a, X["tally"][k].replace("/", " / "), src)
    worse = sum(r["cer"]["full"] > r["cer"]["never"] for r in X["cells"]); N("lolo.worse.than.never", worse, src)
    N("lolo.mean.gap", sg(float(np.mean([r["cer"]["full"] - r["cer"]["in:full"] for r in X["cells"]]))), src)
    N("lolo.nodict.mean.gap", sg(float(np.mean([r["cer"]["router_nodict"] - r["cer"]["in:router_nodict"] for r in X["cells"]]))), src)


def t_b25():
    """appendix: per-configuration coverage, absolute harm and CER at the 25% harm budget; always-accept harm"""
    L = ["\\begin{tabular}{llrrrrrrrrrrrl}", "\\toprule",
         "Test set & Corrector & \\multicolumn{2}{c}{Always accept} & \\multicolumn{3}{c}{Edit cap} & \\multicolumn{3}{c}{Min-confidence} & \\multicolumn{3}{c}{Router} & $\\Delta$CER router$-$cap \\\\",
         " & & harmed & \\% & CER & harmed & acc.\\,\\% & CER & harmed & acc.\\,\\% & CER & harmed & acc.\\,\\% & [95\\% CI]; relative \\\\", "\\midrule"]
    acc = {k: [] for k in ["edit_cap", "conf_min", "FR"]}; rel = []; hr = []
    for (c, d), o in LED.items():
        f = o["devtuned_budget_frontier"]["25pct"]; fr = o["full_router"]; al = o["policies"]["always"]
        cells = ["%d & %.1f" % (al["corrupt"], 100 * al["harm_rate"])]; hr.append(100 * al["harm_rate"])
        for k in ["edit_cap", "conf_min", "FR"]:
            x = f.get(fr if k == "FR" else k)
            if not x: cells.append("— & — & —"); continue
            cells.append("%s & %d & %.1f" % (f4(x["test_cer"]), x["test_corrupt"], 100 * x["accept_rate_attempted"])); acc[k].append(100 * x["accept_rate_attempted"])
        v = f["edit_cap"]["vs_full"]; r = -v["delta"] / f["edit_cap"]["test_cer"]; rel.append(100 * r)
        cells.append("%s [%s, %s]; %s%.1f\\%%" % (sg(-v["delta"]), sg(-v["ci95"][1]), sg(-v["ci95"][0]), "$-$" if r < 0 else "$+$", abs(100 * r)))
        L.append("%s & %s & %s \\\\" % (DS[d], CORR[c], " & ".join(cells)))
    L += ["\\bottomrule", "\\end{tabular}"]; write("tab_b25", "\n".join(L)); src = I2 + "/ledger_*.json"
    for k, kk in [("edit_cap", "cap"), ("conf_min", "conf"), ("FR", "router")]: N("b25.acc.%s" % kk, "%.1f" % np.mean(acc[k]), src)
    N("b25.rel.median", "%.1f" % abs(np.median(rel)), src); N("b25.rel.min", "%.1f" % abs(max(rel)), src); N("b25.rel.max", "%.1f" % abs(min(rel)), src)
    N("always.harm.min", "%.1f" % min(hr), src); N("always.harm.max", "%.1f" % max(hr), src)
    for k, kk in [("edit_cap", "cap"), ("conf_min", "conf"), ("FR", "router")]:   # accept rate at the CER-optimal point
        N("opt.acc.%s" % kk, "%.1f" % np.mean([100 * o["policies"][o["full_router"] if k == "FR" else k]["accept_rate_attempted"] for o in LED.values() if (o["full_router"] if k == "FR" else k) in o["policies"]]), src)



def t_lcurve():
    """appendix: router, cap and min-confidence at the CER-optimal point and at the 25% budget as a function of the number of development documents"""
    import glob
    X5 = os.path.join(I2, "extra5"); src = "results/iter2/extra5/lc_*.json"
    X = sorted([J(f) for f in glob.glob(os.path.join(X5, "lc_*.json"))], key=lambda o: (list(DS).index(o["dataset"]), list(CORR).index(o["corrector"])))
    sizes = sorted({c["dev_docs"] for o in X for c in o["curve"] if c["dev_docs"] < o["n_dev_docs"]}); hdr = ["%d" % n for n in sizes] + ["all"]
    L = ["\\begin{tabular}{ll" + "r" * len(hdr) + "@{\\hskip 10pt}" + "r" * len(hdr) + "}", "\\toprule",
         " & & \\multicolumn{%d}{c}{CER-optimal point} & \\multicolumn{%d}{c}{25\\%% harm budget} \\\\" % (len(hdr), len(hdr)),
         "\\cmidrule(lr){3-%d}\\cmidrule(lr){%d-%d}" % (2 + len(hdr), 3 + len(hdr), 2 + 2 * len(hdr)), "Configuration & Rule & " + " & ".join(hdr + hdr) + " \\\\", "\\midrule"]
    cnt = {}
    for o in X:
        rows = {("all" if c["dev_docs"] == o["n_dev_docs"] else str(c["dev_docs"])): c for c in o["curve"]}
        for i, (k, lab) in enumerate([("full", "Router"), ("edit_cap", "Edit cap"), ("conf_min", "Min-conf.")]):
            if k not in o["curve"][0]: continue
            cells = []
            for m in ["opt_cer", "b25_cer"]:
                for h in hdr:
                    if h not in rows: cells.append("—"); continue
                    v = rows[h][k][m]["mean"]; best = min(rows[h][q][m]["mean"] for q in ["full", "edit_cap", "conf_min"] if q in rows[h])
                    cells.append(("\\textbf{%s}" if v == best else "%s") % f4(v))
            L.append("%s & %s & " % (["%s, %s (%d)" % (DS[o["dataset"]], CORR[o["corrector"]], o["n_dev_docs"]), "\\quad OCR %s" % f4(o["never"]), ""][i], lab) + " & ".join(cells) + " \\\\")
        L.append("\\addlinespace[2pt]")
        for h, c in rows.items():
            for m, mm in [("opt_cer", "opt"), ("b25_cer", "b25")]:
                for q, qq in [("edit_cap", "cap"), ("conf_min", "conf")]:
                    if q in c: cnt.setdefault((mm, qq, h), []).append(c["full"][m]["mean"] < c[q][m]["mean"])
                cnt.setdefault((mm, "ocr", h), []).append(c["full"][m]["mean"] < o["never"])
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]; write("tab_lcurve", "\n".join(L))
    for (mm, qq, h), v in cnt.items(): N("lc.%s.%s.%s" % (mm, qq, h), sum(v), src); N("lc.%s.%s.%s.n" % (mm, qq, h), len(v), src)
    N("lc.opt.aboveocr.5", sum(not x for x in cnt[("opt", "ocr", "5")]), src); N("lc.b25.aboveocr.5", sum(not x for x in cnt[("b25", "ocr", "5")]), src)
    N("lc.reps", max(c["reps"] for o in X for c in o["curve"]), src); N("lc.n", len(X), src)
    sd = lambda k, h: np.mean([c[k]["b25_cer"]["sd"] / c[k]["b25_cer"]["mean"] for o in X for c in o["curve"] if str(c["dev_docs"]) == h and k in c])
    for k, kk in [("full", "router"), ("edit_cap", "cap")]: N("lc.b25.relsd.%s.5" % kk, "%.1f" % (100 * sd(k, "5")), src); N("lc.b25.relsd.%s.20" % kk, "%.1f" % (100 * sd(k, "20")), src)


def t_rank():
    """threshold-free summary with tied scores accepted together, and a paired document bootstrap (Holm across configurations)"""
    import glob
    X = [J(f) for f in glob.glob(os.path.join(I2, "extra5", "rank_*.json"))]; src = "results/iter2/extra5/rank_*.json"
    assert len(X) == len(LED)
    for key, kk, a, b in [("router|edit_cap", "", "router", "edit_cap"), ("router|conf_min", ".conf", "router", "conf_min"), ("router_text|edit_cap", ".text", "router_text", "edit_cap")]:
        Y = [o for o in X if key in o["vs"]]; d = [o["vs"][key]["delta"] for o in Y]; ps = [o["vs"][key]["p_two_sided"] for o in Y]
        N("rank%s.wl" % kk, tally(d, ps), src); N("rank%s.n" % kk, len(Y), src); N("rank.router.lower%s" % kk, sum(x < 0 for x in d), src)
        N("rank%s.first" % kk, f4(np.mean([o["point"][a] for o in Y])), src); N("rank%s.second" % kk, f4(np.mean([o["point"][b] for o in Y])), src)
    N("rank.router", f4(np.mean([o["point"]["router"] for o in X])), src); N("rank.cap", f4(np.mean([o["point"]["edit_cap"] for o in X])), src); N("rank.B", X[0]["B"], src)


def t_fulldev():
    """LTT-style procedure calibrated on the full development set (edit cap and min-confidence need no fitted model)"""
    p = os.path.join(I2, "extra_fulldev.json")
    if not os.path.exists(p): print("skip fulldev"); return
    X = J(p); src = "results/iter2/extra_fulldev.json"
    for a in ["0.1", "0.05", "0.02"]:
        ak = a.replace("0.", "a")
        for k, kk in [("edit_cap", "cap"), ("conf_min", "conf")]:
            v = [r["ltt_full_dev"]["alpha" + a][k] for r in X if k in r["ltt_full_dev"]["alpha" + a]]
            N("lttfull.%s.%s.n" % (kk, ak), len(v), src); N("lttfull.%s.%s.abstain" % (kk, ak), sum(x["abstain"] for x in v), src)
            N("lttfull.%s.%s.viol" % (kk, ak), sum(x["test_risk"] > float(a) for x in v), src)
            ac = [100 * x["accept_rate"] for x in v if not x["abstain"]]
            N("lttfull.%s.%s.acc" % (kk, ak), "%.1f" % np.mean(ac) if ac else "0", src)
    ds = [r["distinct_dev_scores"] for r in X]
    if isinstance(ds[0], dict):
        for k in sorted({k for d in ds for k in d}): N("grid.distinct.%s.min" % k.replace("_", ""), min(d[k] for d in ds if k in d), src)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "generated/figures/data"), exist_ok=True)
    for f in [t_data, t_cer, t_wl, t_same23, t_harm, t_ablation, t_edit, t_verifier, t_ltt, t_transfer, t_seeds, t_newcfg, t_lolo, t_b25, t_lcurve, t_rank, t_fulldev]:
        f()
    with open(os.path.join(ROOT, "generated/numbers.tex"), "w") as fh:
        fh.write("% generated by make_tables.py -- do not edit\n\\makeatletter\n")
        for k, v in sorted(NUM.items()):
            fh.write("\\@namedef{R@%s}{%s}\n" % (k, v["value"].replace("%", "\\%")))
        fh.write("\\newcommand{\\R}[1]{\\@ifundefined{R@#1}{\\textbf{[??#1]}\\PackageWarning{numbers}{undefined number #1}}{\\@nameuse{R@#1}}}\n\\makeatother\n")
    json.dump(NUM, open(os.path.join(ROOT, "generated/numbers.json"), "w"), indent=1, sort_keys=True)
    print(len(NUM), "numbers")
