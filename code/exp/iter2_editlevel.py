#!/usr/bin/env python3
"""B8: edit-level acceptance (CPU only, frozen candidates).

Instead of accepting or rejecting a whole corrected segment, split the OCR->candidate difference into atomic
whitespace-delimited edits and accept each edit individually. Same protocol as the iter-2 ledger:
  * edit router = gradient boosting on per-edit features (+ segment context), 5-fold doc-grouped OOF on dev;
  * thresholds from the dev quantile grid incl. abstain; single test evaluation; micro CER;
  * baselines at edit level: edit-size cap, per-edit min token log-prob (exact token spans, exp/rescore_tokens.py);
  * cascade: segment router accepts the whole segment, edit router handles the segments it rejects;
  * references at segment level: full router, edit-cap, conf-min (from iter2_ledger.scores);
  * paired doc bootstrap (B=10000) + Holm against the segment-level full router and segment edit-cap;
  * dev-tuned harm budgets (a segment is corrupted when its final text is further from gold than the OCR).
Outputs: process/refine-logs/iter2/editlevel_{corr}_{ds}.json, editlevel_summary.md
"""
import os, sys, json, math, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rapidfuzz.distance import Levenshtein
from oclib import DATA, in_dict
from iter2_ledger import CONFIGS, OUT, SEED, B, pval, Split, scores, crossfit, gb, tune, qgrid, core

_tok = re.compile(r"\S+")
al = lambda s: "".join(ch for ch in s if ch.isalnum())
def ind(t): c = core(t); return bool(c) and c.isalpha() and in_dict(c)

def edits_of(o, c):
    """atomic edits as char spans (i1, i2, j1, j2): character alignment cut at whitespace that is aligned to itself,
    so a split/merged word ("York,and" -> "York, and") stays one edit and every edit can be spliced independently"""
    out, cur, dirty = [], None, False
    def close(i, j):
        nonlocal cur, dirty
        if cur is not None and dirty: out.append((cur[0], i, cur[1], j))
        cur, dirty = None, False
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(o, c):
        if tag != "equal":
            if cur is None: cur = (i1, j1)
            dirty = True; continue
        for k in range(i2 - i1):
            if o[i1 + k].isspace(): close(i1 + k, j1 + k)
            elif cur is None: cur = (i1 + k, j1 + k)
    close(len(o), len(c))
    return out

def apply(o, c, eds, keep):
    out, p = [], 0
    for k, (i1, i2, j1, j2) in enumerate(eds):
        out.append(o[p:i1]); out.append(c[j1:j2] if k in keep else o[i1:i2]); p = i2
    return "".join(out) + o[p:]

class ESplit:
    def __init__(s, S):
        s.S = S; s.row, s.F, s.gain, s.eds, s.toks, s.cache = [], [], [], {}, {}, {}
        tp = os.path.join(DATA, "toklp_%s_%s_%s.jsonl" % S.name); TL = None
        if os.path.exists(tp):
            TL = [json.loads(l) for l in open(tp, encoding="utf-8")]
            assert len(TL) == len(S.rows) and all(a["doc_id"] == b["doc_id"] and a.get("seg_idx") == b.get("seg_idx") for a, b in zip(TL, S.rows)), tp
        for i in np.flatnonzero(S.att):
            r = S.rows[i]; o, c = r["ocr"], r["cand"]; eds = edits_of(o, c)
            s.eds[i], s.toks[i] = eds, (o, c)
            tl = TL[i] if TL is not None else None                         # exact token spans (exp/rescore_tokens.py)
            if tl is not None and tl["spans"] is None: tl = None
            if tl is not None: ta = np.array([x[0] for x in tl["spans"]], int); tb = np.array([x[1] for x in tl["spans"]], int); tv = np.array([x[2] for x in tl["spans"]], float)
            ctx = list(S.X[i]) + list(S.A[i]) + (list(S.C[i]) if S.has_conf else [])
            for k, (i1, i2, j1, j2) in enumerate(eds):
                a, b = o[i1:i2], c[j1:j2]; d = Levenshtein.distance(a, b); na, nb = len(a.split()), len(b.split()); one = (na == 1 and nb == 1)
                lp = [np.nan, np.nan]
                if tl is not None:
                    # tokens overlapping the edit's candidate span; a pure deletion (empty span) is scored by the token the
                    # model emitted at that position instead of the deleted text (the end-of-sequence token at the end)
                    m = (ta < j2) & (tb > j1) if j2 > j1 else (ta <= j1) & (tb > j1)
                    seg = tv[m] if m.any() else (np.array([tl["end_lp"]]) if j1 >= len(c) and tl["end_lp"] is not None else tv[:0])
                    if len(seg): lp = [float(seg.min()), float(seg.mean())]
                near = (k > 0 and i1 - eds[k - 1][1] <= 1) or (k + 1 < len(eds) and eds[k + 1][0] - i2 <= 1)
                s.F.append([len(a), len(b), na, nb, d, d / max(1, len(a), len(b)), float(one),
                            float(any(ch.isdigit() for ch in a)), float(any(ch.isdigit() for ch in b)),
                            float(a.lower() == b.lower()), float(al(a) == al(b)), float(a[:1].lower() != b[:1].lower()),
                            float(one and ind(a)), float(one and ind(b)), i1 / max(1, len(o)), len(eds), float(near)] + lp + ctx)
                s.row.append(i); s.gain.append(S.eo[i] - s.err(i, frozenset([k])))
        s.row = np.array(s.row, int); s.F = np.array(s.F, float); s.gain = np.array(s.gain, float)
        s.y = (s.gain > 0).astype(int); s.k = np.concatenate([np.arange(len(s.eds[i])) for i in np.flatnonzero(S.att)]) if len(s.row) else np.array([], int)
        s.has_lp = bool(len(s.F)) and np.isfinite(s.F[:, 17]).mean() > 0.9
    def err(s, i, keep):
        """distance to gold of row i with the edit subset `keep` applied (extremes use the original strings)"""
        if not keep: return int(s.S.eo[i])
        if len(keep) == len(s.eds[i]): return int(s.S.ec[i])
        key = (i, keep)
        if key not in s.cache:
            to, tc = s.toks[i]; s.cache[key] = Levenshtein.distance(apply(to, tc, s.eds[i], keep), s.S.rows[i]["gold"])
        return s.cache[key]
    def deltas(s, sc):
        """per-edit error change when edits of a row are applied in decreasing score order; accepting all edits
        with score>=t is always a prefix of that order, so row error(t) = eo + sum of deltas with score>=t"""
        d = np.zeros(len(sc)); o = np.lexsort((-sc, s.row)); prev_row, keep, prev = -1, set(), 0
        j = 0
        while j < len(o):
            i = s.row[o[j]]; keep, prev = set(), int(s.S.eo[i]); m = j
            while m < len(o) and s.row[o[m]] == i:
                n = m                                           # tied scores enter together
                while n < len(o) and s.row[o[n]] == i and sc[o[n]] == sc[o[m]]: keep.add(int(s.k[o[n]])); n += 1
                e = s.err(i, frozenset(keep)); d[o[m]] = e - prev; prev = e; m = n
            j = m
        return d
    def oracle(s, nmax=8):
        """best edit subset per row: exhaustive for <= nmax edits, otherwise greedy from the best of {none, all,
        individually helpful}; never worse than the segment oracle, a lower bound on the true optimum for long rows"""
        from itertools import combinations
        e = s.S.eo.astype(float)
        for i, eds in s.eds.items():
            n = len(eds)
            if n <= nmax:
                e[i] = min(s.err(i, frozenset(c)) for r in range(n + 1) for c in combinations(range(n), r)); continue
            g = s.gain[s.row == i]; cur = min((frozenset(), frozenset(range(n)), frozenset(np.flatnonzero(g > 0).tolist())), key=lambda k: s.err(i, k))
            while True:
                best = min((cur ^ {k} for k in range(n)), key=lambda k: s.err(i, k))
                if s.err(i, best) >= s.err(i, cur): break
                cur = best
            e[i] = s.err(i, cur)
        return e
    def errs(s, sc, d, t, whole=None):
        """per-row error for: accept edit iff score>=t; rows in `whole` take the full candidate"""
        e = s.S.eo.astype(float) + np.bincount(s.row, d * (sc >= t), len(s.S.rows))
        return e if whole is None else np.where(whole, s.S.ec, e)

def metrics(S, e):
    return dict(cer=float(e.sum() / max(1, S.gl.sum())), corrupt=int((e > S.eo).sum()), improved=int((e < S.eo).sum()))

def etune(E, sc, d, k=101, budget=None, whole=None):
    """same grid as the segment policies: k dev quantiles of the (finite) scores + abstain"""
    best = None; fin = sc[np.isfinite(sc)]
    for t in np.r_[np.unique(np.quantile(fin, np.linspace(0, 1, k))), np.inf] if len(fin) else [np.inf]:
        e = E.errs(sc, d, t, whole)
        if budget is not None and (e > E.S.eo).sum() > budget: continue
        c = e.sum()
        if best is None or c < best[0] - 1e-9: best = (c, t)
    return best[1] if best else np.inf

def run(ds, corr):
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test"); Ed, Et = ESplit(dv), ESplit(te)
    Sseg, full = scores(dv, te)
    # edit-level scores (dev OOF / test)
    ES = {"edit_router": crossfit(gb, Ed.F, Ed.y, np.ones(len(Ed.y), bool), dv.di[Ed.row], Et.F),
          "edit_size": (-Ed.F[:, 5], -Et.F[:, 5])}
    if Ed.has_lp and Et.has_lp:
        f = lambda F: np.where(np.isfinite(F[:, 17]), F[:, 17], -np.inf)   # unscored edits (rows without token spans) are rejected
        ES["edit_lp"] = (f(Ed.F), f(Et.F))
    Dd = {k: Ed.deltas(v[0]) for k, v in ES.items()}; Dt = {k: Et.deltas(v[1]) for k, v in ES.items()}
    P, info = {}, {}                                                       # test per-row error vectors
    P["never"] = te.eo.astype(float); P["always"] = np.where(te.att, te.ec, te.eo).astype(float)
    P["oracle_seg"] = np.minimum(te.eo, te.ec).astype(float)
    P["oracle_edit"] = Et.oracle()
    segkeys = [k for k in (full, "edit_cap", "conf_min") if k in Sseg]
    for k in segkeys:
        sd, st = Sseg[k]; t = tune(dv, sd); P["seg_" + k] = np.where(te.att & (st >= t), te.ec, te.eo).astype(float); info["seg_" + k] = dict(thr=float(t))
    for k, (sd, st) in ES.items():
        t = etune(Ed, sd, Dd[k]); P[k] = Et.errs(st, Dt[k], t); info[k] = dict(thr=float(t), edit_accept_rate=float((st >= t).mean()) if len(st) else 0.0)
    # cascade: whole segment if the segment router is confident, edit router otherwise
    sd, st = Sseg[full]; best = None
    for ts in qgrid(sd, dv.att):                                           # includes both standalone optima (same grids)
        w = dv.att & (sd >= ts); t = etune(Ed, ES["edit_router"][0], Dd["edit_router"], whole=w)
        c = Ed.errs(ES["edit_router"][0], Dd["edit_router"], t, w).sum()
        if best is None or c < best[0] - 1e-9: best = (c, ts, t)
    P["cascade"] = Et.errs(ES["edit_router"][1], Dt["edit_router"], best[2], te.att & (st >= best[1]))
    info["cascade"] = dict(seg_thr=float(best[1]), edit_thr=float(best[2]))
    res = {k: {**metrics(te, e), **info.get(k, {})} for k, e in P.items()}
    # paired doc bootstrap vs the segment-level references
    rng = np.random.default_rng(SEED); W = rng.multinomial(te.nd, np.ones(te.nd) / te.nd, size=B); gl_b = W @ te.G
    def boot_vs(ref, comps, Pm):
        out = {}; er = (W @ np.bincount(te.di, Pm[ref], te.nd)) / gl_b; c0 = Pm[ref].sum() / te.gl.sum()
        for k in comps:
            d = (W @ np.bincount(te.di, Pm[k], te.nd)) / gl_b - er
            out[k] = dict(delta=float(Pm[k].sum() / te.gl.sum() - c0), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                          p_two_sided=pval(d))
        order = sorted(out, key=lambda k: out[k]["p_two_sided"]); m = len(order); mx = 0
        for i, k in enumerate(order): mx = max(mx, min(1.0, (m - i) * out[k]["p_two_sided"])); out[k]["p_holm"] = mx
        return out
    elev = list(ES) + ["cascade"]
    boot = {r: boot_vs(r, elev, P) for r in ["seg_" + full, "seg_edit_cap"]}
    # dev-tuned harm budgets
    frontier = {}; a_dev = dv.corrupt(dv.att)
    for frac in [0.10, 0.25, 0.50]:
        bud = int(frac * a_dev); U = {}
        for k in segkeys:
            sd, st = Sseg[k]; U["seg_" + k] = np.where(te.att & (st >= tune(dv, sd, qgrid(sd, dv.att, 201), budget=bud)), te.ec, te.eo).astype(float)
        for k, (sd, st) in ES.items(): U[k] = Et.errs(st, Dt[k], etune(Ed, sd, Dd[k], 201, budget=bud))
        bt = {r: boot_vs(r, list(ES), U) for r in ["seg_" + full, "seg_edit_cap"]}
        frontier[f"{int(frac*100)}pct"] = {k: dict(test_cer=float(e.sum() / te.gl.sum()), test_corrupt=int((e > te.eo).sum()),
                                                   **{f"vs_{r}": bt[r][k] for r in bt if k in bt[r]}) for k, e in U.items()}
    # headroom diagnostics on test
    hp = np.bincount(Et.row, Et.gain > 0, len(te.rows)); hn = np.bincount(Et.row, Et.gain < 0, len(te.rows))
    diag = dict(n_edits_test=int(len(Et.row)), edits_per_attempted_seg=float(len(Et.row) / max(1, te.att.sum())),
                edit_helpful=float((Et.gain > 0).mean()), edit_harmful=float((Et.gain < 0).mean()), edit_neutral=float((Et.gain == 0).mean()),
                mixed_segments=float(((hp > 0) & (hn > 0))[te.att].mean()),
                oracle_exhaustive_rows=float(np.mean([len(v) <= 8 for v in Et.eds.values()])) if Et.eds else 1.0, has_edit_lp=bool(Ed.has_lp and Et.has_lp),
                edit_router_auc=None)
    if len(set(Et.y)) > 1:
        from sklearn.metrics import roc_auc_score
        diag["edit_router_auc"] = float(roc_auc_score(Et.y, ES["edit_router"][1]))
    out = dict(dataset=ds, corrector=corr, full_router=full, n_test=len(te.rows), n_test_docs=te.nd, policies=res, bootstrap=boot,
               devtuned_budget_frontier=frontier, diagnostics=diag,
               protocol="atomic whitespace-delimited edits (character alignment); edit router cross-fitted (5-fold doc-grouped OOF) on dev; thresholds on dev quantile grid "
                        "incl. abstain; single test eval; micro CER; paired doc bootstrap B=10000; Holm")
    json.dump(out, open(os.path.join(OUT, f"editlevel_{corr}_{ds}.json"), "w"), indent=1)
    return out

def summary():
    import glob
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "editlevel_*.json")))]
    R.sort(key=lambda o: [(d, c) for d, c in CONFIGS].index((o["dataset"], o["corrector"])))
    L = ["# B8 — edit-level acceptance (iter-2 protocol)", "",
         "Test micro CER. `seg_*` accept whole segments; `edit_*` accept atomic whitespace-delimited edits (character alignment); `cascade` = segment router, then edit router on rejected segments.", "",
         "| dataset | corrector | never | always | seg router | seg edit-cap | edit router | edit size | edit lp | cascade | oracle seg | oracle edit | mixed seg | edit AUC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda o, k: f"{o['policies'][k]['cer']:.4f}" if k in o["policies"] else "—"
    for o in R:
        d = o["diagnostics"]; fr = "seg_" + o["full_router"]
        L.append(f"| {o['dataset']} | {o['corrector']} | " + " | ".join(f(o, k) for k in ["never", "always", fr, "seg_edit_cap", "edit_router", "edit_size", "edit_lp", "cascade", "oracle_seg", "oracle_edit"])
                 + f" | {d['mixed_segments']:.2f} | " + (f"{d['edit_router_auc']:.3f}" if d["edit_router_auc"] else "—") + " |")
    def tally(get):
        w = l = n = 0
        for o in R:
            b = get(o)
            if b is None: continue
            if b["p_holm"] < 0.05 and b["delta"] < 0: w += 1
            elif b["p_holm"] < 0.05 and b["delta"] > 0: l += 1
            else: n += 1
        return f"{w} / {l} / {n}"
    L += ["", "## Significant wins / losses / n.s. (Holm p<0.05, paired doc bootstrap), edit-level policy vs segment-level reference", "",
          "| comparison | CER-optimal | 10% budget | 25% budget | 50% budget |", "|---|---|---|---|---|"]
    for pol in ["edit_router", "edit_size", "edit_lp", "cascade"]:
        for ref, lab in [("full", "seg router"), ("seg_edit_cap", "seg edit-cap")]:
            rk = lambda o: "seg_" + o["full_router"] if ref == "full" else ref
            cells = [tally(lambda o: o["bootstrap"][rk(o)].get(pol))]
            for fz in ["10pct", "25pct", "50pct"]:
                cells.append("—" if pol == "cascade" else tally(lambda o: o["devtuned_budget_frontier"][fz].get(pol, {}).get("vs_" + rk(o))))
            L.append(f"| {pol} vs {lab} | " + " | ".join(cells) + " |")
    md = lambda xs: float(np.mean(xs)) if xs else float("nan")
    L += ["", "## Means over configurations", "",
          f"- mean CER: seg router {md([o['policies']['seg_' + o['full_router']]['cer'] for o in R]):.4f}, seg edit-cap {md([o['policies']['seg_edit_cap']['cer'] for o in R]):.4f}, "
          f"edit router {md([o['policies']['edit_router']['cer'] for o in R]):.4f}, edit size {md([o['policies']['edit_size']['cer'] for o in R]):.4f}, "
          f"cascade {md([o['policies']['cascade']['cer'] for o in R]):.4f}, oracle seg {md([o['policies']['oracle_seg']['cer'] for o in R]):.4f}, "
          f"oracle edit {md([o['policies']['oracle_edit']['cer'] for o in R]):.4f}",
          f"- mean share of attempted segments containing both helpful and harmful edits: {md([o['diagnostics']['mixed_segments'] for o in R]):.3f}",
          f"- `oracle edit` = best edit subset per segment (exhaustive for ≤8 edits, {md([o['diagnostics']['oracle_exhaustive_rows'] for o in R]):.1%} of attempted segments; greedy otherwise, so a lower bound on the headroom)",
          "- `edit lp` = minimum token log-prob over the tokens that produce the edit, from exact token-to-character spans (`exp/rescore_tokens.py`: teacher-forced byte log-probs of the stored candidate for ByT5, tokenizer offsets for the LLM correctors); a pure deletion is scored by the token emitted at the deletion point; rows without spans are unscored and rejected."]
    open(os.path.join(OUT, "editlevel_summary.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    only = sys.argv[1:]
    if only == ["summary"]: summary(); sys.exit()
    for ds, corr in CONFIGS:
        if only and f"{corr}_{ds}" not in only: continue
        if not os.path.exists(os.path.join(DATA, f"candidates_{corr}_{ds}_dev.jsonl")): print("skip", ds, corr); continue
        o = run(ds, corr); p = o["policies"]
        print(ds, corr, " ".join(f"{k}={v['cer']:.4f}" for k, v in p.items()), flush=True)
