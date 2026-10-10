#!/usr/bin/env python3
"""Downstream NER-F1 delta experiment (CLEF-HIPE hipe2020-en).
Fixed NER model (dslim/bert-base-NER) run on OCR vs ByT5-corrected vs ROUTER-selected text.
Gold entity spans (char offsets in OCR text); predicted spans on corrected text are mapped
back to OCR offsets via char alignment. Metric: type-agnostic overlap entity F1 (entity
DETECTION). Router trained on dev (text-only features) to accept correction where it
improves per-segment entity detection; evaluated on held-out test."""
import os, json, numpy as np
from difflib import SequenceMatcher
import torch
from transformers import pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oclib import features, load_jsonl, DATA

NER_MODEL = "dslim/bert-base-NER"
dev_id = 0 if torch.cuda.is_available() else -1
ner = pipeline("token-classification", model=NER_MODEL, aggregation_strategy="simple",
               device=dev_id, batch_size=64)

def load_split(split):
    segs = {s_i: s for s_i, s in enumerate(load_jsonl(os.path.join(DATA, f"segments_hipe_en_{split}.jsonl")))}
    cand = load_jsonl(os.path.join(DATA, f"candidates_byt5base8_hipe_en_{split}.jsonl"))
    rows = []
    for i, s in segs.items():
        c = cand[i]
        assert c["ocr"] == s["text"], f"align {split} {i}"
        rows.append(dict(ocr=s["text"], cand=c["cand"], score=float(c.get("score", 0.0)),
                         gold=[tuple(e) for e in s["entities"]]))
    return rows

def ner_spans(texts):
    """Return list of [(start,end)] predicted entity char-spans per text (type-agnostic)."""
    out = []
    preds = ner(texts)
    for p in preds:
        out.append([(e["start"], e["end"]) for e in p])
    return out

def build_charmap(a, b):
    """Map char index in b (corrected) -> char index in a (ocr) via alignment."""
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    m = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            m[j + k] = i + k
    def mp(idx):  # nearest mapped
        if idx in m: return m[idx]
        keys = [k for k in (idx-1, idx+1, idx-2, idx+2) if k in m]
        return m[keys[0]] if keys else min(len(a), idx)
    return mp

def overlap(s1, s2):
    return max(0, min(s1[1], s2[1]) - max(s1[0], s2[0])) > 0

def prf(pred_spans_per_seg, rows):
    """corpus type-agnostic overlap entity F1: pred mapped to OCR coords vs gold."""
    tp = fp = fn = 0
    for ps, r in zip(pred_spans_per_seg, rows):
        gold = [(a, b) for a, b, t in r["gold"]]
        matched_g = set(); used_p = set()
        for pi, p in enumerate(ps):
            hit = next((gi for gi, g in enumerate(gold) if gi not in matched_g and overlap(p, g)), None)
            if hit is not None: tp += 1; matched_g.add(hit); used_p.add(pi)
        fp += len(ps) - len(used_p); fn += len(gold) - len(matched_g)
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2*P*R/max(1e-9, P+R)
    return P, R, F

def seg_detect(pred_spans, gold):
    g = [(a, b) for a, b, t in gold]
    return sum(1 for gg in g if any(overlap(p, gg) for p in pred_spans))

print("running NER on dev/test (OCR + corrected)...")
data = {}
for split in ["dev", "test"]:
    rows = load_split(split)
    ocr_pred = ner_spans([r["ocr"] for r in rows])
    cand_pred_raw = ner_spans([r["cand"] for r in rows])
    # map corrected-text predictions back to OCR coords
    cand_pred = []
    for r, ps in zip(rows, cand_pred_raw):
        mp = build_charmap(r["ocr"], r["cand"])
        cand_pred.append([(mp(a), mp(b)) for a, b in ps])
    data[split] = (rows, ocr_pred, cand_pred)
    print(f"  {split}: {len(rows)} segs done")

# router: train on dev, label = corrected improves per-seg entity detection (segs with gold)
rows_d, ocr_d, cand_d = data["dev"]
X, y = [], []
for r, op, cp in zip(rows_d, ocr_d, cand_d):
    if not r["gold"]: continue
    d_ocr = seg_detect(op, r["gold"]); d_cand = seg_detect(cp, r["gold"])
    X.append(features(r["ocr"], r["cand"]) + [r["score"]]); y.append(int(d_cand > d_ocr))
X, y = np.array(X), np.array(y)
clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=20260526)
clf.fit(X, y)
print(f"\nrouter trained on {len(y)} entity-bearing dev segs; accept-rate label mean={y.mean():.2f}")

def evaluate(split):
    rows, ocr_pred, cand_pred = data[split]
    feats = np.array([features(r["ocr"], r["cand"]) + [r["score"]] for r in rows])
    p_accept = clf.predict_proba(feats)[:, 1]
    # threshold tuned on dev for max F1
    return rows, ocr_pred, cand_pred, p_accept

# tune threshold on dev
rd, od, cd, pad = evaluate("dev")
def route_pred(ocr_pred, cand_pred, p, thr):
    return [cand_pred[i] if p[i] >= thr else ocr_pred[i] for i in range(len(ocr_pred))]
best_thr = max(np.linspace(0.2, 0.8, 31), key=lambda t: prf(route_pred(od, cd, pad, t), rd)[2])

print("\n=== Downstream NER detection F1 (type-agnostic overlap) ===")
for split in ["dev", "test"]:
    rows, ocr_pred, cand_pred, p = (evaluate(split))
    f_never = prf(ocr_pred, rows); f_always = prf(cand_pred, rows)
    f_router = prf(route_pred(ocr_pred, cand_pred, p, best_thr), rows)
    print(f"  [{split}] never P/R/F={f_never[0]:.3f}/{f_never[1]:.3f}/{f_never[2]:.3f} | "
          f"always {f_always[2]:.3f} | router {f_router[2]:.3f} (thr={best_thr:.2f})")
    if split == "test":
        # segment bootstrap (B=2000) on F1 deltas
        rng = np.random.default_rng(20260527)
        routed = route_pred(ocr_pred, cand_pred, p, best_thr)
        idx0 = list(range(len(rows)))
        d_ra, d_nr, d_na = [], [], []
        for _ in range(2000):
            samp = rng.choice(idx0, size=len(idx0), replace=True).tolist()
            rs = [rows[i] for i in samp]
            fn = prf([ocr_pred[i] for i in samp], rs)[2]
            fa = prf([cand_pred[i] for i in samp], rs)[2]
            fr = prf([routed[i] for i in samp], rs)[2]
            d_ra.append(fr - fa); d_nr.append(fr - fn); d_na.append(fn - fa)
        def ci(x): return [float(np.percentile(x,2.5)), float(np.percentile(x,97.5))]
        print(f"      bootstrap: router-always {f_router[2]-f_always[2]:+.3f} CI{ci(d_ra)} "
              f"{'SIG' if ci(d_ra)[0]>0 else 'ns'}; always-vs-never {f_never[2]-f_always[2]:+.3f} CI{ci(d_na)} "
              f"{'SIG(blind hurts)' if ci(d_na)[0]>0 else 'ns'}; router-never {f_router[2]-f_never[2]:+.3f} CI{ci(d_nr)}")
        json.dump(dict(never_f=f_never[2], always_f=f_always[2], router_f=f_router[2], thr=float(best_thr),
                       never_PR=f_never[:2], always_PR=f_always[:2], router_PR=f_router[:2],
                       ci_router_minus_always=ci(d_ra), ci_never_minus_always=ci(d_na), ci_router_minus_never=ci(d_nr)),
                  open(os.path.join(os.path.dirname(__file__),"..","refine-logs","ner_routing_test.json"),"w"), indent=2)
