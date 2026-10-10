#!/usr/bin/env python3
"""Document-level cluster bootstrap 95% CI on test CER deltas (ByT5 corrector).
Router trained on dev, threshold tuned on dev (fixed); resample TEST documents."""
import os, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
RNG = np.random.default_rng(20260526)
DATASET = os.environ.get("DATASET", "bln600")

def build(corr, split):
    R = []
    for r in load_jsonl(os.path.join(DATA, f"candidates_{corr}_{DATASET}_{split}.jsonl")):
        cb, ca = cer(r["ocr"], r["gold"]), cer(r["cand"], r["gold"])
        R.append(dict(doc=r["doc_id"], ed_ocr=Levenshtein.distance(r["ocr"], r["gold"]),
                      ed_cand=Levenshtein.distance(r["cand"], r["gold"]), glen=len(r["gold"]),
                      attempted=(r["ocr"] != r["cand"]), helped=int(ca < cb-1e-9),
                      feat=features(r["ocr"], r["cand"])))
    return R

corr = os.environ.get("CORR", "byt5")
dev, test = build(corr, "dev"), build(corr, "test")
Xd = np.array([r["feat"] for r in dev]); yd = np.array([r["helped"] for r in dev])
ad = np.array([r["attempted"] for r in dev])
clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                     l2_regularization=1.0, random_state=20260526)
clf.fit(Xd[ad], yd[ad])
pd_ = clf.predict_proba(Xd)[:, 1]
def cer_choice(R, use_cand):  # use_cand: bool array
    e = sum(r["ed_cand"] if u else r["ed_ocr"] for r, u in zip(R, use_cand))
    l = sum(r["glen"] for r in R); return e / max(1, l)
# tune thr on dev
thr = min(np.linspace(0.2,0.8,61),
          key=lambda t: cer_choice(dev, [ad[i] and pd_[i]>=t for i in range(len(dev))]))
pt = clf.predict_proba(np.array([r["feat"] for r in test]))[:,1]
at = np.array([r["attempted"] for r in test])
route = np.array([at[i] and pt[i]>=thr for i in range(len(test))])

by_doc = defaultdict(list)
for i, r in enumerate(test): by_doc[r["doc"]].append(i)
docs = list(by_doc)
def metrics(idx):
    sub = [test[i] for i in idx]
    never = cer_choice(sub, [False]*len(sub))
    always = cer_choice(sub, [True]*len(sub))
    router = cer_choice(sub, [route[i] for i in idx])
    return never, always, router
idx_all = list(range(len(test)))
n0, a0, r0 = metrics(idx_all)
B = 2000; d_always, d_never = [], []
for _ in range(B):
    samp = RNG.choice(docs, size=len(docs), replace=True)
    idx = [i for d in samp for i in by_doc[d]]
    nv, al, ro = metrics(idx)
    d_always.append(al - ro)   # always - router (positive = router better)
    d_never.append(nv - ro)    # never - router
def ci(x): return np.percentile(x, [2.5, 97.5])
print(f"[{DATASET}] ByT5 router (thr={thr:.2f}) on held-out test, doc-cluster bootstrap B={B}  (n_docs={len(docs)})")
print(f"  point CER: never {n0:.4f} | always {a0:.4f} | router {r0:.4f}")
ca = ci(d_always); cn = ci(d_never)
print(f"  CER(always) - CER(router) = {a0-r0:+.4f}  95% CI [{ca[0]:+.4f}, {ca[1]:+.4f}]  "
      f"{'SIGNIFICANT' if ca[0]>0 else 'n.s.'}")
print(f"  CER(never)  - CER(router) = {n0-r0:+.4f}  95% CI [{cn[0]:+.4f}, {cn[1]:+.4f}]  "
      f"{'SIGNIFICANT' if cn[0]>0 else 'n.s.'}")
json.dump(dict(thr=float(thr), never=n0, always=a0, router=r0,
               delta_always=[a0-r0, *ca.tolist()], delta_never=[n0-r0, *cn.tolist()]),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"stage2_bootstrap_{DATASET}_{corr}.json"),"w"), indent=2)
