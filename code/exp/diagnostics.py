#!/usr/bin/env python3
"""Stage-3 integrity diagnostics defending the ByT5 router result:
confusion matrix, feature-group ablation, label-permutation sanity check,
threshold sensitivity, full baseline frontier curves. All on held-out test."""
import os, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, FEATNAMES, load_jsonl, DATA
RNG = np.random.default_rng(7)
CORR = "byt5"

def build(split):
    R = []
    for r in load_jsonl(os.path.join(DATA, f"candidates_{CORR}_{split}.jsonl")):
        cb, ca = cer(r["ocr"], r["gold"]), cer(r["cand"], r["gold"])
        R.append(dict(doc=r["doc_id"], ed_ocr=Levenshtein.distance(r["ocr"], r["gold"]),
                      ed_cand=Levenshtein.distance(r["cand"], r["gold"]), glen=len(r["gold"]),
                      attempted=(r["ocr"] != r["cand"]),
                      helped=int(ca < cb-1e-9), hurt=int(ca > cb+1e-9),
                      feat=features(r["ocr"], r["cand"])))
    return R

dev, test = build("dev"), build("test")
Xd = np.array([r["feat"] for r in dev]); yd = np.array([r["helped"] for r in dev]); ad = np.array([r["attempted"] for r in dev])
Xt = np.array([r["feat"] for r in test]); yt = np.array([r["helped"] for r in test]); at = np.array([r["attempted"] for r in test])

def clf(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=20260526)
def cer_choice(R, use):
    e = sum(r["ed_cand"] if u else r["ed_ocr"] for r, u in zip(R, use)); l = sum(r["glen"] for r in R); return e/max(1,l)
def tune_thr(R, a, p):
    return min(np.linspace(0.2,0.8,61), key=lambda t: cer_choice(R, [a[i] and p[i]>=t for i in range(len(R))]))

# ---- full router ----
c = clf(); c.fit(Xd[ad], yd[ad]); pt = c.predict_proba(Xt)[:,1]; pd_ = c.predict_proba(Xd)[:,1]
thr = tune_thr(dev, ad, pd_)
route = np.array([at[i] and pt[i]>=thr for i in range(len(test))])
auc_full = roc_auc_score(yt[at], pt[at])

# (1) confusion matrix on test ATTEMPTED segments: accept/reject x helped/hurt/neutral
acc = route[at]; el = [r for r in test if r["attempted"]]
hel = np.array([r["helped"] for r in el]); hur = np.array([r["hurt"] for r in el]); neu = 1-hel-hur
def cnt(cond): return int(np.sum(cond))
print(f"== (1) Acceptance confusion (test attempted={len(el)}, accept_rate={acc.mean():.2%}) ==")
print(f"   helped&accepted {cnt(acc&(hel==1))}  helped&rejected {cnt((~acc)&(hel==1))}")
print(f"   hurt&accepted   {cnt(acc&(hur==1))}  hurt&rejected   {cnt((~acc)&(hur==1))}")
print(f"   neutral&accepted{cnt(acc&(neu==1))}  neutral&rejected{cnt((~acc)&(neu==1))}")
print(f"   -> accepted-precision(helped among accepted) {cnt(acc&(hel==1))/max(1,cnt(acc)):.2%}; "
      f"hurt-recall(rejected among hurt) {cnt((~acc)&(hur==1))/max(1,cnt(hur==1)):.2%}")

# (2) feature provenance + ablation
ORIG = [0,1,2,3,8,9,10,11,12]   # OCR-only features (pre-correction)
CAND = [4,5,6,7]                # candidate/diff features (need the candidate)
print("\n== (2) Feature-group ablation (test AUC | routed test CER) ==")
for name, cols in [("all", list(range(len(FEATNAMES)))), ("orig-only(no candidate)", ORIG), ("candidate/diff-only", CAND)]:
    ci = clf(); ci.fit(Xd[ad][:,cols], yd[ad]); p = ci.predict_proba(Xt[:,cols])[:,1]; pdd = ci.predict_proba(Xd[:,cols])[:,1]
    t = tune_thr(dev, ad, pdd); rr = [at[i] and p[i]>=t for i in range(len(test))]
    print(f"   {name:24s} AUC {roc_auc_score(yt[at], p[at]):.3f} | CER {cer_choice(test, rr):.4f}")

# (3) label-permutation sanity (shuffle helped within dev docs) -> AUC should ~0.5
dperm = yd.copy(); d2i = defaultdict(list)
for i,r in enumerate(dev): d2i[r["doc"]].append(i)
for ids in d2i.values():
    a_ids=[i for i in ids if ad[i]]; perm=RNG.permutation(dperm[a_ids]);
    for j,i in enumerate(a_ids): dperm[i]=perm[j]
cp = clf(); cp.fit(Xd[ad], dperm[ad]); pp = cp.predict_proba(Xt)[:,1]
# global shuffle (breaks all structure incl. doc base-rate) -> must be ~0.5 if no leakage
gperm = yd[ad].copy(); RNG.shuffle(gperm)
cg = clf(); cg.fit(Xd[ad], gperm); pg = cg.predict_proba(Xt)[:,1]
print(f"\n== (3) Permutation sanity (real AUC={auc_full:.3f}) ==")
print(f"   within-doc shuffle  AUC = {roc_auc_score(yt[at], pp[at]):.3f} (doc base-rate survives by design)")
print(f"   GLOBAL shuffle      AUC = {roc_auc_score(yt[at], pg[at]):.3f} (must be ~0.5 -> no feature leakage)")

# (4) threshold sensitivity (routed test CER over thr) - not knife-edge?
print("\n== (4) Threshold sensitivity (routed test CER) ==")
for t in [0.35,0.40,0.45,0.50,0.55,0.60,0.65]:
    rr=[at[i] and pt[i]>=t for i in range(len(test))]; print(f"   thr={t:.2f}  CER {cer_choice(test, rr):.4f}", end="")
print(f"   [dev-tuned thr={thr:.2f}]")

# (5) full baseline frontiers (min achievable test CER, not one point)
edens = np.array([r["feat"][6] for r in test]); docq=defaultdict(list)
for r in test: docq[r["doc"]].append(r["feat"][6])
dscore = np.array([np.mean(docq[r["doc"]]) for r in test])
def best_cer(score, attempted_mask):
    best=1.0
    for q in np.unique(np.quantile(score, np.linspace(0,1,120))):
        rr=[(attempted_mask[i] and score[i]>=q) for i in range(len(test))]; best=min(best, cer_choice(test, rr))
    return best
print("\n== (5) Full baseline frontiers (MIN achievable test CER over all thresholds) ==")
print(f"   our router (util)     : {best_cer(pt, at):.4f}")
print(f"   error-detection router: {best_cer(edens, at):.4f}")
print(f"   doc-QE router         : {best_cer(dscore, np.ones(len(test),bool)):.4f}")
print(f"   always {cer_choice(test,[True]*len(test)):.4f} | never {cer_choice(test,[False]*len(test)):.4f} | "
      f"oracle {cer_choice(test,[r['ed_cand']<r['ed_ocr'] for r in test]):.4f}")

json.dump(dict(accept_rate=float(acc.mean()), auc_full=float(auc_full),
               perm_auc=float(roc_auc_score(yt[at], pp[at])), dev_thr=float(thr)),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs","stage3_diagnostics.json"),"w"), indent=2)
