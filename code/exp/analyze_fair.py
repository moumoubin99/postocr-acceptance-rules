#!/usr/bin/env python3
"""ARS must-fix #1: re-test C4 with FAIR baselines (all trained GBDTs, dev-tuned
operating points). DATASET via env (bln600|icdar), ByT5 corrector.
  - FULL router         : GBDT on all 13 features (sees candidate), segment-level
  - ERROR-DET (fair)    : GBDT on OCR-only pre-correction features (no candidate)
  - DOC-QE (fair)       : GBDT on per-document mean of 13 features, routes whole docs
All trained on dev, threshold tuned on dev, evaluated on held-out test."""
import os, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
DATASET = os.environ.get("DATASET", "bln600"); CORR = os.environ.get("CORR", "byt5")
ORIG = [0,1,2,3,8,9,10,11,12]  # OCR-only (pre-correction) feature indices
def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=20260526)

def build(split):
    R=[]
    for r in load_jsonl(os.path.join(DATA, f"candidates_{CORR}_{DATASET}_{split}.jsonl")):
        cb,ca=cer(r["ocr"],r["gold"]),cer(r["cand"],r["gold"])
        R.append(dict(doc=r["doc_id"], ed_ocr=Levenshtein.distance(r["ocr"],r["gold"]),
                      ed_cand=Levenshtein.distance(r["cand"],r["gold"]), glen=len(r["gold"]),
                      attempted=(r["ocr"]!=r["cand"]), helped=int(ca<cb-1e-9), feat=features(r["ocr"],r["cand"])))
    return R
dev,test=build("dev"),build("test")
def micro(R,use):
    e=sum(r["ed_cand"] if u else r["ed_ocr"] for r,u in zip(R,use)); return e/max(1,sum(r["glen"] for r in R))
def corr_ct(R,use): return sum(1 for r,u in zip(R,use) if u and r["ed_cand"]>r["ed_ocr"])
ad=np.array([r["attempted"] for r in dev]); at=np.array([r["attempted"] for r in test])
yd=np.array([r["helped"] for r in dev]); yt=np.array([r["helped"] for r in test])

def seg_router(cols):
    Xd=np.array([r["feat"] for r in dev])[:,cols]; Xt=np.array([r["feat"] for r in test])[:,cols]
    c=gb(); c.fit(Xd[ad],yd[ad]); pd_=c.predict_proba(Xd)[:,1]; pt=c.predict_proba(Xt)[:,1]
    thr=min(np.linspace(0.2,0.8,61), key=lambda t: micro(dev,[ad[i] and pd_[i]>=t for i in range(len(dev))]))
    use=[at[i] and pt[i]>=thr for i in range(len(test))]
    auc=roc_auc_score(yt[at],pt[at]) if len(set(yt[at]))>1 else float("nan")
    return micro(test,use), corr_ct(test,use), auc, thr

def doc_qe():
    # per-doc mean features; label = doc benefits from always-correct (doc ed_cand-sum < ed_ocr-sum)
    def docagg(R):
        d=defaultdict(list)
        for r in R: d[r["doc"]].append(r)
        X={}; y={}
        for doc,rs in d.items():
            X[doc]=np.mean([r["feat"] for r in rs],axis=0)
            y[doc]=int(sum(r["ed_cand"] for r in rs) < sum(r["ed_ocr"] for r in rs))
        return d,X,y
    dd,Xd,yd_=docagg(dev); dt,Xt,_=docagg(test)
    docs_d=list(Xd); c=gb(); c.fit(np.array([Xd[k] for k in docs_d]),np.array([yd_[k] for k in docs_d]))
    pd_={k:p for k,p in zip(docs_d,c.predict_proba(np.array([Xd[k] for k in docs_d]))[:,1])}
    docs_t=list(Xt); pt={k:p for k,p in zip(docs_t,c.predict_proba(np.array([Xt[k] for k in docs_t]))[:,1])}
    def route(R,score,thr): return [score[r["doc"]]>=thr for r in R]
    thr=min(np.linspace(0.2,0.8,61), key=lambda t: micro(dev,route(dev,pd_,t)))
    use=route(test,pt,thr); return micro(test,use), corr_ct(test,use), thr

never=micro(test,[False]*len(test)); always=micro(test,[True]*len(test))
oracle=micro(test,[r["ed_cand"]<r["ed_ocr"] for r in test])
full=seg_router(list(range(13))); errdet=seg_router(ORIG); docqe=doc_qe()
print(f"=== DATASET={DATASET}  FAIR-baseline C4 re-test (ByT5, dev-tuned operating points, held-out test) ===")
print(f"  never {never:.4f} | always {always:.4f} | oracle {oracle:.4f}")
print(f"  FULL router (all feats, sees candidate) : CER {full[0]:.4f}  corrupt {full[1]}  AUC {full[2]:.3f}  thr {full[3]:.2f}")
print(f"  ERROR-DET (fair, OCR-only GBDT)         : CER {errdet[0]:.4f}  corrupt {errdet[1]}  AUC {errdet[2]:.3f}  thr {errdet[3]:.2f}")
print(f"  DOC-QE (fair, trained doc-level GBDT)   : CER {docqe[0]:.4f}  corrupt {docqe[1]}  thr {docqe[2]:.2f}")
verdict = ("router STILL beats both fair baselines" if (full[0]<errdet[0]-1e-5 and full[0]<docqe[0]-1e-5)
           else "router does NOT clearly beat fair baselines — C4 must be scoped down")
print(f"  => {verdict}")
json.dump(dict(dataset=DATASET, never=never, always=always, oracle=oracle,
               full=dict(cer=full[0],corrupt=full[1],auc=full[2],thr=full[3]),
               errdet_fair=dict(cer=errdet[0],corrupt=errdet[1],auc=errdet[2],thr=errdet[3]),
               docqe_fair=dict(cer=docqe[0],corrupt=docqe[1],thr=docqe[2]), verdict=verdict),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"fair_baselines_{DATASET}.json"),"w"),indent=2)
