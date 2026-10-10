#!/usr/bin/env python3
"""Multi-corrector SELECTION router. Per segment choose best of {keep-OCR, SymSpell,
ByT5-small, ByT5-base, (ByT5-large)}. Two selectors compared:
  (R) independent ΔED regression + argmax
  (J) JOINT multiclass argmin: features = all candidates' diffs concatenated, the
      model sees every candidate together and predicts which is best (incl. keep-OCR).
Trained on dev, evaluated on held-out test. DATASET via env."""
import os, json, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from rapidfuzz.distance import Levenshtein
from oclib import features, load_jsonl, DATA
DS = os.environ.get("DATASET", "bln600")
POOL = [c for c in ["symspell","byt5","byt5base","byt5base8","byt5large"]
        if os.path.exists(os.path.join(DATA, f"candidates_{c}_{DS}_test.jsonl"))]
ACTIONS = ["ocr"] + POOL
print(f"DATASET={DS}  pool: {POOL}")

def load(split):
    spine = load_jsonl(os.path.join(DATA, f"segments_{DS}_{split}.jsonl"))
    cands = {c: load_jsonl(os.path.join(DATA, f"candidates_{c}_{DS}_{split}.jsonl")) for c in POOL}
    segs=[]
    for i,s in enumerate(spine):
        ocr,gold=s["ocr"],s["gold"]
        row=dict(doc=s["doc_id"], glen=len(gold), ed_ocr=Levenshtein.distance(ocr,gold), cand={})
        for c in POOL:
            r=cands[c][i]; assert r["ocr"]==ocr, f"misalign {c} {split} {i}"
            row["cand"][c]=dict(ed=Levenshtein.distance(r["cand"],gold), feat=features(ocr,r["cand"]))
        segs.append(row)
    return segs
dev,test=load("dev"),load("test")
def ed_of(s,a): return s["ed_ocr"] if a=="ocr" else s["cand"][a]["ed"]
def micro(segs,picks): return sum(ed_of(s,p) for s,p in zip(segs,picks))/max(1,sum(s["glen"] for s in segs))

# ---- (J) joint multiclass argmin ----
def joint_feat(s):
    f=[];
    for c in POOL: f+=s["cand"][c]["feat"]
    return f
Xd=np.array([joint_feat(s) for s in dev]); Xt=np.array([joint_feat(s) for s in test])
yd=np.array([int(np.argmin([ed_of(s,a) for a in ACTIONS])) for s in dev])
clf=HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=500,
                                   l2_regularization=1.0, random_state=20260526)
clf.fit(Xd,yd)
pj=clf.predict(Xt); picks_J=[ACTIONS[k] for k in pj]
# ---- (R) independent regression + argmax ----
cidx={c:k for k,c in enumerate(POOL)}
def Xy(segs):
    X,y=[],[]
    for s in segs:
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1; X.append(s["cand"][c]["feat"]+oh); y.append(s["ed_ocr"]-s["cand"][c]["ed"])
    return np.array(X,float),np.array(y,float)
Xrd,yrd=Xy(dev); reg=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.08,max_iter=400,l2_regularization=1.0,random_state=20260526); reg.fit(Xrd,yrd)
def reg_pick(segs):
    rows=[]; idx=[]
    for si,s in enumerate(segs):
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1; rows.append(s["cand"][c]["feat"]+oh); idx.append((si,c))
    pr=reg.predict(np.array(rows,float)); best={}
    for (si,c),u in zip(idx,pr):
        if u>best.get(si,(None,0.0))[1]: best[si]=(c,u)
        best.setdefault(si,("ocr",0.0))
    return [best[si][0] if best[si][1]>0 else "ocr" for si in range(len(segs))]
picks_R=reg_pick(test)

never=micro(test,["ocr"]*len(test))
single_always={c:micro(test,[c]*len(test)) for c in POOL}
single_oracle={c:micro(test,[c if s["cand"][c]["ed"]<s["ed_ocr"] else "ocr" for s in test]) for c in POOL}
multi_oracle=micro(test,[min(ACTIONS,key=lambda a:ed_of(s,a)) for s in test])
cer_J=micro(test,picks_J); cer_R=micro(test,picks_R)
best_single_router=min(single_oracle.values())  # proxy lower-ref; real single-router from analyze
print(f"\n=== {DS} held-out test CER ===")
print(f"  never                  : {never:.4f}")
for c in POOL: print(f"  always {c:9s}       : {single_always[c]:.4f}  (oracle {single_oracle[c]:.4f})")
print(f"  MULTI oracle (ceiling) : {multi_oracle:.4f}")
print(f"  SELECT-regression (R)  : {cer_R:.4f}")
print(f"  SELECT-joint  (J)      : {cer_J:.4f}   <- joint multiclass")
print(f"  best single-corr oracle: {min(single_oracle.values()):.4f}")
print(f"  joint vs never         : {never-cer_J:+.4f} ({(never-cer_J)/never:.1%} rel.)")
print(f"  joint pick dist        : {dict(Counter(picks_J))}")
json.dump(dict(dataset=DS, pool=POOL, never=never, single_always=single_always, single_oracle=single_oracle,
               multi_oracle=multi_oracle, select_regression=cer_R, select_joint=cer_J,
               joint_picks=dict(Counter(picks_J))),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"multi_select_{DS}.json"),"w"),indent=2)
