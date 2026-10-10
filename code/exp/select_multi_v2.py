#!/usr/bin/env python3
"""Multi-corrector selection v2 (Codex-recommended): calibrated per-candidate utility
regression + ANCHORED override. Default = best-single-corrector router; switch to another
candidate only if predicted-utility margin > delta (tuned on dev). Guarantees >= single-best
at delta=inf, so it can capture multi-oracle headroom without regressing. DATASET via env."""
import os, json, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingRegressor
from rapidfuzz.distance import Levenshtein
from oclib import features, load_jsonl, DATA
DS = os.environ.get("DATASET", "bln600")
ANCHOR = "byt5base8"   # best single corrector
POOL = [c for c in ["symspell","byt5","byt5base","byt5base8"]
        if os.path.exists(os.path.join(DATA, f"candidates_{c}_{DS}_test.jsonl"))]
cidx = {c:k for k,c in enumerate(POOL)}

def load(split):
    spine = load_jsonl(os.path.join(DATA, f"segments_{DS}_{split}.jsonl"))
    cands = {c: load_jsonl(os.path.join(DATA, f"candidates_{c}_{DS}_{split}.jsonl")) for c in POOL}
    segs=[]
    for i,s in enumerate(spine):
        ocr,gold=s["ocr"],s["gold"]; row=dict(glen=len(gold), ed_ocr=Levenshtein.distance(ocr,gold), c={})
        for c in POOL:
            r=cands[c][i]; assert r["ocr"]==ocr
            row["c"][c]=dict(ed=Levenshtein.distance(r["cand"],gold), feat=features(ocr,r["cand"]))
        segs.append(row)
    return segs
dev,test=load("dev"),load("test")
def ed_of(s,a): return s["ed_ocr"] if a=="ocr" else s["c"][a]["ed"]
def micro(segs,picks): return sum(ed_of(s,p) for s,p in zip(segs,picks))/max(1,sum(s["glen"] for s in segs))

# per-candidate utility regressor: y = chars saved vs OCR
def Xy(segs):
    X,y=[],[]
    for s in segs:
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1; X.append(s["c"][c]["feat"]+oh); y.append(s["ed_ocr"]-s["c"][c]["ed"])
    return np.array(X,float), np.array(y,float)
Xd,yd=Xy(dev); reg=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.06,max_iter=500,l2_regularization=1.0,random_state=20260526); reg.fit(Xd,yd)
def utils(segs):
    rows=[];
    for s in segs:
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1; rows.append(s["c"][c]["feat"]+oh)
    pr=reg.predict(np.array(rows,float)).reshape(len(segs),len(POOL))
    return pr  # [nseg, ncorr] predicted chars saved
Ud, Ut = utils(dev), utils(test)

def anchored_picks(segs, U, delta):
    picks=[]
    for si,s in enumerate(segs):
        u={c:U[si,cidx[c]] for c in POOL}
        # base = anchor router decision: anchor if util>0 else ocr
        base = ANCHOR if u[ANCHOR] > 0 else "ocr"; base_u = max(u[ANCHOR], 0.0)
        # best alternative among others (+ocr)
        alts = [(c,u[c]) for c in POOL if c!=ANCHOR] + [("ocr",0.0)]
        a,a_u = max(alts, key=lambda t:t[1])
        picks.append(a if a_u > base_u + delta else base)
    return picks

# tune delta on dev
grid=np.concatenate([[1e9], np.linspace(0,30,61)])
best=min(grid, key=lambda d: micro(dev, anchored_picks(dev,Ud,d)))
picks_t=anchored_picks(test,Ut,best)
cer_anch=micro(test,picks_t)
# references
never=micro(test,["ocr"]*len(test))
anchor_router=micro(test,[ANCHOR if (Ut[i,cidx[ANCHOR]]>0) else "ocr" for i in range(len(test))])
multi_oracle=micro(test,[min(["ocr"]+POOL,key=lambda a:ed_of(s,a)) for s in test])
print(f"DATASET={DS} pool={POOL} anchor={ANCHOR}")
print(f"  never                : {never:.4f}")
print(f"  single-best router   : {anchor_router:.4f}  (anchor={ANCHOR})")
print(f"  ANCHORED multi (v2)  : {cer_anch:.4f}  (dev-tuned delta={best:.1f})")
print(f"  multi-corrector oracle: {multi_oracle:.4f}")
print(f"  pick dist: {dict(Counter(picks_t))}")
gain = anchor_router - cer_anch
print(f"  v2 vs single-best router: {gain:+.4f} ({'IMPROVES' if gain>1e-5 else 'no gain (falls back)'})")
json.dump(dict(dataset=DS, never=never, single_best_router=anchor_router, anchored_multi=cer_anch,
               multi_oracle=multi_oracle, delta=float(best), picks=dict(Counter(picks_t))),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"multi_select_v2_{DS}.json"),"w"),indent=2)
