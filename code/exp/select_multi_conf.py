#!/usr/bin/env python3
"""Multi-corrector selection WITH confidence: does adding each corrector's decoding
log-prob (field 'score') let selection exploit the multi-oracle headroom that text-only
features could not? Pool = neural correctors (all have confidence) + keep-OCR. Anchored
to best corrector. Compares to single-best router and (text-only) multi-select. DATASET via env."""
import os, json, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingRegressor
from rapidfuzz.distance import Levenshtein
from oclib import features, load_jsonl, DATA
DS = os.environ.get("DATASET", "bln600"); ANCHOR = "byt5base8"
POOL = [c for c in ["byt5base","byt5base8"]   # both carry real decoding confidence
        if os.path.exists(os.path.join(DATA, f"candidates_{c}_{DS}_test.jsonl"))]
cidx={c:k for k,c in enumerate(POOL)}
def load(split):
    spine=load_jsonl(os.path.join(DATA,f"segments_{DS}_{split}.jsonl"))
    cd={c:load_jsonl(os.path.join(DATA,f"candidates_{c}_{DS}_{split}.jsonl")) for c in POOL}
    segs=[]
    for i,s in enumerate(spine):
        ocr,gold=s["ocr"],s["gold"]; row=dict(glen=len(gold),ed_ocr=Levenshtein.distance(ocr,gold),c={})
        for c in POOL:
            r=cd[c][i]; assert r["ocr"]==ocr
            row["c"][c]=dict(ed=Levenshtein.distance(r["cand"],gold), feat=features(ocr,r["cand"]), score=float(r.get("score",0.0)))
        segs.append(row)
    return segs
dev,test=load("dev"),load("test")
def ed_of(s,a): return s["ed_ocr"] if a=="ocr" else s["c"][a]["ed"]
def micro(segs,picks): return sum(ed_of(s,p) for s,p in zip(segs,picks))/max(1,sum(s["glen"] for s in segs))
def Xy(segs, with_conf):
    X,y=[],[]
    for s in segs:
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1
            f=s["c"][c]["feat"]+oh+([s["c"][c]["score"]] if with_conf else [])
            X.append(f); y.append(s["ed_ocr"]-s["c"][c]["ed"])
    return np.array(X,float),np.array(y,float)
def utils(segs, reg, with_conf):
    rows=[]
    for s in segs:
        for c in POOL:
            oh=[0]*len(POOL); oh[cidx[c]]=1
            rows.append(s["c"][c]["feat"]+oh+([s["c"][c]["score"]] if with_conf else []))
    return reg.predict(np.array(rows,float)).reshape(len(segs),len(POOL))
def anchored(segs,U,delta):
    picks=[]
    for si,s in enumerate(segs):
        u={c:U[si,cidx[c]] for c in POOL}
        base=ANCHOR if u[ANCHOR]>0 else "ocr"; base_u=max(u[ANCHOR],0.0)
        alts=[(c,u[c]) for c in POOL if c!=ANCHOR]+[("ocr",0.0)]; a,a_u=max(alts,key=lambda t:t[1])
        picks.append(a if a_u>base_u+delta else base)
    return picks
def eval_variant(with_conf):
    Xd,yd=Xy(dev,with_conf); reg=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.06,max_iter=500,l2_regularization=1.0,random_state=20260526); reg.fit(Xd,yd)
    Ud,Ut=utils(dev,reg,with_conf),utils(test,reg,with_conf)
    grid=np.concatenate([[1e9],np.linspace(0,30,61)])
    d=min(grid,key=lambda d:micro(dev,anchored(dev,Ud,d)))
    return micro(test,anchored(test,Ut,d)), d, anchored(test,Ut,d)
never=micro(test,["ocr"]*len(test))
sbr=micro(test,[ANCHOR if (test_i["c"][ANCHOR]["ed"]<test_i["ed_ocr"]) else "ocr" for test_i in test])  # oracle-ish single; replaced below
# single-best router via its own util model (text-only, anchor only)
multi_oracle=micro(test,[min(["ocr"]+POOL,key=lambda a:ed_of(s,a)) for s in test])
notc_cer,notc_d,_=eval_variant(False); conf_cer,conf_d,picks=eval_variant(True)
print(f"=== {DS} multi-select WITH vs WITHOUT confidence (pool={POOL}, anchor={ANCHOR}) ===")
print(f"  never {never:.4f} | multi-oracle {multi_oracle:.4f}")
print(f"  anchored multi, text-only : {notc_cer:.4f} (delta={notc_d:.1f})")
print(f"  anchored multi, +CONFIDENCE: {conf_cer:.4f} (delta={conf_d:.1f})")
print(f"  confidence effect: {notc_cer-conf_cer:+.4f}")
print(f"  pick dist (+conf): {dict(Counter(picks))}")
json.dump(dict(dataset=DS,never=never,multi_oracle=multi_oracle,text_only=notc_cer,with_conf=conf_cer,pool=POOL),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"multi_conf_{DS}.json"),"w"),indent=2)
