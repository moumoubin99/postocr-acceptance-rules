#!/usr/bin/env python3
"""Per-language doc-cluster bootstrap for the multilingual router (always−router, never−router)."""
import os, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
LANGS=["EN","DE","FR","ES","NL"]; CORR="byt5base8"; RNG=np.random.default_rng(20260526)
def build(lang,split):
    R=[]
    for r in load_jsonl(os.path.join(DATA,f"candidates_{CORR}_icdar_{lang}_{split}.jsonl")):
        cb,ca=cer(r["ocr"],r["gold"]),cer(r["cand"],r["gold"])
        R.append(dict(doc=r["doc_id"],eo=Levenshtein.distance(r["ocr"],r["gold"]),ec=Levenshtein.distance(r["cand"],r["gold"]),
                      glen=len(r["gold"]),att=(r["ocr"]!=r["cand"]),helped=int(ca<cb-1e-9),feat=features(r["ocr"],r["cand"])))
    return R
def micro(R,use): return sum(r["ec"] if u else r["eo"] for r,u in zip(R,use))/max(1,sum(r["glen"] for r in R))
dev_all=[];
for L in LANGS: dev_all+=build(L,"dev")
Xd=np.array([r["feat"] for r in dev_all]); yd=np.array([r["helped"] for r in dev_all]); ad=np.array([r["att"] for r in dev_all])
clf=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.08,max_iter=400,l2_regularization=1.0,random_state=20260526)
clf.fit(Xd[ad],yd[ad]); pdv=clf.predict_proba(Xd)[:,1]
thr=min(np.linspace(0.2,0.8,61),key=lambda t:micro(dev_all,[ad[i] and pdv[i]>=t for i in range(len(dev_all))]))
out={}
print(f"per-language bootstrap (B=2000, thr={thr:.2f}):")
for L in LANGS:
    test=build(L,"test"); Xt=np.array([r["feat"] for r in test]); at=np.array([r["att"] for r in test])
    pt=clf.predict_proba(Xt)[:,1]; route=np.array([at[i] and pt[i]>=thr for i in range(len(test))])
    by=defaultdict(list)
    for i,r in enumerate(test): by[r["doc"]].append(i)
    docs=list(by)
    def m(idx,mode):
        sub=[test[i] for i in idx]
        if mode=="never": use=[False]*len(sub)
        elif mode=="always": use=[True]*len(sub)
        else: use=[route[i] for i in idx]
        return micro(sub,use)
    idx0=list(range(len(test))); n0,a0,r0=m(idx0,"never"),m(idx0,"always"),m(idx0,"router")
    da,dn=[],[]
    for _ in range(2000):
        s=RNG.choice(docs,size=len(docs),replace=True); idx=[i for d in s for i in by[d]]
        da.append(m(idx,"always")-m(idx,"router")); dn.append(m(idx,"never")-m(idx,"router"))
    ca=np.percentile(da,[2.5,97.5]); cn=np.percentile(dn,[2.5,97.5])
    sig_a="sig" if ca[0]>0 else "ns"; sig_n="sig" if cn[0]>0 else "ns"
    print(f"  {L}: never-router {n0-r0:+.4f}[{cn[0]:+.4f},{cn[1]:+.4f}]{sig_n}  always-router {a0-r0:+.4f}[{ca[0]:+.4f},{ca[1]:+.4f}]{sig_a}")
    out[L]=dict(never_router=[n0-r0,*ca.tolist()] if False else [n0-r0,*cn.tolist()], always_router=[a0-r0,*ca.tolist()], sig_never=sig_n, sig_always=sig_a)
json.dump(out,open(os.path.join(os.path.dirname(__file__),"..","refine-logs","multilingual_bootstrap.json"),"w"),indent=2)
