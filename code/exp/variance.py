#!/usr/bin/env python3
"""ARS must-fix #7: seed variance of the ICDAR ByT5-small result (3 seeds).
For each seed: train router on dev, dev-tune threshold, eval held-out test."""
import os, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
DS = "icdar"
def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=20260526)
def build(corr, split):
    R=[]
    for r in load_jsonl(os.path.join(DATA, f"candidates_{corr}_{DS}_{split}.jsonl")):
        cb,ca=cer(r["ocr"],r["gold"]),cer(r["cand"],r["gold"])
        R.append(dict(eo=Levenshtein.distance(r["ocr"],r["gold"]), ec=Levenshtein.distance(r["cand"],r["gold"]),
                      glen=len(r["gold"]), att=(r["ocr"]!=r["cand"]), helped=int(ca<cb-1e-9), feat=features(r["ocr"],r["cand"])))
    return R
def micro(R,use): return sum(r["ec"] if u else r["eo"] for r,u in zip(R,use))/max(1,sum(r["glen"] for r in R))
rows=[]
for seed in [1,2,3]:
    corr=f"byt5s{seed}"; dev,test=build(corr,"dev"),build(corr,"test")
    Xd=np.array([r["feat"] for r in dev]); yd=np.array([r["helped"] for r in dev]); ad=np.array([r["att"] for r in dev])
    Xt=np.array([r["feat"] for r in test]); at=np.array([r["att"] for r in test])
    c=gb(); c.fit(Xd[ad],yd[ad]); pdv=c.predict_proba(Xd)[:,1]; pt=c.predict_proba(Xt)[:,1]
    thr=min(np.linspace(0.2,0.8,61), key=lambda t: micro(dev,[ad[i] and pdv[i]>=t for i in range(len(dev))]))
    use=[at[i] and pt[i]>=thr for i in range(len(test))]
    never=micro(test,[False]*len(test)); always=micro(test,[True]*len(test))
    oracle=micro(test,[r["ec"]<r["eo"] for r in test]); router=micro(test,use)
    rows.append((never,always,router,oracle)); print(f"  seed {seed}: never {never:.4f} always {always:.4f} router {router:.4f} oracle {oracle:.4f}")
a=np.array(rows)
labels=["never","always","router","oracle"]
print("\n=== ICDAR ByT5-small seed variance (n=3) ===")
for i,l in enumerate(labels):
    print(f"  {l:7s}: {a[:,i].mean():.4f} ± {a[:,i].std(ddof=1):.4f}  (min {a[:,i].min():.4f}, max {a[:,i].max():.4f})")
print(f"  router gain vs never: {(a[:,0]-a[:,2]).mean():+.4f} ± {(a[:,0]-a[:,2]).std(ddof=1):.4f}")
print(f"  always penalty vs never: {(a[:,1]-a[:,0]).mean():+.4f} ± {(a[:,1]-a[:,0]).std(ddof=1):.4f}")
json.dump(dict(seeds=[1,2,3], rows=a.tolist(), labels=labels,
               router_mean=float(a[:,2].mean()), router_std=float(a[:,2].std(ddof=1)),
               always_mean=float(a[:,1].mean()), always_std=float(a[:,1].std(ddof=1))),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs","icdar_seed_variance.json"),"w"),indent=2)
