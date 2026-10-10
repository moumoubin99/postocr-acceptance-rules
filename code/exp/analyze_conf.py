#!/usr/bin/env python3
"""Confidence-feature experiment: does adding the corrector's own decoding confidence
(mean token log-prob, field 'score') improve the single-corrector acceptance router?
Compares router WITHOUT vs WITH the confidence feature. DATASET via env; corrector byt5base8."""
import os, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
DS = os.environ.get("DATASET", "bln600"); CORR = "byt5base8"
def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=20260526)
def build(split):
    R=[]
    for r in load_jsonl(os.path.join(DATA, f"candidates_{CORR}_{DS}_{split}.jsonl")):
        cb,ca=cer(r["ocr"],r["gold"]),cer(r["cand"],r["gold"])
        R.append(dict(eo=Levenshtein.distance(r["ocr"],r["gold"]), ec=Levenshtein.distance(r["cand"],r["gold"]),
                      glen=len(r["gold"]), att=(r["ocr"]!=r["cand"]), helped=int(ca<cb-1e-9),
                      feat=features(r["ocr"],r["cand"]), score=float(r.get("score", 0.0))))
    return R
dev,test=build("dev"),build("test")
def micro(R,use): return sum(r["ec"] if u else r["eo"] for r,u in zip(R,use))/max(1,sum(r["glen"] for r in R))
ad=np.array([r["att"] for r in dev]); at=np.array([r["att"] for r in test])
yd=np.array([r["helped"] for r in dev]); yt=np.array([r["helped"] for r in test])
def run(with_conf):
    Xd=np.array([r["feat"]+([r["score"]] if with_conf else []) for r in dev])
    Xt=np.array([r["feat"]+([r["score"]] if with_conf else []) for r in test])
    c=gb(); c.fit(Xd[ad],yd[ad]); pdv=c.predict_proba(Xd)[:,1]; pt=c.predict_proba(Xt)[:,1]
    thr=min(np.linspace(0.2,0.8,61), key=lambda t: micro(dev,[ad[i] and pdv[i]>=t for i in range(len(dev))]))
    use=[at[i] and pt[i]>=thr for i in range(len(test))]
    auc=roc_auc_score(yt[at],pt[at]) if len(set(yt[at]))>1 else float("nan")
    return micro(test,use), auc, thr
never=micro(test,[False]*len(test)); always=micro(test,[True]*len(test)); oracle=micro(test,[r["ec"]<r["eo"] for r in test])
base_cer,base_auc,base_thr=run(False); conf_cer,conf_auc,conf_thr=run(True)
print(f"=== {DS} (corrector={CORR}) confidence-feature ablation, held-out test ===")
print(f"  never {never:.4f} | always {always:.4f} | oracle {oracle:.4f}")
print(f"  router WITHOUT confidence : CER {base_cer:.4f}  AUC {base_auc:.3f}")
print(f"  router WITH confidence    : CER {conf_cer:.4f}  AUC {conf_auc:.3f}")
print(f"  delta from confidence     : CER {base_cer-conf_cer:+.4f}  AUC {conf_auc-base_auc:+.3f}")
json.dump(dict(dataset=DS, never=never, always=always, oracle=oracle,
               router_base=base_cer, router_conf=conf_cer, auc_base=base_auc, auc_conf=conf_auc),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"conf_feature_{DS}.json"),"w"),indent=2)
