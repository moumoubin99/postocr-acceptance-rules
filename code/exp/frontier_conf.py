#!/usr/bin/env python3
"""Corruption-constrained operating points: does the confidence-aware acceptance router
give a better CER-corruption frontier than text-only — especially in the high-precision
'do no harm' regime? Single corrector ByT5-base-8ep. Dev-trained router; test frontier.
DATASET via env."""
import os, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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
                      feat=features(r["ocr"],r["cand"]), score=float(r.get("score",0.0))))
    return R
dev,test=build("dev"),build("test")
ad=np.array([r["att"] for r in dev]); yd=np.array([r["helped"] for r in dev]); at=np.array([r["att"] for r in test])
glen_tot=sum(r["glen"] for r in test)
def cer_corr(p, thr):
    """net CER and induced-corruption count at threshold thr (accept attempted & p>=thr)."""
    ed=corr=0
    for i,r in enumerate(test):
        acc = r["att"] and p[i]>=thr
        ed += r["ec"] if acc else r["eo"]
        if acc and r["ec"]>r["eo"]: corr+=1
    return ed/glen_tot, corr
def train_router(with_conf):
    Xd=np.array([r["feat"]+([r["score"]] if with_conf else []) for r in dev])
    Xt=np.array([r["feat"]+([r["score"]] if with_conf else []) for r in test])
    c=gb(); c.fit(Xd[ad],yd[ad]); return c.predict_proba(Xt)[:,1]
pt_text=train_router(False); pt_conf=train_router(True)
never=cer_corr(pt_text, 2.0)[0]  # thr>1 -> accept none -> never
always_corr=cer_corr(pt_text,-1.0)[1]  # thr<0 -> accept all attempted -> always corruption count
grid=np.linspace(0,1,201)
def frontier(p): return [cer_corr(p,t) for t in grid]
ft=frontier(pt_text); fc=frontier(pt_conf)
def cer_at_budget(front, budget):
    cs=[c for c,k in front if k<=budget]; return min(cs) if cs else float("nan")
def min_corr_beating_never(front):
    cand=[(c,k) for c,k in front if c<never-1e-9]; return min(cand,key=lambda x:x[1]) if cand else None
print(f"=== {DS} CER-corruption frontier: text-only vs +confidence (ByT5-base-8ep) ===")
print(f"  never CER {never:.4f}; always-correct induced-corruption count = {always_corr}")
print(f"  CER at matched corruption budgets (lower=better):")
for lab,b in [("always",always_corr),("50%",always_corr//2),("25%",always_corr//4),("10%",always_corr//10)]:
    print(f"    budget={lab:6s}({b:4d}): text {cer_at_budget(ft,b):.4f} | +conf {cer_at_budget(fc,b):.4f}")
print(f"  high-precision 'do no harm' (fewest corruptions while still beating never on CER):")
mt=min_corr_beating_never(ft); mc=min_corr_beating_never(fc)
print(f"    text : CER {mt[0]:.4f} at {mt[1]} corruptions" if mt else "    text : none")
print(f"    +conf: CER {mc[0]:.4f} at {mc[1]} corruptions" if mc else "    +conf: none")
if mt and mc: print(f"    -> confidence cuts corruptions {mt[1]}->{mc[1]} ({(mt[1]-mc[1])/max(1,mt[1]):+.0%}) at the safe operating point")
json.dump(dict(dataset=DS, never=never, always_corr=int(always_corr),
               budgets={lab:[cer_at_budget(ft,b),cer_at_budget(fc,b)] for lab,b in [("always",always_corr),("50",always_corr//2),("25",always_corr//4),("10",always_corr//10)]},
               safe_text=mt, safe_conf=mc, frontier_text=ft, frontier_conf=fc),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs",f"frontier_conf_{DS}.json"),"w"),indent=2)
