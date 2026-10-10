#!/usr/bin/env python3
"""Multilingual generalization: ONE multilingual ByT5-base-8ep corrector + ONE router
(trained on combined multilingual dev) evaluated PER LANGUAGE on held-out test.
Tests whether selective acceptance generalizes cross-lingually."""
import os, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
from oclib import cer, features, load_jsonl, DATA
LANGS = ["EN", "DE", "FR", "ES", "NL"]
CORR = "byt5base8"

def build(lang, split):
    p = os.path.join(DATA, f"candidates_{CORR}_icdar_{lang}_{split}.jsonl")
    R = []
    for r in load_jsonl(p):
        cb, ca = cer(r["ocr"], r["gold"]), cer(r["cand"], r["gold"])
        R.append(dict(eo=Levenshtein.distance(r["ocr"], r["gold"]), ec=Levenshtein.distance(r["cand"], r["gold"]),
                      glen=len(r["gold"]), att=(r["ocr"] != r["cand"]), helped=int(ca < cb-1e-9), feat=features(r["ocr"], r["cand"])))
    return R

def micro(R, use): return sum(r["ec"] if u else r["eo"] for r, u in zip(R, use))/max(1, sum(r["glen"] for r in R))

# train ONE router on combined multilingual dev
dev_all = []
for L in LANGS: dev_all += build(L, "dev")
Xd = np.array([r["feat"] for r in dev_all]); yd = np.array([r["helped"] for r in dev_all]); ad = np.array([r["att"] for r in dev_all])
clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=400, l2_regularization=1.0, random_state=20260526)
clf.fit(Xd[ad], yd[ad]); pdv = clf.predict_proba(Xd)[:, 1]
thr = min(np.linspace(0.2, 0.8, 61), key=lambda t: micro(dev_all, [ad[i] and pdv[i] >= t for i in range(len(dev_all))]))

print(f"Multilingual: 1 corrector (ByT5-base-8ep on 5 langs) + 1 router (combined dev, thr={thr:.2f})")
print(f"{'lang':5s} {'never':>7s} {'always':>7s} {'router':>7s} {'oracle':>7s} {'AUC':>6s} {'help%':>6s} {'rel↓':>6s}")
rows = {}
for L in LANGS:
    test = build(L, "test")
    Xt = np.array([r["feat"] for r in test]); yt = np.array([r["helped"] for r in test]); at = np.array([r["att"] for r in test])
    pt = clf.predict_proba(Xt)[:, 1]
    use = [at[i] and pt[i] >= thr for i in range(len(test))]
    never = micro(test, [False]*len(test)); always = micro(test, [True]*len(test))
    oracle = micro(test, [r["ec"] < r["eo"] for r in test]); router = micro(test, use)
    auc = roc_auc_score(yt[at], pt[at]) if len(set(yt[at])) > 1 else float("nan")
    helped = sum(r["helped"] for r in test if r["att"]); elig = sum(r["att"] for r in test)
    rel = (never-router)/never
    print(f"{L:5s} {never:7.4f} {always:7.4f} {router:7.4f} {oracle:7.4f} {auc:6.3f} {helped/max(1,elig):6.1%} {rel:6.1%}")
    rows[L] = dict(never=never, always=always, router=router, oracle=oracle, auc=auc, rel_gain=rel)
# macro average
ma = {k: float(np.mean([rows[L][k] for L in LANGS])) for k in ["never","always","router","oracle","rel_gain"]}
print(f"{'MACRO':5s} {ma['never']:7.4f} {ma['always']:7.4f} {ma['router']:7.4f} {ma['oracle']:7.4f} {'':6s} {'':6s} {ma['rel_gain']:6.1%}")
json.dump(dict(thr=float(thr), per_lang=rows, macro=ma),
          open(os.path.join(os.path.dirname(__file__),"..","refine-logs","multilingual_results.json"),"w"), indent=2)
