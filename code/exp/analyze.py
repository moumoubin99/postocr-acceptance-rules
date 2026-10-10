#!/usr/bin/env python3
"""
M1/M3/M4/M5 analysis. For each corrector with candidate files:
  - heterogeneity (help/hurt/neutral), oracle headroom
  - router trained on DEV, threshold tuned on DEV, evaluated on TEST (no test leakage)
  - frontier vs collision baselines (doc-QE, error-detection) at matched corruption budget
  - corrector-strength comparison + cross-corrector transfer
"""
import os, sys, json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from oclib import cer, features, FEATNAMES, load_jsonl, DATA
from rapidfuzz.distance import Levenshtein
RNG = 20260526
DATASET = os.environ.get("DATASET", "bln600")

def build(corrector, split):
    p = os.path.join(DATA, f"candidates_{corrector}_{DATASET}_{split}.jsonl")
    rows = load_jsonl(p)
    R = []
    for r in rows:
        cb, ca = cer(r["ocr"], r["gold"]), cer(r["cand"], r["gold"])
        R.append(dict(doc=r["doc_id"], ocr=r["ocr"], gold=r["gold"], cand=r["cand"],
                      glen=len(r["gold"]), cb=cb, ca=ca,
                      attempted=(r["ocr"] != r["cand"]),
                      helped=int(ca < cb - 1e-9), feat=features(r["ocr"], r["cand"])))
    return R

def micro(R, choices):
    e = l = 0
    for r, c in zip(R, choices):
        s = r["cand"] if c == "cand" else r["ocr"]
        e += Levenshtein.distance(s, r["gold"]); l += r["glen"]
    return e / max(1, l)

def corrupt(R, choices):
    return sum(1 for r, c in zip(R, choices) if c == "cand" and r["ca"] > r["cb"] + 1e-9)

def front_at_budget(R, score, attempted, budget, n=120):
    pts = []
    for thr in np.unique(np.quantile(score, np.linspace(0, 1, n))):
        ch = ["cand" if (attempted[i] and score[i] >= thr) else "ocr" for i in range(len(R))]
        pts.append((micro(R, ch), corrupt(R, ch)))
    cand = [c for c, k in pts if k <= budget]
    return min(cand) if cand else float("nan")

def analyze(corrector):
    dev, test = build(corrector, "dev"), build(corrector, "test")
    Xd = np.array([r["feat"] for r in dev]); yd = np.array([r["helped"] for r in dev])
    ad = np.array([r["attempted"] for r in dev])
    Xt = np.array([r["feat"] for r in test]); yt = np.array([r["helped"] for r in test])
    at = np.array([r["attempted"] for r in test])
    elig = [r for r in test if r["attempted"]]
    h = sum(r["helped"] for r in elig); hu = sum(1 for r in elig if r["ca"] > r["cb"]+1e-9)
    ne = len(elig) - h - hu
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                         l2_regularization=1.0, random_state=RNG)
    clf.fit(Xd[ad], yd[ad])
    pd_, pt = clf.predict_proba(Xd)[:, 1], clf.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt[at], pt[at]) if len(set(yt[at])) > 1 else float("nan")
    # tune threshold on DEV
    best = min(np.linspace(0.2, 0.8, 61),
               key=lambda thr: micro(dev, ["cand" if (ad[i] and pd_[i] >= thr) else "ocr" for i in range(len(dev))]))
    ch_t = ["cand" if (at[i] and pt[i] >= best) else "ocr" for i in range(len(test))]
    never = micro(test, ["ocr"]*len(test)); always = micro(test, ["cand"]*len(test))
    oracle = micro(test, ["cand" if r["ca"] < r["cb"] else "ocr" for r in test])
    router = micro(test, ch_t)
    ct_always = corrupt(test, ["cand"]*len(test)); ct_router = corrupt(test, ch_t)
    # collision baselines on TEST at always-corruption budget
    edens = np.array([r["feat"][6] for r in test])           # edit_density -> error-detection proxy
    docq = defaultdict(list)
    for r in test: docq[r["doc"]].append(r["feat"][6])
    dscore = np.array([np.mean(docq[r["doc"]]) for r in test])  # doc-QE proxy
    b_util = front_at_budget(test, pt, at, ct_always)
    b_err = front_at_budget(test, edens, at, ct_always)
    b_doc = front_at_budget(test, dscore, np.ones(len(test), bool), ct_always)
    return dict(corrector=corrector, n_test=len(test), eligible=len(elig),
                helped=h, hurt=hu, neutral=ne, auc=auc, thr=float(best),
                never=never, always=always, oracle=oracle, router=router,
                headroom_always_oracle=always-oracle, router_gain_vs_never=never-router,
                ct_always=ct_always, ct_router=ct_router,
                budget_util=b_util, budget_err=b_err, budget_doc=b_doc,
                _clf=clf, _test=test)

def transfer(src, dst):
    """Train router on src corrector (dev), apply to dst corrector (test)."""
    dev = build(src, "dev"); test = build(dst, "test")
    Xd = np.array([r["feat"] for r in dev]); yd = np.array([r["helped"] for r in dev])
    ad = np.array([r["attempted"] for r in dev])
    Xt = np.array([r["feat"] for r in test]); yt = np.array([r["helped"] for r in test])
    at = np.array([r["attempted"] for r in test])
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                         l2_regularization=1.0, random_state=RNG)
    clf.fit(Xd[ad], yd[ad]); pt = clf.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt[at], pt[at]) if len(set(yt[at])) > 1 else float("nan")
    return auc

if __name__ == "__main__":
    correctors = [c for c in ["symspell", "byt5", "byt5base", "byt5base8", "byt5large", "qwen"]
                  if os.path.exists(os.path.join(DATA, f"candidates_{c}_{DATASET}_test.jsonl"))]
    print(f"DATASET={DATASET}  correctors:", correctors)
    results = {}
    for c in correctors:
        r = analyze(c); results[c] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"\n===== corrector = {c}  (test segs {r['n_test']}, eligible {r['eligible']}) =====")
        print(f"  heterogeneity: helped {r['helped']} hurt {r['hurt']} neutral {r['neutral']} "
              f"({r['helped']/max(1,r['eligible']):.0%}/{r['hurt']/max(1,r['eligible']):.0%}/{r['neutral']/max(1,r['eligible']):.0%})")
        print(f"  TEST CER: never {r['never']:.4f} | always {r['always']:.4f} | "
              f"router {r['router']:.4f} (thr={r['thr']:.2f}) | oracle {r['oracle']:.4f}")
        print(f"  oracle headroom (always-oracle) = {r['headroom_always_oracle']:.4f} ; "
              f"router gain vs never = {r['router_gain_vs_never']:+.4f}")
        print(f"  AUC(test) = {r['auc']:.3f} ; corruption: always {r['ct_always']} router {r['ct_router']}")
        print(f"  @always-corruption budget CER:  util {r['budget_util']:.4f} | "
              f"err-det {r['budget_err']:.4f} | doc-QE {r['budget_doc']:.4f}")
    # M4 corrector-strength claim
    if "symspell" in results and "byt5" in results:
        print("\n===== M4: does router advantage widen with stronger corrector? =====")
        for c in ["symspell", "byt5"]:
            print(f"  {c:9s}: oracle_headroom={results[c]['headroom_always_oracle']:.4f}  "
                  f"router_gain_vs_never={results[c]['router_gain_vs_never']:+.4f}")
        # M5 transfer
        print("\n===== M5: cross-corrector router transfer (AUC on dst test) =====")
        print(f"  train symspell -> test byt5 : {transfer('symspell','byt5'):.3f}")
        print(f"  train byt5 -> test symspell : {transfer('byt5','symspell'):.3f}")
    json.dump(results, open(os.path.join(os.path.dirname(__file__), "..", "refine-logs",
              f"stage2_results_{DATASET}.json"), "w"), indent=2)
    print(f"\nsaved -> refine-logs/stage2_results_{DATASET}.json")
