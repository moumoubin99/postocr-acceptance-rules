#!/usr/bin/env python3
"""Iter-4 additions requested in review (CPU only, frozen candidates, no bootstrap):
  * Learn-then-Test-style calibration of the two one-parameter rules (edit cap, min-token confidence) on the FULL
    development set (they need no fitting half), next to the 50/50 protocol of iter2_ledger;
  * number of distinct development score values of the one-parameter rules (granularity of the threshold search).
Output: process/refine-logs/iter2/extra_fulldev.json
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from oclib import DATA
from iter2_ledger import CONFIGS, OUT, ED, Split, ltt

res = []
for ds, corr in CONFIGS:
    if not os.path.exists(os.path.join(DATA, f"candidates_{corr}_{ds}_dev.jsonl")): continue
    dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test")
    harm_d, harm_t = dv.ec > dv.eo, te.ec > te.eo; natt = max(1, int(te.att.sum()))
    rules = {"edit_cap": (-dv.X[:, ED], -te.X[:, ED])}
    if dv.has_conf: rules["conf_min"] = (dv.C[:, 1], te.C[:, 1])
    r = dict(dataset=ds, corrector=corr, n_dev_docs=int(dv.nd), n_test_docs=int(te.nd), distinct_dev_scores={k: int(len(np.unique(v[0][dv.att]))) for k, v in rules.items()}, ltt_full_dev={})
    for alpha in [0.02, 0.05, 0.10]:
        for k, (sd, st) in rules.items():
            t = ltt(sd, harm_d, dv.att, dv.di, dv.nd, dv.nseg, np.arange(dv.nd), alpha); u = te.att & (st >= t)
            r["ltt_full_dev"].setdefault(f"alpha{alpha}", {})[k] = dict(
                abstain=bool(not np.isfinite(t)), test_risk=float((np.bincount(te.di, u & harm_t, te.nd) / np.maximum(1, te.nseg)).mean()),
                test_cer=te.cer(u), never_cer=te.cer(np.zeros(len(te.rows), bool)), accept_rate=float(u[te.att].sum() / natt))
    res.append(r); print(ds, corr, r["ltt_full_dev"]["alpha0.1"], flush=True)
json.dump(res, open(os.path.join(OUT, "extra_fulldev.json"), "w"), indent=1)
