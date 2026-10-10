#!/usr/bin/env python3
"""Fig. 2 data: test CER vs realised harm when the acceptance threshold is swept (descriptive frontier), plus the
dev-chosen operating points (CER-optimal and 10/25/50% harm budgets). Segment-level rules on frozen candidates.
Usage: python3 paper/make_frontier_data.py <corr>_<ds> ...   -> paper/figures/data/frontier_<corr>_<ds>.json"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); args = sys.argv[1:]; sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "exp"))
from iter2_ledger import Split, scores, tune, qgrid
OUT = os.path.join(HERE, "figures", "data")
for cfg in args:
    corr, ds = cfg.split("_", 1); dv, te = Split(corr, ds, "dev"), Split(corr, ds, "test")
    S, full = scores(dv, te); a_dev = dv.corrupt(dv.att)
    rules = {"router": full, "edit_cap": "edit_cap", "conf_min": "conf_min", "router_nodict": "router_nodict"}
    out = dict(corrector=corr, dataset=ds, n_test=len(te.rows), attempted=int(te.att.sum()),
               never=dict(cer=te.cer(np.zeros(len(te.rows), bool)), harm=0), always=dict(cer=te.cer(te.att), harm=te.corrupt(te.att)),
               oracle=dict(cer=te.cer(te.ec < te.eo), harm=0), curves={}, points={})
    for name, key in rules.items():
        if key not in S: continue
        sd, st = S[key]; ts = np.unique(np.r_[np.quantile(st[te.att], np.linspace(0, 1, 401)), np.inf])
        cur = sorted({(te.corrupt(te.att & (st >= t)), te.cer(te.att & (st >= t))) for t in ts})
        out["curves"][name] = [[int(h), float(c)] for h, c in cur]
        pts = {"opt": tune(dv, sd)}
        for b in (10, 25, 50): pts[f"{b}pct"] = tune(dv, sd, qgrid(sd, dv.att, 201), budget=int(b / 100 * a_dev))
        out["points"][name] = {k: dict(harm=int(te.corrupt(te.att & (st >= t))), cer=float(te.cer(te.att & (st >= t)))) for k, t in pts.items()}
    json.dump(out, open(os.path.join(OUT, f"frontier_{cfg}.json"), "w")); print(cfg, "done", flush=True)
