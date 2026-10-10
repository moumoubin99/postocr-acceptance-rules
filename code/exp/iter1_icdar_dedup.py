#!/usr/bin/env python3
"""Iteration-1 ICDAR leakage check (CPU only, frozen candidates).
ICDAR-2019 doc IDs are disjoint across splits, but OCR segment text recurs verbatim
between train and dev/test. This drops every dev/test segment whose OCR string occurs
in segments_icdar_train.jsonl ("icdarclean") and re-runs the claim ledger on it.
NOTE: correctors were still trained on the full train split; a proper fix re-splits/
dedups and retrains (GPU). Outputs: process/refine-logs/iter1/clean_icdar/
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = os.path.join(ROOT, "data"); C = os.path.join(ROOT, "process", "refine-logs", "iter1", "clean_icdar", "data")
os.makedirs(C, exist_ok=True)
os.environ.setdefault("OUT", os.path.join(ROOT, "process", "refine-logs", "iter1", "clean_icdar"))
sys.path.insert(0, HERE)
import iter1_claim_ledger as M
CORRS = ["symspell", "byt5", "byt5base", "byt5base8"]
trs = {json.loads(l)["ocr"] for l in open(f"{P}/segments_icdar_train.jsonl")}
stats = {}
for corr in CORRS:
    for sp in ["dev", "test"]:
        src = [l for l in open(f"{P}/candidates_{corr}_icdar_{sp}.jsonl")]
        rows = [l for l in src if json.loads(l)["ocr"] not in trs]
        stats[f"{corr}_{sp}"] = {"n_all": len(src), "n_clean": len(rows)}
        open(f"{C}/candidates_{corr}_icdarclean_{sp}.jsonl", "w").writelines(rows)
json.dump(stats, open(os.path.join(os.environ["OUT"], "dedup_counts.json"), "w"), indent=2)
M.DATA = C
for corr in CORRS:
    o = M.run("icdarclean", corr); p = o["policies"]; b = o["bootstrap_vs_router_text"]
    print(corr, {k: round(v["cer"], 4) for k, v in p.items()})
    print("   boot:", {k: (round(v["delta"], 4), round(v["p_holm"], 3)) for k, v in b.items()})
