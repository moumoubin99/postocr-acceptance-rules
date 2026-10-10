#!/usr/bin/env python3
"""Generate SymSpell candidate corrections for a split (cheap corrector baseline).
Usage: python exp/gen_symspell.py dev[,test]   Env: DATASET, DATA_DIR, OUT_DIR."""
import os, sys, json
from oclib import symspell_correct, load_jsonl, DATA

DATA = os.environ.get("DATA_DIR", DATA)
OUT_DIR = os.environ.get("OUT_DIR", DATA)
DATASET = os.environ.get("DATASET", "bln600")
PASS = ("seg_idx", "ocr_sep", "gold_sep", "orig_id", "lang", "group", "dup_train")
for split in sys.argv[1].split(","):
    rows = load_jsonl(os.path.join(DATA, f"segments_{DATASET}_{split}.jsonl"))
    out = os.path.join(OUT_DIR, f"candidates_symspell_{DATASET}_{split}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            rec = {"doc_id": r["doc_id"], "ocr": r["ocr"], "gold": r["gold"], "cand": symspell_correct(r["ocr"])}
            rec.update({p: r[p] for p in PASS if p in r})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} -> {os.path.basename(out)}")
