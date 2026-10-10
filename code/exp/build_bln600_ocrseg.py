#!/usr/bin/env python3
"""B3: OCR-only (gold-free) BLN600 segmentation. The iter-1 segments were cut with a
word-level OCR/gold SequenceMatcher and filtered on gold length (oracle information).
Here boundaries depend on the OCR text only (exp/segutil.py); gold is attached afterwards
(for scoring) via a whole-document Levenshtein character map, segments keep exact spans and
separators, so both documents reconstruct exactly and reassembled-document CER is exact.

Same document split as build_segments.py (same doc order + seed 20260526).
Variants (env VARIANTS, default all):
  c70   cut at an OCR space once >= 70 OCR chars (ByT5 setting)   -> segments_blnocr_*
  w50   50 OCR words per window (LLM context ablation)            -> segments_blnocrw50_*
  w100  100 OCR words per window                                  -> segments_blnocrw100_*
Also docs_bln600_{split}.jsonl (whole whitespace-normalised documents)."""
import os, glob, re, json, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segutil import segment, map_from_opcodes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "data", "Ground Truth")
OCR = os.path.join(ROOT, "data", "OCR Text")
OUT = os.environ.get("OUT", os.path.join(ROOT, "data"))
VARIANTS = os.environ.get("VARIANTS", "c70,w50,w100").split(",")
NAMES = {"c70": "blnocr", "w50": "blnocrw50", "w100": "blnocrw100"}

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    docs = []
    for gp in sorted(glob.glob(os.path.join(GT, "*.txt"))):
        fid = os.path.splitext(os.path.basename(gp))[0]
        op = os.path.join(OCR, fid + ".txt")
        if not os.path.exists(op): continue
        gold = re.sub(r"\s+", " ", open(gp, errors="ignore").read().replace("\n", " ")).strip()
        ocr = re.sub(r"\s+", " ", open(op, errors="ignore").read().replace("\n", " ")).strip()
        if gold and ocr: docs.append((fid, ocr, gold))
    random.seed(20260526)
    random.shuffle(docs)
    n = len(docs)
    splits = {"train": docs[:int(.67*n)], "dev": docs[int(.67*n):int(.83*n)], "test": docs[int(.83*n):]}
    for name, dl in splits.items():
        dump(os.path.join(OUT, f"docs_bln600_{name}.jsonl"),
             [dict(doc_id=f, ocr=o, gold=g) for f, o, g in dl])
        for v in VARIANTS:
            rows = []
            for fid, ocr, gold in dl:
                m = map_from_opcodes(ocr, gold)   # asserts exact reconstruction inside segment()
                for k, s in enumerate(segment(ocr, gold, m, v)):
                    rows.append(dict(doc_id=fid, seg_idx=k, **s))
            dump(os.path.join(OUT, f"segments_{NAMES[v]}_{name}.jsonl"), rows)
            print(f"{name:5s} {v:5s}: {len(dl):3d} docs {len(rows):6d} segs")

if __name__ == "__main__":
    main()
