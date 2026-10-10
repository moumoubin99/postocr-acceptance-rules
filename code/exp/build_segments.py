#!/usr/bin/env python3
"""Build segment-level (ocr, gold, doc_id) JSONL from BLN600 with a fixed
document-level train/dev/test split (no leakage between corrector & router)."""
import os, glob, re, json, random
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "data", "Ground Truth")
OCR = os.path.join(ROOT, "data", "OCR Text")
OUT = os.path.join(ROOT, "data")
SEG_WORDS = 12
random.seed(20260526)

def seg_pair(ocr, gold, target=SEG_WORDS):
    ow, gw = ocr.split(), gold.split()
    sm = SequenceMatcher(a=ow, b=gw, autojunk=False)
    segs, ob, gb = [], [], []
    def fl():
        if gb: segs.append((" ".join(ob), " ".join(gb)))
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ob.append(ow[i1 + k]); gb.append(gw[j1 + k])
                if len(gb) >= target: fl(); ob, gb = [], []
        else:
            ob.extend(ow[i1:i2]); gb.extend(gw[j1:j2])
    fl()
    return [(o, g) for o, g in segs if len(g.strip()) >= 8]

docs = []
for gp in sorted(glob.glob(os.path.join(GT, "*.txt"))):
    fid = os.path.splitext(os.path.basename(gp))[0]
    op = os.path.join(OCR, fid + ".txt")
    if not os.path.exists(op): continue
    gold = re.sub(r"\s+", " ", open(gp, errors="ignore").read().replace("\n", " ")).strip()
    ocr = re.sub(r"\s+", " ", open(op, errors="ignore").read().replace("\n", " ")).strip()
    if gold and ocr: docs.append((fid, ocr, gold))

random.shuffle(docs)
n = len(docs)
splits = {"train": docs[:int(.67*n)], "dev": docs[int(.67*n):int(.83*n)], "test": docs[int(.83*n):]}
for name, dl in splits.items():
    path = os.path.join(OUT, f"segments_bln600_{name}.jsonl")
    cnt = 0
    with open(path, "w") as f:
        for fid, ocr, gold in dl:
            for o, g in seg_pair(ocr, gold):
                f.write(json.dumps({"doc_id": fid, "ocr": o, "gold": g}) + "\n"); cnt += 1
    print(f"{name:5s}: {len(dl):3d} docs  {cnt:6d} segments  -> {os.path.basename(path)}")
