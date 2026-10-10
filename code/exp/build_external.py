#!/usr/bin/env python3
"""B5: external English test corpora from the ICDAR-2026 HIPE-OCRepair benchmark
(https://github.com/hipe-eval/HIPE-OCRepair-2026-data, data/v0.9), used ONLY for
evaluation (zero-shot transfer of BLN600-trained correctors/routers):
  ext_impresso_en : impresso-snippets EN (BL newspapers 1800-1959; CC BY-NC-SA 4.0;
                    new in 2026, manually corrected paragraphs)
  ext_overproof_en: OverProof (SMH + Chronicling America; licence not stated, research use)
Layout normalisation as in the HIPE scorer: '¬'+newline -> joined, newlines -> spaces.
OCR-only c70 segmentation + whole-unit character alignment (same as build_bln600_ocrseg).
Writes segments_{name}_{split}.jsonl, docs_{name}_{split}.jsonl (split in train/dev/test/all)
and ext_overlap_report.json (near-duplicate check of external docs vs BLN600 docs)."""
import os, json, glob, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIPE = os.environ.get("HIPE_DATA", os.path.join(ROOT, "raw", "HIPE-OCRepair-2026-data"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "data"))
SETS = {"ext_impresso_en": "data/v0.9/impresso-snippets/en/*_{split}_en.jsonl",
        "ext_overproof_en": "data/v0.9/overproof/en/*_{split}_en.jsonl"}

def layout(s): return " ".join(re.sub(r"¬\s*\n", "", s).replace("\n", " ").split())

import sys; sys.path.insert(0, os.path.join(ROOT, "exp"))
from segutil import segment, map_from_opcodes

def shingles(s, k=8):
    s = s.lower(); return {s[i:i+k] for i in range(max(1, len(s) - k + 1))}

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

bln = []
for sp in ("train", "dev", "test"):
    p = os.path.join(OUT, f"docs_bln600_{sp}.jsonl")
    if os.path.exists(p): bln += [json.loads(l) for l in open(p)]
bln_sh = [(d["doc_id"], shingles(d["gold"])) for d in bln]
report = {}
for name, pat in SETS.items():
    allseg, alldoc = [], []; rep = {}
    for split in ("train", "dev", "test"):
        fs = glob.glob(os.path.join(HIPE, pat.format(split=split)))
        assert len(fs) == 1, (name, split, fs)
        segs, docs = [], []
        for ln, l in enumerate(open(fs[0], encoding="utf-8")):
            r = json.loads(l)
            orig = r["document_metadata"]["document_id"]
            did = f"{split}:{ln}:{orig}"   # OverProof reuses ids for different articles
            o, g = layout(r["ocr_hypothesis"]["transcription_unit"]), layout(r["ground_truth"]["transcription_unit"])
            if not o or not g: continue
            docs.append(dict(doc_id=did, orig_id=orig, split=split, ocr=o, gold=g, date=r["document_metadata"].get("date")))
            for k, sg in enumerate(segment(o, g, map_from_opcodes(o, g), "c70")):
                segs.append(dict(doc_id=did, seg_idx=k, orig_id=orig, split=split, **sg))
        dump(os.path.join(OUT, f"segments_{name}_{split}.jsonl"), segs)
        dump(os.path.join(OUT, f"docs_{name}_{split}.jsonl"), docs)
        allseg += segs; alldoc += docs
        rep[split] = dict(docs=len(docs), segs=len(segs))
    dump(os.path.join(OUT, f"segments_{name}_all.jsonl"), allseg)
    dump(os.path.join(OUT, f"docs_{name}_all.jsonl"), alldoc)
    hits = []
    for d in alldoc:
        sh = shingles(d["gold"])
        for bid, bs in bln_sh:
            j = len(sh & bs) / max(1, min(len(sh), len(bs)))
            if j >= 0.3: hits.append((d["doc_id"], bid, round(j, 3)))
    rep["near_dup_vs_bln600(containment>=0.3)"] = hits
    report[name] = rep
    print(name, json.dumps(rep)[:400])
json.dump(report, open(os.path.join(OUT, "ext_overlap_report.json"), "w"), indent=1)
