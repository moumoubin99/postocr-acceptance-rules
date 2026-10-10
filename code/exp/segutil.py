#!/usr/bin/env python3
"""OCR-only segmentation shared by the iter-2 builders (BLN600, ICDAR-clean, external).

Boundaries are chosen from the OCR string ONLY. Gold is attached afterwards through a
character map m (m[i] = gold index aligned with OCR char i, len(ocr)+1 entries, m[0]=0,
m[-1]=len(gold)), so gold pieces partition the gold document. Every segment keeps its exact
text plus the separator that follows it, so both documents reconstruct exactly:
    ocr_doc  == "".join(s["ocr"]  + s["ocr_sep"]  for s in segs)
    gold_doc == "".join(s["gold"] + s["gold_sep"] for s in segs)
A system hypothesis for the document is "".join(cand + ocr_sep) (the separator is not
editable by a segment-level corrector; its errors only show up in document-level CER)."""
from rapidfuzz.distance import Levenshtein

MAXCHARS = 250

def map_from_opcodes(ocr, gold):
    m = [0] * (len(ocr) + 1)
    for op in Levenshtein.opcodes(ocr, gold):
        if op.tag in ("equal", "replace"):
            for k in range(op.src_end - op.src_start): m[op.src_start + k] = op.dest_start + k
        elif op.tag == "delete":
            for k in range(op.src_end - op.src_start): m[op.src_start + k] = op.dest_start
    m[0] = 0; m[len(ocr)] = len(gold)          # leading/trailing gold insertions stay inside
    return m

def map_from_aligned(oa, ga, gap="@"):
    """ICDAR-style aligned strings of equal length; gap char marks missing positions."""
    assert len(oa) == len(ga)
    m = []; gi = 0
    for a, g in zip(oa, ga):
        if a != gap: m.append(gi)
        if g != gap: gi += 1
    m.append(gi)
    if m: m[0] = 0
    return m

def cuts(ocr, variant="c70"):
    """OCR-only cut list of (sep_start, sep_end). 'cNN': cut at a space once >= NN chars
    since segment start (forced cut without separator at MAXCHARS); 'wNN': every NN words."""
    out = []; start = 0; last = len(ocr) - 1   # never cut at the final char (no empty tail)
    if variant.startswith("c"):
        target = int(variant[1:])
        for i, ch in enumerate(ocr):
            if i < start: continue
            if i >= last: break
            if ch == " " and i - start >= target:
                out.append((i, i + 1)); start = i + 1
            elif i - start >= MAXCHARS:
                out.append((i, i)); start = i
    else:
        nw = int(variant[1:]); words = 0
        for i, ch in enumerate(ocr):
            if ch == " " and 0 < i < last and ocr[i-1] != " ":
                words += 1
                if words >= nw: out.append((i, i + 1)); words = 0
    return out

def segment(ocr, gold, m, variant="c70"):
    if not ocr: return [dict(ocr="", ocr_sep="", gold=gold, gold_sep="")]
    bounds = cuts(ocr, variant); segs = []; a = 0
    for (s, e) in bounds + [(len(ocr), len(ocr))]:
        segs.append(dict(ocr=ocr[a:s], ocr_sep=ocr[s:e], gold=gold[m[a]:m[s]], gold_sep=gold[m[s]:m[e]]))
        a = e
    assert "".join(x["ocr"] + x["ocr_sep"] for x in segs) == ocr
    assert "".join(x["gold"] + x["gold_sep"] for x in segs) == gold
    return segs

def reassemble(segs, field="cand"):
    return "".join(s[field] + s.get("ocr_sep", " ") for s in segs)
