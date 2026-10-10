#!/usr/bin/env python3
"""Parse HIPE hipe2020-en TSV into sentence segments with the reconstructed OCR text
and character-offset gold entity spans (NE-COARSE-LIT). Output segments_hipe_en_{split}.jsonl
with {doc_id, text, entities:[[start,end,type],...]}. Char offsets survive re-tokenization,
so we can score NER F1 after correction changes token boundaries."""
import os, json, glob
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def parse(path):
    segs=[]; doc=None
    toks=[]  # (token, ne_coarse, nospace_after)
    def flush():
        if not toks: return
        text=""; spans=[]; cur=None
        for i,(t,ne,nsa) in enumerate(toks):
            start=len(text); text+=t; end=len(text)
            if ne.startswith("B-"):
                if cur: spans.append(cur)
                cur=[start,end,ne[2:]]
            elif ne.startswith("I-") and cur and ne[2:]==cur[2]:
                cur[1]=end
            else:
                if cur: spans.append(cur); cur=None
            if not nsa and i<len(toks)-1: text+=" "
        if cur: spans.append(cur)
        if text.strip(): segs.append(dict(doc_id=doc, text=text, ocr=text, gold="", entities=spans))
    for line in open(path,encoding="utf-8",errors="ignore"):
        if line.startswith("# hipe2022:document_id"):
            flush(); toks=[]; doc=line.split("=",1)[1].strip(); continue
        if line.startswith("#") or line.startswith("TOKEN\t"): continue
        line=line.rstrip("\n")
        if not line.strip(): continue
        c=line.split("\t")
        if len(c)<2: continue
        tok, ne = c[0], c[1]; misc = c[-1] if len(c)>=10 else ""
        nsa = "NoSpaceAfter" in misc
        toks.append((tok, ne, nsa))
        if "EndOfSentence" in misc or "EndOfLine" in misc:
            flush(); toks=[]
    flush()
    return segs

for f in sorted(glob.glob(os.path.join(DATA,"hipe","hipe2020_en_*.tsv"))):
    split=f.split("_")[-1].replace(".tsv","")
    segs=parse(f)
    out=os.path.join(DATA,f"segments_hipe_en_{split}.jsonl")
    with open(out,"w") as fh:
        for s in segs: fh.write(json.dumps(s)+"\n")
    ne=sum(len(s["entities"]) for s in segs)
    print(f"{split}: {len(segs)} segments, {ne} entities -> segments_hipe_en_{split}.jsonl")
    for s in segs[:2]:
        print("  text:", s["text"][:90])
        print("  ents:", [(s['text'][a:b], t) for a,b,t in s['entities'][:4]])
