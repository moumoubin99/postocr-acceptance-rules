#!/usr/bin/env python3
"""Build ICDAR-2019 segments for a given LANG (env), capped to MAXFILES for
comparability/speed. Writes segments_icdar_{LANG}_{train,dev,test}.jsonl.
Uses char-aligned [OCR_aligned]/[ GS_aligned] lines (@ = gap)."""
import os, json, random, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
BASE = "https://huggingface.co/datasets/FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction/resolve/main/"
LANG = os.environ.get("LANG_CODE", "DE")
MAXFILES = int(os.environ.get("MAXFILES", 350))
random.seed(20260526)

def lang_files():
    d = json.load(urllib.request.urlopen(
        "https://huggingface.co/api/datasets/FrancophonIA/ICDAR_2019_Competition_Post-OCR_Text_Correction", timeout=30))
    fs = [s["rfilename"] for s in d["siblings"] if f"/{LANG}/" in s["rfilename"] and s["rfilename"].endswith(".txt")]
    random.shuffle(fs)
    return fs[:MAXFILES]

def fetch(f):
    try: return f, urllib.request.urlopen(BASE + urllib.parse.quote(f), timeout=30).read().decode("utf-8","ignore")
    except Exception: return f, None

def parse(txt):
    oa=ga=None
    for line in txt.splitlines():
        if line.startswith("[OCR_aligned]"): oa=line[len("[OCR_aligned]"):].strip("\n")
        elif line.startswith("[ GS_aligned]"): ga=line[len("[ GS_aligned]"):].strip("\n")
    return oa,ga

def window(oa,ga,target=70):
    segs,start,glen=[],0,0; n=min(len(oa),len(ga))
    for i in range(n):
        if ga[i]!="@": glen+=1
        if ga[i]==" " and glen>=target:
            o=oa[start:i].replace("@","").strip(); g=ga[start:i].replace("@","").strip()
            if len(g)>=8: segs.append((o,g))
            start=i; glen=0
    o=oa[start:n].replace("@","").strip(); g=ga[start:n].replace("@","").strip()
    if len(g)>=8: segs.append((o,g))
    return segs

files=lang_files(); print(f"LANG={LANG}: {len(files)} files (cap {MAXFILES}) — downloading...")
docs=[]
with ThreadPoolExecutor(max_workers=16) as ex:
    for f,txt in ex.map(fetch, files):
        if not txt: continue
        oa,ga=parse(txt)
        if oa and ga: docs.append((f,oa,ga))
random.shuffle(docs); n=len(docs)
splits={"train":docs[:int(.67*n)],"dev":docs[int(.67*n):int(.83*n)],"test":docs[int(.83*n):]}
for name,dl in splits.items():
    path=os.path.join(OUT,f"segments_icdar_{LANG}_{name}.jsonl"); cnt=0
    with open(path,"w") as fh:
        for fid,oa,ga in dl:
            for o,g in window(oa,ga): fh.write(json.dumps({"doc_id":fid,"ocr":o,"gold":g})+"\n"); cnt+=1
    print(f"  {name:5s}: {len(dl):3d} docs {cnt:6d} segs -> segments_icdar_{LANG}_{name}.jsonl")
