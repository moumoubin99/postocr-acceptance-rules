"""Check document partition hashes and ICDAR normalized-OCR duplicate flags."""
from pathlib import Path
import gzip, json, collections
import numpy as np
P=Path(__file__).resolve().parent
maps=collections.defaultdict(set)
for line in gzip.open(P/'splits/document_split_map.jsonl.gz','rt'):
    o=json.loads(line);maps[(o['dataset'],o['split'])].add(o['doc_sha256'])
errors=[];count=0
for ds in {d for d,s in maps}:
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if maps[(ds,a)]&maps[(ds,b)]:errors.append('partition:'+ds+':'+a+':'+b)
languages=['EN','DE','FR','ES','NL']
sets={j:{bytes(x) for x in np.load(P/'splits/train_gold_hashes'/('icdarc_'+l+'.npz'),allow_pickle=False)['normalized_ocr_sha256']} for j,l in enumerate(languages)}
for f in (P/'numeric').glob('*/evaluation.json'):
    o=json.loads(f.read_text());ds=o['dataset']
    for sp in ['dev','test']:
        z=np.load(f.parent/(sp+'.npz'),allow_pickle=False)
        if {bytes(x).hex() for x in z['doc_sha256']}-maps[(ds,sp)]:errors.append('missing-map:'+f.parent.name+':'+sp)
        if ds.startswith('icdarc_'):
            dup=np.asarray([bytes(x) in sets[int(j)] for x,j in zip(z['normalized_ocr_sha256'],z['language_code'])],bool)
            if not np.array_equal(dup,z['dup_train'].astype(bool)):errors.append('dup_train:'+f.parent.name+':'+sp)
            count+=len(dup)
r={'pass':not errors,'icdar_duplicate_flags_checked':count,'errors':errors,'scope':'Document partitions and same-language normalized-OCR training-overlap flags. Not near-duplicate text clustering.'}
print(json.dumps(r,indent=2));raise SystemExit(bool(errors))
