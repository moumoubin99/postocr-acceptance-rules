"""Verify supplied hashes, schema and selected segment CER/harm from numeric evidence."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument('--report'); args=ap.parse_args()
errors=[]; count=0; tests=0
mf=ROOT/'SHA256SUMS.json'
if mf.exists():
    for name,want in json.loads(mf.read_text()).items():
        f=ROOT/name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=want:errors.append('hash:'+name)
        tests+=1
for cfg in sorted((ROOT/'numeric').iterdir()):
    if not cfg.is_dir():continue
    meta=json.loads((cfg/'evaluation.json').read_text())
    for sp in ['dev','test']:
        z=np.load(cfg/(sp+'.npz'),allow_pickle=False); n=meta['row_counts'][sp]
        count+=n
        for k in z.files:
            if z[k].dtype.kind not in 'biuf':errors.append('non-numeric:'+cfg.name+':'+k)
            if z[k].shape[0]!=n:errors.append('length:'+cfg.name+':'+k)
            tests+=2
        if (z['ocr_error']<0).any() or (z['candidate_error']<0).any() or (z['gold_length']<0).any():errors.append('negative:'+cfg.name)
        for k in ['doc_sha256','segment_sha256','ocr_sha256','gold_sha256','candidate_sha256']:
            if z[k].shape!=(n,32) or z[k].dtype!=np.uint8:errors.append('digest-schema:'+cfg.name+':'+k)
        if sp!='test':continue
        for name,pol in meta['policies'].items():
            for op,info in pol.items():
                u=z['accept_'+name+'_'+op].astype(bool)
                t=np.inf if info['threshold_is_positive_infinity'] else info['threshold']
                if not np.array_equal(u,z['attempted'].astype(bool)&(z['score_'+name]>=t)):errors.append('decision:'+cfg.name+':'+name+':'+op)
                cer=float(np.where(u,z['candidate_error'],z['ocr_error']).sum()/max(1,z['gold_length'].sum()))
                harm=int((u&(z['candidate_error']>z['ocr_error'])).sum())
                if abs(cer-info['cpu_recheck']['cer'])>1e-12 or harm!=info['cpu_recheck']['corrupt']:errors.append('metric:'+cfg.name+':'+name+':'+op)
                tests+=2
        original=json.loads((ROOT/'results/iter2'/('ledger_'+cfg.name+'.json')).read_text())
        for pol,u in [('never',np.zeros(n,bool)),('always',z['attempted'].astype(bool))]:
            cer=float(np.where(u,z['candidate_error'],z['ocr_error']).sum()/max(1,z['gold_length'].sum()))
            if abs(cer-original['policies'][pol]['cer'])>1e-12:errors.append('original-baseline:'+cfg.name+':'+pol)
            tests+=1
report={'pass':not errors,'numeric_rows':count,'checks':tests,'errors':errors,'scope':'File hashes, numeric-only schema, frozen candidate/OCR error aggregates, supplied CPU recheck decisions. Not original model inference or all secondary analyses.'}
if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2)); raise SystemExit(bool(errors))
