"""Optional CPU re-fit of the main segment router using released precomputed features.
This does not load source text, corrector weights, or call a service.
"""
from pathlib import Path
import os, json
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parent
def fit(X,y,T):
    if len(y)<10 or len(set(y))<2:return np.full(len(T),float(y.mean()) if len(y) else 0.)
    m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.08,max_iter=300,l2_regularization=1.,random_state=20260526)
    m.fit(X,y);return m.predict_proba(T)[:,1]
def features(z,full):
    X=np.c_[z['features_text'],z['features_alignment']]
    return np.c_[X,z['features_confidence']] if full=='router_conf' else X
reports=[]
for cfg in sorted((ROOT/'numeric').iterdir()):
    if not cfg.is_dir():continue
    m=json.loads((cfg/'evaluation.json').read_text()); dv=np.load(cfg/'dev.npz',allow_pickle=False); te=np.load(cfg/'test.npz',allow_pickle=False)
    X,T=features(dv,m['full_router']),features(te,m['full_router']); y=(dv['candidate_error']<dv['ocr_error']).astype(int);mask=dv['attempted'].astype(bool);g=dv['doc_group_index']
    pt=fit(X[mask],y[mask],T);pd=np.zeros(len(X));ng=len(set(g))
    if ng<2:pd=fit(X[mask],y[mask],X)
    else:
        for tr,va in GroupKFold(min(5,ng)).split(X,y,g):
            tr=tr[mask[tr]];pd[va]=fit(X[tr],y[tr],X[va])
    out={'config':cfg.name,'max_dev_score_difference':float(np.max(np.abs(pd-dv['score_router_cpu_recheck']))),'max_test_score_difference':float(np.max(np.abs(pt-te['score_router_cpu_recheck'])))}
    reports.append(out);print(json.dumps(out),flush=True)
(ROOT/'generated').mkdir(exist_ok=True)
(ROOT/'generated/router_refit_report.json').write_text(json.dumps(reports,indent=2)+'\n')
