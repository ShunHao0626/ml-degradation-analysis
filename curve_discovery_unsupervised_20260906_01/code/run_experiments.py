#!/usr/bin/env python3
"""Unsupervised, unsmoothed PvkSOM extension. No morphology labels enter training.
Run with /usr/bin/python3 code/run_experiments.py from the result directory.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/curve_discovery_mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json, hashlib, itertools, time, sys, importlib.metadata
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score
from minisom import MiniSom
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'lab-v2/som_references/accepted/samples_test'
SEEDS = [11, 29, 47, 71, 101]
KS = list(range(1,17))
NORMS = ['minmax', 'maxabs', 'zscore']

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def load_data():
    curves, rows, excluded = [], [], []
    for f in sorted(SOURCE.rglob('*.csv')):
        df=pd.read_csv(f); arr=df[['x','y']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        valid=np.isfinite(arr).all(axis=1); clean=arr[valid]
        if len(clean)<2 or np.ptp(clean[:,0])<=0:
            excluded.append({'path':str(f),'reason':'fewer than two finite distinct x values'});continue
        clean=clean[np.argsort(clean[:,0],kind='stable')];x,y=clean.T
        meta_path=f.parent.parent/'validation_result.json'
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
        axis=meta.get('axis',{});xm=axis.get('x',{});ym=axis.get('y',{})
        name=str(xm.get('name') or '');unit=str(xm.get('unit') or '')
        unit_clean=unit.lower().replace('(','').replace(')','').replace(' ','')
        factor={'h':1,'hr':1,'hour':1,'hours':1,'d':24,'day':24,'days':24,'min':1/60}.get(unit_clean)
        if factor is None and any(z in name.lower() for z in ['(h)','hours','heat hour']):factor=1
        status='time_verified' if factor is not None else ('cycles' if 'cycl' in name.lower() else 'unknown')
        u=(x-x[0])/(x[-1]-x[0]);span=float(np.ptp(y));scale=float(np.max(np.abs(y)))
        cid=f'C{len(curves)+1:03d}'
        row={'curve_id':cid,'source_path':str(f.resolve()),'relative_path':str(f.relative_to(SOURCE)),
             'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'figure':f.parent.parent.name,
             'x_name':name,'x_unit':unit,'y_name':ym.get('name'),'y_unit':ym.get('unit'),
             'axis_status':status,'hours_factor':factor,'n_points':len(x),'invalid_points':int((~valid).sum()),
             'duplicate_x_points':int(len(x)-len(np.unique(x))),'x_min':float(x[0]),'x_max':float(x[-1]),
             'duration_h':float((x[-1]-x[0])*factor) if factor is not None else None,
             'y_min':float(y.min()),'y_max':float(y.max()),'relative_y_range':span/max(scale,1e-15),
             'constant_y':bool(span==0)}
        rows.append(row);curves.append({'id':cid,'x':x.tolist(),'y':y.tolist(),'u':u.tolist(),'meta':row})
    dump(ROOT/'data/raw_curves.json',curves)
    pd.DataFrame(rows).to_csv(ROOT/'data/curve_inventory.csv',index=False)
    dump(ROOT/'data/excluded.json',excluded)
    return curves,pd.DataFrame(rows)

def values(c,norm):
    y=np.asarray(c['y']);u=np.asarray(c['u'])
    if norm=='minmax':return (y-y.min())/np.ptp(y) if np.ptp(y)>0 else np.full_like(y,.5)
    if norm=='maxabs':return y/max(np.max(np.abs(y)),1e-15)
    # Exact time-weighted mean/variance of the unsmoothed piecewise-linear curve.
    dx=np.diff(u);mean=np.sum(dx*(y[:-1]+y[1:])/2)
    second=np.sum(dx*(y[:-1]**2+y[:-1]*y[1:]+y[1:]**2)/3)
    std=np.sqrt(max(0,second-mean**2))
    return (y-mean)/std if std>1e-12 else np.zeros_like(y)

def representation(curves,norm):
    """Isometry of exact piecewise-linear L2 geometry; no PCA variance truncation.
    Two-point Gauss quadrature exactly integrates the product of linear segments.
    Repeated x retain their ordered vertical jumps; jumps have zero L2 measure.
    """
    knots=np.unique(np.concatenate([np.asarray(c['u']) for c in curves]))
    mid=(knots[1:]+knots[:-1])/2;half=np.diff(knots)/2
    q=np.ravel(np.column_stack([mid-half/np.sqrt(3),mid+half/np.sqrt(3)]))
    w=np.repeat(half,2)
    F=np.array([np.interp(q,c['u'],values(c,norm)) for c in curves])*np.sqrt(w)
    gram=F@F.T
    e,v=eigh(gram);keep=e>max(e.max()*1e-13,1e-15)
    X=v[:,keep]*np.sqrt(e[keep])
    err=float(np.max(np.abs(X@X.T-gram)))
    np.savez_compressed(ROOT/f'data/representation_{norm}.npz',X=X,knots=knots,
                        eigenvalues=e,gram_max_error=err)
    return X,{'rank':int(keep.sum()),'union_knots':len(knots),'gram_max_error':err}

def grid(k,line=False):
    a=1 if line else max(d for d in range(1,int(np.sqrt(k))+1) if k%d==0)
    return a,k//a

def train(X,k,sigma,lr,seed,iterations=5000,line=False):
    a,b=grid(k,line)
    som=MiniSom(a,b,X.shape[1],sigma=sigma,learning_rate=lr,random_seed=seed)
    som.random_weights_init(X)
    som.train(X,iterations,random_order=True)
    W=som.get_weights().reshape(k,-1);D=cdist(X,W);labels=D.argmin(axis=1)
    qe=float(D.min(axis=1).mean());occ=len(np.unique(labels))
    sil=float(silhouette_score(X,labels)) if 1<occ<len(X) else None
    # MiniSom's rectangular adjacency includes diagonals (threshold sqrt(2)).
    coords=np.array(list(itertools.product(range(a),range(b))))
    if k>1:
        best=np.argsort(D,axis=1)[:,:2]
        te=float(np.mean(np.linalg.norm(coords[best[:,0]]-coords[best[:,1]],axis=1)>1.42))
    else:te=None
    return {'k':k,'sigma':sigma,'lr':lr,'seed':seed,'iterations':iterations,'grid':f'{a}x{b}',
            'qe':qe,'occupied':occ,'silhouette':sil,'topographic_error':te},labels,W

def elbow(ks,qe):
    """Unsmoothed normalized chord-distance operationalization, NOT full Kneedle."""
    ks=np.asarray(ks);qe=np.asarray(qe)
    x=(ks-ks[0])/(ks[-1]-ks[0]);y=(qe-qe[-1])/max(qe[0]-qe[-1],1e-15)
    gap=1-x-y;i=int(np.argmax(gap))
    return int(ks[i]),gap.tolist()

def piecewise_elbow(ks,qe):
    x=np.asarray(ks,float);y=np.asarray(qe,float);scores=[]
    for j in range(1,len(x)-1):
        b=x[j];A=np.column_stack([np.ones(len(x)),x,np.maximum(0,x-b)])
        fit=A@np.linalg.lstsq(A,y,rcond=None)[0];scores.append((float(np.sum((fit-y)**2)),int(b)))
    return min(scores)[1]

def run():
    t=time.time();curves,inventory=load_data()
    conf={'source':str(SOURCE),'smoothing':False,'truncate':False,'extrapolate':False,
          'target_labels_used':False,'norms':NORMS,'ks':KS,'screen_seeds':SEEDS[:3],
          'final_seeds':SEEDS,'sigma':[.15,.3,.5,1.0],'lr':[.03,.1,.3],
          'screen_iterations':5000,'final_iterations':50000,
          'primary_representation':'minmax','primary_reason':'shape independent of vertical offset and amplitude',
          'parameter_rule':'minimum mean across k of seed-median QE / k=1 seed-median QE',
          'k_rule':'maximum unsmoothed normalized chord gap for k=1..16; report other ranges and shape-overlap review',
          'created_before_training':True}
    dump(ROOT/'experiments/config.json',conf)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','minisom','matplotlib','plotly','threadpoolctl']}
    dump(ROOT/'experiments/environment.json',{'python':sys.version,'executable':sys.executable,'packages':versions})
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    info={};Xs={}
    for norm in NORMS:
        Xs[norm],info[norm]=representation(curves,norm)
    dump(ROOT/'data/representation_checks.json',info)
    print('DATA',len(curves),'points',inventory.n_points.sum(),'axes',inventory.axis_status.value_counts().to_dict(),flush=True)
    print('REPRESENTATIONS',info,flush=True)
    allrows=[];params={}
    for norm in NORMS:
        X=Xs[norm];rows=[]
        for sigma,lr in itertools.product(conf['sigma'],conf['lr']):
            for k,seed in itertools.product(KS,SEEDS[:3]):
                r,_,_=train(X,k,sigma,lr,seed);r['norm']=norm;rows.append(r)
            print('SCREEN',norm,sigma,lr,'seconds',round(time.time()-t,1),flush=True)
        df=pd.DataFrame(rows);allrows+=rows
        table=df.groupby(['sigma','lr','k']).qe.median().unstack('k')
        score=table.div(table[1],axis=0).mean(axis=1)
        best=score.idxmin();params[norm]={'sigma':float(best[0]),'lr':float(best[1])}
        table['normalized_area_score']=score;table.to_csv(ROOT/f'experiments/parameter_ranking_{norm}.csv')
        pd.DataFrame(allrows).to_csv(ROOT/'experiments/screening.csv',index=False)
    dump(ROOT/'experiments/selected_parameters.json',params)
    finalrows=[];summaries={}
    for norm in NORMS:
        X=Xs[norm];byk={};ws={}
        for k in KS:
            for seed in SEEDS:
                r,l,W=train(X,k,**params[norm],seed=seed,iterations=50000)
                r['norm']=norm;finalrows.append(r);byk[k,seed]=l;ws[k,seed]=W
            print('FINAL',norm,k,flush=True)
        df=pd.DataFrame([r for r in finalrows if r['norm']==norm]);q=df.groupby('k').qe.median()
        k,gap=elbow(q.index,q.values)
        best=df[df.k==k].sort_values('qe').iloc[0];seed=int(best.seed)
        lab=byk[k,seed];W=ws[k,seed]
        stability=[adjusted_rand_score(byk[k,a],byk[k,b]) for a,b in itertools.combinations(SEEDS,2)]
        ranges={str(end):elbow(q.index[q.index<=end],q[q.index<=end].values)[0] for end in [8,10,12,16]}
        seed_elbows={str(s):elbow(KS,df[df.seed==s].sort_values('k').qe.to_numpy())[0] for s in SEEDS}
        summary={'selected_k':k,'selected_seed':seed,'params':params[norm],
                 'qe':float(best.qe),'qe_over_k1':float(best.qe/q.loc[1]),
                 'silhouette':float(best.silhouette),'seed_ari_mean':float(np.mean(stability)),
                 'seed_ari_min':float(np.min(stability)),'elbow_by_range':ranges,
                 'elbow_by_seed':seed_elbows,'piecewise_elbow':piecewise_elbow(KS,q.values),
                 'occupied':int(best.occupied),'cluster_sizes':np.bincount(lab,minlength=k).tolist(),
                 'chord_gap':gap}
        summaries[norm]=summary
        # Every k is preserved for adjacent-cluster and overlap inspection.
        bestlabels={};bestweights={}
        for kk in KS:
            ss=int(df[df.k==kk].sort_values('qe').iloc[0].seed)
            bestlabels[f'labels_k{kk}']=byk[kk,ss];bestweights[f'weights_k{kk}']=ws[kk,ss]
        np.savez_compressed(ROOT/f'experiments/models_{norm}.npz',labels=lab,weights=W,
                            **bestlabels,**bestweights,
                            **{f'seed{s}_labels':byk[k,s] for s in SEEDS})
        print('SELECTED',norm,summary,flush=True)
    pd.DataFrame(finalrows).to_csv(ROOT/'experiments/final_sweep.csv',index=False)
    dump(ROOT/'experiments/selection.json',summaries)
    primary=summaries['minmax'];X=Xs['minmax'];base=np.load(ROOT/'experiments/models_minmax.npz')['labels']
    # Group resampling respects common source figures. No outcome labels.
    groups=inventory.figure.to_numpy();ug=np.unique(groups);boot=[];bootlabels=[]
    for b in range(20):
        rng=np.random.default_rng(1000+b);draw=rng.choice(ug,size=len(ug),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==g) for g in draw]);q=[]
        for k in KS:
            r,l,W=train(X[idx],k,**params['minmax'],seed=1000+b,iterations=10000)
            q.append(r['qe'])
            if k==primary['selected_k']:
                out=cdist(X,W).argmin(axis=1);bootlabels.append(out)
                ari=float(adjusted_rand_score(base,out))
        bk,_=elbow(KS,q);boot.append({'replicate':b,'elbow_k':bk,'ari_at_selected_k':ari,'qe':q})
        print('GROUP_BOOTSTRAP',b,bk,ari,flush=True)
    dump(ROOT/'experiments/group_bootstrap.json',boot)
    np.save(ROOT/'experiments/group_bootstrap_labels.npy',np.array(bootlabels))
    # Time-only sensitivity keeps the all-curve result intact.
    idx=np.flatnonzero(inventory.axis_status.to_numpy()=='time_verified');rtime=[];models={}
    for k,seed in itertools.product(KS,SEEDS[:3]):
        r,l,W=train(X[idx],k,**params['minmax'],seed=seed,iterations=30000)
        rtime.append(r);models[k,seed]=(l,W)
    df=pd.DataFrame(rtime);df.to_csv(ROOT/'experiments/time_only_sweep.csv',index=False)
    q=df.groupby('k').qe.median();k,_=elbow(KS,q.values);seed=int(df[df.k==k].sort_values('qe').iloc[0].seed)
    l,W=models[k,seed]
    np.savez_compressed(ROOT/'experiments/time_only_model.npz',indices=idx,labels=l,weights=W)
    dump(ROOT/'experiments/time_only_selection.json',{'n':len(idx),'selected_k':k,'seed':seed,
        'ari_vs_all_curve_primary':float(adjusted_rand_score(base[idx],l)),
        'duration_h_range':[float(inventory.duration_h.min()),float(inventory.duration_h.max())]})
    print('DONE',round(time.time()-t,1),'seconds',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
