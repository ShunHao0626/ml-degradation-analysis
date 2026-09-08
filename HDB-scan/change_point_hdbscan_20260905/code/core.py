"""Label-free raw-curve change point and density clustering primitives."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_key] = '1'
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt
import hdbscan
from scipy.stats import theilslopes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, mutual_info_score)
warnings.filterwarnings('ignore', category=FutureWarning)
SMOOTHING = False
SEED = 20260905
MIN_POINTS = 12
EPS = 1e-12

def dump(obj, path):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,Path):return str(x)
        return x
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False))

def audit(source, out):
    import shutil
    rows, curves = [], {}
    for k, file in enumerate(sorted(Path(source).rglob('*.csv'))):
        sid = f'S{k+1:04d}'
        frame = pd.read_csv(file)
        axis = json.loads((file.parent.parent/'validation_result.json').read_text()).get('axis', {})
        xx = str(axis.get('x', {})).lower()
        unit = 'cycles' if 'cycl' in xx else 'days' if "'(d)'" in xx else 'hours' if any(z in xx for z in ['hour', "'(h)'", "'hr'", "'(hr)'", "'h'", 'time (h)', '( h']) else 'unknown'
        arr = frame[['x','y']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        valid = np.isfinite(arr).all(axis=1)
        clean = pd.DataFrame(arr[valid], columns=['x','y'])
        exact_dups = int(clean.duplicated().sum())
        conflicting = int((clean.groupby('x').y.nunique()>1).sum())
        duplicate_timestamps = int(clean.x.duplicated().sum())
        clean = clean.drop_duplicates().groupby('x', as_index=False, sort=True).y.median()
        x, y = clean.x.to_numpy(), clean.y.to_numpy()
        factor = 24. if unit == 'days' else 1.
        t = (x-x[0])*factor
        flags = []
        if np.any(y<0): flags.append('negative_output')
        yy = str(axis.get('y',{})).lower()
        if 'norm' not in yy and '%' in yy and np.any(y>100): flags.append('absolute_efficiency_gt100')
        if t[-1] > 1e6: flags.append('extreme_axis_span_review')
        if x.min()<0: flags.append('negative_start_retained_shifted')
        row = dict(sample_id=sid, source_path=str(file.resolve()), figure_id=file.parent.parent.name,
            x_axis=json.dumps(axis.get('x',{}),ensure_ascii=False), y_axis=json.dumps(axis.get('y',{}),ensure_ascii=False),
            unit=unit, hours_factor=factor if unit in ['days','hours'] else np.nan,
            original_n=len(arr), clean_n=len(t), start=float(x[0]), end=float(x[-1]), duration_native=float(x[-1]-x[0]),
            duration_analysis=float(t[-1]), median_interval=float(np.median(np.diff(t))),
            missing_fraction=float(np.isnan(arr).any(axis=1).mean()), non_finite_values=int((~np.isfinite(arr)).sum()),
            parse_errors=int((pd.DataFrame(arr).isna().to_numpy() & ~frame[['x','y']].isna().to_numpy()).sum()),
            exact_duplicate_records=exact_dups, duplicate_timestamps=duplicate_timestamps, conflicting_timestamps=conflicting,
            negative_output_count=int((y<0).sum()), minimum_output=float(y.min()), maximum_output=float(y.max()),
            qc_flags=';'.join(flags), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        rows.append(row)
        curves[sid] = dict(t=t, y=y, x=x, unit=unit, figure=row['figure_id'], flags=flags)
        rawdest=out/'raw_input'/file.relative_to(source); rawdest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(file,rawdest)
        pd.DataFrame(dict(x_original=x, elapsed=t, y=y)).to_csv(out/'cleaned_input'/f'{sid}.csv',index=False)
    qc=pd.DataFrame(rows);qc.to_csv(out/'qc_summary.csv',index=False)
    for f in Path(source).rglob('validation_result.json'):
        dst=out/'raw_input'/f.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    return curves,qc

def window_curves(curves, window, cohort='hours', resample=False):
    selected={}
    for sid,c in curves.items():
        if cohort=='hours' and c['unit'] not in ['hours','days']: continue
        if cohort=='cycles' and c['unit']!='cycles': continue
        t,y=c['t'],c['y']
        if window is not None:
            if t[-1] < window: continue
            keep=t<=window;t,y=t[keep],y[keep]
        if len(t)<MIN_POINTS or t[-1]<=0:continue
        selected[sid]={**c,'t':t.copy(),'y':y.copy(),'window':window}
    if resample and selected:
        # Same grid across curves; spacing at least median native interval.
        relative=window is None
        dt=np.median([np.median(np.diff(c['t']/(c['t'][-1] if relative else 1))) for c in selected.values()])
        end=1. if relative else float(window)
        grid=np.arange(0,end+EPS,dt)
        for sid,c in list(selected.items()):
            tt=c['t']/(c['t'][-1] if relative else 1)
            gg=grid[grid<=tt[-1]+EPS]
            if len(gg)<MIN_POINTS: del selected[sid];continue
            selected[sid]={**c,'t':gg*c['t'][-1] if relative else gg,'y':np.interp(gg,tt,c['y']), 'grid_dt':float(dt)}
    return selected

def normalize(y, mode):
    if mode=='initial': den=y[0];offset=0.
    elif mode=='max': den=np.max(y);offset=0.
    elif mode=='robust': den=np.quantile(y,.95)-np.quantile(y,.05);offset=np.median(y)
    else:raise ValueError(mode)
    if abs(den)<EPS: return np.full_like(y,np.nan)
    return (y-offset)/den

def cp_detect(t,y,cfg):
    n=len(t);u=(t-t[0])/(t[-1]-t[0])
    scale=np.quantile(y,.95)-np.quantile(y,.05)
    z=(y-np.median(y))/max(scale,EPS)
    if np.ptp(y)<EPS:return [n]
    minimum=max(3,int(np.ceil(n*cfg['min_fraction'])))
    if n<2*minimum:return [n]
    signal=np.column_stack([z,np.ones(n),u]) if cfg['cost']=='linear' else z.reshape(-1,1)
    common=dict(model=cfg['cost'],min_size=minimum,jump=1)
    if cfg['algorithm']=='pelt':model=rpt.Pelt(**common)
    elif cfg['algorithm']=='binseg':model=rpt.Binseg(**common)
    else:model=rpt.Window(width=min(2*max(minimum,int(np.ceil(n*.1))),2*((n-1)//2)),**common)
    model.fit(signal)
    # BIC-like penalty in the cost's own units, scaled to unsplit dispersion.
    baseline=max(float(model.cost.error(0,n))/n,EPS)
    penalty=cfg['penalty']*baseline*np.log(n)
    return model.predict(pen=penalty)

def cp_times(t,b):
    return np.array([(t[i-1]+t[i])/2 for i in b[:-1]])

def cp_match(a,b,tolerance):
    if len(a)==len(b)==0:return 1.,np.array([],int),np.array([],float)
    hits=np.zeros(len(a),int);errors=np.full(len(a),np.nan)
    if len(a) and len(b):
        i,j=linear_sum_assignment(np.abs(a[:,None]-b[None,:]))
        for ii,jj in zip(i,j):
            err=abs(a[ii]-b[jj])
            if err<=tolerance:hits[ii]=1;errors[ii]=err
    return 2*hits.sum()/max(len(a)+len(b),1),hits,errors

def linear_slope(t,y):
    if len(t)<2 or np.ptp(t)<EPS:return np.nan
    return float(np.dot(t-t.mean(),y-y.mean())/np.sum((t-t.mean())**2))

def cp_quality(t,y,b):
    # Common Gaussian piecewise-affine BIC allows comparisons between costs.
    z=(y-np.median(y))/max(np.quantile(y,.95)-np.quantile(y,.05),EPS)
    u=t/t[-1];rss=0.;start=0
    for end in b:
        xx=u[start:end];yy=z[start:end];s=linear_slope(xx,yy)
        rss+=np.sum((yy-(yy.mean()+s*(xx-xx.mean())))**2);start=end
    return len(y)*np.log(max(rss/len(y),1e-10))+(3*len(b)-1)*np.log(len(y))

def cp_screen_one(cfg,curve_items):
    rows=[]
    for sid,c in curve_items:
        t,y=c['t'],c['y'];b=cp_detect(t,y,cfg);base=cp_times(t,b)
        scores=[]
        for frac in [.05,.1]:
            rng=np.random.default_rng(SEED+int(sid[1:]));keep=np.ones(len(t),bool)
            keep[rng.choice(np.arange(1,len(t)-1),max(1,int(round(frac*len(t)))),replace=False)]=False
            bp=cp_detect(t[keep],y[keep],cfg)
            scores.append(cp_match(base,cp_times(t[keep],bp),max(.03*t[-1],np.median(np.diff(t))))[0])
        rows.append(dict(config_id=cfg['id'],sample_id=sid,bic=cp_quality(t,y,b),segments=len(b),cp_f1=np.mean(scores)))
    return rows

def stable_segments(t,y,cfg,repeats=10,seed=SEED):
    b=cp_detect(t,y,cfg);times=cp_times(t,b)
    hits=np.zeros(len(times));errs=[[] for _ in times];runs=[]
    tol=max(.03*(t[-1]-t[0]),np.median(np.diff(t)))
    rng=np.random.default_rng(seed)
    for frac in [.05,.1]:
        for rep in range(repeats):
            keep=np.ones(len(t),bool)
            keep[rng.choice(np.arange(1,len(t)-1),max(1,int(round(frac*len(t)))),replace=False)]=False
            bb=cp_detect(t[keep],y[keep],cfg);tt=cp_times(t[keep],bb)
            f1,hh,ee=cp_match(times,tt,tol);hits+=hh
            for i,e in enumerate(ee):
                if np.isfinite(e):errs[i].append(float(e))
            runs.append(dict(drop_fraction=frac,repeat=rep,cp_f1=f1,cp_count=len(tt),cp_times=json.dumps(tt.tolist())))
    support=hits/(2*repeats)
    # Segment reliability is the lower support of its two boundaries; endpoints fixed.
    bs=np.r_[1.,support,1.]
    ss=np.minimum(bs[:-1],bs[1:])
    chosen=sorted(sorted(range(len(b)),key=lambda j:(-ss[j],j))[:4])
    info=[dict(cp_index=j,cp_time=float(v),support=float(support[j]),location_error_median=float(np.median(errs[j])) if errs[j] else np.nan,tolerance=tol) for j,v in enumerate(times)]
    return b,chosen,ss,info,runs

def features(t,y,b,chosen,ss):
    # Time-based predictors use elapsed fraction, eliminating observation duration as a label cue.
    u=(t-t[0])/(t[-1]-t[0]);dy=np.diff(y);sl=dy/np.diff(u)
    imax=int(np.argmax(y));imin=int(np.argmin(y));amp=max(np.ptp(y),EPS)
    f=dict(initial=y[0],final=y[-1],global_max=y.max(),global_min=y.min(),peak_time=u[imax],valley_time=u[imin],
        net_change=y[-1]-y[0],total_variation=np.abs(dy).sum(),area=np.trapz(y,u),positive_fraction=(dy>0).mean(),negative_fraction=(dy<0).mean(),
        max_positive_slope=max(0.,sl.max()),max_negative_slope=min(0.,sl.min()),median_positive_slope=np.median(sl[sl>0]) if (sl>0).any() else 0.,
        median_negative_slope=np.median(sl[sl<0]) if (sl<0).any() else 0.,segment_count=len(b),
        max_to_next_min=y[imax]-y[imax:].min(),min_to_next_max=y[imin:].max()-y[imin],
        recovery_amplitude=np.max(y-np.minimum.accumulate(y)),recovery_ratio=np.max(y-np.minimum.accumulate(y))/amp,
        pre_peak_slope=linear_slope(u[:imax+1],y[:imax+1]),immediate_post_peak_slope=sl[imax] if imax<len(sl) else np.nan,
        late_time_slope=linear_slope(u[-max(3,int(np.ceil(len(y)*.2))):],y[-max(3,int(np.ceil(len(y)*.2))):]),
        pre_valley_slope=linear_slope(u[:imin+1],y[:imin+1]),post_valley_slope=linear_slope(u[imin:],y[imin:]))
    segrows=[];start=0
    for j,end in enumerate(b):
        xx,yy=u[start:end],y[start:end]
        slope=linear_slope(xx,yy);rob=float(theilslopes(yy,xx)[0]) if len(xx)>1 else np.nan
        segrows.append(dict(segment_index=j,start_time=float(t[start]),end_time=float(t[end-1]),duration=float(xx[-1]-xx[0]),
            ordinary_slope=slope,theil_sen_slope=rob,start_value=yy[0],end_value=yy[-1],amplitude_change=yy[-1]-yy[0],
            normalized_amplitude_change=(yy[-1]-yy[0])/amp,local_variance=np.var(yy),raw_point_count=len(yy),stability=ss[j],
            physical_slope=slope/(t[-1]-t[0])))
        start=end
    slopes=np.array([s['theil_sen_slope'] for s in segrows]);rev=np.where(slopes[:-1]*slopes[1:]<0)[0]
    mags=np.abs(np.diff(slopes))[rev]
    strongest=rev[np.argsort(mags)[-2:]] if len(rev)>=2 else []
    f.update(largest_slope_reversal=float(mags.max()) if len(mags) else 0.,slope_sign_reversals=len(rev),
        time_between_strongest_reversals=float(abs(cp_times(u,b)[strongest[0]]-cp_times(u,b)[strongest[1]])) if len(strongest)==2 else np.nan)
    fields=['duration','ordinary_slope','theil_sen_slope','start_value','end_value','amplitude_change','normalized_amplitude_change','local_variance','raw_point_count']
    for slot in range(4):
        for field in fields:f[f'seg{slot+1}_{field}']=segrows[chosen[slot]][field] if slot<len(chosen) else np.nan
    return f,segrows

def extract_one(sid,c,cfg,repeats):
    t,y=c['t'],c['y'];b,chosen,ss,info,runs=stable_segments(t,y,cfg,repeats,SEED+int(sid[1:]))
    allfeatures={};segments={}
    for norm in ['initial','max','robust']:
        yy=normalize(y,norm)
        if np.isfinite(yy).all():allfeatures[norm],segments[norm]=features(t,yy,b,chosen,ss)
    return dict(sid=sid,features=allfeatures,segments=segments,cp=info,cp_runs=runs,breaks=b,chosen=chosen,support=ss.tolist())

def transform(frame, representation, train_idx=None):
    a=frame.to_numpy(float);a[~np.isfinite(a)]=np.nan
    idx=np.arange(len(a)) if train_idx is None else np.asarray(train_idx)
    keep=np.mean(np.isnan(a[idx]),axis=0)<=.4
    a=a[:,keep];names=frame.columns[keep].tolist()
    med=np.nanmedian(a[idx],axis=0);a=np.where(np.isnan(a),med,a)
    nonconstant=np.ptp(a[idx],axis=0)>EPS;a=a[:,nonconstant];names=[n for n,k in zip(names,nonconstant) if k]
    scaler=RobustScaler().fit(a[idx]);x=scaler.transform(a)
    pca=None
    if representation!='full':
        pca=PCA(n_components=float(representation),svd_solver='full').fit(x[idx]);x=pca.transform(x)
    return np.asarray(x,dtype=np.float64),dict(feature_names=names,imputation_medians=med[nonconstant],retained_features=len(names),dimensions=x.shape[1],scaler=scaler,pca=pca)

def fit_cluster(x,params):
    model=hdbscan.HDBSCAN(**params,core_dist_n_jobs=1,gen_min_span_tree=True,approx_min_span_tree=False)
    model.fit(x);return model

def metrics(x,model,metric):
    lab=model.labels_;mask=lab>=0;unique,sizes=np.unique(lab[mask],return_counts=True);k=len(unique)
    d=dict(n=len(lab),n_clusters=k,noise_fraction=float((~mask).mean()),cluster_sizes=json.dumps(sizes.tolist()),
        balance=float(sizes.min()/sizes.max()) if k else 0.,dbcv=np.nan,silhouette=np.nan,davies_bouldin=np.nan,calinski_harabasz=np.nan)
    if 1<k<mask.sum():
        distance='cityblock' if metric=='manhattan' else metric
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:d['dbcv']=float(hdbscan.validity.validity_index(x,lab,metric=distance))
            except (ValueError,ZeroDivisionError,FloatingPointError):pass
        d['silhouette']=float(silhouette_score(x[mask],lab[mask],metric=metric))
        d['davies_bouldin']=float(davies_bouldin_score(x[mask],lab[mask]))
        d['calinski_harabasz']=float(calinski_harabasz_score(x[mask],lab[mask]))
    return d

def agreement(a,b):
    a,b=np.asarray(a),np.asarray(b);assigned=(a>=0)&(b>=0)
    if len(a)==0:return dict(ari=np.nan,ari_assigned=np.nan,vi=np.nan,common_assigned_fraction=np.nan)
    def ari(x,y):return float(adjusted_rand_score(x,y)) if len(np.unique(x[x>=0]))>=2 and len(np.unique(y[y>=0]))>=2 else np.nan
    def entropy(x):
        p=np.unique(x,return_counts=True)[1]/len(x);return -np.sum(p*np.log(p))
    return dict(ari=ari(a,b),ari_assigned=ari(a[assigned],b[assigned]) if assigned.sum()>=5 else np.nan,
        vi=float(entropy(a)+entropy(b)-2*mutual_info_score(a,b)),common_assigned_fraction=float(assigned.mean()))

def bootstrap(frame,representation,params,base_labels,repeats=5,groups=None,mode='bootstrap'):
    rows=[];n=len(frame)
    for rep in range(repeats):
        rng=np.random.default_rng(SEED+rep)
        if mode=='seed':idx=rng.permutation(n)
        elif groups is not None:
            g=np.unique(groups);draw=rng.choice(g,len(g),replace=True);idx=np.concatenate([np.where(groups==gg)[0] for gg in draw])
        else:idx=rng.choice(n,n,replace=True)
        x,_=transform(frame,representation,idx);lab=fit_cluster(x[idx],params).labels_
        unique=np.unique(idx);bl=[]
        for i in unique:
            ll,ct=np.unique(lab[idx==i],return_counts=True);bl.append(ll[np.argmax(ct)])
        d=agreement(np.asarray(base_labels)[unique],bl)
        bl=np.asarray(bl)
        for label in sorted(set(base_labels)-{-1}):
            reference=np.asarray(base_labels)[unique]==label
            jaccards=[np.sum(reference & (bl==other))/max(1,np.sum(reference | (bl==other))) for other in set(bl)-{-1}]
            d[f'cluster_{label}_jaccard']=max(jaccards,default=0.) if reference.any() else np.nan
        d.update(repeat=rep,n_unique=len(unique),n_clusters=len(set(lab)-{-1}),noise_fraction=float((lab<0).mean()))
        rows.append(d)
    return rows

def hdb_params(n):
    sizes=[s for s in [10,20,30,50,75,100] if 2*s<=n]
    if n<100:sizes=sorted(set([max(5,int(round(.05*n))),max(5,int(round(.1*n)))]+sizes))
    for size in sizes:
        for samples in [None,5,10,20]:
            for method in ['eom','leaf']:
                for metric in ['euclidean','manhattan']:
                    yield dict(min_cluster_size=size,min_samples=samples,cluster_selection_method=method,metric=metric)

def grid_one(frame,representation,tag):
    x,prep=transform(frame,representation);rows=[];labelmap={};cache={}
    for i,params in enumerate(hdb_params(len(frame))):
        model=fit_cluster(x,params);lab=model.labels_
        signature=(params['metric'],tuple(lab))
        if signature not in cache:cache[signature]=metrics(x,model,params['metric'])
        d={**cache[signature],**params,**tag,'representation':representation,'dimensions':x.shape[1]}
        cid=f"{tag['dataset']}__{tag['normalization']}__{representation}__{i:03d}"
        d['candidate_id']=cid
        if d['n_clusters']>=2:
            bs=bootstrap(frame,representation,params,lab,repeats=5)
            d['bootstrap_ari']=float(np.nanmean([r['ari'] for r in bs])) if any(np.isfinite(r['ari']) for r in bs) else np.nan
            d['bootstrap_vi']=float(np.mean([r['vi'] for r in bs]))
            d['bootstrap_valid_fraction']=float(np.mean([np.isfinite(r['ari']) for r in bs]))
        else:d.update(bootstrap_ari=np.nan,bootstrap_vi=np.nan,bootstrap_valid_fraction=0.)
        rows.append(d);labelmap[cid]=lab
    return rows,labelmap

def dropped_frames(curves,data,cfg,frac,rep):
    """Refit changepoints after deletion; boundary-reliability calibration is frozen."""
    reference={r['sid']:r for r in data};rows={n:{} for n in ['initial','max','robust']}
    for sid,c in curves.items():
        t,y=c['t'],c['y'];rng=np.random.default_rng(SEED+1000*rep+int(sid[1:]))
        keep=np.ones(len(t),bool)
        keep[rng.choice(np.arange(1,len(t)-1),max(1,int(round(frac*len(t)))),replace=False)]=False
        tt,yy=t[keep],y[keep];b=cp_detect(tt,yy,cfg)
        old=reference[sid]['cp'];new=cp_times(tt,b);support=[]
        for v in new:
            if old:
                j=int(np.argmin([abs(v-r['cp_time']) for r in old]));support.append(old[j]['support'] if abs(v-old[j]['cp_time'])<=old[j]['tolerance'] else 0.)
            else:support.append(0.)
        bs=np.r_[1.,support,1.];ss=np.minimum(bs[:-1],bs[1:])
        chosen=sorted(sorted(range(len(b)),key=lambda j:(-ss[j],j))[:4])
        for norm in rows:
            z=normalize(yy,norm)
            if np.isfinite(z).all():rows[norm][sid]=features(tt,z,b,chosen,ss)[0]
    return {norm:pd.DataFrame(r).T for norm,r in rows.items()}

def ranking(table):
    s=table.copy();valid=(s.n_clusters>=2)&s.dbcv.notna()
    # Fixed, label-free score; full-cohort coverage prevents small-cohort wins by omission.
    s['selection_score']=(.45*s.bootstrap_ari.fillna(-1).clip(-1,1)+.15*(1-s.noise_fraction)+
        .20*s.dbcv.fillna(-1)+.05*s.silhouette.fillna(-1)+.05*s.balance+.10*s.coverage)
    s.loc[~valid,'selection_score']=-np.inf
    return s.sort_values(['selection_score','dbcv','candidate_id'],ascending=[False,False,True])
