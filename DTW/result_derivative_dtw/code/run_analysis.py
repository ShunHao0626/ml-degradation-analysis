#!/usr/bin/env python3
"""Reproducible unsupervised, unsmoothed variable-length derivative-aware DTW study."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'code/.mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,shutil,ctypes,time,platform,itertools
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score,adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE=ROOT.parent/'samples_test'
SEED=20260906; REPS=50
NORMS=['y0','ymax','z','robust']; RADII=[.02,.05,.1,.15,.2,-1.]
LINKS=['average','complete','weighted'];LAMBDAS=[.25,.5,1.,2.,4.]
for d in ['raw','aligned','features','distance_matrices','clustering_results','stability','figures','audit']:(ROOT/d).mkdir(exist_ok=True)
lib=ctypes.CDLL(str(ROOT/'code/libdtw.dylib'))
lib.distances.argtypes=[np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_double,ctypes.c_int,np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')]
def savecsv(df,p): df.to_csv(ROOT/p,index=False)
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def norm(y,kind):
 if kind=='y0': center,scale=0,y[0]
 elif kind=='ymax':center,scale=0,np.max(y)
 elif kind=='z':center,scale=np.mean(y),np.std(y)
 else:center,scale=np.median(y),np.quantile(y,.75)-np.quantile(y,.25)
 # Zero-amplitude curves stay constant; record degeneracy in feature audit.
 if abs(scale)<1e-12:scale=1.
 return (y-center)/scale

def make_seq(curves,kind,mode,lam,three=False):
 seq=[]
 for c in curves:
  t,y=c['t'],norm(c['y'],kind)
  der=np.diff(y)/np.diff(t)
  if three:
   slopes=[]
   for i in range(1,len(t)):
    ids=np.arange(max(0,i-1),min(len(t),i+2));xx=t[ids];yy=y[ids];xx=xx-xx.mean();slopes.append(np.dot(xx,yy-yy.mean())/np.dot(xx,xx))
   der=np.array(slopes)
  seq.append(np.ascontiguousarray(np.column_stack([y[1:] if mode!='derivative' else np.zeros(len(der)),lam*der if mode!='classic' else np.zeros(len(der))]),dtype=np.float64))
 return seq

def distance(seq,r,rms=True):
 offsets=np.r_[0,np.cumsum([len(x) for x in seq])].astype(np.int64);data=np.ascontiguousarray(np.vstack(seq));out=np.zeros((len(seq),len(seq)))
 lib.distances(data,offsets,len(seq),r,int(rms),out)
 if not np.isfinite(out).all():raise ValueError('Infeasible DTW path')
 return out

def cluster(D,method,k):return cut_tree(linkage(squareform(D,checks=False),method=method),n_clusters=[int(k)]).ravel()
def metrics(D,l):
 n=len(l);counts=np.bincount(l);same=l[:,None]==l[None,:];within=D[same];between=D[~same]
 return dict(silhouette=silhouette_score(D,l,metric='precomputed'),dunn=float(between.min()/max(within.max(),1e-15)),min_cluster=int(counts.min()),max_fraction=float(counts.max()/n))
def path(a,b,r):
 na,nb=len(a),len(b);band=2 if r<0 else max(r,.5/(na-1)+.5/(nb-1)+1e-12)
 D=np.full((na+1,nb+1),np.inf);D[0,0]=0;ptr=np.zeros((na,nb),np.int8)
 for i in range(na):
  lo=max(0,int(np.ceil((i/(na-1)-band)*(nb-1)-1e-10)));hi=min(nb-1,int(np.floor((i/(na-1)+band)*(nb-1)+1e-10)))
  for j in range(lo,hi+1):
   v=[D[i,j],D[i,j+1],D[i+1,j]];p=int(np.argmin(v));D[i+1,j+1]=v[p]+np.sum((a[i]-b[j])**2);ptr[i,j]=p
 i,j=na-1,nb-1;res=[]
 while i>=0 and j>=0:
  res.append((i,j));p=ptr[i,j]
  if p==0:i-=1;j-=1
  elif p==1:i-=1
  else:j-=1
 return np.array(res[::-1]),D[-1,-1]

def audit():
 manifest=[];records=[];dtrows=[];hashes=[]
 for p in sorted(SOURCE.rglob('*')):
  if p.is_file() and p.name!='.DS_Store':
   rel=p.relative_to(SOURCE);dst=ROOT/'raw'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
   hashes.append(dict(path=str(rel),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 for i,p in enumerate(sorted(SOURCE.rglob('*.csv'))):
  sid=f'C{i+1:03d}';meta=p.parent.parent/'validation_result.json';m=json.loads(meta.read_text()) if meta.exists() else {};a=m.get('axis',{});ax=a.get('x',{});ay=a.get('y',{})
  xu=str(ax.get('unit') or '').lower().strip('() ');xn=str(ax.get('name') or '').lower();yn=str(ay.get('name') or '').lower()
  iscycle='cycle' in xn;fac=24 if xu in ['d','day','days'] else 1 if xu in ['h','hr','hrs','hour','hours'] or '(h)' in xn or 'hour' in xn else None
  isy='pce' in yn or 'efficiency' in yn
  df=pd.read_csv(p);v=df[['x','y']].apply(pd.to_numeric,errors='coerce').to_numpy();valid=np.isfinite(v).all(axis=1);v=v[valid];original=v.copy();v=v[np.argsort(v[:,0],kind='stable')]
  duplicate=len(v)-len(np.unique(v[:,0]));conflict=any(np.ptp(v[v[:,0]==t,1])>1e-12 for t in np.unique(v[:,0]));_,idx=np.unique(v[:,0],return_index=True);v=v[np.sort(idx)]
  t,y=v.T;dt=np.diff(t);reason=[]
  if iscycle:reason.append('cycle_axis_not_hours')
  elif fac is None:reason.append('time_unit_unknown')
  if not isy:reason.append('pce_axis_unconfirmed')
  if conflict:reason.append('conflicting_duplicate_timestamps')
  if len(t)<6:reason.append('fewer_than_6_points')
  if np.any(y<=0):reason.append('nonpositive_pce_or_uncalibrated_pixels')
  row=dict(id=sid,group=p.parent.parent.name,source=str(p.relative_to(SOURCE)),x_name=ax.get('name'),x_unit=ax.get('unit'),y_name=ay.get('name'),y_unit=ay.get('unit'),hour_factor=fac,n_raw=len(df),missing_rows=int((~valid).sum()),duplicate_timestamps=duplicate,conflicting_duplicates=conflict,was_sorted=bool(np.all(np.diff(original[:,0])>=0)),n_clean=len(t),x_start=t[0],x_end=t[-1],duration_h=(t[-1]-t[0])*fac if fac else np.nan,median_dt_h=np.median(dt)*fac if fac else np.nan,dt_min_h=dt.min()*fac if fac else np.nan,dt_max_h=dt.max()*fac if fac else np.nan,y_min=y.min(),y_max=y.max(),reasons=';'.join(reason))
  manifest.append(row)
  if not reason:records.append(dict(id=sid,group=row['group'],source=row['source'],t=(t-t[0])*fac,y=y,absolute_t=t*fac,origin_h=t[0]*fac))
  if fac:
   dtrows.extend(dict(id=sid,dt_h=float(d*fac)) for d in dt)
 durations=np.log10([c['t'][-1] for c in records]);med=np.median(durations);mad=np.median(abs(durations-med));upper=med+5*1.4826*mad
 outlier={c['id'] for c in records if np.log10(c['t'][-1])>upper};records=[c for c in records if c['id'] not in outlier]
 for r in manifest:
  if r['id'] in outlier:r['reasons']='extreme_log_duration_above_median_plus_5_scaled_MAD'
  r['eligible']=not bool(r['reasons'])
 savecsv(pd.DataFrame(manifest),'audit/preprocessing_audit.csv');savecsv(pd.DataFrame(dtrows),'audit/sampling_intervals.csv');savecsv(pd.DataFrame(hashes),'raw/source_sha256.csv')
 (ROOT/'audit/qc_rules.json').write_text(json.dumps(dict(min_points=6,duration_log10_median=med,duration_log10_mad=mad,upper_duration_h=10**upper,time_origin='first recorded observation; no time stretching',duplicate_policy='remove identical; exclude conflicting',source=str(SOURCE)),indent=2))
 np.savez_compressed(ROOT/'aligned/eligible_original_sequences.npz',**{f"{c['id']}_{key}":c[key] for c in records for key in ['t','y','absolute_t']})
 return records,pd.DataFrame(manifest)

def windows(records):
 duration=np.array([c['t'][-1] for c in records]);N=len(records);result={};rows=[];membership=[]
 for q in [1,.9,.8,.7,.6,.5,.4,.3,.2]:
  T=float(np.sort(duration)[N-int(np.ceil(q*N))]);wid=f'coverage{int(q*100):03d}';cs=[];covered=0
  for c in records:
   enough=c['t'][-1]>=T-1e-8;inside=c['t']<=T+1e-8;nr=int(inside.sum());keep=enough and nr>=6;covered+=enough
   membership.append(dict(window=wid,T_h=T,id=c['id'],duration_covers=enough,n_original_in_window=nr,included=keep,reason='included' if keep else 'duration_short' if not enough else 'fewer_than_6_original_points_in_window'))
   if keep:
    t=c['t'][inside].copy();y=c['y'][inside].copy()
    if t[-1]<T-1e-8:t=np.r_[t,T];y=np.r_[y,np.interp(T,c['t'],c['y'])]
    cs.append({**c,'t':t,'y':y})
  result[wid]=dict(T=T,curves=cs);rows.append(dict(window=wid,T_h=T,target_coverage=q,duration_covered=covered,duration_coverage=covered/N,n=len(cs),effective_coverage=len(cs)/N,groups=len(set(c['group'] for c in cs)),median_original_points=np.median([sum(c['t']<T) for c in cs])))
  np.savez_compressed(ROOT/f'aligned/{wid}_ragged.npz',**{f"{c['id']}_{key}":c[key] for c in cs for key in ['t','y']})
  for no in NORMS:
   normalized=[];derivs=[]
   for c in cs:
    z=norm(c['y'],no);d=np.diff(z)/np.diff(c['t'])
    normalized.extend(dict(id=c['id'],time_h=t,y_norm=y) for t,y in zip(c['t'],z))
    derivs.extend(dict(id=c['id'],time_right_h=t,dt_h=dt,dy_dt=dv) for t,dt,dv in zip(c['t'][1:],np.diff(c['t']),d))
   savecsv(pd.DataFrame(normalized),f'aligned/{wid}_{no}_normalized_long.csv');savecsv(pd.DataFrame(derivs),f'features/{wid}_{no}_derivatives.csv')
   # NaN-padded matrices explicitly retain variable-length original sampling; never impute padding.
   maxn=max(len(c['t']) for c in cs);ym=np.full((len(cs),maxn),np.nan);tm=ym.copy()
   for i,c in enumerate(cs):ym[i,:len(c['t'])]=norm(c['y'],no);tm[i,:len(c['t'])]=c['t']
   np.savez_compressed(ROOT/f'aligned/{wid}_{no}_matrices.npz',ids=[c['id'] for c in cs],time_h=tm,y_normalized=ym)
 savecsv(pd.DataFrame(rows),'audit/window_coverage.csv');savecsv(pd.DataFrame(membership),'audit/window_membership.csv')
 return result,pd.DataFrame(rows)

def sweep(W,coverage):
 rows=[];labels={};distances={};cid=0
 for wid,w in W.items():
  n=len(w['curves']);eff=float(coverage.set_index('window').loc[wid,'effective_coverage']);log(f'Sweep {wid}: {w["T"]:.3f} h, n={n}')
  for no in NORMS:
   for mode,lam in [('classic',0),('derivative',1)]+[('multivariate',l) for l in LAMBDAS]:
    seq=make_seq(w['curves'],no,mode,lam)
    for rad in RADII:
     key=f'{wid}_{no}_{mode}_l{lam:g}_r{rad:g}';D=distance(seq,rad);distances[key]=D
     np.savez_compressed(ROOT/f'distance_matrices/{key}.npz',distance=D,ids=[c['id'] for c in w['curves']])
     for method in LINKS:
      Z=linkage(squareform(D,checks=False),method=method);all_l=cut_tree(Z,n_clusters=list(range(2,11)))
      for j,k in enumerate(range(2,11)):
       l=all_l[:,j];cid+=1;name=f'M{cid:06d}';labels[name]=l
       m=metrics(D,l);rows.append(dict(candidate=name,distance_key=key,window=wid,T_h=w['T'],n=n,effective_coverage=eff,normalization=no,mode=mode,lambda_value=lam,radius=rad,linkage=method,K=k,**m,eligible_structure=m['min_cluster']>=max(3,int(np.ceil(.05*n))) and m['max_fraction']<=.95))
  log(f'{wid} complete: {cid} cluster candidates accumulated')
 df=pd.DataFrame(rows)
 # No label templates enter screening: ranks of two internal criteria only, within window.
 df['internal_score']=df.groupby('window').silhouette.rank(pct=True)*.75+df.groupby('window').dunn.rank(pct=True)*.25
 savecsv(df,'clustering_results/internal_metrics_all.csv');np.savez_compressed(ROOT/'clustering_results/labels_all_K2_to_K10.npz',**labels)
 return df,labels,distances

def shortlist(df,labels):
 selected=[]
 for wid,g in df.groupby('window',sort=False):
  pool=g[(g.radius>0)&g.eligible_structure];pool=pool.sort_values(['internal_score','radius'],ascending=[False,True])
  if len(pool)==0:pool=g[g.radius>0].sort_values('internal_score',ascending=False)
  for no in NORMS:
   p=pool[(pool.normalization==no)&(pool['mode']=='multivariate')];seen=set();count=0
   for _,r in p.iterrows():
    signature=tuple((labels[r.candidate][:,None]==labels[r.candidate][None,:]).ravel())
    if signature in seen:continue
    seen.add(signature);selected.append(r.candidate);count+=1
    if count==2:break
  # Guarantee every lambda is represented in stability, even if near-tied internally.
  for lam in LAMBDAS:
   p=pool[(pool['mode']=='multivariate')&(pool.lambda_value==lam)]
   if len(p):selected.append(p.iloc[0].candidate)
  for mo in ['classic','derivative']:
   p=pool[pool['mode']==mo]
   if len(p):selected.append(p.iloc[0].candidate)
 return df[df.candidate.isin(set(selected))].copy()

def perturb(cs,fraction,rng):
 out=[];rates=[]
 for c in cs:
  n=len(c['t']);k=max(1,int(round(fraction*(n-2))));remove=rng.choice(np.arange(1,n-1),size=k,replace=False);keep=np.ones(n,bool);keep[remove]=False
  out.append({**c,'t':c['t'][keep],'y':c['y'][keep]});rates.append(k/n)
 return out,np.mean(rates)

def stability(short,W,labels,Ds):
 details=[];summaries=[];consensus={}
 for z,(_,r) in enumerate(short.iterrows()):
  cs=W[r.window]['curves'];n=len(cs);base=labels[r.candidate];D=Ds[r.distance_key];seq=make_seq(cs,r.normalization,r['mode'],r.lambda_value)
  log(f'Stability {z+1}/{len(short)} {r.candidate} {r.window} {r.normalization} {r["mode"]} K={r.K}')
  for mode in ['bootstrap90','bootstrap95','drop05','drop10']:
   rng=np.random.default_rng(SEED+['bootstrap90','bootstrap95','drop05','drop10'].index(mode));num=np.zeros((n,n));den=np.zeros((n,n));aris=[];rates=[]
   for rep in range(REPS):
    rate=np.nan
    if mode.startswith('bootstrap'):
     # Genuine bootstrap WITH replacement, as requested. Score each observed curve once.
     frac=.9 if mode.endswith('90') else .95;draw=rng.choice(n,size=int(np.ceil(n*frac)),replace=True)
     lb=cluster(D[np.ix_(draw,draw)],r.linkage,r.K);ids,first=np.unique(draw,return_index=True);l=lb[first]
    else:
     pc,rate=perturb(cs,.05 if mode=='drop05' else .1,rng);DD=distance(make_seq(pc,r.normalization,r['mode'],r.lambda_value),r.radius);l=cluster(DD,r.linkage,r.K);ids=np.arange(n)
    ari=adjusted_rand_score(base[ids],l);aris.append(ari);rates.append(rate);num[np.ix_(ids,ids)]+=(l[:,None]==l[None,:]);den[np.ix_(ids,ids)]+=1
    details.append(dict(candidate=r.candidate,window=r.window,mode=mode,replicate=rep,ARI=ari,unique_curves=len(ids),actual_deleted_fraction=rate))
   C=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);consensus[r.candidate+'_'+mode]=C
   np.savez_compressed(ROOT/f'stability/{r.candidate}_{mode}_consensus.npz',consensus=C,numerator=num,denominator=den,ids=[c['id'] for c in cs])
   summaries.append(dict(candidate=r.candidate,mode=mode,replicates=REPS,ARI_mean=np.mean(aris),ARI_median=np.median(aris),ARI_p025=np.quantile(aris,.025),ARI_p975=np.quantile(aris,.975),ARI_min=min(aris),actual_deleted_fraction=np.nanmean(rates) if not mode.startswith('bootstrap') else np.nan))
  savecsv(pd.DataFrame(details),'stability/replicate_metrics.csv');savecsv(pd.DataFrame(summaries),'stability/stability_summary.csv')
 return pd.DataFrame(summaries),consensus

def warp_audit(r,cs,D,base):
 seq=make_seq(cs,r.normalization,r['mode'],r.lambda_value);rows=[];medoids={}
 for k in np.unique(base):
  ids=np.flatnonzero(base==k);medoids[int(k)]=int(ids[np.argmin(D[np.ix_(ids,ids)].sum(axis=1))])
  for i in ids:
   j=medoids[int(k)]
   if i==j:continue
   p,cost=path(seq[i],seq[j],r.radius);a,b=p.T;idxwarp=abs(a/(len(seq[i])-1)-b/(len(seq[j])-1));tw=abs(cs[i]['t'][1:][a]-cs[j]['t'][1:][b])/r.T_h
   # Count raw derivative sign changes collapsed onto one target sample (zeros omitted).
   changes=0;collapsed=0
   for axis,other,source in [(a,b,seq[i][:,1]),(b,a,seq[j][:,1])]:
    signs=np.sign(source);nz=np.flatnonzero(signs);turns=nz[1:][signs[nz[1:]]!=signs[nz[:-1]]];changes+=len(turns)
    mapped=[int(np.median(other[axis==v])) for v in turns if np.any(axis==v)];collapsed+=len(mapped)-len(set(mapped))
   rows.append(dict(candidate=r.candidate,id=cs[i]['id'],medoid=cs[j]['id'],cluster=int(k)+1,path_length=len(p),path_length_ratio=len(p)/max(len(seq[i]),len(seq[j])),max_index_warp=idxwarp.max(),mean_index_warp=idxwarp.mean(),max_time_warp_fraction=tw.max(),mean_time_warp_fraction=tw.mean(),boundary_fraction=float(np.mean(idxwarp>=max(r.radius,.5/(len(seq[i])-1)+.5/(len(seq[j])-1))-1e-8)) if r.radius>0 else np.nan,turn_collapse_fraction=collapsed/max(changes,1),turns=changes))
 return pd.DataFrame(rows),medoids

def main():
 log('Start audit');records,a=audit();W,cov=windows(records)
 log(f'{len(a)} source curves -> {len(records)} eligible; {len(W)} windows')
 df,labels,Ds=sweep(W,cov);short=shortlist(df,labels);savecsv(short,'stability/shortlisted_candidates.csv')
 log(f'Stability shortlist: {len(short)} candidates x 4 perturbations x {REPS} repeats')
 stab,cons=stability(short,W,labels,Ds)
 means=stab.pivot(index='candidate',columns='mode',values='ARI_mean');rank=short.set_index('candidate').join(means)
 rank['stability_mean']=means.mean(axis=1);rank['stability_worst']=means.min(axis=1)
 warps=[]
 for cid,r in rank.iterrows():
  r=r.copy();r['candidate']=cid;wa,med=warp_audit(r,W[r.window]['curves'],Ds[r.distance_key],labels[cid]);warps.append(wa)
  rank.loc[cid,'warp_mean_time']=wa.mean_time_warp_fraction.mean();rank.loc[cid,'warp_p95_max_time']=wa.max_time_warp_fraction.quantile(.95);rank.loc[cid,'turn_collapse_mean']=wa.turn_collapse_fraction.mean()
 warp=pd.concat(warps,ignore_index=True);savecsv(warp,'stability/warping_audit_shortlist.csv')
 # Fixed, phenotype-free score. ARI carries most weight; penalize extreme alignment.
 rank['selection_score']=.45*rank.stability_worst+.25*((rank.silhouette+1)/2)+.1*(rank.dunn/(1+rank.dunn))+.2*rank.effective_coverage-.2*rank.warp_mean_time
 rank['extreme_warp']=(rank.warp_p95_max_time>.35)|(rank.turn_collapse_mean>.25)
 rank['selection_score']-=rank.extreme_warp*.15
 rank=rank.sort_values('selection_score',ascending=False);savecsv(rank.reset_index(),'stability/final_candidate_ranking.csv')
 pool=rank[(rank['mode']=='multivariate')&rank.eligible_structure&(~rank.extreme_warp)]
 if len(pool)==0:pool=rank[rank['mode']=='multivariate']
 best=pool.iloc[0].copy();best['candidate']=pool.index[0]
 payload={k:(v.item() if isinstance(v,np.generic) else v) for k,v in best.to_dict().items()};payload['seed']=SEED;payload['status']='parameters frozen before curve interpretation'
 (ROOT/'clustering_results/frozen_parameters.json').write_text(json.dumps(payload,indent=2))
 np.savez_compressed(ROOT/'code/analysis_state.npz',**labels)
 (ROOT/'code/runtime_environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,platform=platform.platform()),indent=2))
 log(f'FROZEN {best.candidate}: {best.T_h:.3f}h K={best.K} score={best.selection_score:.4f}')
 # Persist reusable state in trusted local pickle; no input pickles are read.
 import pickle
 with (ROOT/'code/state.pkl').open('wb') as f:pickle.dump((records,a,W,cov,df,labels,Ds,rank,best),f,protocol=4)

if __name__=='__main__':main()
