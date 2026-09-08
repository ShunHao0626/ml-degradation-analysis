from pathlib import Path
import json,hashlib,pickle
import numpy as np,pandas as pd
import run_analysis as m
R=m.ROOT;checks=[]
def check(name,condition):
 if not condition:raise AssertionError(name)
 checks.append({'check':name,'passed':True})
manifest=pd.read_csv(R/'raw/source_sha256.csv')
check('Every source file unchanged',all(hashlib.sha256((m.SOURCE/r.path).read_bytes()).hexdigest()==r.sha256 for _,r in manifest.iterrows()))
check('Every raw copy identical to source',all(hashlib.sha256((R/'raw'/r.path).read_bytes()).hexdigest()==r.sha256 for _,r in manifest.iterrows()))
a=pd.read_csv(R/'audit/preprocessing_audit.csv');check('218 audited, 134 eligible',len(a)==218 and a.eligible.sum()==134)
with (R/'code/state.pkl').open('rb') as f:records,a,W,cov,df,labels,Ds,rank,best=pickle.load(f)
check('1512 distances and 40824 labeled candidates',len(Ds)==1512 and len(df)==40824 and len(labels)==40824)
for wid,w in W.items():
 for c in w['curves']:
  assert np.all(np.diff(c['t'])>0) and np.isclose(c['t'][0],0) and np.isclose(c['t'][-1],w['T'])
 for no in m.NORMS:
  normalized=pd.read_csv(R/f'aligned/{wid}_{no}_normalized_long.csv');der=pd.read_csv(R/f'features/{wid}_{no}_derivatives.csv')
  for c in w['curves']:
   d=der[der.id==c['id']];z=m.norm(c['y'],no);assert np.allclose(d.dy_dt,np.diff(z)/np.diff(c['t']))
check('All exported derivatives use true dt and matching normalization',True)
for key,D in Ds.items():assert np.isfinite(D).all() and np.allclose(D,D.T) and np.allclose(np.diag(D),0) and np.min(D)>=0
check('All distance matrices finite symmetric nonnegative with zero diagonal',True)
for _,r in df.iterrows():assert len(labels[r.candidate])==r.n and len(np.unique(labels[r.candidate]))==r.K
check('All label vectors have advertised sample count and K',True)
st=pd.read_csv(R/'stability/replicate_metrics.csv');check('105 candidates, four modes, exactly 50 repeats each',len(st)==21000 and len(st.candidate.unique())==105 and (st.groupby(['candidate','mode']).size()==50).all())
for f in (R/'stability').glob('*_consensus.npz'):
 b=np.load(f);C=b['consensus'];num=b['numerator'];den=b['denominator'];assert np.allclose(num,num.T) and np.allclose(den,den.T);assert np.all(num<=den);assert np.allclose(C[den>0],num[den>0]/den[den>0]);assert np.isnan(C[den==0]).all()
check('Consensus numerators denominators symmetry and missing-pair handling',True)
k=pd.read_csv(R/'stability/final_K_sweep/replicate_metrics.csv');check('K2..10 each receives four modes x 50 repeats',len(k)==1800 and (k.groupby(['K','mode']).size()==50).all())
D=Ds[best.distance_key];l=m.cluster(D,best.linkage,best.K);final=pd.read_csv(R/'clustering_results/final_labels.csv');check('Final labels exactly reproduce frozen linkage',np.array_equal(l+1,final.cluster.to_numpy()))
check('Final cluster counts 84/7',sorted(final.cluster.value_counts().tolist())==[7,84])
for _,r in final[final.is_medoid].iterrows():
 i=np.where(final.id==r.id)[0][0];ii=np.where(final.cluster==r.cluster)[0];assert i==ii[np.argmin(D[np.ix_(ii,ii)].sum(axis=1))]
check('Medoids are real samples minimizing within-cluster distance sum',True)
rng=np.random.default_rng(2)
for n,k,r in [(3,55,.02),(4,7,.05),(17,11,.2),(10,10,-1),(35,65,.02)]:
 aa=rng.normal(size=(n,2));bb=rng.normal(size=(k,2));p,cost=m.path(aa,bb,r);dd=m.distance([aa,bb],r);assert np.isclose(dd[0,1],np.sqrt(cost/len(p)))
check('C++ DTW agrees with independent Python recurrence and backtracking',True)
t=np.array([0.,.2,2.,6.]);seq=m.make_seq([{'t':t,'y':3*t+1}],'y0','multivariate',1)[0];check('Irregular-time linear curve has exact constant slope',np.allclose(seq[:,1],3))
check('Constant curves have zero distance',np.allclose(m.distance([np.ones((5,2)),np.ones((8,2))],.02),0))
for p in (R/'figures').glob('*.png'):assert p.stat().st_size>10000
check('At least 18 original figure groups rendered to PNG and SVG',len(list((R/'figures').glob('*.png')))>=18 and len(list((R/'figures').glob('*.svg')))>=18)
(R/'audit/verification_results.json').write_text(json.dumps(checks,indent=2));print('PASS:',len(checks),'verification categories')
