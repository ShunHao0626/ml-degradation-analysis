import run_analysis as m
import pickle,numpy as np,pandas as pd,json
from sklearn.metrics import adjusted_rand_score
with (m.ROOT/'code/state.pkl').open('rb') as f:records,a,W,cov,df,labels,Ds,rank,best=pickle.load(f)
cs=W[best.window]['curves'];D=Ds[best.distance_key];n=len(cs);out=m.ROOT/'stability/final_K_sweep';out.mkdir(exist_ok=True)
ks=range(2,11);bl={k:m.cluster(D,best.linkage,k) for k in ks};rows=[]
for mode in ['bootstrap90','bootstrap95','drop05','drop10']:
 rng=np.random.default_rng(m.SEED+['bootstrap90','bootstrap95','drop05','drop10'].index(mode));nums={k:np.zeros_like(D) for k in ks};den=np.zeros_like(D)
 for rep in range(50):
  if mode.startswith('bootstrap'):
   draw=rng.choice(n,size=int(np.ceil(n*(.9 if mode.endswith('90') else .95))),replace=True);DD=D[np.ix_(draw,draw)];ids,first=np.unique(draw,return_index=True)
  else:
   pc,rate=m.perturb(cs,.05 if mode=='drop05' else .1,rng);DD=m.distance(m.make_seq(pc,best.normalization,best['mode'],best.lambda_value),best.radius);ids=np.arange(n);first=ids
  den[np.ix_(ids,ids)]+=1
  for k in ks:
   l=m.cluster(DD,best.linkage,k)[first];rows.append(dict(K=k,mode=mode,replicate=rep,ARI=adjusted_rand_score(bl[k][ids],l)));nums[k][np.ix_(ids,ids)]+=l[:,None]==l[None,:]
 for k in ks:np.savez_compressed(out/f'K{k}_{mode}_consensus.npz',consensus=np.divide(nums[k],den,out=np.full_like(D,np.nan),where=den>0),numerator=nums[k],denominator=den,ids=[c['id'] for c in cs])
r=pd.DataFrame(rows);r.to_csv(out/'replicate_metrics.csv',index=False);s=r.groupby(['K','mode']).ARI.agg(['mean','min','median']).reset_index();s.to_csv(out/'stability_summary.csv',index=False)
q=[]
for k in ks:q.append(dict(K=k,**m.metrics(D,bl[k]),**{mode:r[(r.K==k)&(r['mode']==mode)].ARI.mean() for mode in r['mode'].unique()}))
pd.DataFrame(q).to_csv(out/'K_comparison.csv',index=False)
print(pd.DataFrame(q).to_string(index=False))
