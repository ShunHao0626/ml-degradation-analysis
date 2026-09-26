import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_samples, adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'shape_only_four_group_exploration_20260925'))
from run import read_rows,INDEX,SHAPES,original_observations,encode,fit
rows=read_rows(INDEX)
with np.load(SHAPES) as a: raw=a['raw_rank']
orig=[original_observations(r) for r in rows]
n=np.array([len(y) for y in orig]);rough=np.array([np.sum(abs(np.diff(y)))/max(np.ptp(y),.005) for y in orig])
noisy=[]
for sd in (.003,.005):
 rng=np.random.default_rng(9001)
 arr=[]
 for y in orig:
  z=y+rng.normal(0,sd,len(y));z=z/z[0]
  arr.append(np.interp(np.linspace(0,len(y)-1,64),np.arange(len(y)),z))
 noisy.append(np.array(arr))
from collections import Counter
for min_n,max_rough in [(4,3),(5,3),(6,3),(7,3),(8,3),(6,2),(7,2),(8,2),(8,2.5)]:
 idx=np.flatnonzero((n>=min_n)&(rough<=max_rough))
 X,_,_,_=encode(raw[idx]); lab,_,_=fit(X)
 rng=np.random.default_rng(123);sample=rng.choice(len(idx),size=min(1000,len(idx)),replace=False)
 sil=silhouette_samples(X[sample],lab[sample]);smean=float(np.mean(sil));per=[float(np.mean(sil[lab[sample]==k])) if np.any(lab[sample]==k) else None for k in range(4)]
 stab=[]
 for z in noisy:
  X2,_,_,_=encode(z[idx]);l2,_,_=fit(X2)
  jac=[]
  for k in range(4):
   a=lab==k
   jac.append(max(np.sum(a&(l2==j))/np.sum(a|(l2==j)) for j in range(4)))
  stab.append({'ari':float(adjusted_rand_score(lab,l2)),'jaccard':jac})
 out={'min_obs':min_n,'max_rough':max_rough,'kept':len(idx),'sizes':np.bincount(lab).tolist(),'sil':smean,'sil_clusters':per,'stability':stab}
 print(json.dumps(out),flush=True)
