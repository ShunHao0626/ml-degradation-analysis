import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'shape_only_four_group_exploration_20260925'))
from run import read_rows,INDEX,SHAPES,original_observations,encode,fit
rows=read_rows(INDEX)
with np.load(SHAPES) as a: raw=a['raw_rank']
orig=[original_observations(r) for r in rows]
n=np.array([len(y) for y in orig]);rough=np.array([np.sum(abs(np.diff(y)))/max(np.ptp(y),.005) for y in orig])
configs=[(4,3),(6,3),(6,2.5),(8,3)]
base=[]
for mn,rr in configs:
 idx=np.flatnonzero((n>=mn)&(rough<=rr));X,_,_,_=encode(raw[idx]);lab,_,_=fit(X)
 base.append((idx,lab))
results={str(cfg):[] for cfg in configs}
for sd in (.003,.005):
 for seed in range(5):
  rng=np.random.default_rng(55000+seed)
  noisy=[]
  for y in orig:
   z=y+rng.normal(0,sd,len(y));z=z/z[0]
   noisy.append(np.interp(np.linspace(0,len(y)-1,64),np.arange(len(y)),z))
  noisy=np.array(noisy)
  for cfg,(idx,lab) in zip(configs,base):
   X2,_,_,_=encode(noisy[idx]);l2,_,_=fit(X2)
   jac=[]
   for k in range(4):
    a=lab==k
    jac.append(max(np.sum(a&(l2==j))/np.sum(a|(l2==j)) for j in range(4)))
   results[str(cfg)].append({'sd':sd,'seed':seed,'ari':float(adjusted_rand_score(lab,l2)),'min_jac':float(min(jac)),'mean_jac':float(np.mean(jac)),'jac':jac})
for cfg in configs:
 z=results[str(cfg)]
 print(json.dumps({'config':cfg,'keep':len(base[configs.index(cfg)][0]),'sizes':np.bincount(base[configs.index(cfg)][1]).tolist(),
 'mean_ari':float(np.mean([a['ari'] for a in z])),'mean_min_jac':float(np.mean([a['min_jac'] for a in z])),
 'mean_jac':float(np.mean([a['mean_jac'] for a in z])),'by_sd':{str(sd):{
 'ari':float(np.mean([a['ari'] for a in z if a['sd']==sd])),
 'min_jac':float(np.mean([a['min_jac'] for a in z if a['sd']==sd]))} for sd in (.003,.005)} }),flush=True)
