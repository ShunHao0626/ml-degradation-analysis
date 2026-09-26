#!/usr/bin/env python3
"""Median SOM with raw variable-length elastic distances. No barycenter smoothing.
An extension inspired by dissimilarity SOM and SOMTimeS, NOT a SOMTimeS/KASBA reproduction.
"""
from run_experiments import *
import ctypes, subprocess
from scipy.optimize import linear_sum_assignment

def build_distances(curves):
    so=ROOT/'code/elastic_distances.dylib'
    subprocess.run(['/usr/bin/clang++','-O3','-std=c++17','-dynamiclib',str(ROOT/'code/elastic_distances.cpp'),'-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so));ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.raw_dtw.argtypes=[ptr,ptr,ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_double];lib.raw_dtw.restype=ctypes.c_double
    lib.raw_msm.argtypes=[ptr,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_double];lib.raw_msm.restype=ctypes.c_double
    yy=[np.ascontiguousarray(values(c,'minmax')) for c in curves]
    uu=[np.ascontiguousarray(c['u'],dtype=float) for c in curves]
    n=len(curves);out={}
    for method,param in [('dtw',0),('dtw',.25),('dtw',1),('msm',.01),('msm',.1)]:
        name=f'{method}_{param}';D=np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                if method=='dtw':d=lib.raw_dtw(yy[i],uu[i],len(yy[i]),yy[j],uu[j],len(yy[j]),param)
                else:d=lib.raw_msm(yy[i],len(yy[i]),yy[j],len(yy[j]),param)
                D[i,j]=D[j,i]=d
        assert np.isfinite(D).all() and np.all(D>=0)
        np.save(ROOT/f'data/distance_{name}.npy',D);out[name]=D
        print('DISTANCE',name,'max',D.max(),flush=True)
    return out

def median_som(D,k,sigma,seed,epochs=100):
    rng=np.random.default_rng(seed);n=len(D)
    # Distance-based k-means++ style initialization (squared dissimilarities).
    med=[int(rng.integers(n))]
    while len(med)<k:
        p=D[:,med].min(axis=1)**2;p[med]=0
        if p.sum()>0:med.append(int(rng.choice(n,p=p/p.sum())))
        else:med.append(int(rng.choice(np.setdiff1d(np.arange(n),med))))
    med=np.array(med);a,b=grid(k);coords=np.array(list(itertools.product(range(a),range(b))))
    g2=cdist(coords,coords,'sqeuclidean')
    for e in range(epochs):
        labels=D[:,med].argmin(axis=1);s=sigma/(1+2*e/epochs)
        H=np.exp(-g2/(2*s*s));cost=H[labels].T@D
        # Unique sample prototypes avoid duplicate medoids/empty nodes caused solely by ties.
        _,med=linear_sum_assignment(cost)
    labels=D[:,med].argmin(axis=1);occ=len(np.unique(labels))
    return {'k':k,'sigma':sigma,'seed':seed,'qe_elastic':float(D[:,med].min(axis=1).mean()),
            'occupied':occ,'silhouette_elastic':float(silhouette_score(D,labels,metric='precomputed')) if occ>1 else None},labels,med

def main():
    curves=json.loads((ROOT/'data/raw_curves.json').read_text());Ds=build_distances(curves)
    X=np.load(ROOT/'data/representation_minmax.npz')['X'];rows=[];models={};summaries={}
    for name,D in Ds.items():
        for sigma,k,seed in itertools.product([.3,.5,.8],KS,SEEDS):
            r,l,m=median_som(D,k,sigma,seed);r['method']=name
            # Same common paper-style L2 space for comparison across elastic geometries.
            centers=np.array([X[l==j].mean(axis=0) if np.any(l==j) else X[m[j]] for j in range(k)])
            r['assigned_l2_centroid_error']=float(np.linalg.norm(X-centers[l],axis=1).mean())
            r['silhouette_l2']=float(silhouette_score(X,l)) if r['occupied']>1 else None
            rows.append(r);models[name,sigma,k,seed]=(l,m)
        df=pd.DataFrame([r for r in rows if r['method']==name])
        tab=df.groupby(['sigma','k']).qe_elastic.median().unstack('k');score=tab.div(tab[1],axis=0).mean(axis=1)
        sigma=float(score.idxmin());sub=df[df.sigma==sigma];q=sub.groupby('k').qe_elastic.median()
        k,_=elbow(KS,q.values);seed=int(sub[sub.k==k].sort_values('qe_elastic').iloc[0].seed)
        l,m=models[name,sigma,k,seed];st=[adjusted_rand_score(models[name,sigma,k,a][0],models[name,sigma,k,b][0]) for a,b in itertools.combinations(SEEDS,2)]
        summaries[name]={'selected_k':k,'sigma':sigma,'seed':seed,'seed_ari_mean':float(np.mean(st)),
            'elbow_by_range':{str(end):elbow(KS[:end],q.values[:end])[0] for end in [8,10,12,16]},
            'cluster_sizes':np.bincount(l,minlength=k).tolist(),
            'paper_criterion_status':'QE elbow generalized to elastic distance; NOT original Euclidean QE',
            'selection_row':sub[(sub.k==k)&(sub.seed==seed)].iloc[0].to_dict()}
        np.savez_compressed(ROOT/f'experiments/elastic_model_{name}.npz',labels=l,medoids=m)
        pd.DataFrame(rows).to_csv(ROOT/'experiments/elastic_sweep.csv',index=False)
        print('ELASTIC_SELECTED',name,summaries[name],flush=True)
    dump(ROOT/'experiments/elastic_selection.json',summaries)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
