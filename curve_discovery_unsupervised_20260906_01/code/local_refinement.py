"""Uniform, unsupervised within-cluster SOM audit; every parent gets same rule.
Local QE elbows describe substructure, not a replacement global optimum.
No target morphology scores or names are read by this script.
"""
from run_experiments import *
from render_results import panel

def main():
    curves=json.loads((ROOT/'data/raw_curves.json').read_text());X=np.load(ROOT/'data/representation_minmax.npz')['X']
    parent=np.load(ROOT/'experiments/models_minmax.npz')['labels'];p=json.loads((ROOT/'experiments/selected_parameters.json').read_text())['minmax']
    rows=[];summaries=[];out=pd.DataFrame({'curve_id':[c['id'] for c in curves],'parent_cluster':parent+1,'child_cluster':0})
    for j in np.unique(parent):
        idx=np.flatnonzero(parent==j);ks=list(range(1,min(8,len(idx)//3)+1));models={}
        for k,seed in itertools.product(ks,SEEDS):
            r,l,w=train(X[idx],k,**p,seed=seed,iterations=30000);r['parent']=int(j+1);rows.append(r);models[k,seed]=l
        df=pd.DataFrame([r for r in rows if r['parent']==j+1]);q=df.groupby('k').qe.median();k,_=elbow(ks,q.values)
        seed=int(df[df.k==k].sort_values('qe').iloc[0].seed);l=models[k,seed];out.loc[idx,'child_cluster']=l+1
        stability=[adjusted_rand_score(models[k,a],models[k,b]) for a,b in itertools.combinations(SEEDS,2)]
        stats=panel([curves[i] for i in idx],l,X[idx],ROOT/f'figures/local_parent_{j+1}.png',f'Exploratory parent {j+1} substructure: local QE elbow k={k}')
        summaries.append({'parent':int(j+1),'n':len(idx),'local_k':k,'seed_ari_mean':float(np.mean(stability)),'representatives':stats})
        print(summaries[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'experiments/local_sweep.csv',index=False);out.to_csv(ROOT/'local_assignments.csv',index=False)
    dump(ROOT/'experiments/local_selection.json',summaries)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
