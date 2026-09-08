from run_experiments import *

def main():
    X=np.load(ROOT/'data/representation_minmax.npz')['X'];inv=pd.read_csv(ROOT/'data/curve_inventory.csv')
    params=json.loads((ROOT/'experiments/selected_parameters.json').read_text())['minmax'];rows=[];summ={}
    masks={'line_topology':np.arange(len(X)),
           'documented_time_without_extreme_scale':np.flatnonzero((inv.axis_status=='time_verified')&(inv.duration_h<1e5))}
    for name,idx in masks.items():
        for k,seed in itertools.product(KS,SEEDS[:3]):
            r,l,w=train(X[idx],k,**params,seed=seed,iterations=30000,line=name=='line_topology')
            r['check']=name;rows.append(r)
        df=pd.DataFrame([r for r in rows if r['check']==name]);q=df.groupby('k').qe.median();k,_=elbow(KS,q.values)
        summ[name]={'n':len(idx),'selected_k':k,'elbow_by_range':{str(e):elbow(KS[:e],q.values[:e])[0] for e in [8,10,12,16]}}
        print(name,summ[name],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'experiments/additional_sweep.csv',index=False);dump(ROOT/'experiments/additional_checks.json',summ)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
