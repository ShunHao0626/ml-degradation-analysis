"""Run from any directory; checkpointed, fully unsupervised analysis."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/method1_mpl'
from core import *
import argparse, itertools, pickle, time, shutil, sys, importlib.metadata
from joblib import Parallel, delayed

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'samples_test'
JOBS=6

def log(s):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s
    print(line,flush=True)
    with (ROOT/'execution.log').open('a') as f:f.write(line+'\n')

def cached(path, fn):
    if path.exists():return pickle.loads(path.read_bytes())
    result=fn();path.write_bytes(pickle.dumps(result));return result

def extract_dataset(name,cs,cfg,repeats=10):
    path=ROOT/'cache'/f'features_{name}_{cfg["id"]}_r{repeats}.pkl'
    data=cached(path,lambda: Parallel(n_jobs=JOBS)(delayed(extract_one)(sid,c,cfg,repeats) for sid,c in cs.items()))
    frames={}
    for norm in ['initial','max','robust']:
        frames[norm]=pd.DataFrame({r['sid']:r['features'][norm] for r in data if norm in r['features']}).T
        frames[norm].index.name='sample_id'
        frames[norm].to_csv(ROOT/'features'/f'{name}_{norm}_r{repeats}.csv')
    return frames,data

def get_params(row):
    ms=row['min_samples']
    return dict(min_cluster_size=int(row['min_cluster_size']),min_samples=None if pd.isna(ms) else int(ms),
        metric=row['metric'],cluster_selection_method=row['cluster_selection_method'])

def initialize():
    for d in ['raw_input','cleaned_input','window_inputs','normalized_input','resampled_input','features','search','stability','sensitivity','final','figures','cache']:
        (ROOT/d).mkdir(exist_ok=True)
    if not (ROOT/'strategy_original.md').exists():shutil.copy2(ROOT.parent/'01_change_point_hdbscan.md',ROOT/'strategy_original.md')
    curves,qc=cached(ROOT/'cache'/'audit.pkl',lambda:audit(SOURCE,ROOT))
    known=window_curves(curves,None)
    configurations=[]
    for i,(algo,cost,frac,pen) in enumerate(itertools.product(['pelt','binseg','window'],['l2','linear','rbf'],[.03,.05,.08,.10],[.5,1,2,3,5,8,10])):
        configurations.append(dict(id=f'CP{i:03d}',algorithm=algo,cost=cost,min_fraction=frac,penalty=pen))
    protocol=dict(smoothing=False,labels_used=False,random_seed=SEED,minimum_points=MIN_POINTS,
        main_cohort='explicit hours and days metadata; days multiplied by 24',time_origin='first valid observed timestamp',
        candidate_windows_hours=[50,100,200,300,500,750,1000,1500,2000],minimum_window_cohort=30,
        fixed_length_curve_features='elapsed-time fraction; raw point counts retained as required; duration excluded from clustering',
        cp_search=configurations,cp_screen_repeats_per_drop=1,cp_feature_repeats_per_drop=10,final_cp_repeats_per_drop=50,
        cp_selection='per-curve BIC percentile averaged, 80%; deletion CP F1, 20%; no target change point count',
        segment_slots='choose up to four segments with highest minimum boundary support; retain chronological order; NaN pad',
        hdb_bootstrap_screen_repeats=5,finalist_bootstrap_repeats=50,final_repeated_seed_runs=50,
        selection_score='0.45 bootstrap ARI + 0.15 nonnoise + 0.20 DBCV + 0.05 silhouette + 0.05 min/max cluster size + 0.10 cohort coverage',
        scope='staged search, not full Cartesian product of every CP configuration and HDBSCAN configuration',
        finalist_selection_score='0.30 bootstrap ARI + 0.15 deletion ARI + 0.15 normalization ARI + 0.10 nonnoise + 0.15 DBCV + 0.05 silhouette + 0.10 coverage',
        inferential_status='exploratory; no independent confirmatory test set; same-figure dependence checked by group bootstrap')
    dump(protocol,ROOT/'protocol.json')
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in ['numpy','pandas','scipy','scikit-learn','ruptures','hdbscan','matplotlib','joblib'])+'\n')
    return curves,qc,known,configurations

def cp_stage(known,configs):
    path=ROOT/'search'/'cp_per_curve.csv'
    if not path.exists():
        log(f'CP screening: {len(configs)} configurations x {len(known)} curves, two deletion probes each')
        results=Parallel(n_jobs=JOBS,verbose=10)(delayed(cp_screen_one)(cfg,list(known.items())) for cfg in configs)
        pd.DataFrame([r for group in results for r in group]).to_csv(path,index=False)
    d=pd.read_csv(path)
    d['bic_percentile']=d.groupby('sample_id').bic.rank(pct=True,method='average')
    agg=d.groupby('config_id').agg(mean_bic=('bic','mean'),bic_percentile=('bic_percentile','mean'),cp_f1=('cp_f1','mean'),mean_segments=('segments','mean')).reset_index()
    agg=agg.merge(pd.DataFrame(configs),left_on='config_id',right_on='id')
    agg['cp_score']=.8*(1-agg.bic_percentile)+.2*agg.cp_f1
    agg=agg.sort_values(['cp_score','mean_segments'],ascending=[False,True]);agg.to_csv(ROOT/'search'/'cp_parameter_search.csv',index=False)
    choices=agg.groupby(['algorithm','cost'],sort=False).head(1)
    cfgs={r['id']:r for r in configs}
    best=cfgs[agg.iloc[0].id]
    dump(best,ROOT/'search'/'selected_cp.json');log(f'CP selected {best}')
    return best,[cfgs[x] for x in choices.id]

def build_datasets(curves,known,cfg):
    specs=[('full_duration',None)]
    coverage=[]
    for w in [50,100,200,300,500,750,1000,1500,2000]:
        cs=window_curves(curves,w);coverage.append(dict(window_hours=w,n=len(cs),coverage=len(cs)/len(known),eligible=len(cs)>=30))
        if len(cs)>=30:specs.append((f'{w}h',w))
    pd.DataFrame(coverage).to_csv(ROOT/'search'/'window_coverage.csv',index=False)
    datasets={}
    for name,w in specs:
        cs=window_curves(curves,w);log(f'Feature extraction {name}: {len(cs)} curves; no resampling')
        frames,data=extract_dataset(name,cs,cfg)
        datasets[name]=dict(curves=cs,frames=frames,data=data,window=w,coverage=len(cs)/len(known))
        destination=ROOT/'window_inputs'/name;destination.mkdir(exist_ok=True)
        for sid,c in cs.items():pd.DataFrame(dict(elapsed=c['t'],y=c['y'])).to_csv(destination/f'{sid}.csv',index=False)
        for norm in ['initial','max','robust']:
            destination=ROOT/'normalized_input'/name/norm;destination.mkdir(parents=True,exist_ok=True)
            for sid,c in cs.items():pd.DataFrame(dict(elapsed=c['t'],elapsed_fraction=c['t']/c['t'][-1],normalized_y=normalize(c['y'],norm))).to_csv(destination/f'{sid}.csv',index=False)
    return datasets

def grid_stage(datasets):
    allrows=[];alllabels={}
    for name,ds in datasets.items():
        path=ROOT/'cache'/f'grid_{name}.pkl'
        def compute():
            log(f'HDBSCAN grid {name}: three normalizations x four representations x adaptive full parameter grid')
            return Parallel(n_jobs=JOBS,verbose=5)(delayed(grid_one)(frame,rep,dict(dataset=name,normalization=norm,coverage=ds['coverage'])) for norm,frame in ds['frames'].items() for rep in ['full','0.9','0.95','0.99'])
        results=cached(path,compute)
        for rows,labels in results:allrows.extend(rows);alllabels.update(labels)
        pd.DataFrame(allrows).to_csv(ROOT/'search'/'hdbscan_parameter_search.csv',index=False)
        log(f'{name} grid complete; total evaluated candidates {len(allrows)}')
    ranking(pd.DataFrame(allrows)).to_csv(ROOT/'search'/'screen_ranking.csv',index=False)
    return pd.DataFrame(allrows),alllabels

def finalists_stage(table,labels,datasets,cfg):
    ranked=ranking(table)
    # Best candidate per window AND normalization, plus global top six distinct partitions.
    top=ranked[np.isfinite(ranked.selection_score)].groupby(['dataset','normalization'],sort=False).head(1)
    top=pd.concat([top,ranked.head(6)]).drop_duplicates('candidate_id')
    for name in top.dataset.unique():
        log(f'Full change-point support validation {name}: 50 repeats at each of 5% and 10% deletion')
        frames,data=extract_dataset(name,datasets[name]['curves'],cfg,repeats=50)
        datasets[name]['frames']=frames;datasets[name]['data']=data
        ds=datasets[name]
        ds['drop_screen']=cached(ROOT/'cache'/f'drop_screen_{name}.pkl',lambda:Parallel(n_jobs=JOBS)(delayed(dropped_frames)(ds['curves'],ds['data'],cfg,frac,rep) for frac in [.05,.1] for rep in range(5)))
    rows=[];runs=[]
    for _,r in top.iterrows():
        path=ROOT/'cache'/f'finalist_{r.candidate_id}.pkl'
        frame=datasets[r.dataset]['frames'][r.normalization]
        params=get_params(r)
        x,_=transform(frame,str(r.representation));model=fit_cluster(x,params)
        labels[r.candidate_id]=model.labels_
        def compute():
            log(f'50-bootstrap finalist {r.candidate_id}')
            bs=bootstrap(frame,str(r.representation),params,labels[r.candidate_id],50)
            return bs
        bs=cached(path,compute)
        d={**r.to_dict(),**metrics(x,model,params['metric']),'dimensions':x.shape[1]};v=np.array([b['ari'] for b in bs]);valid=v[np.isfinite(v)]
        d.update(bootstrap_ari=float(valid.mean()) if len(valid) else np.nan,
            bootstrap_ari_p05=float(np.quantile(valid,.05)) if len(valid) else np.nan,
            bootstrap_ari_p95=float(np.quantile(valid,.95)) if len(valid) else np.nan,
            bootstrap_vi=np.mean([b['vi'] for b in bs]),bootstrap_valid_fraction=len(valid)/50)
        norm_scores=[]
        for norm,other in datasets[r.dataset]['frames'].items():
            if norm==r.normalization:continue
            other=other.loc[frame.index];ox,_=transform(other,str(r.representation))
            norm_scores.append(agreement(model.labels_,fit_cluster(ox,params).labels_)['ari'])
        drop_scores=[]
        for other in datasets[r.dataset]['drop_screen']:
            ox,_=transform(other[r.normalization].loc[frame.index],str(r.representation))
            drop_scores.append(agreement(model.labels_,fit_cluster(ox,params).labels_)['ari'])
        d['normalization_ari']=float(np.mean(np.nan_to_num(norm_scores,nan=-1.)))
        d['deletion_ari']=float(np.mean(np.nan_to_num(drop_scores,nan=-1.)))
        rows.append(d);runs.extend([{**b,'candidate_id':r.candidate_id} for b in bs])
    finalrank=pd.DataFrame(rows)
    finalrank['selection_score']=(.30*finalrank.bootstrap_ari.fillna(-1)+.15*finalrank.deletion_ari+.15*finalrank.normalization_ari+
        .10*(1-finalrank.noise_fraction)+.15*finalrank.dbcv.fillna(-1)+.05*finalrank.silhouette.fillna(-1)+.10*finalrank.coverage)
    finalrank.loc[(finalrank.n_clusters<2)|finalrank.dbcv.isna(),'selection_score']=-np.inf
    finalrank=finalrank.sort_values(['selection_score','candidate_id'],ascending=[False,True]);finalrank.to_csv(ROOT/'search'/'finalist_ranking.csv',index=False)
    pd.DataFrame(runs).to_csv(ROOT/'stability'/'finalist_bootstrap_runs.csv',index=False)
    return finalrank.iloc[0].to_dict()

def main(stage):
    curves,qc,known,configs=initialize()
    log(f'QC: {len(curves)} total curves, {len(known)} eligible known-hour curves')
    cfg,alternatives=cp_stage(known,configs)
    if stage=='cp':return
    datasets=build_datasets(curves,known,cfg)
    table,labels=grid_stage(datasets)
    if stage=='grid':return
    winner=finalists_stage(table,labels,datasets,cfg)
    dump(winner,ROOT/'search'/'selected_candidate.json')
    log('Candidate frozen before physical interpretation: '+winner['candidate_id'])
    from finalize import finalize
    finalize(ROOT,curves,qc,datasets,cfg,alternatives,winner,table,labels,log)
    from export_result_gallery import render
    render(ROOT)
    log('ALL STAGES COMPLETE')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['cp','grid','all'],default='all');args=parser.parse_args();main(args.stage)
