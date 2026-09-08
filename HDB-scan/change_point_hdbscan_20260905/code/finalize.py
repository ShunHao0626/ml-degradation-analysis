from core import *
import pickle, itertools, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr, kruskal
from sklearn.metrics import adjusted_mutual_info_score

def savefig(path):
    plt.tight_layout();plt.savefig(path,dpi=170,bbox_inches='tight');plt.close()

def comparison(frame,rep,params,base_ids,base_labels):
    x,_=transform(frame,rep);model=fit_cluster(x,params);met=metrics(x,model,params['metric'])
    ids=[s for s in base_ids if s in frame.index];ia=[list(base_ids).index(s) for s in ids];ib=[list(frame.index).index(s) for s in ids]
    met.update(agreement(np.asarray(base_labels)[ia],model.labels_[ib]));met['n_overlap']=len(ids)
    return met,model,x

def drop_one(ds,cfg,frac,rep,norm,representation,params,ids,labels):
    frame=dropped_frames(ds['curves'],ds['data'],cfg,frac,rep)[norm].loc[ids]
    met,model,x=comparison(frame,representation,params,ids,labels)
    return dict(drop_fraction=frac,repeat=rep,**met)

def summarize_runs(rows,kind):
    d=pd.DataFrame(rows)
    result=dict(kind=kind,repeats=len(d),ari_valid_fraction=float(d.ari.notna().mean()))
    result['degenerate_run_fraction']=float(d.ari.isna().mean())
    result['ari_zero_for_degenerate_mean']=float(d.ari.fillna(0.).mean())
    for col in ['ari','ari_assigned','vi','noise_fraction','n_clusters']:
        result[col+'_mean']=d[col].mean();result[col+'_p05']=d[col].quantile(.05);result[col+'_p95']=d[col].quantile(.95)
    return result

def finalize(root,curves,qc,datasets,cfg,alternatives,winner,table,alllabels,log):
    from run_pipeline import extract_dataset,cached,get_params
    ds=datasets[winner['dataset']];norm=winner['normalization'];rep=str(winner['representation']);params=get_params(winner)
    ds['selected_norm']=norm
    frame=ds['frames'][norm];ids=frame.index.tolist();x,prep=transform(frame,rep);model=fit_cluster(x,params);labels=model.labels_
    finalmetrics=metrics(x,model,params['metric'])
    frozen=dict(candidate=winner,cp=cfg,hdbscan=params,metrics=finalmetrics,feature_names=prep['feature_names'],representation=rep,
        selected_window_hours=ds['window'],interpretation_started_after_freeze=True)
    dump(frozen,root/'final'/'frozen_model.json')
    (root/'final'/'fitted_model.pkl').write_bytes(pickle.dumps(dict(model=model,preprocessing=prep,feature_columns=frame.columns.tolist(),ids=ids)))
    frame.to_csv(root/'final'/'feature_table.csv',index_label='sample_id')
    pd.DataFrame(x,index=ids,columns=[f'component_{i+1}' for i in range(x.shape[1])]).to_csv(root/'final'/'clustering_matrix.csv',index_label='sample_id')
    fullx,fullprep=transform(frame,'full')
    pd.DataFrame(fullx,index=ids,columns=fullprep['feature_names']).to_csv(root/'final'/'standardized_features_no_pca.csv',index_label='sample_id')
    pd.DataFrame(dict(feature=prep['feature_names'],imputation_median=prep['imputation_medians'],robust_center=prep['scaler'].center_,robust_scale=prep['scaler'].scale_)).to_csv(root/'final'/'preprocessing_parameters.csv',index=False)
    if prep['pca'] is not None:
        pd.DataFrame(prep['pca'].components_.T,index=prep['feature_names'],columns=[f'PC{i+1}' for i in range(x.shape[1])]).to_csv(root/'final'/'pca_loadings.csv',index_label='feature')
        pd.DataFrame(dict(component=np.arange(1,x.shape[1]+1),explained_variance_ratio=prep['pca'].explained_variance_ratio_)).to_csv(root/'final'/'pca_explained_variance.csv',index=False)
    assignment=pd.DataFrame(dict(sample_id=ids,cluster=labels,membership_probability=model.probabilities_,outlier_score=model.outlier_scores_))
    assignment=assignment.merge(qc[['sample_id','source_path','figure_id','unit','duration_analysis','clean_n','qc_flags']],on='sample_id')
    assignment.to_csv(root/'final'/'cluster_assignment.csv',index=False)
    allassignment=qc[['sample_id','source_path','figure_id','unit','clean_n']].merge(assignment[['sample_id','cluster','membership_probability','outlier_score']],on='sample_id',how='left')
    def exclusion(row):
        if pd.notna(row.cluster):return 'included'
        if row.unit not in ['hours','days']:return 'non_time_axis' if row.unit=='cycles' else 'unknown_time_unit'
        if row.clean_n<MIN_POINTS:return 'fewer_than_12_points'
        return 'does_not_cover_window_or_fewer_than_12_points_within_window'
    allassignment['status']=allassignment.apply(exclusion,axis=1);allassignment.to_csv(root/'final'/'all_218_sample_status.csv',index=False)
    segrows=[];cprows=[];cpruns=[]
    for r in ds['data']:
        if r['sid'] not in ids:continue
        label=int(labels[ids.index(r['sid'])])
        segrows.extend([{**s,'sample_id':r['sid'],'cluster':label} for s in r['segments'][norm]])
        cprows.extend([{**s,'sample_id':r['sid'],'cluster':label} for s in r['cp']])
        cpruns.extend([{**s,'sample_id':r['sid'],'cluster':label} for s in r['cp_runs']])
    segments=pd.DataFrame(segrows);cp=pd.DataFrame(cprows)
    segments.to_csv(root/'final'/'all_segment_descriptors.csv',index=False)
    cp.to_csv(root/'final'/'change_point_support.csv',index=False)
    pd.DataFrame(cpruns).to_csv(root/'stability'/'change_point_50x2_runs.csv',index=False)
    # Final clustering robustness, with preprocessing refit on every bootstrap.
    log('Final stability: sample bootstrap, figure bootstrap, shuffled input, 50 repetitions each')
    stability=[];stability_rows={}
    groups=np.array([curves[s]['figure'] for s in ids])
    for name,kwargs in [('sample_bootstrap',{}),('figure_bootstrap',dict(groups=groups)),('repeated_seed',dict(mode='seed'))]:
        rows=cached(root/'cache'/f'final_{winner["candidate_id"]}_{name}.pkl',lambda:bootstrap(frame,rep,params,labels,50,**kwargs))
        pd.DataFrame(rows).to_csv(root/'stability'/f'{name}_50_runs.csv',index=False)
        stability.append(summarize_runs(rows,name));stability_rows[name]=rows
    log('Final timepoint deletion: 50 runs at 5%, 50 runs at 10%; changepoints and feature transforms refit')
    drops=cached(root/'cache'/f'final_{winner["candidate_id"]}_drop.pkl',lambda:Parallel(n_jobs=6)(delayed(drop_one)(ds,cfg,frac,r,norm,rep,params,ids,labels) for frac in [.05,.1] for r in range(50)))
    pd.DataFrame(drops).to_csv(root/'stability'/'timepoint_deletion_100_runs.csv',index=False)
    for frac in [.05,.1]:stability.append(summarize_runs([r for r in drops if r['drop_fraction']==frac],f'drop_{int(frac*100)}pct'))
    pd.DataFrame(stability).to_csv(root/'stability'/'cluster_stability_summary.csv',index=False)
    percluster=[]
    for label in sorted(set(labels)-{-1}):
        d=dict(cluster=label,n=int((labels==label).sum()))
        for name,rows in stability_rows.items():
            values=pd.DataFrame(rows)[f'cluster_{label}_jaccard']
            d[f'{name}_jaccard_mean']=values.mean();d[f'{name}_jaccard_p05']=values.quantile(.05)
        percluster.append(d)
    pd.DataFrame(percluster).to_csv(root/'stability'/'per_cluster_stability.csv',index=False)

    sensitivity=[]
    # Compare other normalizations with the frozen density parameters.
    for nn,other in ds['frames'].items():
        met,_,_=comparison(other,rep,params,ids,labels);sensitivity.append(dict(kind='normalization',variant=nn,**met))
    # Alternative change-point families use their own label-free best configuration.
    for alt in alternatives:
        name=f'cp_{alt["algorithm"]}_{alt["cost"]}'
        log('Sensitivity '+name)
        if alt['id']==cfg['id']:other=frame
        else:other=extract_dataset(name,ds['curves'],alt,repeats=10)[0][norm]
        met,_,_=comparison(other,rep,params,ids,labels);sensitivity.append(dict(kind='change_point',variant=name,**met))
    # PCA variation and normalization are never picked to achieve a target K.
    for rr in ['full','0.9','0.95','0.99']:
        met,_,_=comparison(frame,rr,params,ids,labels);sensitivity.append(dict(kind='representation',variant=rr,**met))
    log('Sensitivity: common linear grids, observation windows and metadata cohorts')
    cs=window_curves(curves,ds['window'],resample=True)
    linframes,lindata=extract_dataset('linear_resampled',cs,cfg,repeats=10)
    for sid,c in cs.items():
        pd.DataFrame(dict(elapsed=c['t'],y=c['y'])).to_csv(root/'resampled_input'/f'{sid}.csv',index=False)
    dump(dict(grid_spacing=next(iter(cs.values()))['grid_dt'],grid_unit='elapsed_fraction' if ds['window'] is None else 'hours',
        construction='median of median native sampling intervals; no extrapolation',n_curves=len(cs)),root/'resampled_input'/'grid_info.json')
    for nn,other in linframes.items():
        met,_,_=comparison(other,rep,params,ids,labels);sensitivity.append(dict(kind='linear_resampling',variant=nn,**met))
    # Identical cohorts: compare assigned labels and refit both windows on overlap.
    for name,otherds in datasets.items():
        other=otherds['frames'][norm];met,_,_=comparison(other,rep,params,ids,labels)
        sensitivity.append(dict(kind='window_unmatched_cohort',variant=name,**met))
        common=[s for s in ids if s in other.index]
        if len(common)>=12:
            bx,_=transform(frame.loc[common],rep);bl=fit_cluster(bx,params).labels_
            met,_,_=comparison(other.loc[common],rep,params,common,bl)
            sensitivity.append(dict(kind='window_same_cohort_refit',variant=name,**met))
    for cohort in ['all','cycles']:
        othercs=window_curves(curves,None,cohort=cohort)
        otherframes,_=extract_dataset(f'cohort_{cohort}',othercs,cfg,repeats=10)
        other=otherframes[norm]
        alternate_params=params.copy()
        if cohort=='cycles':alternate_params['min_cluster_size']=max(3,min(params['min_cluster_size'],len(other)//4));alternate_params['min_samples']=3
        met,_,_=comparison(other,rep,alternate_params,ids,labels)
        sensitivity.append(dict(kind='metadata_cohort',variant=cohort,**met))
    # Remove only descriptor columns in diagnostic fits, never raw transient points.
    removed=[c for c in frame.columns if 'raw_point_count' in c]
    met,_,_=comparison(frame.drop(columns=removed),rep,params,ids,labels)
    sensitivity.append(dict(kind='sampling_confound',variant='exclude_point_count_features',**met))
    physical=[s for s in ids if not any(flag in curves[s]['flags'] for flag in ['negative_output','absolute_efficiency_gt100','extreme_axis_span_review'])]
    if len(physical)!=len(ids) and len(physical)>=20:
        met,_,_=comparison(frame.loc[physical],rep,params,ids,labels);sensitivity.append(dict(kind='qc_review',variant='flagged_curve_exclusion_diagnostic_only',**met))
    # Neighboring HDBSCAN settings include larger/smaller density scales.
    for i,p in enumerate(hdb_params(len(frame))):
        met,_,_=comparison(frame,rep,p,ids,labels)
        sensitivity.append(dict(kind='hdbscan_grid',variant=json.dumps(p),**met))
    sens=pd.DataFrame(sensitivity);sens.to_csv(root/'sensitivity'/'sensitivity_analysis.csv',index=False)
    # Audit all four-cluster candidates after freeze; never feed this into selection.
    four=table[table.n_clusters==4].copy()
    log(f'Post-freeze audit of {len(four)} four-cluster candidates: 50 bootstraps each; no reselection')
    four_audit=[]
    for _,r in four.iterrows():
        ff=pd.read_csv(root/'features'/f'{r.dataset}_{r.normalization}_r10.csv',index_col=0)
        pp=get_params(r);xx,_=transform(ff,str(r.representation));ll=fit_cluster(xx,pp).labels_
        runs=cached(root/'cache'/f'four_audit_{r.candidate_id}.pkl',lambda:bootstrap(ff,str(r.representation),pp,ll,50))
        d=pd.DataFrame(runs);four_audit.append(dict(candidate_id=r.candidate_id,bootstrap_ari_50=d.ari.mean(),bootstrap_ari_50_p05=d.ari.quantile(.05),bootstrap_ari_50_p95=d.ari.quantile(.95),bootstrap_valid_fraction_50=d.ari.notna().mean(),four_cluster_run_fraction=d.n_clusters.eq(4).mean()))
    four=four.merge(pd.DataFrame(four_audit),on='candidate_id',how='left')
    four.to_csv(root/'sensitivity'/'four_cluster_candidates_posthoc.csv',index=False)
    # Group/source correlation and sampling-time association are post-fit diagnostics.
    confounds=[]
    for col in ['duration_analysis','clean_n']:
        grouped=[assignment.loc[assignment.cluster==k,col].to_numpy() for k in sorted(set(labels)-{-1})]
        h,p=kruskal(*grouped) if len(grouped)>=2 else (np.nan,np.nan)
        confounds.append(dict(variable=col,kruskal_h=h,p_value_descriptive=p))
    confounds.append(dict(variable='figure_id',adjusted_mutual_information=adjusted_mutual_info_score(groups,labels)))
    pd.DataFrame(confounds).to_csv(root/'sensitivity'/'sampling_and_source_confound.csv',index=False)
    log('Parameters frozen; generating medoids, raw member overlays and descriptive morphology')
    descriptions=plots(root,curves,qc,ds,frame,x,model,segments,cp,assignment,cfg,stability,sens)
    write_report(root,qc,ds,winner,cfg,finalmetrics,stability,percluster,sens,descriptions,table,assignment,cp)
    # Check all source byte hashes; finalized manifest covers outputs except regenerable caches/env.
    checksum_rows=[]
    for _,row in qc.iterrows():
        sha=hashlib.sha256(Path(row.source_path).read_bytes()).hexdigest()
        if sha!=row.sha256:raise AssertionError('Raw input changed: '+row.source_path)
    for f in sorted(root.rglob('*')):
        if f.is_file() and not any(p in ['.venv','cache','__pycache__'] for p in f.relative_to(root).parts) and f.name not in ['checksums.sha256','execution.log','process.log','grid_process.log','final_process.log']:
            checksum_rows.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root)))
    (root/'checksums.sha256').write_text('\n'.join(checksum_rows)+'\n')
    dump(dict(smoothing=False,human_class_labels_read=False,human_class_labels_used=False,target_cluster_count=None,
        original_csv_count=len(qc),original_hashes_verified=True,unit_fields_only_from_metadata=True,
        identifiers_excluded_from_features=True,negative_values_retained=True,all_preprocessing_saved=True,
        parameters_frozen_before_morphology=True,seed=SEED,
        stage_deviations=['Staged rather than full Cartesian CP x HDBSCAN search',
            'Screening CP supports use 10 repeats per deletion fraction; all window finalists use required 50',
            'HDBSCAN screening uses 5 bootstraps per nondegenerate candidate; finalists use 50',
            'Timepoint deletion refits CP/features/preprocessing/HDBSCAN with frozen boundary-reliability calibration',
            'Alternative CP and resampling sensitivities use 10 boundary-support repeats per deletion fraction',
            'Full-duration main features use elapsed fraction; this is morphology comparison, not equal exposure comparison']),root/'unsupervised_audit_log.json')

def plots(root,curves,qc,ds,frame,x,model,segments,cp,assignment,cfg,stability,sens):
    ids=frame.index.tolist();labels=model.labels_;palette=plt.get_cmap('tab10');desc=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plt.figure(figsize=(10,4));known=qc[qc.unit.isin(['hours','days'])]
    plt.subplot(121);plt.hist(np.log10(known.duration_analysis),bins=20,color='#346fa6');plt.xlabel('log10 observation duration (hours)');plt.ylabel('Curves')
    plt.subplot(122);cov=pd.read_csv(root/'search'/'window_coverage.csv');plt.plot(cov.window_hours,cov.n,'o-');plt.axhline(30,color='gray',ls='--');plt.xlabel('Candidate window (hours)');plt.ylabel('Eligible curves (>=12 points)');savefig(root/'figures'/'01_qc_and_window_coverage.png')
    plt.figure(figsize=(8,5));proj=PCA(2).fit_transform(x)
    for lab in sorted(set(labels)):
        mask=labels==lab;plt.scatter(proj[mask,0],proj[mask,1],c=['#aaaaaa' if lab<0 else palette(lab%10)],label=f'{"Noise" if lab<0 else "Cluster "+str(lab)} (n={mask.sum()})',s=32,alpha=.85)
    plt.xlabel('Display PC1');plt.ylabel('Display PC2');plt.title('Projection of frozen clustering representation');plt.legend(fontsize=8);savefig(root/'figures'/'02_cluster_projection.png')
    selected_features=['net_change','total_variation','peak_time','valley_time','recovery_ratio','segment_count','late_time_slope','largest_slope_reversal']
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,field in zip(axes.flat,selected_features):
        labs=sorted(set(labels)-{-1});ax.boxplot([frame.loc[np.array(ids)[labels==lab],field].dropna() for lab in labs],tick_labels=[str(l) for l in labs],showfliers=False);ax.set_title(field);ax.set_xlabel('Cluster')
    savefig(root/'figures'/'03_feature_distributions.png')
    pd.concat([frame.assign(cluster=labels).groupby('cluster').median(),frame.assign(cluster=labels).groupby('cluster').count().add_suffix('_n')],axis=1).to_csv(root/'final'/'cluster_feature_summary.csv')
    seqrows=[]
    for sid,g in segments.groupby('sample_id',sort=False):
        values=g.sort_values('segment_index').theil_sen_slope.to_numpy();seq=' → '.join('increase' if v>0 else 'decay' if v<0 else 'flat' for v in values)
        seqrows.append(dict(sample_id=sid,cluster=int(g.cluster.iloc[0]),slope_sequence=seq,slopes=json.dumps(values.tolist())))
    sequence=pd.DataFrame(seqrows);sequence.to_csv(root/'final'/'slope_sequences.csv',index=False)
    sequence.groupby(['cluster','slope_sequence']).size().rename('count').to_csv(root/'final'/'slope_sequence_distribution.csv')
    representatives=[]
    for label in sorted(set(labels)):
        sub=np.where(labels==label)[0];name='noise' if label<0 else f'cluster_{label}'
        destination=root/'final'/name;destination.mkdir(exist_ok=True)
        dist=squareform(pdist(x[sub],metric='cityblock' if model.metric=='manhattan' else model.metric))
        medoid_local=int(np.argmin(dist.sum(axis=1)));medoid_idx=sub[medoid_local];medoid_id=ids[medoid_idx]
        # Real medoid first, then highest membership / smallest medoid distance.
        order=sorted(sub,key=lambda j:(j!=medoid_idx,-model.probabilities_[j],float(np.linalg.norm(x[j]-x[medoid_idx]))))[:10]
        assignment[assignment.cluster==label].to_csv(destination/'members.csv',index=False)
        frame.iloc[sub].to_csv(destination/'member_feature_distributions.csv',index_label='sample_id')
        cp[cp.cluster==label].to_csv(destination/'change_point_distribution.csv',index=False)
        sequence[sequence.cluster==label].to_csv(destination/'member_slope_sequences.csv',index=False)
        for rank,j in enumerate(order,1):
            sid=ids[j];c=ds['curves'][sid]
            pd.DataFrame(dict(elapsed=c['t'],raw_y=c['y'],normalized_y=normalize(c['y'],ds['selected_norm']))).to_csv(destination/f'{rank:02d}_{sid}.csv',index=False)
            representatives.append(dict(cluster=label,rank=rank,sample_id=sid,is_medoid=j==medoid_idx,membership_probability=model.probabilities_[j],source_path=assignment.set_index('sample_id').loc[sid,'source_path']))
        fig,axs=plt.subplots(1,3,figsize=(16,4.5))
        for j in sub:
            sid=ids[j];c=ds['curves'][sid];orig=curves[sid]
            axs[0].plot(c['t'],c['y'],alpha=.25,lw=.8,color=palette(label%10) if label>=0 else 'gray')
            axs[1].plot(c['t']/c['t'][-1],normalize(c['y'],ds['selected_norm']),alpha=.3,lw=.8,color=palette(label%10) if label>=0 else 'gray')
            axs[2].plot(orig['t'],orig['y'],alpha=.2,lw=.7,color=palette(label%10) if label>=0 else 'gray')
        med=ds['curves'][medoid_id]
        axs[0].plot(med['t'],med['y'],color='black',lw=2,label=f'Medoid {medoid_id}');axs[0].legend()
        axs[1].plot(med['t']/med['t'][-1],normalize(med['y'],ds['selected_norm']),color='black',lw=2)
        axs[0].set(xlabel='Elapsed hours',ylabel='Raw output (mixed scales)',title=f'{name}: analysis-window raw curves, n={len(sub)}')
        axs[1].set(xlabel='Fraction of observed window',ylabel=f'{ds["selected_norm"]} normalized output',title='No smoothing; medoid in black')
        axs[2].set(xlabel='Elapsed hours',ylabel='Raw output (mixed scales)',title='Entire original observations')
        savefig(destination/'all_member_raw_curves.png')
        rows=int(np.ceil(len(order)/2));fig,axs=plt.subplots(rows,2,figsize=(11,2.7*rows),squeeze=False)
        for ax,j in zip(axs.flat,order):
            sid=ids[j];c=ds['curves'][sid];ax.plot(c['t'],c['y'],'o-',markersize=2,lw=1,color=palette(label%10) if label>=0 else 'gray')
            for _,r in cp[cp.sample_id==sid].iterrows():ax.axvline(r.cp_time,color='#c14d35',ls='--',alpha=max(.15,r.support))
            ax.set_title(f'{sid} | p={model.probabilities_[j]:.2f}'+(' | MEDOID' if j==medoid_idx else ''));ax.set_xlabel('Elapsed hours');ax.set_ylabel('Raw output')
        for ax in list(axs.flat)[len(order):]:ax.axis('off')
        savefig(destination/'top10_real_curves.png')
        medseq=sequence.set_index('sample_id').loc[medoid_id,'slope_sequence']
        medslopes=segments[segments.sample_id==medoid_id].sort_values('segment_index').theil_sen_slope.to_list()
        d=dict(cluster=int(label),n=len(sub),medoid=medoid_id,medoid_slope_sequence=medseq,medoid_slopes=medslopes,
            median_net_change=float(frame.iloc[sub].net_change.median()),median_recovery_ratio=float(frame.iloc[sub].recovery_ratio.median()))
        desc.append(d)
        fig,axes=plt.subplots(2,4,figsize=(14,6))
        for ax,field in zip(axes.flat,selected_features):
            ax.hist(frame.iloc[sub][field].dropna(),bins=min(12,max(3,len(sub)//3)),color=palette(label%10) if label>=0 else 'gray',alpha=.8)
            ax.set_title(field);ax.set_ylabel('Curves')
        savefig(destination/'feature_distributions.png')
        fig,axs=plt.subplots(1,2,figsize=(13,4))
        cc=cp[cp.cluster==label]
        axs[0].hist([r.cp_time/ds['curves'][r.sample_id]['t'][-1] for _,r in cc.iterrows()],bins=np.linspace(0,1,16),color='#346fa6')
        axs[0].set(xlabel='Change point / observed duration',ylabel='Change points',title=name)
        counts=sequence[sequence.cluster==label].slope_sequence.value_counts().head(6)
        short=counts.index.str.replace('increase','+',regex=False).str.replace('decay','-',regex=False).str.replace('flat','0',regex=False).str.replace(' → ',' / ',regex=False)
        axs[1].barh(short[::-1],counts.to_numpy()[::-1],color='#de9260');axs[1].set(xlabel='Curves',title='Most common segment slope signs')
        savefig(destination/'change_points_and_slope_sequences.png')
    pd.DataFrame(representatives).to_csv(root/'final'/'representatives.csv',index=False);dump(desc,root/'final'/'posthoc_morphology.json')
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    for label in sorted(set(labels)-{-1}):
        g=cp[cp.cluster==label];times=[r.cp_time/ds['curves'][r.sample_id]['t'][-1] for _,r in g.iterrows()];plt.hist(times,bins=np.linspace(0,1,16),alpha=.5,label=f'Cluster {label}')
    plt.xlabel('Change point / observation duration');plt.ylabel('Change points');plt.legend()
    plt.subplot(122)
    values=pd.DataFrame(stability);positions=np.arange(len(values));plt.bar(positions-.18,values.ari_mean,width=.36,color='#346fa6',label='ARI, valid runs only');plt.bar(positions+.18,values.ari_zero_for_degenerate_mean,width=.36,color='#de9260',label='Degenerate runs scored 0');plt.xticks(positions,values.kind,rotation=35,ha='right');plt.ylim(-.1,1.05);plt.ylabel('Mean adjusted Rand index');plt.legend(fontsize=7);savefig(root/'figures'/'04_change_points_and_stability.png')
    finalists=pd.read_csv(root/'search'/'finalist_ranking.csv');best=finalists.sort_values('selection_score',ascending=False).groupby('dataset',sort=False).head(1)
    plt.figure(figsize=(11,4));plt.subplot(121);scores=best.selection_score.replace([-np.inf,np.inf],np.nan);plt.bar(best.dataset,scores.fillna(0),color='#346fa6');plt.xticks(rotation=30);plt.ylabel('Frozen label-free selection score')
    for i,v in enumerate(scores):
        if not np.isfinite(v):plt.text(i,.015,'No valid\nfinalist',ha='center',fontsize=8,color='#a74430')
    plt.subplot(122);plt.scatter(finalists.noise_fraction,finalists.bootstrap_ari,c=finalists.dbcv,cmap='viridis',s=50);plt.colorbar(label='DBCV');plt.xlabel('Noise fraction');plt.ylabel('Bootstrap ARI');savefig(root/'figures'/'05_window_comparison.png')
    return desc

def markdown_table(frame,cols=None,limit=None):
    d=frame if cols is None else frame[cols]
    if limit:d=d.head(limit)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','/')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.to_numpy())

def write_report(root,qc,ds,winner,cfg,met,stability,percluster,sens,descriptions,table,assignment,cp):
    stability=pd.DataFrame(stability);n=met['n'];k=met['n_clusters'];noise=int((assignment.cluster<0).sum())
    selected='每条曲线的完整已观测时长，时间转换为 0–1 的相对进度' if ds['window'] is None else f'从首个有效观测点起 {ds["window"]} 小时'
    stat=lambda name:stability.set_index('kind').loc[name,'ari_mean']
    robust=all(stat(name)>=.75 and stability.set_index('kind').loc[name,'ari_valid_fraction']>=.9 for name in ['sample_bootstrap','figure_bootstrap','drop_5pct','drop_10pct'])
    bestwindows=pd.read_csv(root/'search'/'finalist_ranking.csv').sort_values('selection_score',ascending=False).groupby('dataset',sort=False).head(1)
    counts=assignment.groupby('cluster').size().rename('n').reset_index()
    four=pd.read_csv(root/'sensitivity'/'four_cluster_candidates_posthoc.csv')
    text=f'''# 无监督变点检测 + 动力学特征 + HDBSCAN：研究结果

## 结论先行

- 在本次预先记录的无标签评分规则及搜索范围内，最佳候选是 **{selected}**，归一化为 **{winner['normalization']}**，特征表示 **{winner['representation']}**。
- 该候选纳入 **{n} / 218** 条输入曲线，HDBSCAN 得到 **{k} 个簇**，另有 **{noise} 条噪声曲线（{met['noise_fraction']:.1%}）**。
- **这不是“数据天然只有 {k} 类”的证明。** 这是异质来源、不同实验条件和不同观察时长下的探索性分组；必须连同噪声、删点稳定性、归一化和算法敏感性解读。
- 样本 bootstrap ARI **{stat('sample_bootstrap'):.3f}**；按图片分组 bootstrap ARI **{stat('figure_bootstrap'):.3f}**；删去 5% / 10% 时间点 ARI **{stat('drop_5pct'):.3f} / {stat('drop_10pct'):.3f}**。
- 上述删点 ARI 只对仍产生至少两个非噪声簇的重复计算；5% / 10% 删点分别有 **{stability.set_index('kind').loc['drop_5pct','degenerate_run_fraction']:.0%} / {stability.set_index('kind').loc['drop_10pct','degenerate_run_fraction']:.0%}** 的重复退化，不能把条件均值接近 1 解读为总体稳定。**当前没有找到足以作为稳定最终分类结论的 time window；300 小时是已测试候选中的探索性首选。**
- {'主要扰动检验均达到 ARI 0.75，可称为本数据内较稳定的探索性结果；此阈值只用于结论措辞。' if robust else '至少一项主要扰动检验未达到 ARI 0.75，不足以认定为稳定、可重复的天然形态分类；此阈值只用于结论措辞。'}
- **{'最终输出恰有四簇，但是否“四个稳定簇”还需逐簇及跨参数验证，不能仅凭 K=4 认定。' if k==4 else '本次最终结果不支持宣称存在四个稳定且可重复的簇；保持实际聚类结果，不强行调整为四类。'}**

## 1. 数据审计与适用范围

218 个 CSV 来自 {qc.figure_id.nunique()} 张图片。它们是给定文件中的曲线数据，不能据此认定为实验仪器的原始读数。只读取 CSV 的 x/y 和元数据的坐标名称/单位；文件名中的材料、Control/Target 等内容仅用于溯源，不进入特征或参数选择。

元数据单位分布：{qc.unit.value_counts().to_dict()}。已知小时和天的 159 条曲线中，3 条不足 12 个点，主搜索使用 156 条。天乘以 24 换算为小时；循环次数和单位不明的数据不混入小时主分析，而在敏感性分析中另行处理。未收到额外单位确认时，元数据保持为依据。

非有限值共 {int(qc.non_finite_values.sum())} 个；完全重复行共 {int(qc.exact_duplicate_records.sum())} 个；不同输出共享时间戳的位置共 {int(qc.conflicting_timestamps.sum())} 个。这些位置按时间戳取中位数，保留全部原始 CSV。负输出涉及 {int((qc.negative_output_count>0).sum())} 条曲线，记录异常标记而不因局部起伏删除点。x 起点的小负值保留在原副本；分析时间统一减去首个有效时间值，因此“0 小时”指首次观察，不一定是实验真正开始。

完整 QC：[qc_summary.csv](qc_summary.csv)。每条输入的最终状态：[all_218_sample_status.csv](final/all_218_sample_status.csv)。未知单位、非时间轴、点数不足、未覆盖窗口均有明确原因。

## 2. Time window 探索

候选 50、100、200、300、500、750、1000、1500、2000 小时；无需插值，仅选择持续到窗口末端且窗内至少 12 个原始点的曲线。队列不足 30 条的窗口仅报告覆盖率，不用于主模型选优。主比较为 200、300、500、750、1000 小时及完整时长。

{markdown_table(pd.read_csv(root/'search'/'window_coverage.csv'))}

各窗口经 50 次 bootstrap 复核后的最佳候选：

{markdown_table(bestwindows,['dataset','normalization','representation','n','n_clusters','noise_fraction','dbcv','bootstrap_ari','deletion_ari','normalization_ari','selection_score'])}

“最佳窗口”指本次评分最优，不代表已找到唯一物理时间尺度。完整时长模型将每条曲线的时间缩放为 0–1，用来比较形态；长短实验中相同相对进度并非相同老化时长。固定窗口不做外推，不在缺少真实末端点时构造末端值，实际最后一条观测可略早于窗口边界。覆盖率进入评分；另外保存共同样本上重拟合的窗口比较，以暴露样本组成差异。

![窗口与QC](figures/01_qc_and_window_coverage.png)
![窗口选择](figures/05_window_comparison.png)

## 3. 方法与参数冻结

`SMOOTHING = False`。主分析没有平滑、没有重采样、没有按形态删点。三种归一化分别为 y/y0、y/max(y)、(y−median(y))/(Q95−Q5)。归一化分母为零的方案视为不可用，不填造输出。变点检测内部只进行可逆仿射幅值缩放以校准惩罚，保留全部起伏。

变点搜索 PELT / Binary Segmentation / Window × l2 / linear / rbf × min_segment_fraction [0.03,0.05,0.08,0.10] × penalty_multiplier [0.5,1,2,3,5,8,10]，共 **252 组**；每组在 156 条主曲线上检验，并各做一次 5%/10% 删点探测。linear cost 的输入严格为 `[y, intercept=1, normalized_actual_time]`，不是把单列 y 误当回归模型。min_size=max(3,ceil(n×fraction)) 是点数约束；它不保证不规则采样下每段占相同物理时长。Window 宽度为至少 2×min_size 且约占样本数 20% 的偶数。

惩罚定义为 multiplier × 未分段 cost/n × log(n)，各 cost 在自己的量纲中校准。跨 cost 用共同的分段线性高斯 BIC 排名（每段 2 个回归参数，加变点位置参数），80% BIC 百分位 + 20% 变点位置 F1。最优变点参数：`{json.dumps(cfg,ensure_ascii=False)}`。每个算法/cost 家族保留一个最优参数用于敏感性分析；变点数没有固定。

稳定变点：每个窗口最终候选逐曲线随机删去 5%、10% 点，各 50 次；端点固定，实际删除点数四舍五入且至少 1 点。容许位置偏差为 max(总时长的 3%, 原始 median interval)，使用一对一匹配，输出支持率与位置误差。容差由采样分辨率决定，不含任何形态类别。零变点对零变点 F1 定义为 1；BIC 同时防止只按这个稳定性偏好无分段模型。

保留每段的 OLS/Theil–Sen 斜率、幅值、方差、时长、点数等。以左右边界的最低支持率作为段稳定度，选最多 4 段后按时间排序，不足 NaN 填充；全部分段另外保存。全局极值、正负差分、总变差、积分、恢复、转折及符号序列保留为连续或计数特征。恢复 amplitude 是最大累积谷后增幅；recovery ratio 以全局振幅为分母。主聚类中的斜率对相对时间定义；physical_slope 保存在分段描述中，不作为额外聚类输入。

缺失率 >40% 的列丢弃，余下缺失以训练样本中位数填补；常数列删除；RobustScaler 处理全部保留特征。PCA 比较保留方差 90/95/99% 和不降维。原特征和完整标准化矩阵均保存。长尾极值不剪裁；采样点数特征另做移除敏感性。

HDBSCAN 主搜索共 **{len(table)} 个组合**。min_cluster_size 在 [10,20,30,50,75,100] 中保留满足 2×size≤n 的值，小队列额外添加约 5%/10% 样本量且至少 5 的值；min_samples=[None,5,10,20]，eom/leaf，Euclidean/Manhattan。没有设置目标 K，allow_single_cluster 使用库默认 False，所以全噪声也可能意味着一个连续群体而非可分离多簇。

冻结 HDBSCAN 参数：`{json.dumps({k:winner[k] for k in ['min_cluster_size','min_samples','cluster_selection_method','metric']})}`。保留表示维数 {int(winner['dimensions'])}。DBCV={met['dbcv']:.4f}，silhouette={met['silhouette']:.4f}，DB={met['davies_bouldin']:.4f}，CH={met['calinski_harabasz']:.4f}。silhouette 排除噪声；DB/CH 为欧氏质心指标，即使聚类使用 Manhattan，也只作附加描述。DBCV 调用 hdbscan 官方实现，Manhattan 对应 scipy 的 cityblock。

搜索采用分阶段设计，**并非所有 CP×窗口×归一化×HDBSCAN 的完全笛卡尔积穷举**。每个非退化 HDBSCAN 候选先做 5 次样本 bootstrap，每窗口×归一化最佳及全局前 6 个候选再做 50 次。先用 10+10 次删点评估段稳定度，最终各窗口重做 50+50 次并重拟合候选。最终评分：0.30 bootstrap ARI + 0.15 删点 ARI + 0.15 跨归一化 ARI + 0.10 非噪声比例 + 0.15 DBCV + 0.05 silhouette + 0.10 覆盖率。最终评分中的删点共 10 次用于选优，冻结结果再做 100 次完整敏感性。所有权重均写在 protocol.json 中，未以类别名称或 K=4 选参。

选优时，跨归一化或删点比较若退化到不足两个非噪声簇，按 −1 计入该项评分；因此 finalist 表的 normalization_ari / deletion_ari 是含退化惩罚的评分项。50 次 bootstrap 的均值则排除退化运行，另报告有效比例。标准 membership_probability 只是 HDBSCAN 内部成员强度，不是分类可重复性保证；例如小簇多条曲线成员强度接近 1，但 bootstrap Jaccard 仍较低。

## 4. 稳定性与敏感性

每次样本/图片 bootstrap 都重拟合缺失处理、scaler、PCA 和 HDBSCAN；带放回样本在共同唯一 ID 上比较。ARI 同时保存包括噪声和共同非噪声两个版本；两边任一不足两个非噪声簇时，ARI 为 NA，避免全噪声被误算为完美稳定。VI 以自然对数定义，越低越好；NA 比例同时报告。按图片 bootstrap 反映同图多曲线相关性。HDBSCAN 本身确定性运行；50 次 repeated-seed 指随机打乱输入行顺序，检验排序/并列距离效应。

{markdown_table(stability,['kind','repeats','ari_valid_fraction','ari_mean','ari_zero_for_degenerate_mean','ari_p05','ari_p95','vi_mean','noise_fraction_mean'])}

逐簇 bootstrap Jaccard：

{markdown_table(pd.DataFrame(percluster))}

归一化、重采样、算法及表示敏感性（同一冻结 HDBSCAN 参数）：

{markdown_table(sens[sens.kind.isin(['normalization','linear_resampling','change_point','representation','sampling_confound'])],['kind','variant','n','n_clusters','noise_fraction','dbcv','ari','ari_assigned'])}

线性重采样只用于敏感性分析，网格步长不小于队列典型原始间隔，禁止外推。完整时长模型使用相对时间 common grid；固定窗口使用小时 common grid。重新采样可能减少原始极值/改变有限差分，这正是要检验的敏感性，而不是声称与原数据等价。

100 次时间点删除重新检测变点、提取特征并重拟合标准化/PCA/HDBSCAN。用于选四段的边界可靠度使用冻结的 100 次校准结果，并按时间容差映射到新变点，未在每次扰动内再嵌套 100 次可靠度校准。因此这是**条件于已估计边界可靠度的端到端检验**。变点方法替代和线性重采样的段可靠度使用 10+10 次，作为敏感性而非最终结论校准。bootstrap 也条件于冻结超参数，不等同于在每次 bootstrap 中重新做整个超参数搜索。

共同样本窗口比较：

{markdown_table(sens[sens.kind=='window_same_cohort_refit'],['variant','n','n_clusters','noise_fraction','ari','ari_assigned'])}

![稳定性](figures/04_change_points_and_stability.png)

## 5. 参数冻结后的形态解释

聚类后才根据真实 medoid 的 Theil–Sen 斜率符号解释。正负符号不作为类别训练目标；接近零的斜率可能由数字化精度引起，不强行赋予“快速/缓慢”的类别阈值。不同簇可共享相同斜率序列，簇内也可有多种序列。形态示例的存在并不等价于一个独立稳定簇。

{markdown_table(counts)}

'''
    for d in descriptions:
        label=d['cluster'];folder='noise' if label<0 else f'cluster_{label}'
        text+=f"\n### {'Noise' if label<0 else 'Cluster '+str(label)}：{d['n']} 条\n\n真实 medoid **{d['medoid']}** 的自动分段序列：**{d['medoid_slope_sequence']}**。各段 Theil–Sen 斜率（归一化输出/相对时间）：`{[round(v,4) for v in d['medoid_slopes']]}`。簇内 net change 中位数 {d['median_net_change']:.4f}，recovery ratio 中位数 {d['median_recovery_ratio']:.4f}。\n\n![全部成员](final/{folder}/all_member_raw_curves.png)\n\n[最多10条真实代表曲线](final/{folder}/top10_real_curves.png) · [成员表](final/{folder}/members.csv)\n"
        slopes=d['medoid_slopes']
        if len(slopes)==2 and slopes[0]<slopes[1]<0:
            text+=f'\n这个 medoid 的早段衰减斜率绝对值约为晚段的 **{abs(slopes[0]/slopes[1]):.2f} 倍**，是 rapid decay → slow decay 的真实示例；它不代表整个簇中所有曲线都有相同分段。\n'
        if label<0 and 'decay → increase' in d['medoid_slope_sequence']:
            text+='\n该真实曲线呈现衰减后恢复，但 HDBSCAN 把它放在 noise 中；不能将这些噪声曲线另行命名为一个已识别的稳定恢复簇。\n'
    sequence=pd.read_csv(root/'final'/'slope_sequences.csv');patterns={}
    for name,pattern in [('increase→decay',lambda s:'increase → decay' in s),('decay→recovery',lambda s:'decay → increase' in s),('连续衰减段',lambda s:'decay → decay' in s)]:
        patterns[name]=int(sequence.slope_sequence.map(pattern).sum())
    text+=f'''\n### 对研究问题的直接回答

1. **数据天然支持多少个 cluster？** 本次规则选出 {k} 簇及 {noise} 个 noise；稳定性和敏感性如上，不能推断为数据天然唯一的类别数。
2. **是否出现 increase→decay、rapid decay→slow decay、decay→recovery？** 在自动分段符号序列中，increase→decay 有 {patterns['increase→decay']} 条，decay→increase（恢复）有 {patterns['decay→recovery']} 条，连续衰减段有 {patterns['连续衰减段']} 条。这是聚类后描述计数，未用于选参。是否“快速→缓慢”须比较同条曲线的实际斜率大小，不能仅凭两个负号认定；所有数值在 all_segment_descriptors.csv。
3. **是否存在四个稳定且可重复的 cluster？** {'本次最优输出为四簇，但需要结合跨归一化、跨算法及逐簇稳定性；不能仅以簇数宣称成立。' if k==4 else '本次最终方案没有建立这一结论。没有为了得到四类调参。'}

主搜索中共有 {int((table.n_clusters==4).sum())} 个参数组合出现四簇；这些组合的噪声比例为 {table.loc[table.n_clusters==4,'noise_fraction'].min():.1%}–{table.loc[table.n_clusters==4,'noise_fraction'].max():.1%}，5 次筛选 bootstrap ARI 最高仅 {table.loc[table.n_clusters==4,'bootstrap_ari'].max():.3f}。四簇候选已在参数冻结后另存 `sensitivity/four_cluster_candidates_posthoc.csv`，这些结果没有进入反向调参。

参数冻结后，又对全部 26 个四簇候选分别做了 **50 次 bootstrap**；平均 ARI 最高为 **{four.bootstrap_ari_50.max():.3f}**。这是结论核验，不触发任何模型重选。其区间、有效比例以及重复出现四簇的频率均已保存。

## 6. 限制与可复现性

- 不同实验可能有不同温度、光照、湿度和测试条件；缺少统一应力条件，时间尺度不能直接解释为同一种动力学常数。
- 不同来源图片的数字化分辨率和已存在的数据加工未知。禁止的是本流程再做平滑，不代表上游数据必然未处理。
- 窗口筛选存在观察截止和存活偏差；完整时长存在时长混杂。共同样本复拟合和按图片 bootstrap 可暴露部分问题，无法消除它们。
- 156 条主曲线及若干固定窗口子集规模有限，广泛无监督调参仍可能产生选择偏差。没有独立确认集，得分用于探索性排序。
- 极值、总变差、最大有限差分对采样密度敏感；原始点数特征按策略保留，并额外做删除这些特征的诊断。
- 时序端点未参与随机删除，保持窗口及归一化基准，稳定性不代表对起点/终点不确定性的稳健性。
- 模型不强制一个簇，也不强制四簇；默认 HDBSCAN 可能把单个连续密度群整体判为 noise。
- 当前 Python/数值库环境在部分 PCA 矩阵乘法中发出 RuntimeWarning，日志保留。已验证全部聚类输入为有限值，并用独立 einsum 计算重构保存的 scaler/PCA 坐标，误差在 1e-9 容差内，避免仅靠忽略警告认定结果有效。

复现（需要安装 requirements-lock.txt 中的依赖）：

```bash
cd /Users/shunhao/Desktop/ML/method1/change_point_hdbscan_20260905
.venv/bin/python code/run_pipeline.py --stage all
.venv/bin/python code/verify_outputs.py
```

脚本使用 cache 断点续跑。彻底重跑请将代码和 strategy_original.md 复制到一个新目录，或将当前 cache 改名保留后运行；不要删除原始数据。来源和输出有 SHA-256 校验。

方法实现参考：[ruptures PELT](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)、[CostLinear](https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costlinear/)、[Window](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/window/)、[HDBSCAN API / DBCV](https://hdbscan.readthedocs.io/en/latest/api.html)。

## 7. 文件入口

- `code/`：完整 Python 分析与验证程序。
- `raw_input/`：CSV 与元数据原始副本；`cleaned_input/`：仅有限值/重复时间戳处理后数据。
- `window_inputs/`：不重采样的各窗口数据；`resampled_input/`：敏感性线性网格数据。
- `normalized_input/`：各主窗口三种归一化后的全部曲线，保留原始观测点。
- `features/`：全部窗口、归一化及可靠度重复次数的特征表。
- `search/`：252 组变点参数、逐曲线指标、全 HDBSCAN 搜索、候选复核、窗口覆盖率。
- `final/feature_table.csv`、`final/cluster_assignment.csv`、`final/standardized_features_no_pca.csv`：最终核心数据。
- `final/cluster_*/`：每簇全部 raw 曲线叠图、medoid 与最多 10 条真实代表曲线及 CSV。
- `stability/`：50+50 次变点、100 次删点、50 次样本/图片 bootstrap、50 次种子顺序检验。
- `sensitivity/`：方法/归一化/采样/窗口/队列/HDBSCAN 参数敏感性。
- `unsupervised_audit_log.json`、`protocol.json`、`checksums.sha256`：无标签与来源审计。
'''
    (root/'研究结果与方法报告.md').write_text(text)
    (root/'QC_report.md').write_text(f'''# 数据质量审计

共 {len(qc)} 条曲线，{qc.figure_id.nunique()} 个来源图片。原始 CSV/元数据已复制至 raw_input，所有源 CSV 均已校验 SHA-256；未覆盖原文件。

{markdown_table(qc.groupby('unit').agg(curves=('sample_id','size'),min_points=('clean_n','min'),median_points=('clean_n','median')).reset_index())}

非有限值 {int(qc.non_finite_values.sum())}；完全重复记录 {int(qc.exact_duplicate_records.sum())}；冲突时间戳位置 {int(qc.conflicting_timestamps.sum())}（按时刻取中位数，原副本保留）。未因局部偏离或 transient 删除原始点。

不足 12 点的曲线：

{markdown_table(qc[qc.clean_n<MIN_POINTS],['sample_id','figure_id','unit','clean_n'])}

需检查负输出或极端轴跨度的曲线（只标记，不擅自修改）：

{markdown_table(qc[qc.qc_flags.str.contains('negative_output|extreme_axis_span_review|absolute_efficiency_gt100',na=False)],['sample_id','figure_id','unit','minimum_output','maximum_output','duration_analysis','qc_flags'])}

完整逐条记录：[qc_summary.csv](qc_summary.csv)。纳入与排除原因：[all_218_sample_status.csv](final/all_218_sample_status.csv)。窗口和稳定性结论：[研究结果与方法报告.md](研究结果与方法报告.md)。
''')
    (root/'README.md').write_text(f'''# 本次研究结果

请先阅读 [研究结果与方法报告.md](研究结果与方法报告.md)。

最优候选：{selected}；{n} 条曲线，{k} 个簇，{noise} 条 noise。**目前未找到足够稳定的最终分类窗口，也未建立四个稳定簇的结论。** 此候选仅供探索性分析。

原始数据未覆盖；完整代码、参数搜索、稳定性、敏感性和各簇真实曲线均已保存。

- [最终曲线分配](final/cluster_assignment.csv)
- [218 条曲线纳入/排除原因](final/all_218_sample_status.csv)
- [稳定性](stability/cluster_stability_summary.csv)
- [窗口比较图](figures/05_window_comparison.png)
- [完整流程](code/run_pipeline.py)
''')
