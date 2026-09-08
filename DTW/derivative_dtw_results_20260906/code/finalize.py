#!/usr/bin/env python3
import run_analysis as m
from run_analysis import ROOT,SEED,REPS,NORMS,RADII,savecsv,log,norm,make_seq,distance,cluster,metrics,warp_audit,path
import pickle,json,hashlib,shutil,sys
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram,leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
with (ROOT/'code/state.pkl').open('rb') as f:records,a,W,cov,df,labels,Ds,rank,best=pickle.load(f)
best=best.copy();cs=W[best.window]['curves'];base=labels[best.candidate];D=Ds[best.distance_key];ids=[c['id'] for c in cs];idmap={s:i for i,s in enumerate(ids)}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180})
colors=['#2563a6','#df7932','#4c956c','#a75bb2'];sensitivity=[]

def figsave(fig,name):
 fig.savefig(ROOT/f'figures/{name}.png',bbox_inches='tight');fig.savefig(ROOT/f'figures/{name}.svg',bbox_inches='tight');plt.close(fig)
def sens(name,newcs,no=None,mo=None,lam=None,rad=None,rms=True,three=False):
 no=no or best.normalization;mo=mo or best['mode'];lam=best.lambda_value if lam is None else lam;rad=best.radius if rad is None else rad
 dd=distance(make_seq(newcs,no,mo,lam,three),rad,rms);l=cluster(dd,best.linkage,best.K);pairs=[(j,idmap[c['id']]) for j,c in enumerate(newcs) if c['id'] in idmap]
 if len(pairs)>=3:
  jj,ii=zip(*pairs);ari=adjusted_rand_score(base[list(ii)],l[list(jj)])
 else:ari=np.nan
 sensitivity.append(dict(analysis=name,n=len(newcs),n_overlap=len(pairs),ARI_to_frozen=ari,normalization=no,mode=mo,lambda_value=lam,radius=rad,rms_distance=rms,**metrics(dd,l)))
 savecsv(pd.DataFrame({'id':[c['id'] for c in newcs],'cluster':l+1}),f'clustering_results/sensitivity_{name}_labels.csv')
 return dd,l

log('Frozen-model sensitivities')
for no in NORMS:sens('normalization_'+no,cs,no=no)
for mo in ['classic','derivative']:sens('mode_'+mo,cs,mo=mo,lam=0 if mo=='classic' else 1)
for lam in [.25,.5,1,2,4]:sens('lambda_'+str(lam),cs,lam=lam)
for rad in RADII:sens('radius_'+str(rad),cs,rad=rad)
sens('conventional_sqrt_sum_DTW',cs,rms=False)
sens('three_point_local_linear_slope',cs,three=True)
# Common grid is a secondary control, never the primary representation.
typical_dt=float(np.median([np.median(np.diff(c['t'])) for c in cs]));ngrid=int(np.floor(best.T_h/typical_dt))+1;grid=np.linspace(0,best.T_h,ngrid)
gridcs=[{**c,'t':grid.copy(),'y':np.interp(grid,c['t'],c['y'])} for c in cs]
sens('linear_common_grid',gridcs)
# Absolute clock: use the latest first observation as common start; no extrapolation.
raw={c['id']:c for c in records};start=max(0,max(raw[c['id']]['origin_h'] for c in cs));abscs=[]
for c in cs:
 r=raw[c['id']];tt=r['absolute_t'];inside=(tt>=start)&(tt<=best.T_h)
 if inside.sum()<6 or tt[-1]<best.T_h:continue
 ts=np.unique(np.r_[start,tt[inside],best.T_h]);abscs.append({**r,'t':ts,'y':np.interp(ts,tt,r['y'])})
sens('absolute_time_common_interval',abscs)
# Conflicting duplicate records: first versus last observation, explicitly sensitivity only.
for keep in ['first','last']:
 extra=[]
 for _,row in a[a.reasons=='conflicting_duplicate_timestamps'].iterrows():
  dd=pd.read_csv(m.SOURCE/row.source).drop_duplicates('x',keep=keep).sort_values('x');t=dd.x.to_numpy()*row.hour_factor;y=dd.y.to_numpy();t=t-t[0];inside=t<=best.T_h
  if t[-1]<best.T_h or inside.sum()<6:continue
  ts=np.unique(np.r_[t[inside],best.T_h]);extra.append(dict(id=row.id,group=row.group,t=ts,y=np.interp(ts,t,y)))
 sens('conflicting_duplicates_keep_'+keep,cs+extra)
# Revisit every time window at the SAME frozen parameters and K, with overlap ARI.
for wid,w in W.items():sens('window_fixed_'+wid,w['curves'])
sensdf=pd.DataFrame(sensitivity);savecsv(sensdf,'stability/frozen_model_sensitivity.csv')

log('Source-figure bootstrap')
groups=sorted(set(c['group'] for c in cs));rng=np.random.default_rng(SEED+405);num=np.zeros_like(D);den=np.zeros_like(D);group_rows=[]
for rep in range(REPS):
 drawgroups=rng.choice(groups,size=len(groups),replace=True);draw=np.concatenate([np.array([i for i,c in enumerate(cs) if c['group']==g]) for g in drawgroups]);l=cluster(D[np.ix_(draw,draw)],best.linkage,best.K);u,first=np.unique(draw,return_index=True);lb=l[first];ari=adjusted_rand_score(base[u],lb);num[np.ix_(u,u)]+=lb[:,None]==lb[None,:];den[np.ix_(u,u)]+=1;group_rows.append(dict(replicate=rep,ARI=ari,n_unique=len(u),groups_unique=len(set(drawgroups))))
gc=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);np.savez_compressed(ROOT/'stability/final_group_bootstrap_consensus.npz',consensus=gc,numerator=num,denominator=den,ids=ids);savecsv(pd.DataFrame(group_rows),'stability/source_figure_bootstrap.csv')
# Score-weight sensitivity: fix component definitions and vary weighting assumptions.
weightrows=[]
for ws,wc in [(s,c) for s in [.35,.45,.55] for c in [.1,.2,.3] if s+c<=.85]:
 pool=rank[(rank['mode']=='multivariate')&rank.eligible_structure&(~rank.extreme_warp)].copy();wi=1-ws-wc-.1
 pool['alternative_score']=ws*pool.stability_worst+wi*(pool.silhouette+1)/2+.1*pool.dunn/(1+pool.dunn)+wc*pool.effective_coverage-.2*pool.warp_mean_time
 top=pool.sort_values('alternative_score',ascending=False).iloc[0]
 weightrows.append(dict(stability_weight=ws,coverage_weight=wc,silhouette_weight=wi,selected=top.name,window=top.window,T_h=top.T_h,K=top.K))
savecsv(pd.DataFrame(weightrows),'stability/selection_weight_sensitivity.csv')

log('Representatives and raw descriptive features')
wa,medoids=warp_audit(best,cs,D,base);savecsv(wa,'clustering_results/final_warping_audit.csv')
finalrows=[];features=[];turnrows=[];pairrows=[]
for i,c in enumerate(cs):
 k=int(base[i]);z=norm(c['y'],best.normalization);y0=c['y']/c['y'][0];der=np.diff(y0)/np.diff(c['t']);sign=np.sign(der);nz=np.flatnonzero(sign);turns=nz[1:][sign[nz[1:]]!=sign[nz[:-1]]]
 for q in turns:turnrows.append(dict(id=c['id'],cluster=k+1,time_h=c['t'][q],kind='peak' if sign[q]<0 else 'valley'))
 mid=best.T_h/2;first=der[c['t'][1:]<=mid];last=der[c['t'][1:]>mid]
 features.append(dict(id=c['id'],cluster=k+1,group=c['group'],n_points=len(c['t']),duration_full_h=raw[c['id']]['t'][-1],origin_absolute_h=raw[c['id']]['origin_h'],end_y_over_y0=y0[-1],peak_y_over_y0=y0.max(),valley_y_over_y0=y0.min(),global_peak_time_h=c['t'][np.argmax(y0)],global_valley_time_h=c['t'][np.argmin(y0)],peak_at_boundary=bool(np.argmax(y0) in [0,len(y0)-1]),valley_at_boundary=bool(np.argmin(y0) in [0,len(y0)-1]),n_sign_changes=len(turns),slope_median_per_h=np.median(der),slope_abs_median_per_h=np.median(abs(der)),positive_slope_fraction=np.mean(der>0),positive_slope_time_fraction=np.sum(np.diff(c['t'])[der>0])/best.T_h,first_half_median_slope=np.median(first) if len(first) else np.nan,last_half_median_slope=np.median(last) if len(last) else np.nan))
 finalrows.append(dict(id=c['id'],cluster=k+1,group=c['group'],source=c['source'],is_medoid=i==medoids[k],n_points=len(c['t']),distance_to_medoid=D[i,medoids[k]]))
 for j in range(i):
  if base[j]==k:pairrows.append(dict(id_a=c['id'],id_b=cs[j]['id'],cluster=k+1,DTW=D[i,j]))
fr=pd.DataFrame(finalrows);ft=pd.DataFrame(features);turn=pd.DataFrame(turnrows);pairs=pd.DataFrame(pairrows)
savecsv(fr,'clustering_results/final_labels.csv');savecsv(ft,'features/final_curve_descriptors.csv');savecsv(turn,'features/raw_local_turning_times.csv');savecsv(pairs,'clustering_results/final_within_cluster_distances.csv')
summary=[]
for k in range(int(best.K)):
 sub=ft[ft.cluster==k+1];med=cs[medoids[k]];summary.append(dict(cluster=k+1,n=len(sub),source_figures=sub.group.nunique(),medoid=med['id'],medoid_source=med['source'],end_ratio_median=sub.end_y_over_y0.median(),end_ratio_q25=sub.end_y_over_y0.quantile(.25),end_ratio_q75=sub.end_y_over_y0.quantile(.75),positive_time_fraction_median=sub.positive_slope_time_fraction.median(),median_raw_sign_changes=sub.n_sign_changes.median(),median_peak_h=sub.global_peak_time_h.median(),median_valley_h=sub.global_valley_time_h.median(),within_DTW_median=pairs[pairs.cluster==k+1].DTW.median()))
 dst=ROOT/'clustering_results/representative_raw_curves';dst.mkdir(exist_ok=True);source=m.SOURCE/med['source'];shutil.copy2(source,dst/f'cluster{k+1}_{med["id"]}_raw.csv');shutil.copy2(source.with_suffix('.png'),dst/f'cluster{k+1}_{med["id"]}_raw.png')
 savecsv(pd.DataFrame({'time_h':med['t'],'raw_y':med['y'],'y_over_y0':med['y']/med['y'][0],'z':norm(med['y'],'z')}),f'clustering_results/representative_raw_curves/cluster{k+1}_{med["id"]}_window.csv')
summary=pd.DataFrame(summary);savecsv(summary,'clustering_results/final_cluster_summary.csv')
# Main matrix and complete final K sweep convenient exports.
savecsv(pd.DataFrame(D,index=ids,columns=ids).reset_index(names='id'),'distance_matrices/final_distance_matrix.csv')
Z=linkage(squareform(D,checks=False),method=best.linkage);order=leaves_list(Z)
allk=m.cut_tree(Z,n_clusters=list(range(2,11)));savecsv(pd.DataFrame({'id':ids,**{f'K{k}':allk[:,k-2]+1 for k in range(2,11)}}),'clustering_results/final_parameters_labels_K2_to_K10.csv')
np.save(ROOT/'clustering_results/final_linkage.npy',Z)
# Each of four consensus matrices separately plus co-observation-weighted aggregate.
aggnum=np.zeros_like(D);aggden=np.zeros_like(D);cons={}
for mode in ['bootstrap90','bootstrap95','drop05','drop10']:
 b=np.load(ROOT/f'stability/{best.candidate}_{mode}_consensus.npz');aggnum+=b['numerator'];aggden+=b['denominator'];cons[mode]=b['consensus'];savecsv(pd.DataFrame(b['consensus'],columns=ids).assign(id=ids)[['id']+ids],f'stability/final_consensus_{mode}.csv')
C=aggnum/aggden;savecsv(pd.DataFrame(C,columns=ids).assign(id=ids)[['id']+ids],'stability/final_consensus_combined.csv');np.savez_compressed(ROOT/'stability/final_consensus_combined.npz',consensus=C,numerator=aggnum,denominator=aggden,ids=ids)

log('Plotting')
fig,(ax,tabax)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[2,1]});dur=np.array([c['t'][-1] for c in records]);t=np.r_[0,np.sort(np.unique(dur))];ax.step(t,[np.mean(dur>=x)*100 for x in t],where='post',label='Duration coverage',color='#2563a6');ax.scatter(cov.T_h,cov.effective_coverage*100,color='#df7932',label='Usable: >=6 original points',zorder=3)
ax.axvline(best.T_h,color='black',ls='--',lw=1,label=f'Selected {best.T_h:.2f} h');ax.set_xscale('log');ax.set(xlabel='Elapsed time from first observation (h)',ylabel=f'Percent of {len(records)} QC-eligible curves',ylim=(0,105),xlim=(20,dur.max()*1.1),title='Coverage versus usable sample count');ax.legend(loc='upper right',fontsize=8)
tabax.axis('off');tab=tabax.table(cellText=[[f'{r.T_h:.2f}',f'{r.duration_covered}',f'{r.n}'] for _,r in cov.iterrows()],colLabels=['Window (h)','Covered','Usable'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.7)
for j in range(3):tab[(4,j)].set_facecolor('#ffe3c5')
tabax.set_title('Counts among QC-eligible curves',fontsize=10);fig.tight_layout();figsave(fig,'01_coverage_vs_time')
fig,axs=plt.subplots(1,3,figsize=(12,3.5));eligible=a[a.eligible]
for ax,col,title in zip(axs,['duration_h','median_dt_h','n_clean'],['Curve duration (h)','Median sampling interval (h)','Original point count']):
 vals=eligible[col];ax.hist(np.log10(vals) if col!='n_clean' else vals,bins=20,color='#477ca8',edgecolor='white');ax.set(xlabel=('log10 '+title if col!='n_clean' else title),ylabel='Curves')
fig.suptitle('QC-eligible raw data; no smoothing');fig.tight_layout();figsave(fig,'02_qc_distributions')
# Main visual stays in y/y0 for interpretability, labels came from z-normalized multivariate DTW.
fig,axs=plt.subplots(1,int(best.K),figsize=(12,4.2),squeeze=False)
for k,ax in enumerate(axs[0]):
 for i,c in enumerate(cs):
  if base[i]==k:ax.plot(c['t'],c['y']/c['y'][0],color=colors[k],alpha=.25,lw=.8)
 med=cs[medoids[k]];ax.plot(med['t'],med['y']/med['y'][0],color='black',lw=2,label=f'Observed medoid {med["id"]}')
 ax.set(xlabel='Elapsed observed time (h)',ylabel='Raw PCE / first observed PCE',title=f'Cluster {k+1}: n={(base==k).sum()}, figures={ft[ft.cluster==k+1].group.nunique()}',xlim=(0,best.T_h));ax.legend(fontsize=8)
fig.suptitle(f'Frozen partition: {best.T_h:.2f} h, K={best.K}; raw unsmoothed trajectories');fig.tight_layout();figsave(fig,'03_clusters_raw_y_over_y0')
# Truly unnormalized y separated by source axis scale to avoid comparing percent and fractions.
meta=a.set_index('id');categories={}
for i,c in enumerate(cs):
 yn=str(meta.loc[c['id'],'y_name']).lower();yu=str(meta.loc[c['id'],'y_unit']).lower();cat='relative / normalized' if any(x in yn+' '+yu for x in ['norm','relative','nor.']) else 'absolute PCE (%)'
 # preserve relative-percent separate from fraction display based on explicit metadata or numeric scale
 if cat.startswith('relative'):cat+=': percent scale' if np.median(c['y'])>2 else ': fraction scale'
 categories[c['id']]=cat
cats=sorted(set(categories.values()));fig,axs=plt.subplots(int(best.K),len(cats),figsize=(13,7),squeeze=False)
for k in range(int(best.K)):
 for col,cat in enumerate(cats):
  ax=axs[k,col]
  for i,c in enumerate(cs):
   if base[i]==k and categories[c['id']]==cat:ax.plot(c['t'],c['y'],color=colors[k],alpha=.45,lw=.8)
  ax.set(title=f'Cluster {k+1}: {cat}',xlabel='Elapsed observed time (h)',ylabel='Unnormalized CSV y')
fig.suptitle('All raw values, separated by source response scale (no means or smoothing)');fig.tight_layout();figsave(fig,'04_clusters_unnormalized_raw')
fig,axs=plt.subplots(1,2,figsize=(11,4));im=axs[0].imshow(C[np.ix_(order,order)],vmin=0,vmax=1,cmap='viridis');axs[0].set(title='Combined co-observation weighted consensus',xlabel='Curves (dendrogram order)',ylabel='Curves');fig.colorbar(im,ax=axs[0],fraction=.046);im2=axs[1].imshow(D[np.ix_(order,order)],cmap='magma');axs[1].set(title='DTW distance (same ordering)',xlabel='Curves');fig.colorbar(im2,ax=axs[1],fraction=.046);fig.tight_layout();figsave(fig,'05_consensus_and_distance')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for k in range(int(best.K)):
 for i,c in enumerate(cs):
  if base[i]==k:
   der=np.diff(c['y']/c['y'][0])/np.diff(c['t']);axs[0].plot(c['t'][1:],der,color=colors[k],alpha=.17,lw=.65)
 sub=ft[ft.cluster==k+1];axs[1].scatter(np.full(len(sub),k+1)+np.linspace(-.12,.12,len(sub)),sub.slope_median_per_h,color=colors[k],s=17,alpha=.7)
axs[0].set(yscale='symlog',xlabel='Elapsed observed time (h)',ylabel='Finite-difference d(y/y0)/dt (1/h)',title='Unsmoothed derivative trajectories');axs[1].set(xticks=[1,2],xlabel='Cluster',ylabel='Per-curve median derivative (1/h)',title='One summary per curve');fig.tight_layout();figsave(fig,'06_derivative_distributions')
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for k in range(int(best.K)):
 sub=ft[ft.cluster==k+1]
 for ax,col in zip(axs[:2],['global_peak_time_h','global_valley_time_h']):ax.hist(sub[col],bins=np.linspace(0,best.T_h,13),alpha=.55,color=colors[k],label=f'C{k+1} (n={len(sub)})')
 axs[2].hist(turn[turn.cluster==k+1].time_h,bins=np.linspace(0,best.T_h,13),alpha=.55,color=colors[k],label=f'C{k+1}')
for ax,title in zip(axs,['Global maximum (includes endpoints)','Global minimum (includes endpoints)','All raw sign-change turns; noise-sensitive']):ax.set(title=title,xlabel='Elapsed time (h)',ylabel='Count');ax.legend(fontsize=8)
fig.tight_layout();figsave(fig,'07_peak_valley_times')
fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].boxplot([pairs[pairs.cluster==k+1].DTW for k in range(int(best.K))],tick_labels=['C1','C2'],showfliers=False);axs[0].set(ylabel='Within-cluster DTW',title='Pairwise distances (dependent observations)');axs[1].scatter(wa.mean_time_warp_fraction*best.T_h,wa.max_time_warp_fraction*best.T_h,c=[colors[int(k)-1] for k in wa.cluster],s=18,alpha=.65);axs[1].set(xlabel='Mean physical displacement along path (h)',ylabel='Maximum physical displacement (h)',title='Each curve to its observed medoid');fig.tight_layout();figsave(fig,'08_distances_and_warping')
fig,ax=plt.subplots(figsize=(12,4));dendrogram(Z,ax=ax,no_labels=True,color_threshold=Z[-2,2],above_threshold_color='#444444');ax.set(title=f'{best.linkage.capitalize()} linkage on variable-length DTW distances',ylabel='Linkage distance',xlabel='91 observed curves');figsave(fig,'09_dendrogram')
# Best candidate per window, plus classic controls, not a single incomparable silhouette ranking.
bw=rank[(rank['mode']=='multivariate')&rank.eligible_structure&(~rank.extreme_warp)].reset_index().sort_values('selection_score',ascending=False).groupby('window',sort=False).head(1).sort_values('T_h');savecsv(bw,'clustering_results/best_model_by_window.csv')
fig,axs=plt.subplots(1,3,figsize=(13,3.7));x=np.arange(len(bw));names=[f'{t:.0f}' for t in bw.T_h]
for ax,col,title in zip(axs,['silhouette','stability_worst','selection_score'],['Silhouette','Worst perturbation mean ARI','Fixed composite selection score']):
 ax.plot(x,bw[col],'o-',color='#2563a6');ax.set(xticks=x,xticklabels=names,title=title,xlabel='Window length (h)');ax.tick_params(axis='x',labelrotation=45)
fig.tight_layout();figsave(fig,'10_window_comparison')
fig,ax=plt.subplots(figsize=(10,4));show=sensdf[~sensdf.analysis.str.startswith('window_fixed')];ax.barh(np.arange(len(show)),show.ARI_to_frozen,color='#477ca8');ax.set(yticks=np.arange(len(show)),yticklabels=show.analysis,xlabel='ARI with frozen labels on overlapping curves',xlim=(-.1,1.05),title='Sensitivity at fixed K=2; normalization changes the question');ax.tick_params(axis='y',labelsize=8);fig.set_size_inches(10,8);fig.tight_layout();figsave(fig,'11_parameter_sensitivity')
fig,axs=plt.subplots(1,2,figsize=(11,4));seq=make_seq(cs,best.normalization,best['mode'],best.lambda_value)
for k,ax in enumerate(axs):
 med=medoids[k];sub=wa[wa.cluster==k+1];sid=sub.sort_values('max_time_warp_fraction').iloc[-1].id;i=idmap[sid];p,_=path(seq[i],seq[med],best.radius);a1,b1=p.T
 ax.plot(cs[i]['t'][1:][a1],cs[med]['t'][1:][b1],lw=1.5,color=colors[k]);ax.plot([0,best.T_h],[0,best.T_h],'k--',alpha=.4);ax.set(title=f'C{k+1}: largest max displacement\n{sid} vs medoid {ids[med]}',xlabel=f'{sid} elapsed time (h)',ylabel=f'{ids[med]} elapsed time (h)')
 savecsv(pd.DataFrame({'source_index':a1,'medoid_index':b1,'source_time_h':cs[i]['t'][1:][a1],'medoid_time_h':cs[med]['t'][1:][b1]}),f'clustering_results/representative_path_cluster{k+1}.csv')
fig.tight_layout();figsave(fig,'12_representative_warp_paths')
# Individual original raw plots for ALL retained curves: paginated gallery.
for page,offset in enumerate(range(0,len(cs),16),1):
 fig,axs=plt.subplots(4,4,figsize=(14,10))
 for j,ax in enumerate(axs.flat):
  i=offset+j
  if i>=len(cs):ax.axis('off');continue
  c=cs[i];ax.plot(c['t'],c['y'],color=colors[int(base[i])],lw=1);ax.set_title(f'{c["id"]} | C{base[i]+1} | image {c["group"].replace("图片","")}',fontsize=9);ax.tick_params(labelsize=7);ax.set_xlabel('h',fontsize=8)
 fig.suptitle('Unnormalized CSV values, selected window; scales vary by original axis');fig.tight_layout();figsave(fig,f'gallery_raw_page{page:02d}')
# Feature degeneracy audit and quantitative derivative contribution.
featrows=[]
for no in NORMS:
 for c in cs:
  z=norm(c['y'],no);featrows.append(dict(id=c['id'],normalization=no,raw_sd=np.std(c['y']),raw_IQR=np.quantile(c['y'],.75)-np.quantile(c['y'],.25),normalized_sd=np.std(z),derivative_rms=np.sqrt(np.mean((np.diff(z)/np.diff(c['t']))**2))))
savecsv(pd.DataFrame(featrows),'audit/final_normalization_scales.csv')
classicD=distance(make_seq(cs,best.normalization,'classic',0),best.radius);tri=np.triu_indices(len(cs),1);relative_change=np.median(abs(D[tri]-classicD[tri])/np.maximum(classicD[tri],1e-12))
extra=dict(absolute_common_start_h=start,absolute_common_end_h=float(best.T_h),absolute_sensitivity_n=len(abscs),typical_original_dt_h=typical_dt,common_grid_spacing_h=float(np.diff(grid)[0]),common_grid_points=len(grid),group_bootstrap_ARI_mean=float(pd.DataFrame(group_rows).ARI.mean()),group_bootstrap_ARI_p025=float(pd.DataFrame(group_rows).ARI.quantile(.025)),group_bootstrap_ARI_p975=float(pd.DataFrame(group_rows).ARI.quantile(.975)),median_relative_distance_change_vs_classic=float(relative_change),selected_label_ARI_vs_classic=float(sensdf.set_index('analysis').loc['mode_classic','ARI_to_frozen']),source_hashes_unchanged=all(hashlib.sha256((m.SOURCE/r.path).read_bytes()).hexdigest()==r.sha256 for _,r in pd.read_csv(ROOT/'raw/source_sha256.csv').iterrows()))
(ROOT/'audit/final_checks.json').write_text(json.dumps(extra,indent=2));log(json.dumps(extra))
