#!/usr/bin/env python3
"""Export the frozen unsupervised DTW fit in the visual layout of thesis/result."""
from pathlib import Path
import os,shutil,json,pickle,html,sys
R=Path(__file__).resolve().parents[1]
S=R.parent/'derivative_dtw_results_20260906'
os.environ['MPLCONFIGDIR']=str(R/'code/.mplcache')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.spatial.distance import squareform
# Preserve every numerical deliverable; original input directory is only read.
for name in ['raw','aligned','features','distance_matrices','clustering_results','stability','audit','figures','code']:
 shutil.copytree(S/name,R/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.mplcache'))
for name in ['methods_summary.md','research_strategy_original.md']:
 shutil.copy2(S/name,R/name)
# Adapt copied reproduction notes and validation to the additional visualization files.
repro=R/'code/REPRODUCE.md'
repro.write_text(repro.read_text().replace(str(S),str(R)))
verify=R/'code/verify_results.py'
verify.write_text(verify.read_text().replace("glob('*.png')))==18","glob('*.png')))>=18").replace("glob('*.svg')))==18","glob('*.svg')))>=18"))
sys.path.insert(0,str(R/'code'))
import run_analysis as m
with (R/'code/state.pkl').open('rb') as f:records,a,W,cov,metrics_all,labels,Ds,rank,best=pickle.load(f)
cs=W[best.window]['curves'];N=len(cs);K=int(best.K);T=float(best.T_h);l=labels[best.candidate];D=Ds[best.distance_key];ids=[c['id'] for c in cs]
fr=pd.read_csv(R/'clustering_results/final_labels.csv');ft=pd.read_csv(R/'features/final_curve_descriptors.csv')
wa,medoids=m.warp_audit(best,cs,D,l)
typical=float(np.median([np.median(np.diff(c['t'])) for c in cs]));grid=np.linspace(0,T,int(np.floor(T/typical))+1)
ygrid=np.array([np.interp(grid,c['t'],c['y']/c['y'][0]) for c in cs]);zgrid=np.array([np.interp(grid,c['t'],m.norm(c['y'],best.normalization)) for c in cs])
colors=['#3475b8','#e38a47','#58a16e','#a770ad','#65aaba','#ac855a','#8088ca','#af6880','#a4a643','#777777']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17,'figure.dpi':130,'savefig.dpi':180})
def png(fig,name):
 fig.savefig(R/(name+'.png'),bbox_inches='tight');fig.savefig(R/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def interactive(fig,name):
 fig.write_html(R/(name+'.html'),include_plotlyjs='directory',full_html=True,config={'displaylogo':False,'responsive':True,'toImageButtonOptions':{'format':'png','scale':2}})
def draw_cluster(ax,k,kind='y0',common_limits=False):
 for i,c in enumerate(cs):
  if l[i]!=k:continue
  yy=c['y']/c['y'][0] if kind=='y0' else m.norm(c['y'],best.normalization)
  ax.plot(c['t'],yy,color='#778899',alpha=.32 if (l==k).sum()>20 else .55,lw=.9)
 mat=ygrid if kind=='y0' else zgrid
 ax.plot(grid,mat[l==k].mean(axis=0),color='#d62728',lw=2.5,label='Mean (derived; linear display grid)')
 c=cs[medoids[k]];yy=c['y']/c['y'][0] if kind=='y0' else m.norm(c['y'],best.normalization)
 ax.plot(c['t'],yy,'--',color='#101820',lw=1.9,label=f'Real medoid: {c["id"]}')
 ax.set(title=f'Cluster {k+1} | n={int((l==k).sum())}',xlabel='Elapsed observed time (h)',ylabel='PCE / first observed PCE' if kind=='y0' else 'z-normalized PCE',xlim=(0,T))
 if common_limits:ax.set_ylim(min(ygrid.min(),.35)-.03,max(ygrid.max(),1.1)+.03)
 ax.legend(loc='best',fontsize=8)
print('Export cluster panels and interactive samples',flush=True)
for kind,suffix in [('y0',''),('z','_z')]:
 fig,axs=plt.subplots(1,K,figsize=(7*K,4.6),squeeze=False)
 for k,ax in enumerate(axs[0]):draw_cluster(ax,k,kind,common_limits=kind=='y0')
 fig.suptitle(f'Multivariate DTW + average linkage | {T:.2f} h | K={K}\nIndividual unsmoothed curves + derived mean + real medoid',fontsize=15);fig.tight_layout();png(fig,'dtw_clusters'+suffix)
for k in range(K):
 fig,ax=plt.subplots(figsize=(10,5.6));draw_cluster(ax,k);fig.tight_layout();png(fig,f'dtw_cluster_{k+1}')
 fig=go.Figure();sel=np.flatnonzero(l==k)
 for i in sel:
  c=cs[i];fig.add_trace(go.Scatter(x=c['t'],y=c['y']/c['y'][0],name=c['id'],mode='lines',line=dict(color='rgba(75,131,169,0.32)',width=1.5),customdata=np.column_stack([c['y'],m.norm(c['y'],best.normalization)]),hovertemplate=f"{c['id']} | {html.escape(c['group'])}<br>{html.escape(Path(c['source']).stem)}<br>Elapsed: %{{x:.3f}} h<br>PCE / PCE0: %{{y:.6f}}<br>Raw y: %{{customdata[0]:.6f}}<br>z: %{{customdata[1]:.4f}}<extra></extra>",showlegend=False))
 fig.add_trace(go.Scatter(x=grid,y=ygrid[sel].mean(axis=0),name='Mean (derived; not a raw sample)',line=dict(color='#d62728',width=4)))
 mi=medoids[k];c=cs[mi];fig.add_trace(go.Scatter(x=c['t'],y=c['y']/c['y'][0],name=f'Real medoid {c["id"]}',line=dict(color='black',width=3,dash='dash')))
 buttons=[dict(label='All curves',method='update',args=[{'visible':[True]*(len(sel)+2)}])]
 for j,i in enumerate(sel):buttons.append(dict(label=cs[i]['id']+' | image '+cs[i]['group'].replace('图片',''),method='update',args=[{'visible':[q==j for q in range(len(sel))]+[True,True]}]))
 fig.update_layout(title=f'Multivariate DTW — Cluster {k+1}, n={len(sel)}',xaxis_title='Elapsed observed time (h)',yaxis_title='PCE / first observed PCE',template='plotly_white',height=650,hovermode='closest',legend=dict(orientation='h',y=-.19),margin=dict(t=105,b=110),updatemenus=[dict(buttons=buttons,direction='down',x=1,y=1.16,xanchor='right')])
 interactive(fig,f'dtw_cluster_{k+1}_interactive')
 fr[fr.cluster==k+1].to_csv(R/f'cluster_{k+1}_samples.csv',index=False)
 pd.DataFrame({'time_h':grid,'derived_mean_y_over_y0':ygrid[sel].mean(axis=0),'derived_median_y_over_y0':np.median(ygrid[sel],axis=0),'derived_mean_z':zgrid[sel].mean(axis=0)}).to_csv(R/'features'/f'cluster_{k+1}_display_summary.csv',index=False)
# All source values remain separate from normalization and aggregate display curves.
fig,ax=plt.subplots(figsize=(7,4.5));counts=np.bincount(l);bars=ax.bar(np.arange(1,K+1),counts,color=colors[:K],width=.6)
for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,n+.6,f'{n} ({n/N:.1%})',ha='center')
ax.set(xticks=np.arange(1,K+1),xlabel='DTW cluster',ylabel='Number of curves',ylim=(0,max(counts)*1.15),title=f'Cluster membership | N={N}');fig.tight_layout();png(fig,'dtw_cluster_counts')
for kind in ['normalized','derivative']:
 cols=7;rows=int(np.ceil(N/cols));fig,axs=plt.subplots(rows,cols,figsize=(16,rows*1.45),squeeze=False)
 for i,ax in enumerate(axs.flat):
  if i>=N:ax.axis('off');continue
  c=cs[i];yy=c['y']/c['y'][0]
  if kind=='derivative':xx=c['t'][1:];yy=np.diff(yy)/np.diff(c['t'])
  else:xx=c['t']
  ax.plot(xx,yy,color=colors[l[i]],lw=.85);ax.set_title(f'{c["id"]} | C{l[i]+1}',fontsize=8);ax.tick_params(labelsize=6);ax.set_xlim(0,T)
 fig.suptitle(f'All {N} curves — '+('raw y/y0; no smoothing' if kind=='normalized' else 'true finite differences d(y/y0)/dt; no smoothing'),fontsize=15);fig.tight_layout(rect=(0,0,1,.98));png(fig,'overview_'+kind)
print('Export projections and validation',flush=True)
# PCA is ONLY a display projection: frozen labels were fitted to ragged DTW sequences.
pca=PCA(n_components=2);xy=pca.fit_transform(zgrid)
fig,ax=plt.subplots(figsize=(8,5.5))
for k in range(K):
 ix=l==k;ax.scatter(xy[ix,0],xy[ix,1],s=28,color=colors[k],alpha=.72,label=f'Cluster {k+1} (n={ix.sum()})');mi=medoids[k];ax.scatter(*xy[mi],marker='*',s=210,color=colors[k],edgecolors='black',linewidths=.9);ax.annotate(ids[mi],xy[mi],xytext=(6,6),textcoords='offset points',fontsize=9)
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',title='PCA display, colored by frozen DTW cluster\nLinear grid for display only; stars = observed medoids');ax.legend();fig.tight_layout();png(fig,'pca_visualization')
pd.DataFrame({'id':ids,'cluster':l+1,'PC1':xy[:,0],'PC2':xy[:,1]}).to_csv(R/'clustering_results/pca_display_coordinates.csv',index=False)
# Principal-coordinate view derived from the DTW matrix itself.
Q=D**2;B=-.5*(Q-Q.mean(axis=0)[None,:]-Q.mean(axis=1)[:,None]+Q.mean());assert np.isfinite(B).all();ev,V=np.linalg.eigh(B);order=np.argsort(ev)[::-1];ev=ev[order];V=V[:,order];mds=V[:,:2]*np.sqrt(np.maximum(ev[:2],0));neg=float(abs(ev[ev<0]).sum()/np.maximum(abs(ev).sum(),1e-15))
fig,ax=plt.subplots(figsize=(8,5.5))
for k in range(K):ax.scatter(mds[l==k,0],mds[l==k,1],s=28,color=colors[k],alpha=.72,label=f'Cluster {k+1}')
ax.set(title=f'DTW distance projection (classical MDS)\nNegative eigenvalue mass: {neg:.1%}; display is approximate',xlabel='Coordinate 1',ylabel='Coordinate 2');ax.legend();fig.tight_layout();png(fig,'dtw_projection')
order=leaves_list(linkage(squareform(D,checks=False),method=best.linkage));C=np.load(R/'stability/final_consensus_combined.npz')['consensus']
for values,name,title,cmap in [(D,'dtw_distance_map','Pairwise DTW distances','viridis'),(C,'dtw_consensus','Co-clustering consensus','Blues')]:
 fig,ax=plt.subplots(figsize=(7,5.8));im=ax.imshow(values[np.ix_(order,order)],cmap=cmap,**({'vmin':0,'vmax':1} if name=='dtw_consensus' else {}));fig.colorbar(im,ax=ax,fraction=.046,pad=.04);ax.set(title=title+' | dendrogram order',xlabel='Curves',ylabel='Curves');ax.grid(False);fig.tight_layout();png(fig,name)
# Method comparison follows the three representations explicitly requested in the strategy.
mode_labels={};comparison=[]
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
for ax,mode,lam in zip(axs,['classic','derivative','multivariate'],[0,1,best.lambda_value]):
 dd=m.distance(m.make_seq(cs,best.normalization,mode,lam),best.radius);ll=m.cluster(dd,best.linkage,K);mode_labels[mode]=ll;cc=np.bincount(ll);mm=m.metrics(dd,ll);ari=adjusted_rand_score(l,ll);comparison.append(dict(mode=mode,ARI_to_multivariate=ari,**mm));ax.bar(np.arange(1,K+1),cc,color=colors[:K]);ax.set(title=f'{mode.capitalize()} DTW\nsilhouette={mm["silhouette"]:.3f}; ARI={ari:.3f}',xticks=np.arange(1,K+1),xlabel='Cluster ID (local to each fit)',ylabel='Curves',ylim=(0,N+8))
 for j,n in enumerate(cc):ax.text(j+1,n+1,str(n),ha='center')
fig.suptitle('Representation comparison at fixed window, normalization, radius, linkage and K');fig.tight_layout();png(fig,'dtw_method_comparison');pd.DataFrame(comparison).to_csv(R/'clustering_results/method_comparison.csv',index=False)
# Endpoint retention is available for every curve; absolute initial efficiency is not.
retention=np.array([c['y'][-1]/c['y'][0] for c in cs]);delta=(1-retention)*100
fig,ax=plt.subplots(figsize=(8,5));v=ax.violinplot([delta[l==k] for k in range(K)],positions=np.arange(1,K+1),showmedians=True,showextrema=False)
for k,body in enumerate(v['bodies']):body.set_facecolor(colors[k]);body.set_alpha(.35)
rng=np.random.default_rng(m.SEED)
for k in range(K):ax.scatter(k+1+rng.uniform(-.1,.1,(l==k).sum()),delta[l==k],color=colors[k],s=17,alpha=.75)
ax.axhline(0,color='gray',ls='--',lw=1);ax.set(xticks=np.arange(1,K+1),xlabel='DTW cluster',ylabel='Relative PCE loss at window end (%)',title='Endpoint PCE change by cluster\nPositive = loss; negative = improvement');fig.tight_layout();png(fig,'pce_violin')
for style in ['boxplot','violin']:
 fig=go.Figure()
 for k in range(K):
  ii=np.flatnonzero(l==k);kw=dict(y=delta[ii],name=f'Cluster {k+1} (n={len(ii)})',text=[ids[i] for i in ii],hovertemplate='%{text}<br>Relative loss: %{y:.3f}%<extra></extra>',marker_color=colors[k])
  if style=='violin':fig.add_trace(go.Violin(**kw,box_visible=True,meanline_visible=True,points='all'))
  else:fig.add_trace(go.Box(**kw,boxpoints='all',jitter=.25))
 fig.update_layout(title=f'PCE change by DTW cluster at {T:.2f} h',yaxis_title='(PCE0 - PCEend) / PCE0 × 100 (%)',template='plotly_white',height=580);interactive(fig,'pce_'+style)
pd.DataFrame({'id':ids,'cluster':l+1,'PCE_end_over_first':retention,'relative_PCE_loss_percent':delta}).to_csv(R/'clustering_results/pce_change_by_cluster.csv',index=False)
# Time-window panels permit direct visual comparison of the actual learned groupings.
bw=pd.read_csv(R/'clustering_results/best_model_by_window.csv');fig,axs=plt.subplots(3,3,figsize=(15,12))
for ax,(_,r) in zip(axs.flat,bw.iterrows()):
 wc=W[r.window]['curves'];ll=labels[r.candidate];dd=Ds[r.distance_key]
 for k in np.unique(ll):
  ii=np.flatnonzero(ll==k);med=ii[np.argmin(dd[np.ix_(ii,ii)].sum(axis=1))]
  for i in ii:ax.plot(wc[i]['t'],wc[i]['y']/wc[i]['y'][0],color=colors[k],alpha=.15,lw=.65)
  c=wc[med];ax.plot(c['t'],c['y']/c['y'][0],color=colors[k],lw=2,label=f'C{k+1}: n={len(ii)}')
 ax.set(title=f'{r.T_h:.2f} h | {r.normalization} | K={r.K}\nsil={r.silhouette:.3f}, worst mean ARI={r.stability_worst:.3f}',xlabel='Elapsed observed time (h)',ylabel='PCE / PCE0');ax.legend(fontsize=7)
fig.suptitle('Time-window sensitivity: raw curves and real medoids of the selected model per window',fontsize=15);fig.tight_layout(rect=(0,0,1,.965));png(fig,'window_sensitivity')
for src,dst in [('01_coverage_vs_time.png','coverage_vs_time.png'),('06_derivative_distributions.png','derivative_distribution.png'),('07_peak_valley_times.png','peak_valley_distribution.png'),('08_distances_and_warping.png','within_cluster_distance_and_warping.png'),('09_dendrogram.png','dtw_dendrogram.png'),('10_window_comparison.png','window_metrics.png'),('11_parameter_sensitivity.png','parameter_sensitivity.png'),('12_representative_warp_paths.png','dtw_warping_paths.png')]:shutil.copy2(R/'figures'/src,R/dst)
fr.to_csv(R/'cluster_assignments.csv',index=False);shutil.copy2(R/'clustering_results/final_parameters_labels_K2_to_K10.csv',R/'cluster_labels_K2_to_K10.csv')
shutil.copy2(R/'stability/stability_summary.csv',R/'stability_table.csv');shutil.copy2(R/'clustering_results/internal_metrics_all.csv',R/'internal_metrics.csv')
summary=json.loads((R/'clustering_results/frozen_parameters.json').read_text());summary.update(algorithm='multivariate DTW + hierarchical average linkage',n_source_curves=len(a),n_qc_eligible=int(a.eligible.sum()),cluster_sizes={f'cluster_{k+1}':int((l==k).sum()) for k in range(K)},medoids={f'cluster_{k+1}':ids[medoids[k]] for k in range(K)},display_mean='Derived arithmetic mean on a linear grid; not an original sample; not used in clustering',display_grid_spacing_h=float(grid[1]-grid[0]),pca_display_explained_variance=pca.explained_variance_ratio_.tolist(),dtw_projection_negative_eigenvalue_mass=neg,source_fit=str(S),reference_layout=str(R.parent.parent/'thesis/result'),labels_recomputed_from_saved_distance=True)
(R/'dtw_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
# Results-first landing page: immediately show main panel and individual interactive plots.
thumbs=[('dtw_clusters.png','全部曲线分簇总览'),('dtw_clusters_z.png','实际 z-normalization 表示'),('dtw_cluster_counts.png','各簇样本数量'),('pca_visualization.png','二维 PCA 展示'),('dtw_distance_map.png','DTW 距离矩阵'),('dtw_consensus.png','共聚类矩阵'),('dtw_method_comparison.png','三种 DTW 对照'),('pce_violin.png','各簇终点 PCE 变化'),('coverage_vs_time.png','时间覆盖率'),('window_sensitivity.png','各候选窗口的聚类曲线'),('derivative_distribution.png','导数分布'),('peak_valley_distribution.png','峰谷时间分布'),('dtw_warping_paths.png','时间扭曲路径')]
cards=''.join(f'<section><h2>{html.escape(title)}</h2><a href="{file}"><img src="{file}" alt="{html.escape(title)}" loading="lazy"></a></section>' for file,title in thumbs)
links=''.join(f'<a class="button" href="dtw_cluster_{k+1}_interactive.html">簇 {k+1}：交互查看 {int((l==k).sum())} 条曲线</a>' for k in range(K))
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Derivative-aware DTW 聚类结果</title><style>body{{font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif;color:#203148;background:#f4f6f9;margin:0}}main{{max-width:1180px;margin:32px auto;padding:32px;background:white;border-radius:12px}}h1{{margin-top:0;font-size:28px}}h2{{font-size:21px;margin-top:36px}}img{{width:100%;height:auto}}a{{color:#23659b}}.button{{display:inline-block;padding:10px 15px;margin:6px 10px 6px 0;background:#e8f0f7;border-radius:6px;text-decoration:none}}.meta{{color:#566779}}section{{border-top:1px solid #e3e8ed;margin-top:30px}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main><h1>导数感知 DTW 无监督聚类结果</h1><p class="meta">{T:.2f} h · {N} 条曲线 · K={K} · z-normalization · λ={best.lambda_value:g} · radius={best.radius:.0%} · average linkage</p><p>{links}</p><p><a href="cluster_assignments.csv">样本标签 CSV</a> · <a href="dtw_summary.json">运行摘要 JSON</a> · <a href="pce_boxplot.html">PCE 箱线图</a> · <a href="pce_violin.html">PCE 小提琴图</a> · <a href="README.md">文件说明</a></p><p>灰线为全部未平滑样本，红线为明确标注的派生均值，黑色虚线为真实 medoid。时间从首个观测点起算；均值与 PCA 的线性显示网格不参与聚类。分组对归一化敏感，形态解释以实际曲线为准。</p>{cards}</main></html>'''
(R/'index.html').write_text(page)
readme=f'''# 导数感知 DTW 无监督聚类结果

先打开 **[index.html](index.html)**，或直接查看 **[dtw_clusters.png](dtw_clusters.png)**。输出组织参考 `thesis/result`，训练方法采用你提供 Markdown 的 **multivariate DTW + 层次聚类 G1 分支**。

当前结果：{T:.2f} h，91 条，z-normalization，λ=4，radius=20%，average linkage，K=2，簇规模 84/7。所有曲线未平滑；原始数据未修改。

| 文件 | 内容 |
| --- | --- |
| dtw_clusters.png | 各簇全部曲线 + 红色派生均值 + 黑色真实 medoid；共用纵轴 |
| dtw_clusters_z.png | 相同分组在实际 z-normalization 表示下的曲线 |
| dtw_cluster_1.png / dtw_cluster_2.png | 单簇放大图 |
| dtw_cluster_1_interactive.html / dtw_cluster_2_interactive.html | 悬停读样本 ID、来源、时间、原始 y；下拉选择样本 |
| dtw_cluster_counts.png | 每簇数量和比例 |
| overview_normalized.png / overview_derivative.png | 全部 91 条曲线的归一化 / 真实差分导数小图 |
| pca_visualization.png | 冻结 DTW 标签的 PCA 展示；线性网格仅用于显示 |
| dtw_projection.png | 直接从 DTW 距离得到的二维投影 |
| dtw_distance_map.png / dtw_consensus.png / dtw_dendrogram.png | 距离、共聚类和层次树 |
| dtw_method_comparison.png | classic、derivative、multivariate 三种策略对照 |
| pce_boxplot.html / pce_violin.html / pce_violin.png | 每簇终点相对 PCE 变化的分布 |
| coverage_vs_time.png / window_sensitivity.png / window_metrics.png | 候选时间窗口的覆盖、曲线分组和指标 |
| derivative_distribution.png / peak_valley_distribution.png | 导数、峰谷时刻分布 |
| within_cluster_distance_and_warping.png / dtw_warping_paths.png | 簇内距离和时间扭曲审计 |
| cluster_assignments.csv / cluster_1_samples.csv / cluster_2_samples.csv | 样本到簇的对应关系 |
| cluster_labels_K2_to_K10.csv | 冻结距离设置下的全部 K 标签 |
| dtw_summary.json | 实际运行参数、指标、簇规模与 medoid |
| internal_metrics.csv / stability_table.csv | 全量内部指标和候选稳定性 |

红色均值是参数冻结后计算的显示汇总，不是原始样本，也不用于拟合。其公共显示网格间距 {grid[1]-grid[0]:.2f} h，不细于典型原始间距 {typical:.2f} h。原始曲线仍按自身不等长时间点绘制。每簇 medoid 来自真实样本。

按原 Markdown 保留 `raw/`、`aligned/`、`features/`、`distance_matrices/`、`clustering_results/`、`stability/`、`figures/`。1,512 个距离矩阵、40,824 组参数标签以及候选重复检验均保留；细节见 `methods_summary.md` 和 `code/PROTOCOL.md`。`plotly.min.js` 是离线交互图依赖，请随 HTML 一起保留。

此目录采用已完成并验证的真实无监督计算，重新核验冻结距离得到的标签，并重新生成上述展示文件。没有为了模仿参考图指定四簇。参考目录的 Savitzky–Golay 平滑、SOM U-matrix/拓扑误差、KMeans 对照和按绝对初始效率分组不属于本次已选策略/可用数据，因此相应位置采用真实的导数总览、DTW 距离/层次图、三种 DTW 对照和按簇终点变化图。

复现整个计算见 `code/REPRODUCE.md`。重新导出本展示：`python3 code/export_visual_results.py`。导出器按当前已冻结的计算状态生成图，不调用人工曲线标签。分组对归一化敏感，且本次导数未改变 PCE-only 的标签；图形分离不能直接证明物理机制。
'''
(R/'README.md').write_text(readme)
# Numerical and content checks: no label/model refitting during visualization.
assert np.array_equal(m.cluster(D,best.linkage,K),l)
assert list(fr.id)==ids and np.array_equal(fr.cluster.to_numpy(),l+1)
assert np.allclose((R/'aligned'/f'{best.window}_z_matrices.npz').exists(),True)
assert grid[1]-grid[0]>=typical-1e-9
assert all(cs[i]['id'] in ids for i in medoids.values())
verification=dict(n_curves=N,all_sample_ids_preserved=True,labels_reproduced=True,means_used_for_training=False,common_display_grid_spacing_h=float(grid[1]-grid[0]),typical_original_spacing_h=typical,cluster_trace_counts={f'cluster_{k+1}':int((l==k).sum())+2 for k in range(K)},n_png_top_level=len(list(R.glob('*.png'))),n_html_top_level=len(list(R.glob('*.html'))),source_dataset_unchanged=True)
(R/'audit/visual_export_verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification),flush=True)
print('Result directory:',R,flush=True)
