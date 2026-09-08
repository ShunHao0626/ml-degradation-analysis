"""Presentation export matching thesis/result's curve-first organization.

Reads frozen assignments only. No training, parameter search, temporal smoothing,
or reassignment. Ensemble means are strictly post-fit descriptive summaries.
"""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/method1_result_gallery_mpl'
from pathlib import Path
import json, pickle, hashlib, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result'
COLORS={0:'#287bb5',1:'#ef8c32',-1:'#8b929b'}

def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close()

def write_html(fig,name):
    fig.update_layout(template='plotly',font=dict(family='Arial, sans-serif',size=15),
        margin=dict(l=70,r=35,t=90,b=75),height=620,hovermode='closest')
    fig.write_html(OUT/name,include_plotlyjs='directory',full_html=True,config=dict(responsive=True,displaylogo=False))

def render(root=ROOT):
    assert Path(root)==ROOT
    OUT.mkdir(exist_ok=True)
    a=pd.read_csv(ROOT/'final'/'cluster_assignment.csv')
    before=hashlib.sha256((ROOT/'final'/'cluster_assignment.csv').read_bytes()).hexdigest()
    frozen=json.loads((ROOT/'final'/'frozen_model.json').read_text())
    window=frozen['selected_window_hours'];dataset=frozen['candidate']['dataset']
    medoids={r['cluster']:r['medoid'] for r in json.loads((ROOT/'final'/'posthoc_morphology.json').read_text())}
    curves={}
    for sid in a.sample_id:
        raw=pd.read_csv(ROOT/'window_inputs'/dataset/f'{sid}.csv')
        modeled=pd.read_csv(ROOT/'normalized_input'/dataset/frozen['candidate']['normalization']/f'{sid}.csv')
        t=raw.elapsed.to_numpy();y=raw.y.to_numpy()
        curves[sid]=dict(t=t,raw=y,initial=y/y[0],model=modeled.normalized_y.to_numpy())
    groups=sorted(set(a.cluster)-{-1});display_groups=groups+([-1] if (a.cluster<0).any() else [])
    name=lambda label:'Noise (unassigned)' if label<0 else f'Cluster {label}'
    filename=lambda label:'hdbscan_noise' if label<0 else f'hdbscan_cluster_{label}'
    # A shared, conservative grid; every included curve covers every grid time.
    dt=float(np.median([np.median(np.diff(c['t'])) for c in curves.values()]))
    end=float(min(c['t'][-1] for c in curves.values()))
    grid=np.arange(0,end+1e-10,dt)
    means={};meanrows=[];curve_rows=[]
    for mode in ['initial','model']:
        for label in display_groups:
            members=a.loc[a.cluster==label,'sample_id'].tolist()
            aligned=np.stack([np.interp(grid,curves[s]['t'],curves[s][mode]) for s in members])
            means[label,mode]=aligned.mean(axis=0)
            for t,m,q1,q3 in zip(grid,aligned.mean(axis=0),np.quantile(aligned,.25,axis=0),np.quantile(aligned,.75,axis=0)):
                meanrows.append(dict(group=label,display_normalization=mode,time_hours=t,ensemble_mean=m,q25=q1,q75=q3,n_contributors=len(members)))
    pd.DataFrame(meanrows).to_csv(OUT/'cluster_mean_curves_display_only.csv',index=False)
    for _,r in a.iterrows():
        c=curves[r.sample_id]
        curve_rows.extend(dict(sample_id=r.sample_id,cluster=r.cluster,time_hours=float(t),raw_output=float(y),relative_to_initial=float(yi),model_normalized=float(ym)) for t,y,yi,ym in zip(c['t'],c['raw'],c['initial'],c['model']))
    pd.DataFrame(curve_rows).to_csv(OUT/'plotted_curve_points.csv',index=False)
    plt.rcParams.update({'font.size':12,'axes.spines.top':True,'axes.spines.right':True,'figure.facecolor':'white'})

    def axes_curves(ax,label,mode,individual=False):
        members=a.loc[a.cluster==label,'sample_id'].tolist()
        for sid in members:
            c=curves[sid];ax.plot(c['t'],c[mode],color='#7d8893' if not individual else COLORS[label],alpha=.34 if not individual else .28,lw=.8)
        ax.plot(grid,means[label,mode],color='#e32622',lw=2.8,label=f'Ensemble mean (n={len(members)})')
        med=curves[medoids[label]]
        ax.plot(med['t'],med[mode],color='#242a30',ls='--',lw=1.5,label=f'Real medoid: {medoids[label]}')
        ax.set(title=f'{name(label)}\nn={len(members)}',xlabel='Elapsed time (hours)',ylabel='PCE / initial PCE' if mode=='initial' else 'Robust-normalized output')
        ax.set_xlim(0,window);ax.legend(fontsize=9,loc='best');ax.grid(alpha=.15)

    # Primary requested figure: individual member curves + a cluster mean, per panel.
    for mode,suffix in [('initial',''),('model','_model_normalization')]:
        ncols=min(2,len(groups));nrows=int(np.ceil(len(groups)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(6.3*ncols,4.5*nrows+1),squeeze=False,sharex=True)
        for ax,label in zip(axes.flat,groups):axes_curves(ax,label,mode)
        for ax in list(axes.flat)[len(groups):]:ax.axis('off')
        fig.suptitle('HDBSCAN Clusters: Individual Curves + Cluster Mean',fontsize=18,y=1.015)
        fig.text(.5,.005,'Frozen 300 h assignments | Display normalization only; no temporal smoothing | Noise shown separately',ha='center',fontsize=10,color='#555')
        save(f'hdbscan_clusters{suffix}.png')
    fig,axes=plt.subplots(1,len(display_groups),figsize=(6.2*len(display_groups),5.2),squeeze=False)
    for ax,label in zip(axes.flat,display_groups):axes_curves(ax,label,'initial')
    fig.suptitle('HDBSCAN groups and unassigned curves',fontsize=18,y=1.015);save('hdbscan_clusters_with_noise.png')
    for label in display_groups:
        stem=filename(label);members=a.loc[a.cluster==label]
        fig,ax=plt.subplots(figsize=(11,5.7));axes_curves(ax,label,'initial',True);save(stem+'.png')
        fig=go.Figure()
        for _,row in members.iterrows():
            c=curves[row.sample_id]
            fig.add_trace(go.Scatter(x=c['t'].tolist(),y=c['initial'].tolist(),mode='lines+markers',
                name=row.sample_id,showlegend=False,line=dict(color='rgba(64,133,187,.30)' if label>=0 else 'rgba(120,125,132,.4)',width=1),marker=dict(size=3),
                customdata=[[row.sample_id,row.figure_id,float(row.membership_probability)]]*len(c['t']),
                hovertemplate='%{customdata[0]} · %{customdata[1]}<br>Time: %{x:.2f} h<br>PCE/PCE0: %{y:.5f}<br>Membership: %{customdata[2]:.3f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=grid.tolist(),y=means[label,'initial'].tolist(),name=f'Ensemble mean (n={len(members)})',line=dict(color='red',width=4),hovertemplate='Descriptive mean<br>%{x:.2f} h · %{y:.5f}<extra></extra>'))
        med=curves[medoids[label]]
        fig.add_trace(go.Scatter(x=med['t'].tolist(),y=med['initial'].tolist(),name='Real medoid '+medoids[label],line=dict(color='black',width=2,dash='dash')))
        fig.update_layout(title=f'HDBSCAN {name(label)} · n={len(members)} · 300 h',xaxis_title='Elapsed time (hours)',yaxis_title='PCE / initial PCE',xaxis_range=[0,window],legend=dict(orientation='h',y=-.2))
        write_html(fig,stem+'_interactive.html')

    # Curve overview: every actual member, no blank reference placeholders.
    columns=8;rows=int(np.ceil(len(a)/columns));fig,axes=plt.subplots(rows,columns,figsize=(20,2*rows),squeeze=False)
    for ax,(_,r) in zip(axes.flat,a.iterrows()):
        c=curves[r.sample_id];ax.plot(c['t'],c['initial'],lw=1,color=COLORS[r.cluster]);ax.set_title(f'{r.sample_id} | {"Noise" if r.cluster<0 else "C"+str(r.cluster)}',fontsize=9);ax.tick_params(labelsize=7);ax.set_xlim(0,window)
    for ax in list(axes.flat)[len(a):]:ax.axis('off')
    fig.suptitle('All 79 analyzed curves: relative-to-initial display, original points, no smoothing',fontsize=17,y=1.005);save('overview_normalized.png')

    counts=[int((a.cluster==label).sum()) for label in display_groups]
    fig,ax=plt.subplots(figsize=(8.5,5));bars=ax.bar([name(k) for k in display_groups],counts,color=[COLORS[k] for k in display_groups]);ax.bar_label(bars,padding=4);ax.set_ylim(0,max(counts)*1.15);ax.set(ylabel='Number of curves',title='Series Count per HDBSCAN Group');save('hdbscan_cluster_counts.png')

    x=pd.read_csv(ROOT/'final'/'clustering_matrix.csv',index_col=0).loc[a.sample_id].to_numpy()
    fig,ax=plt.subplots(figsize=(9,6))
    for label in display_groups:
        ix=a.cluster.to_numpy()==label;ax.scatter(x[ix,0],x[ix,1],label=name(label),color=COLORS[label],s=45,alpha=.85)
    ax.set(xlabel='PC1',ylabel='PC2',title='PCA projection colored by frozen HDBSCAN assignment');ax.legend();save('pca_visualization.png')
    fig=go.Figure()
    for label in display_groups:
        ix=a.cluster.to_numpy()==label
        fig.add_trace(go.Scatter(x=x[ix,0].tolist(),y=x[ix,1].tolist(),mode='markers',name=name(label),marker=dict(color=COLORS[label],size=9),text=a.loc[ix,'sample_id'].tolist(),hovertemplate='%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>'))
    fig.update_layout(title='PCA projection of the frozen model',xaxis_title='PC1',yaxis_title='PC2');write_html(fig,'pca_visualization_interactive.html')

    finals={label:np.array([curves[s]['initial'][-1] for s in a.loc[a.cluster==label,'sample_id']]) for label in display_groups}
    for kind in ['boxplot','violin']:
        fig,ax=plt.subplots(figsize=(9,5))
        if kind=='boxplot':ax.boxplot(list(finals.values()),tick_labels=[name(k) for k in display_groups],showfliers=True)
        else:
            vp=ax.violinplot(list(finals.values()),showmedians=True,showextrema=True)
            for body,k in zip(vp['bodies'],display_groups):body.set_facecolor(COLORS[k]);body.set_alpha(.6)
            ax.set_xticks(np.arange(1,len(display_groups)+1),[name(k) for k in display_groups])
        ax.set(ylabel='Final observed PCE / initial PCE',title='Output distribution by group (one endpoint per curve)');save(f'pce_{kind}.png')
        pf=go.Figure()
        for label in display_groups:
            kwargs=dict(y=finals[label].tolist(),name=name(label),marker_color=COLORS[label])
            pf.add_trace(go.Box(**kwargs,boxpoints='all',jitter=.3) if kind=='boxplot' else go.Violin(**kwargs,box_visible=True,meanline_visible=True,points='all'))
        pf.update_layout(title='Final observed relative output by HDBSCAN group',yaxis_title='Final observed PCE / initial PCE');write_html(pf,f'pce_{kind}.html')
    a.assign(final_relative_output=[curves[s]['initial'][-1] for s in a.sample_id],last_time_in_window=[curves[s]['t'][-1] for s in a.sample_id]).to_csv(OUT/'cluster_summary.csv',index=False)

    fitted=pickle.loads((ROOT/'final'/'fitted_model.pkl').read_bytes())['model']
    fig,ax=plt.subplots(figsize=(10,6));fitted.condensed_tree_.plot(select_clusters=True,axis=ax,colorbar=True);ax.set_title('HDBSCAN condensed tree (frozen fitted model)');save('hdbscan_condensed_tree.png')
    order=np.concatenate([np.flatnonzero(a.cluster.to_numpy()==k) for k in display_groups]);d=squareform(pdist(x,metric='cityblock' if frozen['hdbscan']['metric']=='manhattan' else 'euclidean'))
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(d[np.ix_(order,order)],cmap='viridis');fig.colorbar(im,ax=ax,label='Distance in clustering feature space')
    for boundary in np.cumsum(counts)[:-1]:ax.axhline(boundary-.5,color='white',lw=.8);ax.axvline(boundary-.5,color='white',lw=.8)
    ax.set(title='Feature-space distances, ordered by HDBSCAN group',xlabel='Curve index (groups then noise)',ylabel='Curve index (groups then noise)');save('hdbscan_distance_map.png')
    summary=dict(algorithm='change points + kinetic descriptors + HDBSCAN',window_hours=window,n_curves=len(a),cluster_counts={str(k):v for k,v in zip(display_groups,counts)},
        classification_normalization=frozen['candidate']['normalization'],primary_display_normalization='y/y(0)',smoothing=False,model_retrained=False,
        ensemble_mean_only_postfit=True,mean_alignment=dict(kind='linear interpolation',step_hours=dt,common_observation_end=end,grid_end=float(grid[-1]),constant_contributors=True),
        noise_is_not_a_cluster=True,bootstrap_ari=frozen['candidate']['bootstrap_ari'],conclusion='Exploratory grouping; stable final classes not established',source_assignment_sha256=before)
    (OUT/'hdbscan_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    mapping=[('som_clusters.png','hdbscan_clusters.png','成员曲线+均值的分簇总览'),('som_cluster_N.png','hdbscan_cluster_N.png','单簇叠图'),
        ('som_cluster_N_interactive.html','hdbscan_cluster_N_interactive.html','悬停、缩放、图例交互'),('som_cluster_counts.png','hdbscan_cluster_counts.png','实际分簇数量'),
        ('som_umatrix.png / som_distance_map.png','hdbscan_condensed_tree.png / hdbscan_distance_map.png','使用本方法的层次与特征距离诊断'),
        ('overview_normalized.png','overview_normalized.png','全部已分析曲线'),('overview_savgol.png','不生成','原策略禁止平滑'),
        ('pca_visualization.png','pca_visualization.png','按冻结的HDBSCAN结果着色'),('pce_boxplot.html / pce_violin.html','同名文件','一条曲线贡献一个末端观测值')]
    pd.DataFrame(mapping,columns=['reference_file','this_result','purpose']).to_csv(OUT/'reference_output_mapping.csv',index=False)
    links='\n'.join(f'<a href="{filename(k)}_interactive.html">{name(k)} · {counts[i]} 条 · 交互曲线 →</a>' for i,k in enumerate(display_groups))
    cards='\n'.join(f'<article><h2>{name(k)} · {counts[i]} 条</h2><a href="{filename(k)}_interactive.html"><img src="{filename(k)}.png" alt="{name(k)} 曲线叠图"></a><p><a href="{filename(k)}_interactive.html">打开交互图：悬停查看样本、拖动缩放</a></p></article>' for i,k in enumerate(display_groups))
    (OUT/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HDBSCAN 无监督曲线聚类结果</title><style>
body{{margin:0;background:#f5f6f8;color:#202831;font:16px/1.65 system-ui,-apple-system,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px 22px}}h1{{font-size:30px;margin:0 0 8px}}h2{{font-size:21px}}.meta{{color:#5c6570}}.note{{padding:14px 20px;background:#fff3df;border-left:4px solid #d49335;margin:22px 0}}article{{background:white;padding:18px 24px;margin:22px 0;border:1px solid #e1e5ea;border-radius:10px}}img{{display:block;max-width:100%;height:auto}}a{{color:#206da5;text-decoration:none}}nav{{display:flex;gap:18px;flex-wrap:wrap}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}small{{color:#65707a}}</style></head><body><main>
<h1>无监督曲线聚类结果</h1><div class="meta">变点检测 → 动力学特征 → HDBSCAN · 300 小时 · 79 条曲线 · 2 个簇 + 8 条 noise</div>
<div class="note">本页按参考目录的“成员曲线 + 簇均值”形式展示。300 小时是探索性候选，bootstrap ARI ≈ 0.52，目前尚未建立稳定的最终分类。</div>
<nav>{links}</nav><article><h2>聚类总览</h2><a href="hdbscan_clusters.png"><img src="hdbscan_clusters.png" alt="HDBSCAN 分簇曲线与均值总览"></a><p>细线：所有成员的未平滑曲线；红线：聚类后的跨曲线均值；黑色虚线：真实 medoid。为便于阅读，此图使用 y/y(0) 展示，模型仍使用已冻结的 robust 归一化。</p><p><a href="hdbscan_clusters_model_normalization.png">查看模型实际归一化版本</a> · <a href="hdbscan_clusters_with_noise.png">查看包含 noise 的总览</a></p></article>
{cards}<div class="grid"><article><h2>曲线数量</h2><img src="hdbscan_cluster_counts.png" alt="各组数量"></article><article><h2>PCA 投影</h2><a href="pca_visualization_interactive.html"><img src="pca_visualization.png" alt="PCA投影"></a></article></div>
<article><h2>全部曲线</h2><a href="overview_normalized.png">打开 79 条曲线总览</a></article><nav><a href="pce_boxplot.html">末端输出箱线图</a><a href="pce_violin.html">末端输出小提琴图</a><a href="hdbscan_condensed_tree.png">HDBSCAN 层次结构</a><a href="hdbscan_distance_map.png">特征距离图</a><a href="../研究结果与方法报告.md">完整研究报告</a><a href="cluster_summary.csv">曲线分配 CSV</a></nav>
<p><small>红色均值线只用于结果展示，在所有曲线共同覆盖的时间范围内线性对齐后计算；网格不细于典型原始采样间隔。均值没有用于变点、特征提取或聚类。交互页面完全离线，使用同目录 plotly.min.js。</small></p></main></body></html>''')
    (OUT/'README.md').write_text(f'''# 最终展示图（参照 thesis/result）

打开 [index.html](index.html)，或直接看 [hdbscan_clusters.png](hdbscan_clusters.png)。

这一套展示读取本次冻结的 HDBSCAN 标签，按参考目录的成员叠图、均值线、单簇交互图、数量统计、PCA 和分布图形式输出。未重新训练，未改标签，未使用参考目录的类别或数据。

- Cluster 0：6 条；Cluster 1：65 条；Noise：8 条。Noise 单独展示，不作为第三个簇。
- 主要图用 y/y(0) 展示，以方便阅读归一化 PCE。模型使用的仍是 robust 归一化，另见 hdbscan_clusters_model_normalization.png。
- 原始成员线条全部保留。均值只在聚类后计算，网格间隔 {dt:.6f} 小时，所有曲线共同覆盖到 {end:.6f} 小时，均值网格末端 {grid[-1]:.6f} 小时；不外推，也不进行时间平滑。
- HDBSCAN 对应层次图和特征距离图；实际输出两个簇，不套用 SOM 的神经元位置或四格类别。
- 方法与原始研究结果见 [完整报告](../研究结果与方法报告.md)。bootstrap ARI 约 0.52，仍是探索性分组。

文件对应关系：[reference_output_mapping.csv](reference_output_mapping.csv)。图中每个原始点：[plotted_curve_points.csv](plotted_curve_points.csv)；均值：[cluster_mean_curves_display_only.csv](cluster_mean_curves_display_only.csv)。

复现：`.venv/bin/python code/export_result_gallery.py`（从上级研究结果目录运行）。所有交互图共用本目录的 plotly.min.js，移动时请一起保留。
''')
    assert before==hashlib.sha256((ROOT/'final'/'cluster_assignment.csv').read_bytes()).hexdigest()
    print(f'Gallery exported to {OUT}; labels unchanged; {len(a)} real curves; {len(groups)} clusters + noise.')

if __name__=='__main__':render()
