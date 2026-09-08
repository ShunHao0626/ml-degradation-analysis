#!/usr/bin/env python3
from run_experiments import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import html

rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                 'path.simplify':False,'font.size':10,'savefig.dpi':180})
COLORS=['#2563eb','#dc2626','#059669','#9333ea','#ea580c','#0891b2','#a16207','#475569',
        '#db2777','#65a30d','#6366f1','#0d9488','#d97706','#be123c','#7c3aed','#334155']

def panel(curves,labels,X,out,title,norm='minmax',medoids=None):
    ids=np.unique(labels);cols=min(3,int(np.ceil(np.sqrt(len(ids)))));rows=int(np.ceil(len(ids)/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(5*cols,3.35*rows),squeeze=False,sharex=True,sharey=True)
    stats=[]
    for j,ax in zip(ids,axes.flat):
        ix=np.flatnonzero(labels==j);m=int(ix[np.argmin(cdist(X[ix],X[ix]).sum(axis=1))]) if medoids is None else int(medoids[j])
        for i in ix:ax.plot(curves[i]['u'],values(curves[i],norm),color=COLORS[j%16],alpha=.20,lw=.8)
        ax.plot(curves[m]['u'],values(curves[m],norm),color='#111827',lw=1.8,marker='.',ms=3,label=f'Raw medoid: {curves[m]["id"]}')
        ax.set_title(f'Cluster {j+1} | n={len(ix)}');ax.legend(loc='best',fontsize=8);ax.grid(alpha=.15)
        ax.set_xlabel('Relative observed progress (0-1)');ax.set_ylabel(f'Output ({norm})')
        stats.append({'cluster':int(j+1),'n':len(ix),'medoid':curves[m]['id'],'medoid_figure':curves[m]['meta']['figure']})
    for ax in list(axes.flat)[len(ids):]:ax.axis('off')
    fig.suptitle(title+'\nAll original points; no smoothing. Black = an observed curve, not a fitted archetype.',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(out);plt.close(fig)
    return stats

def gallery(curves,labels):
    path=ROOT/'figures/all_curves';path.mkdir(exist_ok=True)
    for page,start in enumerate(range(0,len(curves),12),1):
        fig,axes=plt.subplots(4,3,figsize=(15,13))
        for i,ax in zip(range(start,min(start+12,len(curves))),axes.flat):
            c=curves[i];m=c['meta'];ax.plot(c['u'],values(c,'minmax'),'.-',color=COLORS[labels[i]%16],lw=1,ms=3)
            span=f'{m["duration_h"]:.1f} h' if m['duration_h'] is not None else m['axis_status']
            ax.set_title(f'{c["id"]} | cluster {labels[i]+1} | {span}\n{m["n_points"]} points; relative amplitude {m["relative_y_range"]:.3g}',fontsize=9)
            ax.set_xlabel('Relative observed progress');ax.set_ylim(-.07,1.07);ax.grid(alpha=.15)
        for ax in list(axes.flat)[min(12,len(curves)-start):]:ax.axis('off')
        fig.suptitle(f'Every curve / page {page:02d} - MinMax view, all original points, no smoothing',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(path/f'page_{page:02d}.png');plt.close(fig)

def main():
    curves=json.loads((ROOT/'data/raw_curves.json').read_text());inventory=pd.read_csv(ROOT/'data/curve_inventory.csv')
    sel=json.loads((ROOT/'experiments/selection.json').read_text());df=pd.read_csv(ROOT/'experiments/final_sweep.csv')
    X=np.load(ROOT/'data/representation_minmax.npz')['X'];model=np.load(ROOT/'experiments/models_minmax.npz');labels=model['labels']
    stats=panel(curves,labels,X,ROOT/'figures/01_selected_clusters.png',f'Primary shape SOM: QE elbow selects k={sel["minmax"]["selected_k"]}')
    dump(ROOT/'experiments/cluster_representatives.json',stats)
    panel(curves,labels,X,ROOT/'figures/02_same_clusters_maxabs.png','Same primary membership, amplitude-preserving display',norm='maxabs')
    for norm in ['maxabs','zscore']:
        z=np.load(ROOT/f'experiments/models_{norm}.npz');xx=np.load(ROOT/f'data/representation_{norm}.npz')['X']
        panel(curves,z['labels'],xx,ROOT/f'figures/alternative_{norm}.png',f'{norm} SOM: k={sel[norm]["selected_k"]}',norm=norm)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for norm,ax in zip(NORMS,axes):
        s=df[df.norm==norm].groupby('k').qe.agg(['median','min','max']);ax.plot(s.index,s['median'],'o-',color='#2563eb');ax.fill_between(s.index,s['min'],s['max'],alpha=.2)
        k=sel[norm]['selected_k'];ax.axvline(k,color='#dc2626',ls='--',label=f'Chord elbow: {k}');ax.legend();ax.set_title(f'{norm} | QE across 5 seeds');ax.set_xlabel('Number of SOM nodes');ax.set_ylabel('Mean L2 distance to BMU');ax.grid(alpha=.15)
    fig.suptitle('Paper criterion: quantization-error elbow (unsmoothed error curves)');fig.tight_layout();fig.savefig(ROOT/'figures/03_qe_elbows.png');plt.close(fig)
    k=sel['minmax']['selected_k']
    for kk in sorted(set([max(2,k-1),k+1,k+2])):
        panel(curves,model[f'labels_k{kk}'],X,ROOT/f'figures/neighbor_k{kk}.png',f'Adjacent k={kk}: inspect redundancy and hidden subgroups')
    # Cluster-center similarity is descriptive evidence for the paper's overlap check.
    overlap=[]
    for kk in KS[1:]:
        l=model[f'labels_k{kk}'];cent=[];rad=[]
        for j in np.unique(l):
            xx=X[l==j];c=xx.mean(axis=0);cent.append(c);rad.append(np.linalg.norm(xx-c,axis=1).mean())
        for a,b in itertools.combinations(range(len(cent)),2):
            dist=float(np.linalg.norm(cent[a]-cent[b]));overlap.append({'k':kk,'cluster_a':a+1,'cluster_b':b+1,'center_distance':dist,'within_radius_sum':rad[a]+rad[b],'separation_to_spread':dist/max(rad[a]+rad[b],1e-15)})
    pd.DataFrame(overlap).to_csv(ROOT/'experiments/cluster_overlap.csv',index=False)
    z=np.load(ROOT/'experiments/time_only_model.npz');ids=z['indices'];ts=json.loads((ROOT/'experiments/time_only_selection.json').read_text())
    panel([curves[i] for i in ids],z['labels'],X[ids],ROOT/'figures/04_verified_time_clusters.png',f'Confirmed time axes only: n={len(ids)}, k={ts["selected_k"]}')
    # Actual elapsed hours: no invented extension and no removal from main analysis.
    fig,ax=plt.subplots(figsize=(12,5))
    for i in ids:
        c=curves[i];x=(np.array(c['x'])-c['x'][0])*c['meta']['hours_factor'];ax.plot(x,values(c,'maxabs'),color=COLORS[labels[i]%16],alpha=.25,lw=.8)
    ax.set_xscale('symlog',linthresh=10)
    ax.set_xlabel('Elapsed hours from first observed point (symlog; raw scale anomalies retained)');ax.set_ylabel('Output / max absolute output');ax.set_title('159 curves with documented time units; two image-140 curves have suspect x scales');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(ROOT/'figures/05_actual_hours.png');plt.close(fig)
    elastic=json.loads((ROOT/'experiments/elastic_selection.json').read_text())
    for name,s in elastic.items():
        z=np.load(ROOT/f'experiments/elastic_model_{name}.npz');panel(curves,z['labels'],X,ROOT/f'figures/elastic_{name}.png',f'Median SOM / {name}: generalized QE elbow k={s["selected_k"]}',medoids=z['medoids'])
    assignments=inventory.copy();assignments['primary_cluster']=labels+1
    for norm in ['maxabs','zscore']:assignments[f'{norm}_cluster']=np.load(ROOT/f'experiments/models_{norm}.npz')['labels']+1
    for name in elastic:assignments[f'{name}_cluster']=np.load(ROOT/f'experiments/elastic_model_{name}.npz')['labels']+1
    assignments.to_csv(ROOT/'cluster_assignments.csv',index=False)
    # Each curve retains its exact original and both coordinate transforms.
    export=ROOT/'data/transformed_curves';export.mkdir(exist_ok=True)
    for c,j in zip(curves,labels):
        m=c['meta'];x=np.array(c['x']);hour=(x-x[0])*m['hours_factor'] if m['hours_factor'] is not None else np.full(len(x),np.nan)
        pd.DataFrame({'original_x':x,'original_y':c['y'],'relative_progress':c['u'],'elapsed_h_if_verified':hour,'y_minmax':values(c,'minmax'),'y_maxabs':values(c,'maxabs'),'primary_cluster':j+1}).to_csv(export/f'{c["id"]}.csv',index=False)
    gallery(curves,labels)
    # Offline interactive raw-point viewer, all source data inline; no network dependencies.
    payload=[]
    for c,j in zip(curves,labels):
        cc=dict(c);cc['cluster']=int(j+1);cc['minmax']=values(c,'minmax').tolist();cc['maxabs']=values(c,'maxabs').tolist();payload.append(cc)
    js=get_plotlyjs()
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>无平滑曲线探索</title>
<style>body{font:15px system-ui;margin:24px;background:#f8fafc;color:#172033}h1{font-size:25px}select,button{padding:8px;margin:5px}#plot{height:650px;background:white}.note{max-width:1100px;line-height:1.7}#detail{white-space:pre-wrap;font-size:13px}a{color:#2563eb}</style>
<h1>无监督曲线探索 · 全部 218 条 · 无平滑</h1>
<p class="note">主结果使用论文的 QE 肘部判据。相对进程 0–1 表示各自完整观测区间，不能当作相同真实小时或寿命。MinMax 会放大小波动，可切换 MaxAbs 核对实际相对幅度。原始横轴含小时、循环次数和未知单位。</p>
<p><a href="REPORT_CN.md">完整报告</a> · <a href="cluster_assignments.csv">曲线归属</a> · <a href="figures/01_selected_clusters.png">主结果图片</a> · <a href="figures/03_qe_elbows.png">QE 判断图</a></p>
<label>分组 <select id="group"></select></label><label>曲线 <select id="curve"></select></label>
<button id="prev">上一条</button><button id="next">下一条</button>
<label>横轴 <select id="axis"><option value="relative">相对进程 0–1</option><option value="hours">实际小时（仅确认单位者）</option><option value="original">原始横坐标（注意单位混合）</option></select></label>
<label>纵轴 <select id="norm"><option value="minmax">MinMax 形态</option><option value="maxabs">MaxAbs 相对幅度</option><option value="y">原始输出（注意量纲混合）</option></select></label>
<div id="plot"></div><p class="note" id="detail"></p><script>'''+js+'''</script><script>
const data=PAYLOAD, colors=COLORS;const $=id=>document.getElementById(id);
$('group').innerHTML='<option value="0">全部曲线</option>'+[...new Set(data.map(d=>d.cluster))].map(j=>`<option value="${j}">Cluster ${j}</option>`).join('');
function candidates(){return data.filter(d=>(!+$('group').value||d.cluster===+$('group').value)&&($('axis').value!=='hours'||d.meta.hours_factor!==null));}
function options(){let old=$('curve').value;$('curve').innerHTML='<option value="">叠加显示</option>'+candidates().map(d=>`<option value="${d.id}">${d.id} · ${d.meta.figure} · Cluster ${d.cluster}</option>`).join('');if([...$('curve').options].some(o=>o.value===old))$('curve').value=old;draw();}
function draw(){const mode=$('axis').value,norm=$('norm').value,chosen=$('curve').value;
const ds=candidates(),traces=ds.map(d=>({x:mode==='relative'?d.u:mode==='hours'?d.x.map(x=>(x-d.x[0])*d.meta.hours_factor):d.x,y:d[norm],mode:chosen===d.id?'lines+markers':'lines',type:'scatter',name:d.id+' / C'+d.cluster,opacity:chosen?(chosen===d.id?1:.09):.5,line:{color:chosen===d.id?'#111827':colors[(d.cluster-1)%colors.length],width:chosen===d.id?2.4:1,simplify:false},marker:{size:5},hovertemplate:d.id+' / Cluster '+d.cluster+'<br>x=%{x}<br>y=%{y}<extra></extra>'}));
if(chosen){const idx=ds.findIndex(d=>d.id===chosen);if(idx>=0)traces.push(traces.splice(idx,1)[0]);}
Plotly.react('plot',traces,{title:`${ds.length} 条曲线 · ${chosen||'全部叠加'} · 无平滑`,xaxis:{title:mode==='relative'?'各自完整观测进程（0–1）':mode==='hours'?'自首个观测点起的实际小时':'原始横坐标（单位可能不同）'},yaxis:{title:norm},showlegend:false,hovermode:'closest',margin:{t:55,r:35,b:60,l:70},paper_bgcolor:'white'}, {responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:3}});
let d=data.find(d=>d.id===chosen);$('detail').textContent=d?JSON.stringify({曲线:d.id,主簇:d.cluster,原始信息:d.meta},null,2):'选择一条曲线查看原始点、路径和测量时长。所有连线均为原始点之间的直线。';}
$('group').onchange=options;$('axis').onchange=options;$('curve').onchange=draw;$('norm').onchange=draw;
function step(v){let s=$('curve');s.selectedIndex=Math.max(0,Math.min(s.options.length-1,s.selectedIndex+v));draw();}$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);options();
</script></html>'''.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False)).replace('COLORS',json.dumps(COLORS))
    (ROOT/'index.html').write_text(doc,encoding='utf-8')
    print('RENDERED',stats,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
