"""Recheck all data_all curves and render four target-shape groups."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-shape-overlay-mpl')
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SOURCE = REPO / 'lab/data_all'
PRIOR = REPO / 'lab-v2/ifo_four_shape_data_all_discovery'
spec = importlib.util.spec_from_file_location('prior_data_all_method', PRIOR/'run_pipeline.py')
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
CLASSES = method.CLASSES
PREFIX = dict(zip(CLASSES, 'BHSV'))
WINDOW = 500
COLORS = ['#407b47', '#849327', '#c98b21', '#b88e10']
DESCRIPTIONS = ['先升高，再缓慢下降', '先升高，快速下降后转缓',
                '先快速下降，再缓慢下降', '先下降、回升，再下降']
plt.rcParams.update({'font.family':'Arial Unicode MS', 'axes.unicode_minus':False,
                     'font.size':10})


def evaluate(r):
    k=r['strict_class']; a=float(r['early_slope']); b=float(r['late_slope'])
    field={'IFO-Bridge':'gain_abs','IFO-Hill':'gain_abs',
           'IFO-Slope':'early_drop_abs','IFO-Valley':'recovery_abs'}[k]
    effect=float(r[field])/max(float(r['start_value']), 1e-12)
    notes=[]
    if r['confidence_level']!='high': notes.append('已有规则扰动稳定性不足')
    if not r['prior_features_verified']: notes.append('当前特征与原稳定性评估不一致')
    if effect<.01: notes.append('关键变化幅度不足起始值1%')
    if b>=0: notes.append('末段斜率非负')
    if r['n_observed_in_window']<8: notes.append('500h内不足8个原始采样点')
    if r['first_value']<=0: notes.append('首个输出值非正')
    if k in ('IFO-Slope','IFO-Valley'):
        ratio=abs(a)/max(abs(b),1e-12)
        if a>=0 or ratio<1.5: notes.append('早期下降速度不足末段1.5倍')
    elif k=='IFO-Bridge':
        ratio=a/max(abs(b),1e-12)
        if a<=0 or ratio<1.5: notes.append('早期上升速度不足末段下降速度1.5倍')
    else:
        drop=float(r['early_peak_drop_abs'])/max(200-float(r['t_peak_fraction'])*WINDOW,1)
        ratio=drop/max(abs(b)/WINDOW,1e-12)
        if ratio<1.5: notes.append('峰后下降速度不足末段1.5倍')
    return notes,effect,ratio


def display_data(x,y):
    stop=np.interp(WINDOW,x,y);keep=x<WINDOW
    tx=np.r_[x[keep],float(WINDOW)]; yy=np.r_[y[keep],stop]
    shape=(yy-yy.min())/np.ptp(yy) if np.ptp(yy)>0 else np.full_like(yy,.5)
    return tx, yy/y[0] if y[0]>0 else np.full_like(yy,np.nan), shape


def representatives(rows):
    selected=[];dois=set()
    for r in sorted(rows,key=lambda r:r['template_correlation'],reverse=True):
        if r['doi'] in dois: continue
        selected.append(r);dois.add(r['doi'])
        if len(selected)==3:break
    return selected


def plot(rows,display,name,mode='shape',candidates=False,representative_only=False):
    fig,axes=plt.subplots(2,2,figsize=(15,10.4),sharex=True,sharey=(mode=='shape'))
    for k,ax,color,desc in zip(CLASSES,axes.flat,COLORS,DESCRIPTIONS):
        members=[r for r in rows if r['strict_class']==k]
        reps=representatives([r for r in members if r['status']=='clear_match'])
        ax.axvspan(0,200,color='#dbe8d8',alpha=.5,lw=0)
        ax.axvspan(200,WINDOW,color='#fff0ce',alpha=.45,lw=0)
        ax.axvline(200,color='#777777',ls='--',lw=.9)
        if not representative_only:
            for r in members:
                tx,rel,shape=display[r['curve_id']]
                ax.plot(tx,shape if mode=='shape' else rel,color=color,
                        alpha=max(.025,min(.25,3/np.sqrt(max(len(members),1)))),
                        lw=.7,ls='--' if r['status']=='boundary_candidate' else '-')
        for i,r in enumerate(reps):
            tx,rel,shape=display[r['curve_id']]
            ax.plot(tx,shape if mode=='shape' else rel,
                    color=['#205f87','#aa4c31','#654982'][i],lw=1.9,
                    label=r['plot_id']+' · '+(str(r['record'])[:26]+'…' if len(str(r['record']))>26 else str(r['record'])),
                    marker='.' if representative_only else None,ms=3)
        count_label=f'展示 {len(reps)} 条 / 本类 {len(members)} 条' if representative_only else f'{len(members)} 条'
        ax.set_title(f'{k} · {count_label}'+ ('（候选）' if candidates else '')+'\n'+desc,fontsize=15)
        ax.set(xlim=(0,WINDOW),xlabel='距首个采样点的时间（h）',
               ylabel='曲线内相对幅度（min–max）' if mode=='shape' else '输出 / 首个采样值')
        if mode=='shape':ax.set_ylim(-.03,1.06)
        ax.set_xticks([0,100,200,300,400,500]);ax.tick_params(labelbottom=True)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8,loc='best',framealpha=.9)
    title=f'data_all · 四类形态规则匹配（{len(rows)} 条）'
    if candidates:title=f'data_all · 全部四类形态候选（{len(rows)} 条）'
    if representative_only:title='data_all · 各类代表曲线（每类至多3条）'
    fig.suptitle(title,fontsize=21,y=.985)
    foot='真实小时尺度，首点对齐；展示 0–500 h；绘图不平滑。深色线为代表曲线，编号可追溯至原始 CSV。'
    foot+='\n'+('纵轴逐条缩放，仅用于看形状；须结合相对幅度图判断。' if mode=='shape' else '纵轴除以首值，保留相对变化幅度；各面板纵轴刻度独立。')
    if candidates:foot+=' 虚线为较弱、边界或采样稀疏候选。'
    fig.text(.5,.018,foot,ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.072,1,.952));fig.savefig(OUT/name,dpi=200);plt.close(fig)


def main():
    old=pd.read_csv(PRIOR/'06_final_four_classes/all_input_curves_final_status.csv').fillna('')
    old_features=pd.read_csv(PRIOR/'04_shape_features/final_shape_features_and_labels.csv').set_index('curve_id')
    sources=sorted(SOURCE.rglob('*.csv'))
    assert {str(p.relative_to(REPO)) for p in sources}==set(old.source_file)
    oldmap=old.set_index('source_file').to_dict('index')
    audit=[];eligible=[];smooth=[];display={}
    grid=np.arange(1,WINDOW*6+1)/6
    for p in sources:
        rel=str(p.relative_to(REPO));prev=oldmap[rel]
        vp=p.parent.parent/'validation_result.json'
        meta=json.loads(vp.read_text())
        ok,factor,axis_reason=method.classify_axis(meta,vp,p)
        x,y=method.read_curve(p)
        reasons=[]
        if not ok:reasons.append(axis_reason)
        if len(x)<4:reasons.append('少于4个唯一采样点')
        if not len(y) or not np.isfinite(y).all() or y.max()<=0:reasons.append('输出无效')
        if not np.isfinite(x).all():reasons.append('横坐标无效')
        tx=(x*factor-x.min()*factor) if factor and len(x) else np.array([])
        duration=float(tx.max()) if len(tx) else 0.
        if duration>method.CFG.max_plausible_duration_hours:reasons.append('异常超长时间')
        r={'curve_id':prev['curve_id'],'source_file':rel,'doi':meta.get('doi',''),
           'record':p.parent.parent.name,'series_name':p.stem,
           'x_name':(meta.get('axis',{}).get('x') or {}).get('name'),
           'x_unit':(meta.get('axis',{}).get('x') or {}).get('unit'),
           'y_name':(meta.get('axis',{}).get('y') or {}).get('name'),
           'y_unit':(meta.get('axis',{}).get('y') or {}).get('unit'),
           'time_factor_to_hours':factor,'duration_hours':duration,
           'n_unique_points':len(x),'n_observed_in_window':int(np.sum(tx<=WINDOW)),
           'first_value':float(y[0]) if len(y) else None,
           'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
           'confidence_level':prev['confidence_level'],'status':'',
           'review_reason':'；'.join(reasons)}
        if reasons:r['status']='excluded_axis_or_quality'
        elif duration<WINDOW:r['status']='insufficient_duration'
        else:
            raw=Akima1DInterpolator(tx,y)(grid)
            assert np.isfinite(raw).all()
            smooth.append(savgol_filter(raw/np.max(np.abs(raw)),71,2))
            eligible.append(r);display[r['curve_id']]=display_data(tx,y)
        audit.append(r)
    features,_=method.trajectory_features(np.vstack(smooth),WINDOW,np.array([r['curve_id'] for r in eligible]))
    params=json.loads((PRIOR/'04_shape_features/selected_shape_classifier.json').read_text())['selected_parameters']
    labels=method.assign_by_rules(features,params);strict=method.strict_topology_mask(features,labels)
    templates,_=method.target_templates()
    candidates=[];feature_mismatches=[];step_fraction_differences=[]
    for i,r in enumerate(eligible):
        f=features.iloc[i].to_dict();cid=r['curve_id']
        verified=cid in old_features.index and all(np.isclose(float(v),float(old_features.loc[cid,k]),rtol=1e-5,atol=1e-7)
                     for k,v in f.items() if k not in ('curve_id','fraction_negative_steps'))
        # Negative-step fractions are sensitive to roundoff on flat segments;
        # this diagnostic is not used in the saved classifier or its stability.
        if cid in old_features.index and not np.isclose(f['fraction_negative_steps'],old_features.loc[cid,'fraction_negative_steps'],rtol=1e-5,atol=1e-7):
            step_fraction_differences.append(cid)
        r.update(f);r['prior_features_verified']=verified
        if not verified:feature_mismatches.append(cid)
        r['strict_class']=str(labels[i]) if strict[i] else ''
        if not strict[i]:r['status']='other_or_unresolved_shape';continue
        notes,effect,ratio=evaluate(r)
        r.update(status='boundary_candidate' if notes else 'clear_match',review_reason='；'.join(notes),
                 key_effect_fraction_of_start=effect,rate_contrast=ratio)
        tx,_,shape=display[cid]
        line=np.interp(np.linspace(0,WINDOW,301),tx,shape)
        r['template_correlation']=float(np.corrcoef(line,templates[CLASSES.index(r['strict_class'])])[0,1])
        candidates.append(r)
    candidates.sort(key=lambda r:(CLASSES.index(r['strict_class']),r['source_file']))
    for k in CLASSES:
        for i,r in enumerate([r for r in candidates if r['strict_class']==k],1):r['plot_id']=f'{PREFIX[k]}{i:03d}'
    clear=[r for r in candidates if r['status']=='clear_match']
    boundary=[r for r in candidates if r['status']=='boundary_candidate']
    for name,rows in [('all_2151_classification.csv',audit),('clear_matches.csv',clear),
                      ('boundary_candidates.csv',boundary),('all_shape_candidates.csv',candidates)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    reps=[r for k in CLASSES for r in representatives([r for r in clear if r['strict_class']==k])]
    pd.DataFrame(reps).to_csv(OUT/'representative_curves.csv',index=False,encoding='utf-8-sig')
    plot(clear,display,'four_classes_clear_shapes.png')
    plot(clear,display,'four_classes_relative_output.png',mode='relative')
    plot(candidates,display,'four_classes_all_candidates.png',candidates=True)
    plot(clear,display,'representative_shapes.png',representative_only=True)
    summary={'source_curves':len(sources),'axis_quality_valid':sum(r['status']!='excluded_axis_or_quality' for r in audit),
             'window_eligible':len(eligible),'window_hours':WINDOW,'phase_boundary_hours':200,
             'clear_matches':dict(Counter(r['strict_class'] for r in clear)),
             'boundary_candidates':dict(Counter(r['strict_class'] for r in boundary)),
             'all_candidates':dict(Counter(r['strict_class'] for r in candidates)),
             'status_counts':dict(Counter(r['status'] for r in audit)),
             'recomputed_rule_feature_mismatches':feature_mismatches,
             'unused_negative_step_fraction_differences':len(step_fraction_differences),
             'extra_display_thresholds':{'min_key_effect_fraction':.01,'min_rate_contrast':1.5,'min_observed_points':8}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    table='\n'.join(f'| {k} | {summary["clear_matches"].get(k,0)} | {summary["boundary_candidates"].get(k,0)} |' for k in CLASSES)
    (OUT/'README_CN.md').write_text(f'''# data_all 四类形态归类

重新读取 `lab/data_all` 的全部{len(sources)}条CSV，重新核对轴信息、换算时间并计算形态特征。原始文件未修改，逐条SHA-256写入分类清单。

| 类别 | 本次规则匹配 | 边界候选 |
|---|---:|---:|
{table}

主图共{len(clear)}条；含边界候选共{len(candidates)}条。{len(eligible)}条覆盖统一500h窗口。
全部记录状态：{json.dumps(summary['status_counts'],ensure_ascii=False)}。

## 图与清单

- four_classes_clear_shapes.png：所有通过本次追加规则的曲线按四类叠加于一图；深色线为代表曲线。
- four_classes_relative_output.png：同一批曲线除以首值，保留相对幅度；各面板纵轴刻度独立。
- four_classes_all_candidates.png：全部原形态规则候选；边界候选用虚线。
- representative_shapes.png：每类至多3条代表曲线，便于观察具体形状。
- all_2151_classification.csv：全部输入的分类、来源、轴、时间单位、排除原因与文件哈希。
- clear_matches.csv / boundary_candidates.csv / all_shape_candidates.csv：三个对应集合。
- representative_curves.csv：图中深色代表曲线编号与原始路径。

## 方法与范围

这是按用户示意图进行的形态规则筛选，不是无监督聚类。复用并重新计算 `../ifo_four_shape_data_all_discovery` 已保存500h方案的预处理、峰谷特征与分类规则；没有运行SOM、重新搜索参数或加入合成样本。
与上批samples_test的750h图不同，本批沿用已有500h窗口，0–200h为早期，200h后为后期；因此条数不能和上批直接比较。

分类计算：按x排序，重复x取均值；核对轴含义；时间统一为小时、首点对齐零点；10min网格Akima插值、MaxAbs归一化、Savitzky–Golay(71,2)平滑，提取早期峰谷、增益、恢复及末段斜率。早期斜率拟合首10%（约0–50h），末段拟合末30%（约350–500h）。分钟/天/周分别使用1/60、24、168换算；月/年按730.5/8766小时近似。分区推断的单位与轴标签均保留在清单，不能视为逐篇原图人工确认。

展示主图追加：原规则稳定性为high且当前特征复核一致；关键变化至少为初值1%；末段下降；早期快变与末段下降速度比至少1.5（Hill比较峰后至200h的平均下降速度）；500h内至少8个实际点、首值为正。变化幅度分别是Bridge/Hill升高、Slope早期下降、Valley恢复。
1%、1.5倍、8点是本次可调整筛选阈值，不是统计显著性、文献通用标准或与示意图完全一致的保证。缺少时长/轴信息及不满足规则不等于绝不存在局部相似形态。

绘图重新使用原始点，不平滑；排序、重复x取均值，仅500h边界线性插值，不外推。形状图对窗口内每条曲线单独min–max缩放，可能放大小波动；必须结合相对幅度图。首点对齐后的200h并非核实过的实验绝对起点。
代表曲线从主图集合中按与示意形状的相关性选出，并优先不同DOI；此选择仅便于展示。没有去重相似/重复论文曲线，计数单位是CSV记录，不能当作独立器件数。

本次重算参与规则判定的形态特征与已有特征不一致的记录数：{len(feature_mismatches)}。另有{len(step_fraction_differences)}条的负向微步比例不同，此项不参与分类/规则稳定性判定；平段上的浮点微差会影响该比例，不据此排除曲线。
''')
    assert len({r['source_file'] for r in audit})==len(sources)
    assert len(clear)+len(boundary)==len(candidates)
    assert sum(summary['status_counts'].values())==len(sources)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
