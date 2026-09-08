"""Audit existing IFO topology candidates and render raw-data overlays.

Uses the repository's 750 h topology screening; adds explicit amplitude and
rate-contrast criteria for a clearer illustrative subset. Does not retrain or
force unclassified curves into a target class.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-shape-overlay-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIOR = REPO / 'lab-v2/ifo_four_shape_discovery/06_final_four_classes'
OUT = ROOT / 'accepted/ifo_shape_groups'
CLASSES = ['IFO-Bridge', 'IFO-Hill', 'IFO-Slope', 'IFO-Valley']
COLORS = ['#498349', '#8eaa2c', '#e4a328', '#cb8424']
DESCRIPTIONS = ['先升高，再缓慢下降', '先升高，快速下降后转缓',
                '先快速下降，再缓慢下降', '先下降、回升，再下降']
plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.titlesize': 15})


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def clear_reasons(r):
    kind = r['strict_class']
    early, late = float(r['early_slope']), float(r['late_slope'])
    field = {'IFO-Bridge': 'gain_abs', 'IFO-Hill': 'gain_abs',
             'IFO-Slope': 'early_drop_abs', 'IFO-Valley': 'recovery_abs'}[kind]
    effect = float(r[field]) / float(r['start_value'])
    reasons = []
    if r['confidence_level'] != 'high':
        reasons.append('已有规则扰动稳定性较低')
    if effect < .01:
        reasons.append('关键升高/下降/回升幅度小于起始值的1%')
    if late >= 0:
        reasons.append('末段未呈下降趋势')
    if kind in ('IFO-Slope', 'IFO-Valley'):
        ratio = abs(early) / max(abs(late), 1e-12)
        if early >= 0 or ratio < 1.5:
            reasons.append('早期下降速度不足末段的1.5倍')
    elif kind == 'IFO-Bridge':
        ratio = early / max(abs(late), 1e-12)
        if early <= 0 or ratio < 1.5:
            reasons.append('早期上升速度不足末段下降速度的1.5倍')
    else:
        peak_h = float(r['t_peak_fraction']) * 750
        drop_speed = float(r['early_peak_drop_abs']) / max(200-peak_h, 1)
        ratio = drop_speed / max(abs(late)/750, 1e-12)
        if ratio < 1.5:
            reasons.append('峰后至200h的下降速度不足末段的1.5倍')
    return reasons, effect, ratio


def load_display(r):
    data = read_rows(REPO / r['source_file'])
    x = np.array([float(p['x']) for p in data])
    y = np.array([float(p['y']) for p in data])
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    ux, indices, counts = np.unique(x, return_index=True, return_counts=True)
    y = np.add.reduceat(y, indices) / counts
    x = (ux - ux.min()) * float(r['time_factor_to_hours'])
    assert x[-1] >= 750
    # Only the 750 h endpoint is linearly interpolated. Interior plotted points
    # are the original observations; no smoothing is used for display.
    stop = float(np.interp(750, x, y))
    keep = x < 750
    x, y = np.r_[x[keep], 750.], np.r_[y[keep], stop]
    assert np.isfinite(y).all() and y[0] > 0
    shape = (y-y.min()) / np.ptp(y) if np.ptp(y) else np.full_like(y, .5)
    return x, y/y[0], shape


def plot(rows, filename, mode='shape', candidate=False, single=None):
    if single:
        fig, ax = plt.subplots(figsize=(13, 8.8))
        axes = [ax]
        kinds = [single]
    else:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10.4), sharex=True, sharey=True)
        axes = list(axs.flat)
        kinds = CLASSES
    for ax, kind in zip(axes, kinds):
        members = [r for r in rows if r['strict_class'] == kind]
        clear = sum(r['display_status'] == 'clear_match' for r in members)
        color = COLORS[CLASSES.index(kind)]
        ax.axvspan(0, 200, color='#dbe8d8', alpha=.5, lw=0)
        ax.axvspan(200, 750, color='#fff0ce', alpha=.45, lw=0)
        ax.axvline(200, color='#777777', ls='--', lw=.9)
        for i, r in enumerate(members):
            x, relative, shape = load_display(r)
            weak = r['display_status'] != 'clear_match'
            c = plt.get_cmap('tab20')(i % 20) if len(members) <= 5 or single else color
            label = r['plot_label'] + (' [候选]' if weak else '')
            ax.plot(x, shape if mode == 'shape' else relative,
                    color=c, alpha=.60 if len(members)>5 else .9,
                    lw=1.05 if len(members)>5 else 1.7,
                    ls='--' if weak else '-', label=label,
                    marker='.' if len(members)<=5 else None, ms=3)
        title = f'{kind} · {len(members)} 条'
        if candidate:
            title += f'（较明确 {clear}）'
        ax.set_title(title + '\n' + DESCRIPTIONS[CLASSES.index(kind)])
        ax.set(xlim=(0, 750), xlabel='距首个采样点的时间（h）',
               ylabel='曲线内相对幅度（min–max）' if mode=='shape' else '输出 / 首个采样值')
        ax.set_xticks([0, 200, 400, 600, 750])
        if mode == 'shape':
            ax.set_ylim(-.03, 1.06)
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        if len(members) <= 5:
            ax.legend(fontsize=8, loc='best', framealpha=.88)
        elif single:
            ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.15),
                      ncol=4, frameon=False)
        else:
            ax.text(.98, .96, f'各线对应样本见分类清单\n及 {kind} 单独图',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
    title = ('四类形态候选：全部 61 条' if candidate else
             f'四类形态：较明确的 {len(rows)} 条匹配曲线')
    if single:
        title = f'{single} · 较明确匹配曲线与编号'
    fig.suptitle(title, fontsize=21, y=.986)
    text = '保留真实小时尺度，首点对齐；展示 0–750 h；绘图不平滑。'
    text += ('纵轴逐条缩放，仅比较形状，会放大小波动。' if mode=='shape'
             else '纵轴除以各自首值，保留相对变化幅度。')
    if candidate:
        text += '\n虚线为较弱或边界候选；不等同于已确认的典型形态。'
    else:
        text += '\n规则形态筛选，非无监督聚类；1% 幅度、1.5 倍速率为本次展示阈值。'
    fig.text(.5, .017, text, ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .07 if not single else .075, 1, .947))
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    prior = read_rows(PRIOR / 'strict_four_class_matches.csv')
    all_rows = read_rows(PRIOR / 'all_218_curves_final_status.csv')
    sources = {str(p.relative_to(REPO)) for p in (ROOT/'accepted/samples_test').rglob('*.csv')}
    assert sources == {r['source_file'] for r in all_rows} and len(sources)==218
    manifest = read_rows(REPO/'lab-v2/ifo_four_type_matching_curves_export/export_manifest.csv')
    for m in manifest:
        assert hashlib.sha256((REPO/m['source_csv']).read_bytes()).hexdigest()==m['csv_sha256']
    prior.sort(key=lambda r:(CLASSES.index(r['strict_class']),int(r['record'][2:]),r['series_name']))
    for i, r in enumerate(prior, 1):
        reasons, effect, ratio = clear_reasons(r)
        r['display_status'] = 'boundary_candidate' if reasons else 'clear_match'
        r['review_reason'] = '；'.join(reasons)
        r['key_effect_fraction_of_start'] = effect
        r['rate_contrast'] = ratio
        parts = r['series_name'].split('__')
        r['plot_label'] = f'{i:02d} · {r["record"]} / {parts[-1]}'
    clear = [r for r in prior if r['display_status']=='clear_match']
    plot(clear, 'four_classes_clear_shapes.png')
    plot(clear, 'four_classes_relative_output.png', mode='relative')
    plot(prior, 'four_classes_all_candidates.png', candidate=True)
    for kind in CLASSES:
        plot([r for r in clear if r['strict_class']==kind], f'{kind}.png', single=kind)
    def write_csv(name, rows):
        with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv('clear_matches.csv',clear)
    write_csv('boundary_candidates.csv',[r for r in prior if r['display_status']!='clear_match'])
    lookup = {r['curve_id']:r for r in prior}
    for r in all_rows:
        m=lookup.get(r['curve_id'])
        r['display_status']=m['display_status'] if m else r['final_status']
        r['plot_label']=m['plot_label'] if m else ''
        r['review_reason']=m['review_reason'] if m else r['audit_reasons']
    write_csv('all_218_classification.csv',all_rows)
    summary={'source_curves':218,'verified_source_hashes':len(manifest),
             'clear_matches':dict(Counter(r['strict_class'] for r in clear)),
             'boundary_candidates':dict(Counter(r['strict_class'] for r in prior if r not in clear)),
             'status_counts':dict(Counter(r['display_status'] for r in all_rows)),
             'window_hours':750,'early_boundary_hours':200,
             'min_effect_fraction_of_start':.01,'min_rate_contrast':1.5}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'README_CN.md').write_text('''# 四类形态归类与叠加图

本次复核当前 samples_test 的218条源曲线，逐条来源集合一致；原有61条形态候选的源CSV SHA-256均与已有导出清单一致。没有修改原始数据。

## 结果

四类都有较明确的形态匹配：Bridge 3、Hill 2、Slope 36、Valley 1，共42条。
另保留19条较弱或边界候选：Bridge 0、Hill 1、Slope 15、Valley 3。
剩余26条未满足已有四类拓扑条件；68条不足统一750h窗口；63条轴/质量不合格。
不足750h及轴信息不合格并不表示不存在相似形状，本次不对它们强行分类。

## 文件

- four_classes_clear_shapes.png：42条较明确匹配，四类分别叠加于一张四分图。
- four_classes_relative_output.png：同样42条，纵轴为原始输出除以首个采样值，检查实际相对幅度。
- four_classes_all_candidates.png：原有全部61条候选，新增审查未通过者用虚线。
- IFO-Bridge.png / IFO-Hill.png / IFO-Slope.png / IFO-Valley.png：每类单独叠加，完整编号图例。
- clear_matches.csv / boundary_candidates.csv：类别、图中编号、来源及形态证据。
- all_218_classification.csv：全部218条记录及未纳入主图的原因。

## 分类依据与边界

复用 lab-v2/ifo_four_shape_discovery 的750h形态筛选及已保存的逐条特征，不重新训练。
原方法：核对时间与输出轴；天换算为小时，首点对齐零点；按x排序并平均重复x；Akima插值至10min网格并进行Savitzky–Golay平滑，提取峰谷、幅度与早晚斜率；0–200h为早期。早期斜率实际拟合首10%窗口（约0–75h），晚期斜率拟合末30%（约525–750h）。

本次追加展示筛选：已有规则稳定性为high；关键幅度至少为起始值1%；末段斜率为负。Bridge要求早期上升速度至少为末段下降速度1.5倍；Slope/Valley要求早期下降速度至少为末段1.5倍；Hill比较峰值至200h的平均下降速度与末段，要求至少1.5倍。关键幅度分别是Bridge/Hill的升高、Slope的早期下降、Valley的回升。

1%和1.5倍是本次用于区分明显形态与边界候选的可调整展示阈值，不是文献统一标准、统计显著性或实验真实性判定；这里的“较明确”仅指通过这些规则。阈值改变会改变条数。Valley只有图片114/Series_2进入较明确集合，回升幅度约为首值3.8%；其余三个Valley候选回升不足1%。

## 绘图

重新读取原CSV，不使用平滑曲线绘制；按x排序、平均重复x，单位转换后以首个采样点为零。仅显示0–750h；750h终点用线性插值，其余显示点均来自原始记录。不进行外推。200h虚线是距首个采样点的时间，不等同于已核实的实验绝对起点。
形状图对每条曲线在当前窗口独立min–max缩放；幅度对照图除以首值。时间轴不缩放到0–1。各曲线物理输出指标和实验条件可能不同，图只用于形态观察，不推断材料机理、寿命优劣或统计独立样本数。
''')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
