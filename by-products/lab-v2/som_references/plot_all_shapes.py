"""Plot every accepted curve with per-curve affine scaling for shape inspection."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-shape-overlay-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'accepted' / 'samples_test'
OUT = ROOT / 'accepted' / 'shape_overlay'


def scale(values):
    span = np.ptp(values)
    return (values - values.min()) / span if span else np.full_like(values, 0.5)


def main():
    OUT.mkdir(exist_ok=True)
    curves = []
    for folder in sorted((p for p in SOURCE.iterdir() if p.is_dir()),
                         key=lambda p: int(p.name.removeprefix('图片'))):
        meta = json.loads((folder / 'validation_result.json').read_text())
        for path in sorted(folder.rglob('*.csv')):
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            x = np.array([float(r['x']) for r in rows])
            y = np.array([float(r['y']) for r in rows])
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError(f'Nonfinite values: {path}')
            parts = path.stem.split('__')
            axis = meta.get('axis', {})
            def label(key):
                a = axis.get(key) or {}
                return ' '.join(str(a[k]) for k in ('name', 'unit') if a.get(k)) or '未注明'
            notes = []
            if not (axis.get('x') or {}).get('name'):
                notes.append('横轴信息缺失')
            if not (axis.get('y') or {}).get('name'):
                notes.append('纵轴信息缺失')
            if folder.name in ('图片140', '图片208'):
                notes.append('原始数值尺度异常，待复核')
            if np.ptp(y) == 0:
                notes.append('常数曲线，显示于 y=0.5')
            curves.append(dict(id=len(curves)+1, record=folder.name,
                label=f'{folder.name} · {parts[-2]} · {parts[-1]}',
                source=str(path.relative_to(ROOT)), xaxis=label('x'), yaxis=label('y'),
                xmin=float(x.min()), xmax=float(x.max()), ymin=float(y.min()), ymax=float(y.max()),
                n=len(x), notes=notes,
                xy=np.column_stack((scale(x), scale(y))).round(8).tolist()))

    assert len(curves) == 218
    assert sum(c['n'] for c in curves) == 9674
    assert all(0 <= v <= 1 for c in curves for xy in c['xy'] for v in xy)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    for c in curves:
        xy = np.array(c['xy'])
        ax.plot(xy[:, 0], xy[:, 1], color='#2869a4', alpha=.22, lw=.9)
    ax.set(xlim=(-.015, 1.015), ylim=(-.025, 1.025),
           xlabel='Relative x position within each curve (0–1)',
           ylabel='Relative y position within each curve (0–1)',
           title='All 218 curves · individual min–max scaling')
    ax.grid(alpha=.17)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .025, 'Shape comparison only: original units, durations and amplitudes differ.\n'
             'All 9,674 points retained; no smoothing or interpolation. Constant curves are shown at y = 0.5.',
             ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(OUT / 'all_218_shapes.png', dpi=200)
    plt.close(fig)
    template = (ROOT / 'shape_overlay_template.html').read_text()
    data = json.dumps(curves, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (OUT / 'all_218_shapes.html').write_text(template.replace('__CURVES_JSON__', data))
    (OUT / 'curve_index.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2))
    (OUT / 'README_CN.md').write_text('''# 全部曲线形状叠加图

来源：`../samples_test`，99 个图片目录，218 条曲线，共 9,674 个原始点。所有曲线均包含，源文件未修改。

- `all_218_shapes.png`：全部曲线静态叠加图。
- `all_218_shapes.html`：可离线打开的交互图。悬停查看、点击锁定曲线，也可通过下拉菜单或前后按钮逐条查看；全部曲线始终保留为背景。
- `curve_index.json`：曲线编号、源路径、原始轴标签、范围和显示坐标。

每条曲线独立变换：x′=(x−min(x))/(max(x)−min(x))，y′=(y−min(y))/(max(y)−min(y))。
常数坐标显示于 0.5。保留所有点及原始连接顺序，不平滑、不插值、不截断；点之间用直线连接。

该图用于比较各曲线内部的形状。横轴不是统一的真实时间，纵轴不是统一的 PCE 或效率。
独立缩放会放大原本微小的波动，无法据此比较实际降幅、退化速度或寿命。
元数据缺失、循环次数曲线及疑似数值尺度异常的图片140/208也保留在图中；交互图在选中时显示相应提示。
''')
    print(json.dumps({'curves':len(curves),'points':sum(c['n'] for c in curves),
          'constant_y':[c['label'] for c in curves if c['ymin']==c['ymax']],
          'output':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
