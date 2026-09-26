# 0–200 h 形状优先的四簇实验

目标是在**聚类拟合时不使用四类名称或旧规则标签**，尽量保留可用曲线，让匿名簇呈现 Bridge、Hill、Slope、Valley 所描述的四种形状。这里的“四簇”是模型设定；四类名称只用于完成聚类后的人工解释。特征与筛选门槛的选择参考了目标外观和旧自动候选，因此**整个研究与模型选择过程并非完全无监督**。旧候选也不是独立真值。

## 直接查看

- [逐曲线交互放大页](results/anonymous_cluster_detail_viewer.html)：切换簇、输入 ID、局部放大纵轴；蓝线为插值，橙点为原始测点。
- [四簇统一纵轴图](results/anonymous_four_groups.png)与[曲线叠加局部放大图](results/anonymous_four_groups_zoom.png)（[SVG](results/anonymous_four_groups_zoom.svg)）。放大图按参考风格显示半透明的簇成员曲线与黑色平滑逐序位中位数；横轴为相对观测序位，不标 hour。各面板的纵轴范围不同，超出范围的线段会被裁切。
- [匿名簇归属](results/anonymous_assignments.csv)、[被筛除的 17 条](results/excluded_rough_curves.csv)、[全部 2152 条均保留的对照版本](results/all_curves_variant_assignments.csv)。
- [具体结果与限制](RESULTS.md)。

## 输入与方法

从既有严格 0–200 h 数值轨迹的 2207 条旧自动候选出发，沿用[形状 DTW 实验](../shape_only_unsupervised_20260925/RESULTS.md)制作的 2152 条输入：每条至少 4 个不同观测时刻。55 条少于 4 个时刻，不能可靠显示峰谷转折，未进入本实验。横轴仅用原始测点的**相对序位**；真实 hour 只排序，不进入聚类特征。

每条曲线按自身观测范围缩放，并以 0.01 归一化 PCE 为幅度下限，避免极小噪声被无限放大，同时保留约 1% 的形状变化。曲线在 64 个相对序位插值，轻度平滑后用 5 个 Chebyshev 形状系数表示；稳健缩放、局部密度加权 KMeans 拟合 4 簇。聚类程序 [`run.py`](run.py)不读取旧四类标签、源分组、绝对小时或人工作答。

主结果对原始测点计算总变差 / max(观测范围, 0.005)，仅将比值大于 3 的 17 条高频反复起伏曲线留在“待单独检查”清单。因此正式四簇保留 2135/2152（99.2%）条可聚类输入，约占 2207 条旧数值候选的 96.7%。这 17 条仍可在全部保留的对照版本中找到归属；筛除不等于判定它们是错误数据。

## 复现

在工作区根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/run.py
python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/all_curves_variant.py
python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/posthoc_audit.py
python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/sensitivity.py
python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/make_viewer.py
python3 target_shape_classification/research_early_200h/shape_only_four_group_exploration_20260925/verify.py
```

参数与相对输入路径见 [`config.json`](config.json)。首次执行前需保留上游形状 DTW 实验的 `results/input_index.csv`、`results/shape_inputs.npz` 与原始点文件。修改参数后需重新生成全部结果，旧的事后审计不能视为新运行的验证。
