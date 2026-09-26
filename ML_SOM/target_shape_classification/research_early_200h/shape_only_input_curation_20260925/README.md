# 无标签输入质量筛选后的四簇实验

这是[上一轮四簇探索](../shape_only_four_group_exploration_20260925/README.md)的**独立新结果目录**。本轮固定上一轮的形状表示、密度加权 KMeans、四簇数量及随机种子，只调整输入数据的最低实测点数；聚类拟合与本轮门槛选择都不读取旧四类候选。上一轮的特征／模型方案曾受目标外观引导，故不能把整条研究链称作从零开始的严格无监督发现。

## 看结果

- [按参考风格绘制的四簇放大图 PNG](results/anonymous_four_groups_zoom.png)（[SVG](results/anonymous_four_groups_zoom.svg)）：彩色淡线是该匿名簇全部曲线，黑线是平滑后的逐序位中位数。各面板的纵轴自动按本簇中位数放大，范围外成员片段会被裁切；横轴为相对观测序位，无 hour。
- [统一纵轴图](results/anonymous_four_groups.png)与[逐曲线交互查看器](results/anonymous_cluster_detail_viewer.html)。
- [全部匿名归属](results/anonymous_assignments.csv)、[255 条暂缓输入曲线及原因](results/excluded_input_curves.csv)、[结果与限制](RESULTS.md)。
- [无标签门槛选择记录](SELECTION.md)及[数值核验](results/verification.json)。

## 输入与复现

上游有 2152 条 0–200 h 曲线，各有至少 4 个不同实测时刻。本轮要求至少 8 个不同实测时刻；原始测点总变差 / max(观测范围, 0.005) 仍须不大于 3。结果保留 **1897** 条：238 条因不足 8 点、17 条因高频起伏暂缓。比上一轮 2135 条的主结果额外暂缓 **238** 条，未用满用户允许的约 300 条。

在工作区根目录依次运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 target_shape_classification/research_early_200h/shape_only_input_curation_20260925/run.py
python3 target_shape_classification/research_early_200h/shape_only_input_curation_20260925/posthoc_audit.py
python3 target_shape_classification/research_early_200h/shape_only_input_curation_20260925/sensitivity.py
python3 target_shape_classification/research_early_200h/shape_only_input_curation_20260925/make_viewer.py
python3 target_shape_classification/research_early_200h/shape_only_input_curation_20260925/verify.py
```

无标签筛选复现脚本为 [`unlabeled_screen.py`](unlabeled_screen.py) 与 [`unlabeled_repeat.py`](unlabeled_repeat.py)。参数见 [`config.json`](config.json)。原始 CSV 与上一轮输出未改写。
