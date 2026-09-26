# ML_SOM 工作区

[English](README.md) · 中文

这里保留曲线库构建、0–200 h SOM／形态研究和实验结果图集。固定 1,842 条曲线的四簇密度加权 KMeans 复现由[独立仓库](https://github.com/ShunHao0626/perovskite-early-degradation-clusters)维护，本目录已移除其重复代码和固定输入副本。

| 目录 | 内容 | Git 状态 |
|---|---|---|
| [`target_shape_classification/`](target_shape_classification/README.md) | 0–200 h 输入审计、SOM、形态判读及其上游探索 | 代码与方法文档公开；原始输入和逐曲线结果留本地 |
| [`canonical_2246_full_dataset_20260925/`](canonical_2246_full_dataset_20260925/README.md) | 2,246 条规范曲线的构建与核验 | 构建代码公开；生成数据留本地 |
| [`experiment_cluster_figures_20260926/`](experiment_cluster_figures_20260926/README.md) | 跨实验图集的收集脚本与索引说明 | 脚本公开；复制的图库留本地 |
| `data_final/`、`original_curves_2250/` | 上游曲线和来源文件 | 本地数据，不推送 |
| `si_1842_supplementary/` | 独有的旧 SI 文稿、图表及绘图数值 | 本地补充材料，不推送；复现以独立仓库为准 |

共享的 Hartono/PvkSOM 代码在仓库根目录 [`thesis/`](../thesis/README.md)，导数聚类代码在 [`derivative_clustering/`](../derivative_clustering/README.md)，不在此处保留第二份。总体研究边界见[仓库首页](../README_CN.md)。
