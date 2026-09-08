# 导数感知 DTW 无监督聚类结果

先打开 **[index.html](index.html)**，或直接查看 **[dtw_clusters.png](dtw_clusters.png)**。输出组织参考 `thesis/result`，训练方法采用你提供 Markdown 的 **multivariate DTW + 层次聚类 G1 分支**。

当前结果：399.43 h，91 条，z-normalization，λ=4，radius=20%，average linkage，K=2，簇规模 84/7。所有曲线未平滑；原始数据未修改。

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

红色均值是参数冻结后计算的显示汇总，不是原始样本，也不用于拟合。其公共显示网格间距 21.02 h，不细于典型原始间距 20.89 h。原始曲线仍按自身不等长时间点绘制。每簇 medoid 来自真实样本。

按原 Markdown 保留 `raw/`、`aligned/`、`features/`、`distance_matrices/`、`clustering_results/`、`stability/`、`figures/`。1,512 个距离矩阵、40,824 组参数标签以及候选重复检验均保留；细节见 `methods_summary.md` 和 `code/PROTOCOL.md`。`plotly.min.js` 是离线交互图依赖，请随 HTML 一起保留。

此目录采用已完成并验证的真实无监督计算，重新核验冻结距离得到的标签，并重新生成上述展示文件。没有为了模仿参考图指定四簇。参考目录的 Savitzky–Golay 平滑、SOM U-matrix/拓扑误差、KMeans 对照和按绝对初始效率分组不属于本次已选策略/可用数据，因此相应位置采用真实的导数总览、DTW 距离/层次图、三种 DTW 对照和按簇终点变化图。

复现整个计算见 `code/REPRODUCE.md`。重新导出本展示：`python3 code/export_visual_results.py`。导出器按当前已冻结的计算状态生成图，不调用人工曲线标签。分组对归一化敏感，且本次导数未改变 PCE-only 的标签；图形分离不能直接证明物理机制。
