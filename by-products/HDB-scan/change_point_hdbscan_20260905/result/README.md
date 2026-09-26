# 最终展示图（参照 thesis/result）

打开 [index.html](index.html)，或直接看 [hdbscan_clusters.png](hdbscan_clusters.png)。

这一套展示读取本次冻结的 HDBSCAN 标签，按参考目录的成员叠图、均值线、单簇交互图、数量统计、PCA 和分布图形式输出。未重新训练，未改标签，未使用参考目录的类别或数据。

- Cluster 0：6 条；Cluster 1：65 条；Noise：8 条。Noise 单独展示，不作为第三个簇。
- 主要图用 y/y(0) 展示，以方便阅读归一化 PCE。模型使用的仍是 robust 归一化，另见 hdbscan_clusters_model_normalization.png。
- 原始成员线条全部保留。均值只在聚类后计算，网格间隔 17.467785 小时，所有曲线共同覆盖到 277.015615 小时，均值网格末端 262.016770 小时；不外推，也不进行时间平滑。
- HDBSCAN 对应层次图和特征距离图；实际输出两个簇，不套用 SOM 的神经元位置或四格类别。
- 方法与原始研究结果见 [完整报告](../研究结果与方法报告.md)。bootstrap ARI 约 0.52，仍是探索性分组。

文件对应关系：[reference_output_mapping.csv](reference_output_mapping.csv)。图中每个原始点：[plotted_curve_points.csv](plotted_curve_points.csv)；均值：[cluster_mean_curves_display_only.csv](cluster_mean_curves_display_only.csv)。

复现：`.venv/bin/python code/export_result_gallery.py`（从上级研究结果目录运行）。所有交互图共用本目录的 plotly.min.js，移动时请一起保留。
