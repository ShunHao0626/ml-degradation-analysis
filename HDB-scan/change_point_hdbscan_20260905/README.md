# 本次研究结果

**最终展示图入口：[result/index.html](result/index.html)。分簇曲线与均值总览：[hdbscan_clusters.png](result/hdbscan_clusters.png)。**

请先阅读 [研究结果与方法报告.md](研究结果与方法报告.md)。

最优候选：从首个有效观测点起 300 小时；79 条曲线，2 个簇，8 条 noise。**目前未找到足够稳定的最终分类窗口，也未建立四个稳定簇的结论。** 此候选仅供探索性分析。

原始数据未覆盖；完整代码、参数搜索、稳定性、敏感性和各簇真实曲线均已保存。

- [最终曲线分配](final/cluster_assignment.csv)
- [218 条曲线纳入/排除原因](final/all_218_sample_status.csv)
- [稳定性](stability/cluster_stability_summary.csv)
- [窗口比较图](figures/05_window_comparison.png)
- [完整流程](code/run_pipeline.py)
