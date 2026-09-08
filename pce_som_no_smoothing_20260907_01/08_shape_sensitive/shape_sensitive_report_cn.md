# 形状敏感无监督对照

逐曲线 z-normalization 受 k-Shape 的形状归一化思想启发，但仍使用论文的 SOM、QE elbow 和中心重叠规则；没有 cross-correlation、时间扭曲、平滑或标签。

该对照自动选择 K=4，参数 `paper_lr_sensitivity`。冻结后中心的 IFO-like 描述计数为 `{'Slope-like': 2, 'Bridge-like': 1, 'Valley-like': 1}`。它分离出 Bridge-like 与 Valley-like 中心，但没有独立 Hill-like 中心。

此对照不替代主结果：共同的原始 MaxAbs 空间 silhouette 为 -0.1643，主论文流程为 0.5054。此外 z-normalization 刻意消除了幅度信息。因此主结论仍采用论文 MaxAbs 流程的 K=4。

`posthoc_individual_ifo_candidate_gallery.png` 只是在最终模型冻结后按峰谷形态连续评分列出个体候选，用于回答数据中是否能找到四类外观；它不是聚类标签，不能作为‘无监督发现四个簇’的证据。
