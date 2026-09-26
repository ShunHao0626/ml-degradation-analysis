# 方法文献与本轮取舍

1. Hartono et al., *Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset*, Nature Communications 14, 4869 (2023), DOI: 10.1038/s41467-023-40585-3。原工作使用 150 h、归一化轨迹和 SOM，并以 QE elbow 与中心可区分性判断节点数。本轮保留 SOM/QE 核心，但按用户要求关闭原代码中的 Savitzky–Golay smoothing。
2. Javed et al., *SOMTimeS: Self Organizing Maps for Time Series Clustering*, Data Mining and Knowledge Discovery (2023), DOI: 10.1007/s10618-023-00979-9。该方法说明 DTW 可嵌入 SOM；但本数据的绝对老化时间具有物理意义，宽松时间扭曲可能把 200 h 前后的峰谷对齐成同一形态。本轮先采用保持物理时间的多视角有限差分表征，并把受限 DTW 留作独立敏感性方法。
3. Liang et al., *TimeCSL: Unsupervised Contrastive Learning of General Shapelets for Explorable Time Series Analysis*, PVLDB 17(12), 2024, DOI: 10.14778/3685800.3685907。无监督 shapelet 对局部形态发现有吸引力；但本数据是稀疏、异构的文献图像数字化曲线，且当前环境无其深度学习依赖。直接训练会引入不可核验的增强假设，因此本轮使用可审计的 level/difference 多视角表征作为保守近似。
4. Li et al., *Using dynamic time warping self-organizing maps to characterize diurnal patterns in environmental exposures*, Scientific Reports 12, 2022, DOI: 10.1038/s41598-021-03584-4。支持形状感知 SOM 的可行性，同时提醒相似度定义会改变聚类结构。
5. Paparrizos & Gravano, *k-Shape: Efficient and Accurate Clustering of Time Series*, SIGMOD 2015, DOI: 10.1145/2723372.2737793。逐曲线 z-normalization 有助于减少幅值主导；本轮纳入 z-level，并额外加入未平滑的一阶和二阶有限差分视角。

本轮不使用任何半监督 shapelet、must-link/cannot-link 约束或基于目标四类的特征选择，因为这些会改变“全过程无监督”的性质。
