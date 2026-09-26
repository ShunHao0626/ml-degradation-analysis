# 文献依据与本次取舍

检索日期：2026-09-06。使用原始论文、作者稿或出版社页面；以下不是对所有可行算法的穷尽比较。

1. **Hartono et al., 2023, Nature Communications — Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset.** [正文](https://www.nature.com/articles/s41467-023-40585-3)。本地依据为 `thesis/paper/work.md` 和 `Supplementary.md` 的 “SOM Quantisation Error”。原文使用 MaxAbs、SOM，以及 QE 随类别数变化的肘部，再检查簇形状重叠。本次保留 SOM 和这套判断逻辑，取消其平滑、150 h 截断及短轨迹排除。

2. **Javed et al., online 2023 / volume 2024 — SOMTimeS.** [出版社全文](https://link.springer.com/article/10.1007/s10618-023-00979-9)。将 DTW 用于 SOM 的匹配，并用距离上下界减少计算。本次借鉴“弹性距离与 SOM 结合”；实际实现为原始变长序列上的 median SOM，并非复现 SOMTimeS，尤其不声称复现其加速或基准准确率。

3. **Conan-Guez, Rossi & El Golli, 2006 — Fast algorithm and implementation of dissimilarity self-organizing maps.** [作者稿](https://apiacoa.org/publications/2006/conan-guezrossietal2006neural-networks.pdf)。给出使用不相似度矩阵的批量 SOM，原型取实际观测样本。本次采用此思想，避免用平均原型掩盖代表曲线细节；增加原型唯一性分配约束以处理重复原型，故是明确的实现变体。

4. **Paparrizos & Bogireddy, 2025, PVLDB — Time-Series Clustering: A Comprehensive Study of Data Mining, Machine Learning, and Deep Learning Methods.** [原始论文](https://www.vldb.org/pvldb/vol18/p4380-paparrizos.pdf)。大规模比较提醒：更新、更复杂的模型不自动带来更好的时间序列聚类；其结果不能直接证明在本数据上的优劣。因此本次依靠本地无监督判据和稳定性，不以论文年份选择胜者，也未使用预训练标签模型。

5. **Holder & Bagnall, 2026 — Rock the KASBA: blazingly fast and accurate time series clustering.** [出版社全文](https://link.springer.com/article/10.1007/s10618-026-01189-9)。2026-02-25 正式发表，强调度量弹性距离、MSM、初始化与高效中心更新。本次测试 MSM 距离与 median SOM 的组合，不是 KASBA 全算法。原始不同点数序列上的 MSM 在本数据中呈现点数敏感性，未成为推荐结果。

6. **Satopää et al., 2011 — Finding a Kneedle in a Haystack.** [作者论文](https://www.cs.williams.edu/~jeannie/papers/kneedle-simplex11.pdf)。提供肘部自动识别的背景。本次只用无平滑的归一化端点连线最大偏离法，并以连续分段线性拟合复核，**不是完整 Kneedle 实现**，也不是原论文提供的自动公式。

7. **Time series clustering with random convolutional kernels, 2024.** [出版社全文](https://link.springer.com/article/10.1007/s10618-024-01018-x)。作为表示学习候选检索；本次没有实现。218 条、每条仅 6–105 点且元数据异质的曲线，优先采用可精确保留原始折点、可追踪每条输入的表示，更便于检验无平滑约束。

本次的“相对时间上的精确折线 L2 表示”是针对该数据的工程改进：通过所有原始横坐标的并集和精确积分保留折线几何，不把它包装为上述某篇论文的现成算法。它既没有进行平滑样条拟合，也没有以四种目标形态训练特征。
