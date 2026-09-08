# 最终报告：无监督 PCE 曲线细节搜索

## 直接结论

本轮最终选择 **K=4**，输入为 `w0300_coverage_maxabs_level`，即 300 h、`coverage` 质量子集、`maxabs_level` 表征。SOM 参数为 sigma=0.5、learning rate=0.03、random_order=True。

冻结模型没有同时形成四个一一对应 Bridge/Hill/Slope/Valley 的稳定节点；因此不能宣称已无监督发现目标四类。

冻结后形态计数为 `{'Slope-like': 4}`。这些名称没有进入样本筛选、参数搜索或 K 判断。

最终 10 个随机种子的平均两两 ARI 为 0.9578；20 次样本 bootstrap 相对冻结分区的 ARI 中位数为 0.9182，5%–95% 区间为 0.7312–0.9774。

## 数据与无平滑保证

- 扫描 accepted CSV：2151 条；识别为真实 PCE 且时间有效：2104 条。
- 搜索窗口：[150, 200, 300, 500, 750, 1000] h；coverage/detail/dense 使用预注册的点数、首末覆盖和最大时间空档规则。
- 最终纳入 1590 条曲线。任何曲线都没有因为不像目标四类而删除。
- 所有曲线局部起伏均保留。固定输入网格仅是相邻真实点之间的分段线性插值；没有滤波、样条、Savitzky–Golay、LOWESS 或移动平均。
- 最终代表图 `raw_point_representatives.png` 回到原始数字化散点，避免密集网格造成“细节增加”的错觉。

## 搜索规模

- 数据表示组合：72 个 window × cohort × representation 变体。
- K-means 快速筛选运行：3240 次；每种表征独立保留一个候选进入 SOM。
- SOM 参数运行：4032 次，覆盖 sigma=[0.2, 0.3, 0.5, 0.8, 1.2, 2.0]、learning rate=[0.03, 0.1, 0.3, 0.5]、顺序/随机训练、K=[2, 3, 4, 5, 6, 7, 8] 和多个种子。
- 冻结主模型后，又让全部72个数据变体运行6组代表性 SOM 设置，共9072次训练，并审计432个自动 elbow 模型。
- 类别数没有固定为4。每个 SOM 设置先按 QE 几何 elbow 选 K，再用完全无标签的质量指标进行秩聚合。

## 排名前十二的 SOM 决策

| variant_id | setting_id | elbow_k | silhouette_median | seed_ari | topographic_error_median | min_cluster_fraction_median | som_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w0300_coverage_maxabs_level | sigma0.5_lr0.03_random | 4 | 0.48854 | 0.95852 | 0 | 0.08239 | 309.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.3_sequential | 4 | 0.4954 | 1 | 0 | 0.083648 | 313 |
| w0300_coverage_maxabs_level | sigma2_lr0.3_sequential | 5 | 0.40658 | 1 | 0 | 0.10692 | 314.5 |
| w0300_coverage_maxabs_level | sigma2_lr0.03_sequential | 5 | 0.41023 | 1 | 0 | 0.10377 | 316.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.1_sequential | 4 | 0.4874 | 1 | 0 | 0.083019 | 319.5 |
| w0300_coverage_maxabs_level | sigma2_lr0.1_sequential | 5 | 0.4115 | 1 | 0 | 0.10377 | 321.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.03_sequential | 4 | 0.47064 | 0.9603 | 0 | 0.098742 | 335.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.03_random | 4 | 0.46704 | 0.89664 | 0 | 0.098742 | 338.5 |
| w0300_coverage_maxabs_level | sigma0.5_lr0.03_sequential | 4 | 0.48821 | 0.84896 | 0 | 0.080503 | 348.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.5_sequential | 4 | 0.50395 | 1 | 0 | 0.075472 | 350.5 |
| w0300_coverage_maxabs_level | sigma0.3_lr0.03_random | 4 | 0.50003 | 0.83501 | 0 | 0.074214 | 353.5 |
| w0300_coverage_maxabs_level | sigma0.8_lr0.1_random | 4 | 0.49179 | 0.8883 | 0 | 0.077358 | 358 |

## 冻结后的形态描述

| cluster | n | posthoc_ifo_name | peak_time_h | trough_time_h | rise | post_peak_drop | post_trough_recovery | early_slope | late_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 86 | Slope-like | 0 | 300 | 0 | 0.72745 | 0 | -0.0038693 | -0.0011266 |
| 1 | 883 | Slope-like | 0 | 300 | 0 | 0.070696 | 0 | -0.00024329 | -0.0002244 |
| 2 | 212 | Slope-like | 0 | 300 | 0 | 0.41332 | 0 | -0.0019112 | -0.0008033 |
| 3 | 409 | Slope-like | 0 | 300 | 0 | 0.2451 | 0 | -0.00095636 | -0.0007606 |

## 全窗口扩展与四类可能性

- 主网格的192个 elbow 设置中，exact-four 为0；最多只出现 Bridge/Hill/Slope 三种。
- 全窗口、全质量子集扩展的432个 elbow 模型中，exact-four 仍为0；14个模型覆盖3/4种目标形态。
- 最稳定的三形态候选是300 h、coverage、z-level+d1、sigma=1.2、learning rate=0.5、顺序训练、K=4；seed ARI=1.0，但没有 Valley。
- 部分1000 h z-level 模型能形成 Valley-like 节点，但会缺少 Bridge 或 Hill，说明恢复形态需要长窗口，却没有与另外三种同时成为四个稳定簇。
- `individual_ifo_raw_point_gallery.png` 证明数据中存在四种外观的个体曲线；这是冻结后的目标形态检索，不是聚类标签。

## 科学边界

- 参数搜索很宽并不意味着可以把某个恰好出现四种外观的次优运行升级为主结论；主结论必须服从预注册的无标签选择规则。
- 文献图像数字化曲线的采样密度、实验条件和上游处理并不一致，簇不能直接解释为单一退化机理。
- 一阶/二阶有限差分保留局部变化，但也会放大数字化噪声；因此这些表征和 level-only 表征被共同比较，而不是预先指定为正确答案。
- 150/200 h 结果保留为窗口敏感性，但主结论至少需要 300 h 以观察较长期行为。

## 文件入口

- `05_frozen_selection/frozen_selection.json`：在 IFO 命名前冻结的选择。
- `06_final_model/clusters/all_clusters.png`：所有成员与真实 medoid。
- `07_posthoc_ifo/raw_point_representatives.png`：原始数字化点。
- `07_posthoc_ifo/cluster_shape_descriptors.csv`：冻结后形态解释。
- `07_posthoc_ifo/individual_ifo_raw_point_gallery.png`：四种个体候选的原始散点。
- `08_stability/bootstrap_stability.csv`：20 次样本 bootstrap。
- `03_kmeans_screen/variant_decisions.csv` 与 `04_som_search/setting_decisions.csv`：完整选择证据。
- `10_posthoc_grid_audit/all_setting_posthoc_audit.csv`：主网格192个 elbow 设置的事后审计。
- `11_extended_window_cohort_sensitivity/posthoc_morphology_audit.csv`：全窗口432个 elbow 模型的事后审计。
