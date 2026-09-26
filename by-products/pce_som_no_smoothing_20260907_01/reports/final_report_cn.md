# 最终报告：无平滑、全过程无监督的 PCE 曲线分类

## 结论

主分析由论文规则选择 **K=4**，没有预设为四类。最佳无监督参数配置为 `paper_lr_sensitivity`：`sigma=0.5`、`learning_rate=0.3`、`random_order=False`。最终模型使用 50,000 次更新和 10 个随机种子，选择与其他种子分区平均一致性最高的种子作为可复现输出。

K-means 的独立 inertia elbow 为 K=4。它只用于一致性核查，不覆盖 SOM/论文判据。

150 h 论文窗口对参数更敏感：三组论文参数的 elbow 分别为 K=4 或 K=5；这说明是否观察 200 h 之后会改变细分程度，也是把 500 h 作为本任务主窗口、150 h 仅作为论文对照的原因。

冻结形态名称完全在模型冻结后生成；节点名称计数为 `{'Slope-like': 4}`。它们只是 IFO-like 对照，不是训练标签。

这意味着“最佳类别数恰好是 4”，但**不是**参考图中的四种 IFO 主簇：论文原始 MaxAbs 表达主要按衰减速率分组。

## 形状敏感对照与四类曲线查找

为检查幅度主导是否掩盖峰谷，又运行了逐曲线 z-normalization 的无平滑 SOM；K、参数和模型仍由相同的无监督规则确定。该对照同样选择 K=4，冻结后的中心为两个 Slope-like、一个 Bridge-like 和一个 Valley-like，仍没有独立 Hill-like 中心。它在共同的原始 MaxAbs 空间 silhouette 为 -0.1643，低于主模型的 0.5054，因此不替代主结论。

不过，原始数据中确实可以事后找到四种外观的个体候选，见 `08_shape_sensitive/posthoc_individual_ifo_candidate_gallery.png` 及其 provenance CSV。该步骤在模型冻结后进行，只回答“数据里有没有相似曲线”，不是无监督聚类得到四个 IFO 类的证据。

## 数据范围

- 扫描 accepted CSV：2151 条。
- 500 h 主模型纳入：1202 条；150 h 论文敏感性模型纳入：1944 条。
- 只使用 validation metadata/记录名可识别为 PCE/efficiency 的曲线。
- 各曲线从首个观测记为 elapsed time=0；依据轴元数据把 minute/hour/day/week/month/year 转为小时。
- 使用逐曲线 MaxAbs 归一化，与论文一致。
- 固定长度输入通过观测点之间的分段线性插值获得；没有滤波、样条、Savitzky–Golay 或移动平均。

## 类别数判定

论文规则是：QE elbow 给出候选范围，然后选择能够表达主要形态的最小 K；若更高 K 的中心重叠或不可区分则拒绝。这里用多种子 median QE 计算几何 elbow，并同时输出中心间 RMSE/簇内 RMSE 比、空节点、最小簇比例和 seed ARI。

为落实论文的‘主要形态’表述，自动排名前先要求每个节点至少覆盖 1% 样本且无空节点；不满足者仍完整保留在结果表，但不能仅靠离群小簇获选。

| setting | source | sigma | learning_rate | random_order | elbow_k | qe_at_elbow | silhouette_at_elbow | seed_ari_at_elbow | overlap_ratio_at_elbow | min_cluster_fraction_at_elbow | occupied_nodes_at_elbow | empty_nodes_at_elbow | rank_qe | rank_silhouette | rank_stability | rank_separation | rank_sum | passes_main_shape_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_lr_sensitivity | paper | 0.5 | 0.3 | False | 4 | 1.03952 | 0.509842 | 1 | 2.54438 | 0.0415973 | 4 | 0 | 6 | 1 | 1 | 4 | 12 | True |
| optimized_random_order | optimization | 0.5 | 0.1 | True | 5 | 0.886108 | 0.470932 | 0.920386 | 2.93652 | 0.0291181 | 5 | 0 | 1 | 4 | 6 | 1 | 12 | True |
| optimized_wider_neighborhood | optimization | 1 | 0.1 | True | 4 | 0.969777 | 0.485266 | 0.948243 | 2.7495 | 0.0707155 | 4 | 0 | 4 | 3 | 4 | 2 | 13 | True |
| optimized_wider_faster | optimization | 1 | 0.3 | True | 4 | 0.984451 | 0.493198 | 0.936647 | 2.70999 | 0.0640599 | 4 | 0 | 5 | 2 | 5 | 3 | 15 | True |
| paper_main | paper | 0.5 | 0.1 | False | 5 | 0.88878 | 0.464694 | 0.999598 | 2.51925 | 0.030782 | 5 | 0 | 2 | 5 | 2 | 6 | 15 | True |
| paper_sigma_sensitivity | paper | 0.3 | 0.1 | False | 5 | 0.888937 | 0.464663 | 0.98266 | 2.52135 | 0.030782 | 5 | 0 | 3 | 6 | 3 | 5 | 17 | True |

## 冻结后的形态描述

| cluster | n | start | at_50h | at_200h | end | peak | peak_time_h | trough | trough_time_h | early_slope_per_h | late_slope_per_h | post_trough_recovery | post_peak_drop | range | posthoc_ifo_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 371 | 0.987566 | 0.965758 | 0.878126 | 0.733393 | 0.987566 | 0 | 0.733393 | 500 | -0.000555762 | -0.000477538 | 0 | 0.254174 | 0.254174 | Slope-like |
| 1 | 586 | 0.981992 | 0.973316 | 0.960889 | 0.915198 | 0.981992 | 0 | 0.915198 | 500 | -0.000168203 | -0.000169165 | 0 | 0.0667944 | 0.0667944 | Slope-like |
| 2 | 56 | 0.99992 | 0.723776 | 0.500656 | 0.262394 | 0.99992 | 0 | 0.262394 | 500 | -0.00315027 | -0.000589488 | 0 | 0.737526 | 0.737526 | Slope-like |
| 3 | 189 | 0.99922 | 0.915262 | 0.733952 | 0.497492 | 0.99922 | 0 | 0.497492 | 500 | -0.00153459 | -0.000735203 | 0 | 0.501728 | 0.501728 | Slope-like |

## 解释边界

- 这些数据来自文献图像数字化，采样稀疏且不同实验条件混合，不能把簇直接解释成单一物理退化机理。
- 500 h 窗口用于观察 200 h 后行为；它不是原论文的 150 h 窗口，因此同时给出 150 h 敏感性结果。
- 线性插值不会创造新的极值，但密集网格不等于新增测量；`raw_point_representatives.png` 专门展示真实数字化点。
- 若没有四个独立 IFO-like 节点，不能宣称无监督算法发现了四类。

## 文件导航

- `04_k_selection/setting_and_k_decisions.csv`：每组参数的 elbow 与无监督质量指标。
- `05_final_model/final_model.json`：冻结模型、K 和种子。
- `05_final_model/clusters/all_clusters.png`：所有成员、median 与 SOM 权重。
- `06_posthoc_ifo/raw_point_representatives.png`：未经平滑的原始数字化点。
- `06_posthoc_ifo/representative_provenance.csv`：代表曲线溯源。
- `08_shape_sensitive/shape_sensitive_report_cn.md`：形状敏感无监督对照。
- `08_shape_sensitive/posthoc_individual_ifo_candidate_gallery.png`：四种外观的事后个体候选。
- `literature_review.md`：方法文献与取舍。
