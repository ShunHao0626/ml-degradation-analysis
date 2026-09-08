# 最终决定

## 最优无监督解

- 最佳类别数：**K=4**。
- 最佳配置：MiniSom `2x2`，`sigma=0.5`，`learning_rate=0.3`，顺序更新，
  50,000 次迭代。
- 选择依据：论文的 median QE elbow + 最小可区分主形态原则；同时检查中心重叠、
  空节点、最小簇支持度和多种子稳定性。
- 最终 10 个随机种子的分区 ARI 为 1.0；K-means 独立 elbow 也为 K=4。
- 主模型四簇规模：371、586、56、189。

## 与目标四种 IFO 形态的关系

最佳论文流程虽然选择 K=4，但四个中心都是不同衰减速度的 Slope-like 曲线，
所以不能把它们命名为 Bridge/Hill/Slope/Valley。

形状敏感 z-normalized SOM 也自动选择 K=4；冻结后得到两个 Slope-like、一个
Bridge-like、一个 Valley-like 中心，没有独立 Hill-like 中心。由于其在共同原始
MaxAbs 空间的 silhouette 为 -0.1643，明显低于主模型的 0.5054，因此不替代主模型。

数据中能找到四种外观的个体候选，已保存为事后形态检索结果；这说明四类外观
存在于数据中，但不等于无监督模型支持“四个 IFO 主簇”。

## 关键文件

- `reports/final_report_cn.md`：完整主报告。
- `05_final_model/clusters/all_clusters.png`：最佳 K=4 主结果。
- `06_posthoc_ifo/raw_point_representatives.png`：各主簇真实数字化点代表。
- `08_shape_sensitive/clusters.png`：形状敏感 K=4 对照。
- `08_shape_sensitive/posthoc_individual_ifo_candidate_gallery.png`：四种 IFO-like 个体候选。
- `08_shape_sensitive/posthoc_individual_ifo_candidate_provenance.csv`：候选来源和评分。
- `04_k_selection/setting_and_k_decisions.csv`：所有参数的 K 判定证据。
- `verification_results.json`：无平滑、无标签和产物一致性检查。

全流程没有调用 Savitzky–Golay、移动平均、LOESS、样条或其他平滑/降噪函数。
固定长度 SOM 输入仅使用原始观测点之间的分段线性插值。
