# SOM 2–10类课题组汇报对比包

主分析数据为1,442条曲线。n=2–10均已分别训练，不是只运行n=4。
每个n均使用5个随机种子和50,000次训练。

## 推荐汇报顺序

1. `som_n2_to_n10_centroid_overview.png`：先展示2–10类形状如何逐步细分。
2. `som_n4_n5_n6_candidate_comparison.png`：重点比较SI候选范围4–6类。
3. `som_n2_to_n10_cluster_size_heatmap.png`：展示新增类别的样本占比。
4. `../05_cluster_number_decision/main_high_quality_min10/cluster_number_decision.png`：展示QE、稳定性、质心分离和K-means WCSS。

## 结论

- n=4、5、6均具有很高的随机种子稳定性。
- n=4的最近质心RMSE为0.121，仍高于0.10分离阈值。
- n=5和n=6分别降至约0.074和0.068，开始拆分已有主要形状。
- 因此n=4作为主分类；n=5和n=6作为更细粒度的对照结果保留。

## 全部数值

| n_nodes | topology | qe_mean_across_seeds | qe_relative_improvement_from_previous | mean_pairwise_seed_ARI | silhouette_pca20 | davies_bouldin_pca20 | min_centroid_rmse | min_cluster_fraction | selected_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1x2 | 2.2896 | nan | 1.0 | 0.686 | 0.5816 | 0.2884 | 15.4646 | False |
| 3 | 1x3 | 1.6214 | 0.2918 | 1.0 | 0.5829 | 0.6558 | 0.1521 | 8.3218 | False |
| 4 | 2x2 | 1.3755 | 0.1517 | 1.0 | 0.5436 | 0.7339 | 0.1212 | 4.0915 | True |
| 5 | 1x5 | 1.177 | 0.1443 | 1.0 | 0.4672 | 0.7767 | 0.074 | 3.3981 | False |
| 6 | 2x3 | 1.0875 | 0.076 | 1.0 | 0.4546 | 0.8453 | 0.0679 | 1.8031 | False |
| 7 | 1x7 | 1.001 | 0.0796 | 1.0 | 0.4063 | 0.8915 | 0.0525 | 1.5257 | False |
| 8 | 2x4 | 0.976 | 0.0249 | 0.9993 | 0.4105 | 0.9214 | 0.0525 | 1.5257 | False |
| 9 | 3x3 | 0.9075 | 0.0702 | 1.0 | 0.3671 | 0.9379 | 0.0387 | 1.5257 | False |
| 10 | 2x5 | 0.8769 | 0.0337 | 0.8059 | 0.3608 | 0.9379 | 0.0358 | 1.1789 | False |

## 每个分类数的完整结果

- n=2: `../04_som_results/main_high_quality_min10/n_02/`
- n=3: `../04_som_results/main_high_quality_min10/n_03/`
- n=4: `../04_som_results/main_high_quality_min10/n_04/`
- n=5: `../04_som_results/main_high_quality_min10/n_05/`
- n=6: `../04_som_results/main_high_quality_min10/n_06/`
- n=7: `../04_som_results/main_high_quality_min10/n_07/`
- n=8: `../04_som_results/main_high_quality_min10/n_08/`
- n=9: `../04_som_results/main_high_quality_min10/n_09/`
- n=10: `../04_som_results/main_high_quality_min10/n_10/`