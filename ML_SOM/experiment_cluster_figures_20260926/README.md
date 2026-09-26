# ML_SOM 聚类实验结果图合集

本目录把工作区中已有的**实验聚类结果图**复制到 `figures/`，原文件均未移动或修改。按[图集页面](index.html)浏览，或用[逐图清单](figure_manifest.csv)查每张图的原路径、方法、数据及 SHA-256。下表的“图数”按 PNG/SVG 文件计；同一图的 PNG 与 SVG 算两份格式。

## 方法与结果图

| 实验 | 数据与聚类方法（简述） | 图数 | 代表图 |
|---|---|---:|---|
| 早期 PvkSOM 主结果 | Hartono 2245 条 PCE；平滑、归一化后用 2×2 SOM，附 U-Matrix 与 KMeans 比较 | 9 | [四簇](figures/01_thesis_som/result/som_clusters.png) |
| 早期 PvkSOM 归档运行 | 同一项目的另一次旧运行；图像哈希与主结果不同，单独保存 | 9 | [四簇](figures/01_thesis_som/20230816_run_revision_excN2/som_clusters.png) |
| 原数值输入 SOM | Hartono 发布的数值直接输入 SOM；QE 肘部选择 K=4 | 3 | [簇图](figures/02_preprocessing/test1_raw/01_raw_som_clusters.png) |
| 归一化输入 SOM | 同 2245 条曲线，重采样审计后做 MaxAbs 归一化；自然 K=5 | 3 | [簇图](figures/02_preprocessing/test2_resample_normalize/02_resample_normalize_som_clusters.png) |
| 归一化＋平滑 SOM | 在上一步加 Savitzky–Golay 平滑；自然 K=5 | 3 | [簇图](figures/02_preprocessing/test3_resample_normalize_smooth/03_resample_normalize_smooth_som_clusters.png) |
| 三管线固定四簇对照 | 上述三种输入均固定 2×2 SOM 作形状比较；第四面板是文献参照图，不参与训练 | 1 | [比较图](figures/02_preprocessing/comparison/04_three_pipeline_comparison.png) |
| 目标 PCE 主 SOM | 本项目 PCE，10×10 密度均衡 SOM，早期阶段按输入形状均衡抽样；节点事后解释 | 2 | [四种 motif](figures/03_target_som/som_primary/four_som_motifs.png) |
| 目标 PCE SOM 基线与抽样对照 | 基线、当前审阅集未均衡基线、Gumbel 耦合抽样版本 | 6 | [基线图](figures/03_target_som/som_baseline/four_som_motifs.png) |
| 目标 PCE SOM 参数消融 | 密度幂次 0/1/3、导数权重、上升权重、medoid 命名、12×12 网格；每组各两图 | 14 | [密度消融示例](figures/03_target_som/som_density0/four_som_motifs.png) |
| 冻结 E2 SOM 与节点审计 | 0–200 h PCE 的 10×10 SOM 地图、早期细节版和节点与旧规则候选的事后对应 | 3 | [SOM 节点图](figures/04_research_som/frozen_e2/som_map.png) |
| 冻结 SOM 原型群组数探索 | 对 100 个已训练 SOM 原型做平均连接层次聚类；对照 k=2/4 和真实曲线 | 5 | [四组节点图](figures/04_research_som/prototype_groups/som_groups_k4.png) |
| 实际小时 DTW | 0–200 h PCE 幅值＋局部斜率的受限 DTW 相似图，8 个匿名群 | 1 | [群组概览](figures/05_pce_other_clustering/hour_aware_dtw/anonymous_cluster_overview.png) |
| 相对序位形状 DTW | 不使用绝对小时；按曲线形状的 DTW 细分为 16/32/64 个匿名簇 | 14 | [64 簇放大页](figures/05_pce_other_clustering/shape_dtw_k16_32_64/anonymous_k64_zoom_page3.png) |
| 形状编码四簇探索 | 相对观测序位、Chebyshev 系数、密度加权 KMeans；保留 2135 条 | 3 | [放大图](figures/05_pce_other_clustering/shape_only_four_group_exploration_20260925/anonymous_four_groups_zoom.png) |
| 无标签输入质量筛选四簇 | 点数／起伏质量筛选后，沿用形状编码与 KMeans；保留 1897 条 | 3 | [放大图](figures/05_pce_other_clustering/shape_only_input_curation_20260925/anonymous_four_groups_zoom.png) |
| 有监督输入筛选四簇 | 旧自动候选参与**输入选择**；随后用冻结的无监督形状 KMeans，保留 1842 条 | 3 | [最新放大图](figures/05_pce_other_clustering/supervised_input_frozen_unsupervised_20260926/anonymous_four_groups_zoom.png) |
| UCR 方法复现 | ECG5000、Plane、Trace 小子集；欧氏、DTW、PFD 等距离的层次聚类树 | 15 | [PFD-DTW 示例](figures/06_derivative_comparisons/ucr_Trace/PFD_DTW_HC_dendrogram.png) |
| 导数 DTW 合成数据消融 | 幅值 DTW、一至三阶导数 DTW 与融合距离比较 | 3 | [最佳簇图](figures/06_derivative_comparisons/eg_dtw_derivative/best_clusters.png) |
| SOM／导数功能验证 | 模拟曲线的 SOM、导数和 DTW 固定方法比较及采样压力情景 | 7 | [方法比较](figures/06_derivative_comparisons/functional_validation/comparison.png) |
| 真实采样背景半合成实验 | 注入目标形态，对比普通 DTW、PFD-DTW、有序导数图谱聚类 | 5 | [恢复对照](figures/06_derivative_comparisons/accepted_pfd/03_derivative_graph/recovery_comparison.png) |
| 实际小时双层分组 | 相对进程做形态分组，真实小时单独做响应时间子群；含首轮求解器归档 | 33 | [最终形态组](figures/07_hour_aware/outputs/existing_400/shape_groups_actual_hours.png) |

**合计 145 张。**前四组目录（`01`–`04`，58 张）属于 SOM 本身或其原型后处理；`05`–`07` 是本工作区中的其他聚类方法及对照，不能称作 SOM 结果。原始论文截图、单条源曲线图、审阅图、数据质量图和交互 HTML 未复制；对应实验的原始目录仍可查阅。

这些图展示的是各实验当时的输出，数据集、时窗、预处理、簇数及评估口径不同，**不能把它们的簇编号或比例直接横向比较**。尤其最后一轮“有监督输入筛选”只在拟合阶段保持无标签；旧候选没有独立真值。方法与限制以各原实验的 README/RESULTS 为准。

运行 `python3 experiment_cluster_figures_20260926/collect_figures.py` 可从原位置重新复制并生成清单与图集。脚本在复制时核对每张图的 SHA-256。
