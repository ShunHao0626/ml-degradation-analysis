# 历史实验索引

这里收纳此前位于仓库根目录的探索性实验，原有子目录结构、配置、报告和汇总结果保持不变。该目录是研究记录；当前工作请从根目录 [README](../README.md) 进入。原始曲线、论文全文、模型文件和大型中间矩阵仍保留在本地。

| 方向 | 路径 | 主要问题 |
|---|---|---|
| 文献曲线与 200 h SOM | [`lab/`](lab/) | 从初版到规范化的 200 h 重分析 |
| 四形态 SOM 搜索 | [`lab-v2/`](lab-v2/) | 218 条子集和完整文献曲线上的无监督搜索与事后形态核查 |
| 文献参数与聚类数 | [`pce_hour_curve_taxonomy/`](pce_hour_curve_taxonomy/)、[`pce_curve_pattern_discovery/`](pce_curve_pattern_discovery/) | 参数证据、K 的选择及稳健性 |
| 变化点与密度聚类 | [`HDB-scan/`](HDB-scan/) | 不平滑的 HDBSCAN 对照 |
| 可变长度曲线 | [`curve_discovery_unsupervised_20260906_01/`](curve_discovery_unsupervised_20260906_01/) | 相对进度轴上的无监督实验 |
| 无平滑与广泛搜索 | [`pce_som_no_smoothing_20260907_01/`](pce_som_no_smoothing_20260907_01/)、[`pce_ifo_unsupervised_detail_search_20260907_01/`](pce_ifo_unsupervised_detail_search_20260907_01/) | 500 h 基线、窗口和表示的敏感性 |

大部分脚本原来假定这些目录直接位于仓库根目录。迁入本目录后，带有硬编码相对路径的历史脚本需要从其原目录布局或按其报告记录的路径运行；归档结果本身不受移动影响。
