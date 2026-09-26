# 四类合成曲线 + 原文 SOM 复刻实验报告

## 结论

在这个**人工构造、四类均衡且形态刻意可分**的基准数据上，严格按论文预处理与 SOM 参数运行后，2×2 SOM 成功占用 4 个节点；训练后的一一节点命名得到准确率 **100.00%**、宏平均 F1 **1.0000**、ARI **1.0000**。共正确分类 1000/1,000 条。

固定种子补充复算的平均准确率为 **100.00%**，最低为 **100.00%**；不同种子原始节点分组的平均两两 ARI 为 **1.0000**。

这证明的是：**当数据确实由这四种清晰拓扑组成时，论文的无监督 SOM 流程有能力把它们恢复出来。** 它不证明真实的两组数据天然存在这四个簇，也不能替代真实数据实验。

## 数据与训练边界

- 数据：1,000 条曲线，Bridge/Hill/Slope/Valley 各 250 条，0–150 h。
- 原始点：约 30 min 间隔，带时间抖动、随机缺点、白噪声和相关噪声。
- 训练输入：只包含预处理后的曲线数值，不包含 `true_class`。
- 真值用途：仅用于训练后的 Hungarian 节点命名和指标计算。
- 主实验随机种子：按论文保持未指定（MiniSom `None`）；固定种子只用于稳健性复算。

## 论文参数

- 10 min 重采样；Akima 插值。
- 逐曲线 MaxAbsScaler 等价归一化。
- Savitzky–Golay：window 71，二阶多项式。
- MiniSom 2.2.9，2×2，sigma=0.5，learning_rate=0.1。
- `random_weights_init`，顺序训练 50,000 次。
- 时间窗口严格使用论文的 150 h。

## 主实验节点规模

| 后验类别 | 数量 |
|---|---:|
| IFO-Bridge | 250 |
| IFO-Hill | 250 |
| IFO-Slope | 250 |
| IFO-Valley | 250 |

## 文件导航

- `01_synthetic_data/synthetic_curves_long.csv.gz`：全部原始合成观测。
- `01_synthetic_data/curve_truth_and_generation_parameters.csv`：真值及每条曲线生成参数。
- `02_paper_preprocessing/preprocessed_curves_long.csv.gz`：完整预处理长表。
- `02_paper_preprocessing/preprocessed_matrix_float64.npy`：实际 SOM 输入矩阵。
- `03_paper_som_main/cluster_assignments.csv`：主实验分类结果。
- `03_paper_som_main/main_metrics.json`：主指标。
- `04_seed_stability/`：五个固定种子的补充稳定性结果与逐曲线分配。
- `05_evaluation/`：混淆矩阵、分类报告及误分类清单。
- `06_validation/`：独立复核结果。

## 解释限制

本实验是“方法可恢复性 / positive-control”测试。因为生成器预先定义了四类，不能将其表述为在真实数据上发现了四种自然簇。严谨写法是：论文 SOM 在受控合成数据上能恢复四类，但在现有真实数据上尚未稳定恢复全部四类。
