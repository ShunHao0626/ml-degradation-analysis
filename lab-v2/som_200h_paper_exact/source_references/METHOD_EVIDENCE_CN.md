# 原文参数证据与本次 200h 映射

| 环节 | 原论文/原 notebook | 本次实现 |
|---|---|---|
| 截止时间 | 150h；短于 150h 排除，长曲线只取前 150h | 唯一目标改动：200h；短于 200h 排除，长曲线只取前 200h |
| 最大值位置 | 最大 PCE 在截止时间之后的曲线排除 | 全曲线全局最大值位于 200h 后则排除，并保留审计字段 |
| 时间采样 | 10min | 10min，1200 点（10min 至 200h） |
| 插值 | Akima | Akima |
| 归一化 | MaxAbsScaler，按单条曲线最大绝对值 | 相同数学运算 |
| 平滑 | Savitzky-Golay，window=71，polyorder=2 | 完全相同 |
| SOM | MiniSom 2.2.9；2×2；sigma=0.5；learning_rate=0.1 | 完全相同 |
| 初始化/训练 | random_weights_init；50,000 次；默认顺序；未设置随机种子 | 完全相同 |
| 对照 | TimeSeriesKMeans(n_clusters=4, metric='dtw') | 相同显式参数；并核验默认值均一致：max_iter=50、tol=1e-6、n_init=1、max_iter_barycenter=100、metric_params=None、n_jobs=None、dtw_inertia=False、verbose=0、random_state=None、init='k-means++' |
| PCE 稳定性 | top-3 最大 PCE 均值；截止点 PCE；5 个等数量组 | 截止点改成 200h，其余相同 |

证据位置：`work.md` 的 Data analysis 和 SOM 段落；`Supplementary.md` 的 Supplementary Fig. 2、6、7、8；两个原 notebook 的预处理、SOM 与 DTW k-means 代码单元。

## 数据集导致、且无法伪装成“完全一致”的限制

本地输入是从论文图中数字化的稀疏 `x/y` 曲线，不是作者原始每 2min MPPT 表。因此 10min 重采样与 Akima 合并为“在 10min 网格上做 Akima 重建”，然后才归一化和平滑。曲线级 N₂/air、1 sun、温度、封装、pixel filter、材料堆栈元数据不存在，不能验证或筛选。统计 PCE 分组只在纵轴明确标为绝对百分比且未标为 normalized/relative 的子集上进行，并标为探索性结果。

循环轴、单位不明、非 PCE/效率/功率纵轴、少于 200h、全局最大值在 200h 后，以及明显不合理的超过 100,000h 数字化时间跨度均逐条排除；不按数值范围猜测单位。
