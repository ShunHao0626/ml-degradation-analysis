# 无平滑、变长 PCE 曲线的无监督形态探索

本目录基于 `/Users/shunhao/Desktop/ML/thesis` 的论文方法，分析：

`/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted/samples_test`

目标是在不使用 IFO-Bridge、IFO-Hill、IFO-Slope、IFO-Valley 标签干预训练的情况下，尽量保留不同时长的曲线，以论文采用的 SOM quantization error（QE）肘部逻辑决定类别数，最后再检查四种目标形态是否存在。

最终建议为 **4 个整体主簇**，样本数分别为 46、23、89、60。四种 IFO-like 形态都找到了真实样本，但它们没有分别形成四个互不重叠的主簇。Slope-like 与部分 Valley-like 曲线同属快速衰减主簇；统一局部分析还发现了一个包含 5 条曲线的回升子群。

建议先看：

- [初始值2%与50/100/200 h卡点实验](supplementary_initial2pct_checkpoints_20260907/README.md)：120条曲线的阈值事件模型；按预定全范围QE规则选择5类
- [MaxAbs 数据质量筛选补充实验](supplementary_filtered_maxabs_20260907/README.md)：筛除轴/时长异常后，推荐的 H500 组保留 105 条并稳定选择 3 类
- [完整技术报告](report.html)
- [离线交互曲线查看器](index.html)
- [4 个无监督主簇](figures/01_selected_clusters.png)
- [QE 类别数判断](figures/03_qe_elbows.png)
- [四种 IFO-like 原始样本](figures/00_four_target_examples_minmax.png)
- [所有曲线的聚类归属](cluster_assignments.csv)

## 1. x 轴的 hour 到底怎样处理

### 1.1 没有修改原始 hour

源 CSV 没有被改写。218 条曲线和 9,674 个原始点全部保留，未截断长曲线，也未把短曲线外推到 150 h 或 200 h。

原始坐标可在以下文件复核：

- `data/raw_curves.json`；
- `data/transformed_curves/C001.csv` 至 `C218.csv` 中的 `original_x`、`original_y`；
- `data/curve_inventory.csv` 中的源文件路径、x/y 范围和 SHA-256。

### 1.2 主聚类横轴不是 hour，而是相对观测进程

数据中只有 159 条曲线有可确认的时间单位；17 条横轴是循环次数，42 条无法确认单位。为了让全部 218 条曲线参与形态比较，每条曲线的完整横轴独立映射到 0–1：

```text
u = (x - x_min) / (x_max - x_min)
```

含义是：

- `u=0` 是该曲线的第一个观测点；
- `u=1` 是该曲线的最后一个观测点；
- `u=0.2` 表示走完该曲线自身观测区间的 20%；
- `u=0.2` 不表示 20 h、40 h 或 200 h；
- 两条曲线相同的 u 不保证是相同实际小时。

例如，测量 100 h 和测量 1,000 h 的曲线都会映射到 `u=0…1`。模型比较的是“先上升再下降”“先快速下降再缓慢下降”等阶段顺序和相对轮廓，而不是相同真实时刻下的退化速度。

这样做可以保留短曲线，且不会虚构短曲线未观测的后半段。代价是主模型不能回答“第 200 h 谁退化得更快”。主结果只能解释为**各自完整观测窗口内的相对形态分类**。

### 1.3 实际小时如何保留

时间单位只根据每个图片目录的 `validation_result.json` 判断，不根据 x 数值大小猜测：

| 元数据单位 | hour 换算系数 |
|---|---:|
| h、hr、hour、hours | 1 |
| d、day、days | 24 |
| min | 1/60 |
| cycles 或未知 | 不换算 |

可确认单位时另存：

```text
elapsed_hour = (original_x - original_x_min) × unit_factor
```

真实小时可在 [实际小时图](figures/05_actual_hours.png)、交互查看器和逐曲线 CSV 的 `elapsed_h_if_verified` 列中查看。每条曲线在自己的真实测量终点停止，没有补齐。

图片140的 C048、C049 按现有元数据计算超过 4,100 万小时，疑似数字化横轴尺度异常。实验没有擅自修正或删除，只在 `data/quality_flags.csv` 标记。去掉这两条异常尺度、只分析其余 157 条已确认时间曲线时，完整 k=1–16 搜索仍选 4 类。

## 2. y 轴归一化逻辑

输入同时含 PCE、百分数、Normalized PCE、Normalized Pmax 等纵轴定义，原始数值不可直接放进同一个距离空间。本实验对每条曲线分别建立三种表示。

### 2.1 MinMax：最终主模型

```text
y_minmax = (y - min(y)) / (max(y) - min(y))
```

每条非常数曲线自己的最低点为 0、最高点为 1；常数曲线固定为 0.5，避免除以零。

MinMax 在实验运行前就被设为主表示，因为任务关注形态，而且输入 y 的量纲和是否预归一化并不统一。它去掉绝对输出水平和总振幅，让模型主要比较峰谷位置、阶段顺序、相对快慢及轮廓。

它的限制是会放大小波动。例如实际只从 1.00 变到 0.99 的曲线也会被拉伸到 1–0。本数据有 7 条曲线的相对变化不足 2%，均已标记。因此判断这些曲线时必须同时查看 MaxAbs 或原始 y。

### 2.2 MaxAbs：振幅对照

```text
y_maxabs = y / max(abs(y))
```

若 y 为非负值，最大值约为 1，但最小值不强制为 0。PCE 从 1.00 变到 0.98 时，MaxAbs 仍显示 0.98–1.00，而 MinMax 会显示 1–0。

这更接近原论文的 MaxAbsScaler，用于核对相对变化幅度。它没有作为最终主结果，因为本数据混合了原始 PCE、百分数和已归一化输出。MaxAbs 在完整 k=1–16 范围选出 5，但 k 上限为 8、10、12 时会选 3，说明类别数范围依赖较强。

### 2.3 时间加权 z-score：形态敏感性分析

把相邻原始点视为直线，在相对时间 u 上精确积分得到均值和方差：

```text
mean = integral y(u) du
std² = integral y(u)² du - mean²
y_zscore = (y - mean) / std
```

它同时消除纵向平移和尺度，但会强调局部偏离。完整范围选出 5 类，而不同种子与搜索范围约支持 3–5 类，因此只作为敏感性分析。

| y 表示 | 用途 | 完整 k=1–16 的结果 |
|---|---|---:|
| MinMax | 主形态模型 | 4 |
| MaxAbs | 保留相对振幅的对照 | 5，但范围依赖明显 |
| 时间加权 z-score | 去除平移和尺度后的对照 | 5，但约 3–5 不确定 |

最终选择 MinMax 的 4 类，并不是因为看到它恰好得到 4 才事后选择；`experiments/config.json` 保存了训练前确定的主表示和规则。

## 3. “关闭 smoothing”的具体含义

新管线没有使用 Savitzky–Golay、移动平均、LOESS、Gaussian filter、Akima 平滑、smoothing spline 或人工形态模板。

每条曲线仅按 x 稳定排序，并用相邻原始点之间的直线定义分段线性折线。这种线性连接只用于计算观测点之间的函数值和距离：

- 不改变原始点；
- 不外推；
- 不截断；
- 不增加或删除峰谷；
- 绘图保留全部原始点。

簇图中的黑线是数据中真实存在的 medoid 曲线，不是平滑均值或拟合的理想曲线。

7 条曲线存在重复 x，共 12 个重复点，均原样保留并显示为垂直线。连续时间 L2 中零时长的垂直跳变面积为零，但跳变后的走势仍进入距离；DTW/MSM 对照则直接包含这些点。

## 4. 主实验完整运行逻辑

### 4.1 数据加载与审计

主脚本：`code/run_experiments.py`

1. 递归读取 `samples_test/**/*.csv`。
2. 将 x、y 转为数值，检查至少两个有限点且 x 有非零跨度。
3. 对 x 稳定排序；重复 x 不取平均。
4. 读取 `validation_result.json` 的轴名称和单位。
5. 保存源路径、SHA-256、点数、x/y 范围、单位状态和时长。
6. 生成相对进程 u，同时保留原 x/y。

| 数据审计项 | 数量 |
|---|---:|
| 图片来源 | 99 |
| 曲线 | 218 |
| 原始点 | 9,674 |
| 每条点数 | 6–105 |
| 已确认时间单位 | 159 |
| 循环次数横轴 | 17 |
| 未确认单位 | 42 |
| 排除曲线 | 0 |

### 4.2 变长折线如何变成 SOM 输入

普通 SOM 要求相同维数。这里没有把所有曲线粗略重采样成固定 100 或 1,200 点，而是构造保持折线 L2 距离的等距表示：

1. 合并所有曲线的相对横坐标 u，形成全体折点并集；
2. 在每个相邻折点区间用两点 Gauss 求积；
3. 精确计算所有分段线性曲线的 L2 内积；
4. 形成 218×218 Gram 矩阵；
5. 对 Gram 矩阵特征分解，获得保持该几何的坐标；
6. 只删除数值精度内的零特征值，不按 PCA 方差比例压缩。

SOM 输入空间中的欧氏距离等于相对进程上的折线 L2 距离：

```text
d(f,g) = sqrt(integral_0^1 (f(u)-g(u))² du)
```

这意味着采样密集区域不会仅因点更多而获得更多权重，而按它在相对 x 轴上占据的区间长度加权。独立数值验证的最大距离平方误差为 `8.36e-12`。

### 4.3 SOM 超参数初筛

MinMax、MaxAbs、z-score 分别独立搜索：

| 参数 | 搜索范围 |
|---|---|
| 节点数/类别数 k | 1–16 |
| sigma | 0.15、0.30、0.50、1.00 |
| learning rate | 0.03、0.10、0.30 |
| 初筛 seeds | 11、29、47 |
| 初筛更新次数 | 5,000 |

SOM 网格取尽量接近方形的整数因子，例如 k=4 为 2×2、k=6 为 2×3、k=5 为 1×5。另做统一 1×k 网格检查，MinMax 仍选 4。

每个 `(normalization, sigma, learning_rate)` 组合在每个 k 下取三种子的 QE 中位数，再除以该组合 k=1 的 QE：

```text
score(sigma, lr)
  = mean over k=1..16 [median_seed(QE_k) / median_seed(QE_k=1)]
```

选择 score 最小的组合。该规则只评价相对量化能力，不读取 IFO 标签。

| 表示 | 选中 sigma | 选中 learning rate |
|---|---:|---:|
| MinMax | 0.5 | 0.1 |
| MaxAbs | 0.5 | 0.3 |
| z-score | 1.0 | 0.1 |

### 4.4 最终长训练

每种表示用选中的 sigma/lr 执行：

| 参数 | 设置 |
|---|---|
| k | 1–16 |
| seeds | 11、29、47、71、101 |
| 每次更新 | 50,000 |
| 训练顺序 | 由 seed 控制的随机顺序 |
| 目标标签 | 不使用 |

QE 与论文逻辑一致，是每条曲线到最佳匹配单元权重的平均距离：

```text
QE = mean_i ||x_i - w_BMU(i)||₂
```

silhouette、topographic error 和不同种子间 ARI 同时保存，但只用于辅助检查，不取代 QE。这里的 ARI 比较两个无监督分组的一致性，不是真实标签下的分类准确率。

### 4.5 类别数判断

原论文使用 QE elbow，并检查增加类别后是否出现相似、重叠的簇，但没有提供唯一自动 elbow 函数。本实验将其操作化为：

1. 每个 k 取 5 个 seeds 的 QE 中位数；
2. 将 k 和 QE 归一化到 0–1；
3. 连接 QE 曲线首末端；
4. 选择相对该直线偏离最大的 k；
5. 用连续分段线性拟合、不同 k 上限、逐 seed elbow 和相邻 k 图复核；
6. 检查增加 k 后簇中心是否开始相对簇内分散程度变得接近。

MinMax 的前六个 QE 中位数为：

| k | QE | 相对前一个 k 的下降 |
|---:|---:|---:|
| 1 | 0.195220 | — |
| 2 | 0.152614 | 21.8% |
| 3 | 0.126046 | 17.4% |
| 4 | 0.109364 | 13.2% |
| 5 | 0.104142 | 4.8% |
| 6 | 0.098493 | 5.4% |

k=4 后改善明显减小。复核结果：

- k 上限取 8、10、12、16 均选 4；
- 5 个 seeds 各自均选 4；
- 连续分段线性拟合选 4；
- 统一 1×k 网格选 4；
- 159 条已确认时间曲线选 4；
- 去除两条异常时间尺度后的 157 条曲线，在完整 k=1–16 下选 4；
- k=5/6 主要进一步切分相近的衰减形状。

因此 4 来自 QE 判断，不是因为目标图里有四类。

### 4.6 最终 seed 与主结果

确定 k=4 后，选择 5 个 seeds 中 QE 最低的一次，最终 seed=29：

| 指标 | 数值 |
|---|---:|
| sigma | 0.5 |
| learning rate | 0.1 |
| updates | 50,000 |
| QE | 0.109111 |
| silhouette | 0.327809 |
| seeds 两两 ARI 平均 | 0.891516 |
| seeds 两两 ARI 最低 | 0.801071 |
| 簇规模 | 46、23、89、60 |

### 4.7 按图片来源重采样

同一图片中的多条曲线可能相关，因此以 99 个图片目录为组，做 20 次有放回重采样。每次对 k=1–16 重新训练和找 elbow：

- 16/20 次选择 4；
- 4/20 次选择 5；
- 固定 k=4 时，与主模型分组的平均 ARI 为 0.812。

所以 4 是最常出现的简洁整体粒度，5 是有一定支持的备选细分；不能声称 4 是唯一数学答案。

### 4.8 局部子簇

为了检查少数峰谷形态是否被大簇覆盖，对全部四个主簇使用相同的局部 SOM + QE elbow 规则：

| 主簇 | 样本数 | 局部 k | seeds ARI 平均 |
|---:|---:|---:|---:|
| 1 | 46 | 4 | 1.000 |
| 2 | 23 | 3 | 0.948 |
| 3 | 89 | 3 | 0.712 |
| 4 | 60 | 4 | 0.852 |

这些是探索性子结构，不表示全局最优类别数是 14。主簇4的局部子群3包含 C022、C178、C202、C207、C216 五条下探后回升曲线。

### 4.9 四种 IFO 只在训练后解释

IFO 目标图没有进入数据加载、归一化、距离、超参数、QE elbow 或 seed 选择。模型完成后才人工选择这些真实样本用于解释：

| 事后解释 | 曲线 | 主簇 | 观测跨度 | 相对输出变化 |
|---|---|---:|---:|---:|
| Bridge-like | C096 / 图片194 | 2 | 100.7 h | 3.82% |
| Hill-like | C004 / 图片102 | 1 | 212.8 h | 16.99% |
| Slope-like | C182 / 图片75 | 4 | 1000.6 h | 16.16% |
| Valley-like | C059 / 图片16 | 4 | 643.9 h | 44.97% |

C096 只测到约 101 h，因此只能说其早期上升、平台和缓慢回落类似 Bridge，不能证明“200 h 后”行为。Slope-like 和 Valley-like 同在主簇4，也说明模型并未被强制凑成四个 IFO 标签。

## 5. 文献扩展方法

`code/run_elastic.py` 和 `code/elastic_distances.cpp` 进一步测试变长原始点序列的弹性距离。实现借鉴 SOMTimeS 的 DTW 思路和 dissimilarity SOM 的样本原型，但不是 SOMTimeS 或 KASBA 的完整复现。

### DTW

```text
local_cost(i,j) = (y_i-y_j)² + beta × (u_i-u_j)²
distance = sqrt(total_path_cost / (n+m))
```

测试 `beta=0、0.25、1`：分别选择 3、4、4 类。beta=0 可自由扭曲相对时间，容易把发生在不同阶段的下降或回升对齐；beta 增大后保留更多阶段位置。三者种子稳定性均低于主 MinMax-L2 SOM，因此未取代主结果。

### MSM

Move-Split-Merge 测试 `c=0.01、0.1`，分别选择 6、5 类，但 seeds ARI 约为 0.36，并出现很小的簇。原始点数和数字化采样密度差异对 MSM 影响明显，因此不推荐。

### median SOM 参数

| 参数 | 值 |
|---|---|
| y | MinMax |
| x | 原始点对应的相对进程 u |
| k | 1–16 |
| sigma | 0.3、0.5、0.8 |
| seeds | 11、29、47、71、101 |
| epochs | 100 |
| 原型 | 数据中的实际曲线 medoid |

不同弹性距离的 QE 与 L2 QE 不在同一量纲，不能直接比较绝对值。共同 MinMax-L2 空间下的辅助比较保存在 `method_comparison.csv`。

## 6. 全部参数汇总

### 主 SOM

| 参数 | 值 |
|---|---|
| primary normalization | MinMax per curve |
| x representation | 每条完整观测区间映射到 u=0–1 |
| smoothing | False |
| truncation | False |
| extrapolation | False |
| target labels used | False |
| k | 1–16 |
| sigma search | 0.15、0.3、0.5、1.0 |
| learning-rate search | 0.03、0.1、0.3 |
| screening seeds | 11、29、47 |
| screening updates | 5,000 |
| final seeds | 11、29、47、71、101 |
| final updates | 50,000 |
| selected sigma | 0.5 |
| selected learning rate | 0.1 |
| selected k | 4 |
| selected seed | 29 |

### 其他实验

| 实验 | 参数 |
|---|---|
| 图片来源 bootstrap | 99 组；20 次；k=1–16；每次每 k 更新 10,000；seeds=1000–1019 |
| 时间子集 | 159 条；k=1–16；seeds=11/29/47；更新 30,000 |
| 异常尺度时间子集 | 157 条；k=1–16；seeds=11/29/47；更新 30,000 |
| 局部子簇 | 每个父簇 k=1 至 min(8,floor(n/3))；5 seeds；更新 30,000 |
| 网格敏感性 | 所有 k 强制 1×k；3 seeds；更新 30,000 |

## 7. 运行顺序

```bash
cd /Users/shunhao/Desktop/ML/curve_discovery_unsupervised_20260906_01

/usr/bin/python3 code/run_experiments.py
/usr/bin/python3 code/run_elastic.py
/usr/bin/python3 code/local_refinement.py
/usr/bin/python3 code/additional_checks.py
/usr/bin/python3 code/render_results.py
/usr/bin/python3 code/posthoc_examples.py
/usr/bin/python3 code/verify_numerics.py
/usr/bin/python3 code/finalize_report_data.py
/opt/homebrew/bin/node code/verify_viewer.cjs
```

后续脚本依赖主实验输出，应按此顺序运行。`run_elastic.py` 会用系统 `clang++` 编译 C++ 距离实现。重跑会覆盖本结果目录中的同名生成文件，但不会改写输入数据或 `thesis`。Python 和依赖版本见 `requirements.txt` 与 `experiments/environment.json`。

## 8. 脚本与输出

| 文件 | 运行逻辑 | 主要输出 |
|---|---|---|
| `code/run_experiments.py` | 审计、三种 y 表示、SOM 参数搜索、选 k、bootstrap、时间子集 | `data/`、`selection.json`、模型 NPZ、搜索 CSV |
| `code/run_elastic.py` | DTW/MSM + median SOM | `elastic_sweep.csv`、`elastic_selection.json` |
| `code/local_refinement.py` | 对所有主簇统一细分 | `local_assignments.csv`、`local_selection.json` |
| `code/additional_checks.py` | 1×k 网格和异常尺度敏感性 | `additional_checks.json` |
| `code/render_results.py` | 静态图、交互页、逐曲线 CSV | `figures/`、`index.html`、`cluster_assignments.csv` |
| `code/posthoc_examples.py` | 训练后 IFO-like 实例 | `target_shape_examples.csv`、目标形态图 |
| `code/verify_numerics.py` | 源数据、L2、DTW、MSM 数值核验 | `verification.json` |
| `code/finalize_report_data.py` | 方法比较、报告和交付完整性 | `method_comparison.csv`、`report.html` |
| `code/verify_viewer.cjs` | 执行查看器的数据和控件逻辑 | `viewer_logic_verification.json` |
| `code/run_filtered_supplement.py` | MaxAbs、无平滑的数据质量与时长筛选补充实验 | `supplementary_filtered_maxabs_20260907/` |
| `code/run_initial_threshold_checkpoints.py` | 初始值1/2/3%与50/100/200 h卡点无监督实验 | `supplementary_initial2pct_checkpoints_20260907/` |

共记录 3,787 次训练：主超参数初筛 1,728，最终长训练 240，DTW/MSM 1,200，局部分析 155，时间子集 48，附加检查 96，图片来源 bootstrap 320。

## 9. 结果目录索引

- `cluster_assignments.csv`：每条曲线在主模型和全部对照模型中的归属；
- `local_assignments.csv`：主簇与局部子簇；
- `target_shape_examples.csv`：四种事后解释样本和源文件；
- `method_comparison.csv`：方法、类别数、稳定性和共同空间指标；
- `data/curve_inventory.csv`：原始轴、单位、时长、点数和 SHA-256；
- `data/quality_flags.csv`：异常尺度、负输出、未知单位、低振幅和重复 x；
- `data/transformed_curves/`：原 x/y、u、可确认小时、MinMax 和 MaxAbs；
- `experiments/config.json`：训练前确定的规则和参数；
- `experiments/selected_parameters.json`：选出的 sigma/lr；
- `experiments/selection.json`：最终 k、seed、QE、ARI 和簇规模；
- `experiments/*sweep.csv`：逐次参数结果；
- `figures/all_curves/`：19 页全部 218 条无平滑曲线；
- `references/literature_review.md`：论文与扩展方法依据。
- `supplementary_filtered_maxabs_20260907/`：筛选后的独立补充实验、105 条 H500 推荐曲线及完整复现文件。
- `supplementary_initial2pct_checkpoints_20260907/`：初始值2%卡点策略、120条入选曲线、5类主解与相邻k复核。

## 10. 应如何表述结果

准确表述是：

> 在每条曲线完整观测区间映射为相对进程、每条曲线 MinMax 归一化且不平滑的条件下，论文式 QE elbow 稳定支持 4 个整体形态主簇；数据中存在与 Bridge、Hill、Slope、Valley 相似的原始轨迹，其中回升型还形成了较小的局部子群。

不能表述为：

> 218 条曲线在统一真实小时下被证明恰好分为 IFO-Bridge、IFO-Hill、IFO-Slope、IFO-Valley 四类。

主 x 轴是相对观测进程，MinMax 消除了总振幅，IFO 图没有成为训练标签，四条实例也没有各自落入四个不同主簇。这套实验回答的是“尽可能保留曲线后的无监督形态发现”，不是统一 200 h 的寿命比较。

## 11. 验证与限制

- 218 个源文件和 9,674 个点已通过哈希及逐点检查；
- 折线 L2 表示通过独立积分检查；
- C++ DTW/MSM 与独立 Python 动态规划的 200 组设置一致；
- 查看器的数据、筛选、坐标切换和导航逻辑已自动执行；
- 本会话没有可连接的真实浏览器，未完成真实浏览器点击测试；
- 图片数字化数据的轴元数据和采样密度异质；
- 相对进程忽略真实退化速度，MinMax 会放大小振幅波动；
- 没有真实标签，不能报告四分类准确率；
- 4 类是本数据、候选方法及参数范围下最有依据的简洁解，不是所有算法空间中的全局最优证明。

更完整的结果讨论和文献链接见 [完整技术报告](report.html)。
