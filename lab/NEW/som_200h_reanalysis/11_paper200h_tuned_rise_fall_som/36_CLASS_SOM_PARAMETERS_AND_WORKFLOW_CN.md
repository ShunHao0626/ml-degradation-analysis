# 200 h文献式SOM分成36类：完整参数与流程

## 1. 文档目的与结论边界

本文档记录本项目如何得到一个包含36个节点的SOM，并说明如何在其中识别“先上升后下降”曲线。目标是使该结果可复现、可审计，并避免将探索性选参误写成文献原本的结论。

需要首先明确：

- “36类”是一个 **6×6 SOM中的36个神经元/节点**，不是36套不同参数。36个节点共享同一套SOM训练参数。
- 预处理及SOM基本算法沿用原文献作者流程，唯一的数据窗口改动是由150 h延长到200 h。
- 实际调节的SOM参数只有：节点数 `n_nodes`、邻域宽度 `sigma`、初始学习率 `learning_rate`。
- RTF（Rise Then Fall，先升后降）标签没有输入SOM，也没有参与权重更新；但它参与了参数组合的比较与最终模型选择。因此准确表述是：**无监督SOM训练＋目标导向的训练后选参**。
- 36类不是由QE拐点单独确定，也不是原文献推荐的主模型类别数。它是为了形成高纯度RTF小节点而选出的探索性过聚类结果。

最终选定参数为：

| 参数 | 最终值 |
| --- | ---: |
| SOM节点数 | 36 |
| SOM网格 | 6×6 |
| `sigma` | 0.3 |
| `learning_rate` | 0.1 |
| 最终训练迭代次数 | 50,000 |
| 主随机种子 | 42 |
| 主要RTF节点 | 22 |

## 2. 输入数据与固定预处理

### 2.1 输入文件

固定输入目录：

```text
/Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/03_preprocessed/main_high_quality_min10
```

主要输入文件：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `X_smoothed.npy` | SOM实际输入矩阵 | `8c9b473c1b3a56ffcda2d0b1464bcaf3f969a3cbead871f6cb9103cad8dd0096` |
| `time_grid.csv` | 统一时间网格 | `5395426af31831ac13ab63ceafc1d52831333fb8c8aa59818dbab5058f2cebbd` |
| `curve_metadata.csv` | 曲线身份及来源元数据 | `c6d076940a5f0a4084c33ab6adc36b991ee4308754c7f4f4a66a9a0bb4d3d6c0` |
| `preprocessing_config.json` | 预处理配置 | `386d296d6db17d9c93b81ebb4055f23c448607992ad9e608ad5b26159ee7ba58` |

输入矩阵形状：

```text
1,442条曲线 × 1,201个时间点
```

### 2.2 预处理顺序

每一条入选曲线按以下顺序处理：

1. 将时间统一转换为相对老化时间，起点设为0 h。
2. 使用0–200 h分析窗口。
3. 建立10 min间隔的共同时间网格，即步长为 `1/6 h`，总计1,201个时间点。
4. 使用 `scipy.interpolate.Akima1DInterpolator` 进行Akima插值。
5. 为保证能够插值到200 h，允许保留第一个超过200 h的原始点作为边界括点；不使用平尾填充，也不外推。
6. 每条曲线除以其自身0–200 h插值曲线中的最大PCE：

   ```text
   x_normalized(t) = x_interpolated(t) / max[x_interpolated(0:200 h)]
   ```

7. 对归一化曲线应用Savitzky–Golay平滑。
8. 将平滑后的1,201维曲线直接输入SOM。

### 2.3 固定预处理参数

| 项目 | 参数 |
| --- | --- |
| 分析窗口 | 0–200 h |
| 时间步长 | 10 min（0.1666667 h） |
| 插值 | Akima |
| 归一化 | 每条曲线除以自身0–200 h最大值 |
| Savitzky–Golay窗口长度 | 71个时间点，约11.83 h |
| Savitzky–Golay多项式阶数 | 2 |
| Savitzky–Golay边界模式 | `interp` |
| 最终特征维数 | 1,201 |

以下处理没有用于这个36类模型：

- PCHIP插值；
- 导数特征；
- 手工形状描述符；
- 特定时间段加权；
- PCA降维；
- DTW距离；
- RTF标签作为输入特征。

## 3. 最终SOM的全部训练参数

### 3.1 MiniSom构造参数

实际等效构造如下：

```python
som = MiniSom(
    x=6,
    y=6,
    input_len=1201,
    sigma=0.3,
    learning_rate=0.1,
    decay_function="asymptotic_decay",
    sigma_decay_function="asymptotic_decay",
    neighborhood_function="gaussian",
    topology="rectangular",
    activation_distance="euclidean",
    random_seed=42,
)
```

完整参数解释：

| 参数 | 数值 | 作用 |
| --- | --- | --- |
| `x` | 6 | SOM网格行数 |
| `y` | 6 | SOM网格列数 |
| `input_len` | 1201 | 每条输入曲线的时间点数 |
| `sigma` | 0.3 | 初始Gaussian邻域宽度 |
| `learning_rate` | 0.1 | 初始学习率 |
| `decay_function` | `asymptotic_decay` | 学习率衰减函数，MiniSom默认值 |
| `sigma_decay_function` | `asymptotic_decay` | 邻域宽度衰减函数，MiniSom默认值 |
| `neighborhood_function` | `gaussian` | 邻域函数 |
| `topology` | `rectangular` | 矩形网格拓扑 |
| `activation_distance` | `euclidean` | 输入曲线与节点权重的欧氏距离 |
| `random_seed` | 42 | 最终主模型随机种子 |

`asymptotic_decay` 的实现为：

```text
dynamic_parameter(t)
= initial_parameter / [1 + t / (max_iter / 2)]
```

因此在训练末期，学习率和sigma大约衰减至各自初始值的三分之一。

### 3.2 初始化与训练调用

```python
np.random.seed(42)
som.random_weights_init(curves)
som.train(
    curves,
    50_000,
    random_order=False,
    verbose=False,
    use_epochs=False,
    fixed_points=None,
)
```

| 训练设置 | 数值 |
| --- | --- |
| 权重初始化 | `random_weights_init(curves)` |
| 迭代次数 | 50,000 |
| 样本顺序随机打乱 | 否，`random_order=False` |
| Epoch训练模式 | 否，`use_epochs=False` |
| 固定BMU约束 | 无，`fixed_points=None` |
| 训练日志 | `verbose=False` |

### 3.3 类别分配和编号

每条曲线训练后分配给欧氏距离最近的最佳匹配单元（BMU）：

```python
(node_x, node_y) = som.winner(curve)
cluster_id = node_x * 6 + node_y
```

节点编号按行展开：

```text
 0   1   2   3   4   5
 6   7   8   9  10  11
12  13  14  15  16  17
18  19  20  21  22  23
24  25  26  27  28  29
30  31  32  33  34  35
```

最终SOM权重张量的形状为：

```text
6 × 6 × 1201
```

## 4. RTF诊断标签

### 4.1 基础形状量

对于每条预处理后的曲线 `x(t)`：

```text
peak_index       = argmax[x(t)]
peak_time        = t[peak_index]
initial_gain     = x(peak_time) - x(0 h)
post_peak_drop   = x(peak_time) - x(200 h)
```

峰后趋势以5 h为间隔检查。因为时间网格为10 min，5 h对应30个时间点。峰后相邻采样点满足以下条件时计为“未继续上升”：

```text
delta_x <= 0.002
```

`post_peak_nonincrease_fraction_5h` 是所有峰后5 h差分中满足该条件的比例。

### 4.2 四种RTF严格程度

| 标签 | 峰值时间 | 初始上升量 | 峰后至200 h下降量 | 峰后未上升比例 | 本数据数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RTF broad` | 2–180 h | ≥0.01 | ≥0.02 | ≥0.50 | 145 |
| `RTF medium` | 5–160 h | ≥0.02 | ≥0.03 | ≥0.55 | 97 |
| `RTF core` | 5–150 h | ≥0.03 | ≥0.05 | ≥0.60 | 56 |
| `RTF strong` | 5–120 h | ≥0.05 | ≥0.10 | ≥0.70 | 26 |

本次选参使用 `RTF core` 的56条曲线作为训练后的诊断目标。

再次强调：RTF标签不参与SOM权重训练。它只在每次SOM训练完成后用于计算节点的富集程度。

## 5. 第一步：99组参数初筛

### 5.1 初筛固定设置

| 项目 | 数值 |
| --- | ---: |
| 初筛随机种子 | 42 |
| 每组初筛迭代次数 | 20,000 |
| 初筛组合总数 | 99 |

### 5.2 参数网格

文献范围组合：

- 节点数：`2, 3, 4, 5, 6, 7, 8, 9, 10, 16`
- `(sigma, learning_rate)`：`(0.5, 0.1)`、`(0.3, 0.1)`、`(0.5, 0.3)`

为寻找小型RTF节点而扩展的组合：

- 节点数：`16, 20, 25, 36`
- `sigma`：`0.05, 0.10, 0.30, 0.50, 0.80, 1.20`
- `learning_rate`：`0.05, 0.10, 0.30`

去除重复组合后总计99组。

节点数自动转换为尽可能接近正方形的矩形拓扑。例如：

```text
4 -> 2×2
16 -> 4×4
20 -> 4×5
25 -> 5×5
36 -> 6×6
```

### 5.3 每次运行中如何找RTF节点

对每个节点计算：

```text
precision = 节点中的RTF-core数量 / 节点总曲线数
recall    = 节点中的RTF-core数量 / 56
F1        = 2 × precision × recall / (precision + recall)
```

一次SOM运行中的候选RTF节点依次按以下优先级选择：

1. F1更高；
2. 若F1相同，precision更高；
3. 若仍相同，RTF-core数量更多；
4. 若仍相同，节点总成员数更少。

代码中的等效排序键为：

```text
(F1, precision, target_count, -node_size)
```

### 5.4 初筛评分

```text
screen_score
= 0.50 × target_F1
+ 0.30 × target_precision
+ 0.15 × target_recall
+ 0.05 × centroid_is_RTF_broad
```

其中 `centroid_is_RTF_broad` 为真时取1，否则取0。

进入完整验证的资格条件：

- 目标节点至少包含5条 `RTF core` 曲线；
- 该节点的平均曲线满足 `RTF broad`；
- 在合格组合中按 `screen_score` 排名前8。

此外，无论是否进入前8，都加入文献主参数 `(n=4, sigma=0.5, learning_rate=0.1)` 作为基准。因此完整验证共有9组候选参数。

## 6. 第二步：多随机种子完整验证

### 6.1 验证设置

| 项目 | 数值 |
| --- | --- |
| 随机种子 | 7、21、42、84、168 |
| 每个种子的迭代次数 | 50,000 |
| 每组参数的独立训练次数 | 5 |
| 种子间配对数量 | 10 |
| 主结果种子 | 42 |

### 6.2 验证指标

- **QE（Quantization Error）**：每条曲线与其BMU权重向量之间欧氏距离的平均值。
- **TE（Topographic Error）**：第一和第二BMU在SOM网格中不相邻的样本比例。
- **ARI**：比较两个随机种子对全部1,442条曲线的分组一致性。
- **目标节点Jaccard**：比较两个种子各自最佳RTF节点的全部成员集合，计算交集除以并集。
- **平均目标precision、recall、F1**：五个随机种子各自最佳RTF节点指标的平均值。

### 6.3 高纯度节点门槛

参数组合必须同时满足：

```text
主种子节点中心满足 RTF broad
主种子节点 precision >= 0.70
主种子节点 RTF-core数量 >= 8
五种子平均 precision >= 0.50
五种子全局平均 ARI >= 0.50
五种子目标节点平均 Jaccard >= 0.25
```

满足全部条件时：

```text
stable_high_purity_core_node = True
```

这里的“stable”只是满足预先设置的综合门槛，并不代表每个随机种子的目标节点成员完全相同。

### 6.4 完整验证评分

```text
validation_score
= 0.30 × mean_target_F1
+ 0.25 × mean_target_precision
+ 0.15 × mean_target_recall
+ 0.15 × mean_global_seed_ARI
+ 0.15 × mean_target_node_Jaccard
```

最终排序规则：

1. 首先将 `stable_high_purity_core_node=True` 的模型排在前面；
2. 然后在同一门槛状态内按 `validation_score` 从高到低排列。

因此，36类并不是因为它拥有全体候选中最低QE或单纯最高评分而被选择，而是因为它先通过了高纯度RTF节点门槛，并在通过门槛的候选模型中取得最高综合验证评分。

## 7. 最终36类模型结果

### 7.1 主种子42

| 指标 | 数值 |
| --- | ---: |
| 节点数 | 36 |
| 网格 | 6×6 |
| `sigma` | 0.3 |
| `learning_rate` | 0.1 |
| 迭代次数 | 50,000 |
| QE | 0.5735388 |
| TE | 0.7760055 |
| 最佳RTF节点 | 22 |
| 节点22总成员数 | 9 |
| 节点22中的RTF-core数量 | 8 |
| 节点22 precision | 0.8888889（88.89%） |
| 节点22 recall | 0.1428571（14.29%） |
| 节点22 F1 | 0.2461538 |
| 节点中心峰值时间 | 71 h |
| 节点中心初始上升量 | 0.1263922 |
| 节点中心峰后下降量 | 0.0710147 |

节点22位于零起始坐标 `(3, 4)`，即按人类习惯从1开始计数的第4行、第5列。

### 7.2 五随机种子结果

| 指标 | 数值 |
| --- | ---: |
| 平均QE | 0.5715770 |
| QE标准差 | 0.0079558 |
| 平均目标节点precision | 0.7542113 |
| 平均目标节点recall | 0.1607143 |
| 平均目标节点F1 | 0.2513565 |
| 平均目标节点RTF-core数量 | 9.0 |
| 平均目标节点总大小 | 18.4 |
| 全局平均ARI | 0.6020586 |
| 全局最小ARI | 0.5270728 |
| 目标节点平均Jaccard | 0.3168752 |
| 目标节点最小Jaccard | 0.0163934 |
| 完整验证评分 | 0.4259070 |

## 8. 为什么不是文献主模型的4类

文献式4节点基准参数为：

```text
n_nodes=4, topology=2×2, sigma=0.5, learning_rate=0.1
```

在本数据和RTF-core定义下，种子42的结果为：

| 指标 | 4类基准 | 36类选定模型 |
| --- | ---: | ---: |
| 目标节点总大小 | 831 | 9 |
| 目标节点RTF-core数量 | 37 | 8 |
| precision | 4.45% | 88.89% |
| recall | 66.07% | 14.29% |
| QE | 1.3755 | 0.5735 |

4类模型覆盖了更多RTF曲线，但把它们与大量其他曲线放在同一个大节点中。36类模型牺牲召回率，换取了一个高纯度的小型RTF节点。因此两者回答的是不同问题：

- 4类：适合表达较粗的总体退化模式；
- 36类：适合探索一个较纯的先升后降形状亚型。

## 9. 结果限制与推荐表述

### 9.1 主要限制

1. 节点22只找回56条RTF-core曲线中的8条，召回率仅14.29%。不能将其解释为已经找出全部先升后降曲线。
2. 五种子目标节点最小Jaccard只有0.0164，说明最不一致的两个种子之间，目标节点成员变化很大。
3. 欧氏距离比较完整的1,201维曲线。同样具有先升后降转折的曲线，如果200 h终点或整体衰减幅度不同，仍可能被分配到不同节点。
4. RTF阈值及选参评分由本次研究问题定义，不属于原文献原始训练流程。
5. 随着节点数增加，QE通常会下降。因此不能仅用36类QE低于4类QE来证明36类更科学。

### 9.2 推荐写法

可以写为：

> 在保持文献所用Akima插值、曲线最大值归一化、Savitzky–Golay平滑和MiniSom训练框架不变的条件下，将分析窗口扩展至200 h，并对SOM节点数、邻域宽度及学习率进行参数扫描。SOM训练本身为无监督过程；训练后使用预定义的先升后降形状准则评估各节点的富集程度。最终的6×6 SOM可分离出一个高纯度但低召回率的先升后降曲线亚型节点，因此该结果作为探索性形状亚型分析，而非原文献主分类数的替代。

不推荐写为：

> QE证明36类是最佳分类数。

也不推荐写为：

> SOM自动发现并完整分出了所有先升后降曲线。

## 10. 复现代码与输出文件

### 10.1 执行代码

```text
/Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/code/run_paper200h_tuned_rise_fall_som.py
```

验证代码：

```text
/Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/code/verify_paper200h_tuned_rise_fall_som.py
```

### 10.2 关键结果

| 文件 | 内容 |
| --- | --- |
| `00_fixed_input_and_target/input_manifest.json` | 固定输入与处理参数 |
| `00_fixed_input_and_target/rise_then_fall_diagnostics.csv` | 1,442条曲线的RTF诊断量 |
| `01_parameter_screen/primary_seed_parameter_screen.csv` | 99组初筛结果 |
| `02_multiseed_validation/candidate_multiseed_metrics.csv` | 9组候选参数的五种子验证 |
| `03_selected_model/selected_model.json` | 最终参数和目标节点指标 |
| `03_selected_model/final_curve_assignments.csv` | 每条曲线的最终节点 |
| `03_selected_model/final_cluster_summary.csv` | 36个节点的成员与形状摘要 |
| `03_selected_model/final_centroid_curves.csv` | 36条节点中心曲线 |
| `03_selected_model/final_som_weights.npy` | 6×6×1201权重张量 |
| `03_selected_model/final_som_model.pkl` | 可重新加载的MiniSom模型 |
| `03_selected_model/all_som_nodes.png` | 36节点曲线总览 |
| `03_selected_model/rise_then_fall_node_with_raw_points.png` | 节点22及对应原始数据点 |

所有结果位于：

```text
/Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/11_paper200h_tuned_rise_fall_som
```

## 11. 文献流程核对依据

本次“文献式流程”的本地核对来源为：

```text
/Users/shunhao/Desktop/ML/thesis/paper/work.md
/Users/shunhao/Desktop/ML/thesis/paper/Supplementary.md
/Users/shunhao/Desktop/ML/thesis/20230816_degradation_analysis_revision_11_cleaned.ipynb
```

文献主分析使用较小SOM（主结果为4节点），补充材料考察了不同节点数以及部分 `sigma`、`learning_rate` 组合。36节点及更宽的参数网格属于本项目为识别RTF亚型进行的扩展调参，不应归因于原文献。

## 12. 软件环境

| 软件 | 版本 |
| --- | --- |
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| MiniSom | 2.3.6 |

不同软件版本、随机数实现或MiniSom默认行为可能造成节点编号及成员发生变化。复现时应保持输入文件、软件版本和随机种子一致。
