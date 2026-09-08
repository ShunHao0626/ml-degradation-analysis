# 方案 1：无监督变点检测 + 动力学特征 + HDBSCAN

## 目标
在**不使用任何人工类别标签、不预设 Bridge / Hill / Slope / Valley、不强制 K=4** 的前提下，从钙钛矿太阳能电池 ageing curves 中寻找自然出现的多阶段动力学形态。

本方案最适合识别：
- rapid increase → slow decay
- rapid increase → rapid decay → slow decay
- rapid decay → slow decay
- rapid decay → recovery → slow decay

因为它显式保留和建模“斜率变化、峰/谷、恢复、阶段切换”。

---

## 核心原则
1. **禁止任何平滑处理**
   - 不使用 Savitzky–Golay
   - 不使用 moving average
   - 不使用 LOWESS / LOESS
   - 不使用 Gaussian filter
   - 不使用 spline smoothing
   - 不使用 rolling median 作为主分析输入
2. 保留 raw curve 的全部原始起伏。
3. 不使用 Bridge / Hill / Slope / Valley 作为训练标签、聚类模板或参数选择依据。
4. 不强制聚类数为 4。
5. 参数选择只依赖无监督指标与稳定性。
6. 如果最终不是 4 类，接受结果。
7. 所有预处理前后数据都要保留，严禁覆盖 raw data。

---

# 可直接复制给 AI 的 Prompt

你是一名时间序列机器学习研究员。请基于我提供的 perovskite solar cell ageing time-series dataset，实施一个完全无监督的“change-point detection + kinetic descriptors + HDBSCAN”分析流程。

## A. 研究限制
本实验必须保持 fully unsupervised：
- 禁止使用人工标签。
- 禁止将 Bridge、Hill、Slope、Valley 作为训练目标。
- 禁止定义诸如“recovery > 10% 就是 Valley”的规则。
- 禁止为得到 4 类而人为调参。
- 禁止固定 cluster number = 4。
- 只有在聚类完成后，才允许根据 cluster centroid / medoid / representative curves 进行物理形态解释。
- 参数选择必须基于 label-free metrics、bootstrap stability 和 repeated-seed stability。

## B. 数据预处理
请首先完成数据审计，不要立即修改原始曲线。

### B1. 原始数据检查
对每条曲线记录：
- sample / device ID
- time vector
- output / PCE vector
- 起始时间
- 终止时间
- 总时长
- 原始采样点数
- median sampling interval
- missing fraction
- duplicate timestamps
- non-finite values
- obvious impossible values

生成一份 QC summary table。

### B2. 禁止 smoothing
主分析中：
```python
SMOOTHING = False
```

不得调用：
```python
scipy.signal.savgol_filter
rolling.mean()
rolling.median()
gaussian_filter1d()
lowess()
UnivariateSpline(..., s>0)
```

如果代码中已有 smoothing，请删除或显式关闭。

### B3. 缺失与异常值
仅删除：
- NaN / Inf
- 明确的 parsing error
- 重复 timestamp 中完全重复的记录

不要因为某个点偏离邻近值就删除它，因为该点可能是真实 transient feature。

若同一 timestamp 存在多个不同值：
- 保留原始副本
- 主分析可取该 timestamp 的中位数
- 必须记录发生次数

### B4. 是否重采样
优先尝试**不重采样**的 feature extraction。

如果某个后续算法必须使用统一时间轴：
- 仅使用线性插值到 common grid
- 禁止 cubic spline
- 禁止 Akima 作为主方案
- 禁止高阶插值
- common grid 的时间间隔不得细于数据集中典型原始采样间隔
- 保存 `raw_input` 和 `resampled_input` 两套数据

线性插值仅用于对齐，不视为 smoothing，但必须做 sensitivity analysis。

### B5. 归一化
同时测试以下 3 种，不提前选择：
1. relative-to-initial: `y(t)/y(0)`
2. max normalization: `y(t)/max(y)`
3. robust amplitude normalization: `(y-median(y))/(Q95-Q5)`

不要用“哪一种最容易出现四类”决定方案。
使用 cluster stability 与内部指标选择。

## C. 无监督变点检测

### C1. 原则
变点检测用于寻找 slope / level / variance 改变的位置，不得根据预设四种曲线形态定义变点数。

### C2. 推荐实现
优先使用 `ruptures`。

分别测试：
- PELT
- Binary Segmentation
- Window-based change-point detection

cost function 至少比较：
- `l2`
- `linear`
- `rbf`

### C3. 变点数量
不要固定每条曲线必须 2 个变点或 3 个变点。
允许 0、1、2、3 或更多变点。

为防止噪声造成过度切分，测试：
```text
min_segment_fraction = [0.03, 0.05, 0.08, 0.10]
penalty_multiplier   = [0.5, 1, 2, 3, 5, 8, 10]
```

### C4. 稳定变点
对每条曲线做 bootstrap / jitter sensitivity：
- 随机删去 5% 数据点
- 随机删去 10% 数据点
- 重复 50 次
- 比较变点时间位置是否稳定

定义稳定变点时只依据重复实验中的位置一致性，不依据是否符合 Hill / Valley。

## D. 动力学特征提取

### D1. 全局特征
至少包括：
- initial value
- final value
- global max / min
- time of global max / min
- net change
- total variation
- area under normalized curve
- fraction of positive / negative first differences
- maximum positive / negative finite-difference slope
- median positive / negative slope

### D2. 分段特征
根据变点自动切段后，对每一段计算：
- segment duration
- ordinary slope
- robust Theil–Sen slope
- start / end value
- amplitude change
- normalized amplitude change
- local variance
- number of raw points

为了形成固定长度 feature vector：
- 保存前 4 个最稳定 segment
- 不足部分用 NaN
- 额外保留 `segment_count`

### D3. 转折结构特征
计算连续值，不设人为类别阈值：
- max-to-next-min amplitude
- min-to-next-max amplitude
- recovery amplitude
- recovery ratio
- pre-peak slope
- immediate post-peak slope
- late-time slope
- pre-valley slope
- post-valley slope
- largest slope reversal magnitude
- number of slope sign reversals
- time between strongest slope reversal events

## E. 特征标准化
- 删除缺失率过高的 feature
- 对长尾 feature 使用 RobustScaler
- 保留未降维 feature matrix
- 可选 PCA，仅用于去共线/压缩，不得为了得到四类挑主成分

测试：
```text
variance retained = [90%, 95%, 99%]
```

同时跑：
- HDBSCAN on full standardized features
- HDBSCAN on PCA features

## F. HDBSCAN 聚类
测试：
```text
min_cluster_size = [10, 20, 30, 50, 75, 100]
min_samples = [None, 5, 10, 20]
cluster_selection_method = ["eom", "leaf"]
metric = ["euclidean", "manhattan"]
```

如果样本量较小，自动缩小 `min_cluster_size` 搜索范围。

不得写：
```python
n_clusters = 4
```

保存：
- cluster labels
- membership probabilities
- outlier scores
- cluster sizes

## G. 无监督模型选择
对每一个 preprocessing + feature + HDBSCAN combination 计算：
- DBCV
- silhouette score（排除 noise 后，同时报告 noise fraction）
- Davies–Bouldin index
- Calinski–Harabasz score
- cluster-size balance
- percentage noise
- bootstrap cluster stability
- ARI between perturbation runs
- variation of information between repeated runs

优先选择：
1. 稳定性高
2. noise fraction 合理
3. 内部指标好
4. 对 normalization / 轻微采样变化不敏感

绝对禁止以“这组参数出现了 Bridge/Hill/Slope/Valley”为选参依据。

## H. 结果解释
聚类完成、参数冻结后，才能进行 morphology interpretation。

对每个 cluster 输出：
- 所有成员 raw curves 叠图
- medoid curve
- 10 条最具代表性的真实曲线
- feature distribution
- change-point distribution
- slope-sequence distribution

然后才允许描述：
```text
Cluster X resembles rapid increase → slow decay.
Cluster Y resembles rapid decay → recovery → slow decay.
```

不要反向修改算法。

## I. 必做敏感性分析
至少重复：
1. 不重采样
2. 线性重采样
3. initial normalization
4. max normalization
5. robust normalization
6. 不同 change-point algorithm
7. 不同 HDBSCAN parameter range
8. bootstrap sampling
9. 删除 5% 随机时间点
10. 删除 10% 随机时间点

## J. 最终输出
请输出：
1. 完整 Python notebook / script
2. QC report
3. feature table CSV
4. cluster assignment CSV
5. cluster stability table
6. 每个 cluster 的 representative curves
7. 参数搜索结果
8. sensitivity analysis
9. markdown methods report
10. 一份“全流程未使用 class labels”的 audit log

最后明确回答：
- 数据天然支持多少个 cluster？
- 是否自然出现 increase→decay、rapid decay→slow decay、decay→recovery 等形态？
- 是否存在四个稳定且可重复的 cluster？
- 如果不是四个，保持原结果，不强行调整。
