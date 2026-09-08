# 方案 2：PCE + 导数的 Multivariate DTW + 层次聚类 / DTW-SOM

## 目标
在完全无监督条件下，让算法重点感知：
- 斜率正负
- rapid / slow transition
- peak / valley
- recovery
- 同一种形态在不同时间发生的 time shift

本方案特别适合处理“形状相似但 peak / valley 出现时间不同”的曲线。

---

# 可直接复制给 AI 的 Prompt

请对我提供的 perovskite ageing curves 实施一个 fully unsupervised 的 derivative-aware DTW clustering workflow。

## A. 禁止事项
主分析中禁止：
- Savitzky–Golay
- moving average
- rolling median
- LOWESS
- Gaussian smoothing
- spline smoothing
- wavelet denoising
- 人工 curve labels
- template matching
- 固定 4 clusters
- 根据是否出现预期四类来挑参数

## B. 原始数据保留
建立：
```text
raw/
aligned/
features/
distance_matrices/
clustering_results/
stability/
figures/
```
原始文件永远只读。

## C. 时间轴处理
### C1. QC
检查：
- sampling interval distribution
- curve duration
- point count
- missing values
- duplicated timestamps

### C2. 统一时间范围
不要自动截到 150 h。
根据数据覆盖情况绘制 `coverage vs time`。
候选窗口可自动生成：
- 全部样本都覆盖的最大公共窗口
- 90% 样本覆盖窗口
- 80% 样本覆盖窗口

分别做 sensitivity analysis。

### C3. 对齐
优先：
1. 若实现允许 variable-length DTW，直接使用原始采样序列。
2. 若必须 common grid：
   - 使用线性插值
   - 不使用 spline / Akima
   - grid spacing 不应细于典型原始采样间隔

不得额外 smoothing。

## D. 归一化
并行比较：
1. `y / y0`
2. `y / ymax`
3. z-normalization
4. robust amplitude normalization

## E. 一阶导数表示
因为不允许 smoothing，主分析使用真实 finite difference：
```python
dy_dt = np.diff(y) / np.diff(t)
```

要求：
- 使用真实 `dt`
- 不假设固定 sampling interval

可作为 sensitivity analysis 计算 3 点局部线性斜率，但不要用卷积或滤波器。

构造 multivariate sequence：
`X(t) = [y_norm(t), lambda * dy/dt]`

测试：
```text
lambda = [0.25, 0.5, 1, 2, 4]
```

lambda 的选择依据只能是稳定性与内部指标。

## F. DTW 距离
分别计算：
1. classic DTW on normalized PCE
2. derivative DTW
3. multivariate DTW on [PCE, derivative]

限制 warping：
```text
Sakoe-Chiba radius = [2%, 5%, 10%, 15%, 20% of sequence length]
```

同时报告 unrestricted DTW 作为对照，但不优先使用。

## G. 聚类算法
### G1. 层次聚类
基于 precomputed DTW distance matrix 测试：
- average linkage
- complete linkage
- weighted linkage

不要使用 Ward linkage。

候选 cluster number：
```text
K = 2...10
```

### G2. DTW-SOM
如果实现 DTW-SOM，测试：
```text
1x2, 1x3, 2x2, 2x3, 3x3, 3x4
```
repeated random seeds >= 50。

## H. K 的选择
Hierarchical clustering：
- silhouette with precomputed DTW distance
- Dunn index
- bootstrap ARI
- consensus matrix

DTW-SOM：
- quantization error
- topographic error
- repeated-seed stability
- occupancy
- bootstrap stability

禁止因为 K=4 是目标而直接选 K=4。

## I. Consensus / 稳定性
重复：
- 90% curve bootstrap
- 95% curve bootstrap
- 每条曲线随机删除 5% time points
- 每条曲线随机删除 10% time points

至少 50 次。

构建 co-clustering consensus matrix：
`C_ij = 两条曲线在重复实验中被分到同一 cluster 的比例`。

## J. 结果展示
参数冻结后：
- 每个 cluster 绘制所有 raw curves
- medoid 必须是真实曲线
- mean / median 可显示，但明确不是原始样本
- derivative distribution
- peak / valley time distribution
- within-cluster DTW distance

之后才允许解释：
- increase → slow decay
- increase → fast decay → slow decay
- rapid decay → slow decay
- decay → recovery → slow decay

## K. 防止 DTW 过度对齐
必须检查：
- warping path length
- maximum local warping
- average warping amount
- 是否把多个转折强行对齐成一个转折

如果某组参数需要极端 warping 才产生稳定 cluster，应降权或剔除。

## L. 最终输出
1. preprocessing audit
2. normalized curve matrices
3. derivative sequence files
4. DTW distance matrices
5. clustering labels for K=2...10
6. internal metric table
7. stability table
8. consensus matrix
9. representative raw curves
10. methods summary

最终结论只能描述数据自己支持的 cluster structure；若不是 4 类，不允许修改参数直到出现 4 类。
