# 文献参数边界与类别数判据

## 固定流程

- 仅使用明确的 PCE/efficiency-time 曲线；严格窗口为前 150 h。
- 10 min 重采样，Akima 插值，逐曲线 MaxAbs 归一化，Savitzky-Golay `(71, 2)`。
- MiniSom 2.2.9，`random_weights_init`，50,000 次顺序训练。
- 实际矩阵每条 900 点，对应 `1/6 h ... 150 h`；这与作者发布的 `(2245, 900)` 数组一致。

## 唯一允许搜索的论文参数

- 类别数 `n=2...10`。
- 三组 `(sigma, learning_rate)`：`(0.5,0.1)`、`(0.3,0.1)`、`(0.5,0.3)`。
- `seed=0` 只用于让候选公平复算，不作为搜索变量。

## 类别数规则

论文规则不是固定 K=4，也不是一个额外的自动指标：先看 quantization-error elbow，
再检查较小 K 是否欠分、较大 K 的中心是否开始重叠或不可区分，选择仍能捕获主形态的最小 K。
IFO-Bridge/Hill/Slope/Valley 不参与训练、排序或 K 选择，只能在结果冻结后事后解释。

`n=16` 仅因为 Supplementary Fig. 1 明确展示而作为探索对照；它不进入 `n=2...10`
肘部，也不能被冒充为论文最优类别数。

## 数据审计

- 总 CSV：218。
- 严格纳入：109。
- 排除原因计数（同一曲线可有多项）：`{"x_axis_not_explicit_time": 43, "duration_shorter_than_150h": 25, "x_axis_is_cycles": 17, "y_axis_not_explicit_pce_or_efficiency": 17, "time_unit_not_explicit_hour_or_day": 4, "nonfinite_or_nonpositive_series": 4, "record_140_implausible_hour_scale": 2, "record_35_no_observed_early_0_150h_segment": 1, "constant_series": 1}`。

## 复现歧义

论文没有完整列出每个 n 的 SOM 网格形状。这里采用与补图一致且尽量接近方形的因子布局：
`{"2": [1, 2], "3": [1, 3], "4": [2, 2], "5": [1, 5], "6": [2, 3], "7": [1, 7], "8": [2, 4], "9": [3, 3], "10": [2, 5]}`。布局会影响拓扑，必须与结果一起报告。

## 文献证据

- `thesis/paper/work.md`：Data analysis、Degradation curve shape clustering。
- `thesis/paper/Supplementary.md`：Supplementary Figs. 6、7、15-17 与 SOM Quantisation Error。
