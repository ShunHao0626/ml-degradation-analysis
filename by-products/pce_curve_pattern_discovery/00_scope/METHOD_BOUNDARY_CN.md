# 方法边界与参数白名单

## 固定流程

- 输入根目录：`/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted/samples_test`。
- 仅纳入横轴明确为 hour/day 时间且纵轴明确为 PCE/efficiency 的曲线。
- 前 150 h，10 min 网格，Akima 插值，逐曲线 MaxAbs 归一化。
- Savitzky–Golay `(window=71, order=2)`。
- MiniSom 2.2.9，随机权重初始化，50,000 次顺序训练。

## 唯一扫描的论文参数

- `K=2...10`。
- `(sigma, learning_rate)` 只取论文展示的 `(0.5,0.1)`、`(0.3,0.1)`、`(0.5,0.3)`。
- `seed=0` 只保证 27 个候选公平复算，不参与搜索。

## 类别数规则

按论文方法先看 quantization-error elbow；再检查低 K 是否欠分、高 K 是否产生重叠、
不可区分或极小节点，选择能够覆盖主要形态的最小 K。目标 IFO 名称在 K 冻结前不可使用。
`K=16` 只复现 Supplementary Fig. 1 的探索分辨率，不进入 K=2...10 的最终选择。

## 数据审计结果

- 发现 CSV：218。
- 严格纳入：109。
- 排除原因计数（同一曲线可有多项）：`{"x_axis_not_explicit_time": 43, "duration_shorter_than_150h": 25, "x_axis_is_cycles": 17, "y_axis_not_explicit_pce_or_efficiency": 17, "time_unit_not_explicit_hour_or_day": 4, "nonfinite_or_nonpositive_series": 4, "record_140_implausible_hour_scale": 2, "record_35_no_observed_early_0_150h_segment": 1, "constant_series": 1}`。

## 明确没有做

没有修改 SOM 算法，没有增加导数、峰谷、材料或实验条件变量，没有目标模板，
没有按 Bridge/Hill/Slope/Valley 选择参数，也没有强制 K=4。
