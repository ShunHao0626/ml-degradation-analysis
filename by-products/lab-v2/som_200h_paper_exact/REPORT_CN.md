# 200h PvkSOM 严格参数复现结果

## 结论摘要

对 218 条数字化曲线完成了全量审计，最终有 114 条满足可确认时间/PCE 轴、跨度不少于 200h、全局最大 PCE 不在 200h 之后以及基础数值质量要求。提取后矩阵为 `114 × 1200`，时间分辨率 10min，截止 200h。

SOM 使用论文原参数：MiniSom 2.2.9、2×2、sigma=0.5、learning rate=0.1、50,000 iterations、random_weights_init、顺序训练、random seed=None。该次训练的量化误差为 0.755603，拓扑误差为 0.000000。

- fast_exponential_decay: 3 条（SOM neuron 0,1）
- initial_gain: 54 条（SOM neuron 1,0）
- medium_exponential_decay: 22 条（SOM neuron 0,0）
- slow_exponential_decay: 35 条（SOM neuron 1,1）

DTW k-means 对照状态：`completed`，与 SOM 的 ARI=0.8058。

## 数据筛选

- 原始曲线：218
- 纳入：114
- 排除：104

审计组合原因（同一条曲线可同时命中多项，因此此处是组合计数）：

- `selected`: 114
- `x_axis_or_unit_unverified`: 38
- `duration_shorter_than_200h`: 31
- `x_axis_is_cycles_not_time`: 17
- `global_maximum_pce_after_200h`: 10
- `x_axis_or_unit_unverified;nonfinite_or_nonpositive_pce`: 4
- `implausible_duration_over_100000h`: 2
- `y_axis_not_verified_as_pce_efficiency_or_power`: 2

所有曲线和排除原因见 `data_0_200h/all_218_curves_audit.csv`。前 200h 的原值、归一化值和平滑值分别保存在同目录三个矩阵 CSV 以及 `analysis_matrices.npz`。

## 绝对 PCE 子集

只有 26 条曲线的纵轴能明确确认是绝对 PCE/效率百分比。严格按原 notebook 的 top-3 最大值均值、200h 末点、5 个等数量 PCE 组计算后，组均值回归斜率为 -0.613586，R²=0.559087。由于输入缺少 N₂、1 sun、温度等条件元数据，这一部分只能视作探索性结果，不能与原论文 2,245 个同质器件的统计结论等同。

## 重要边界

输入并非原论文的原始 2min MPPT 数据，而是 218 条文献图数字化曲线，且包含不同研究条件。能保持一致的是公开方法与数值参数；无法从输入核验的实验条件没有被假定为满足。具体参数证据和不可避免的适配见 `source_references/METHOD_EVIDENCE_CN.md`。

## 复现

```bash
cd /Users/shunhao/Desktop/ML/lab-v2/som_200h_paper_exact
MPLCONFIGDIR=/tmp/mplconfig_som200h_paper_exact conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```

原 notebook 没有设置随机种子。本次也没有新增 seed，以保持参数一致；因此重新训练可能得到旋转/置换或局部差异。当前运行的初始权重、最终权重和 pickle 模型均已保存，可用于精确追溯本次结果。
