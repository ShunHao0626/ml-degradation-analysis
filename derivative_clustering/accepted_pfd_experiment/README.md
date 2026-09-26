# Accepted 数据集：hour 统一与复杂半合成 PFD-DTW 实验

本目录用于在不修改 `dataset/accepted` 原始文件的前提下完成：

1. 横坐标单位审计，并只将可证明的时间轴换算为 hour；
2. 建立 0--200h 主分析数据；
3. 生成带真实采样、漂移、相关噪声和局部干扰的四类复杂半合成曲线；
4. 比较普通 DTW、论文 PFD-DTW、多尺度有序导数图和融合方法的注入恢复能力。

所有数据筛选、换算、生成参数和随机种子都会写入输出文件，人工标签只用于聚类完成后的评价。

## 第一步：hour 审计

```bash
python3 audit_hours.py
```

主要输出位于 `outputs/01_hour_audit/`：

- `curve_manifest.csv`：每条曲线的单位判断、换算、纳入状态和排除原因；
- `integrated_hours_long.csv`：所有可明确换算为 hour 的原始观测；
- `primary_0_200h_long.csv`：主实验使用的统一时间网格；
- `audit_summary.json` 和 `audit_overview.png`：统计与视觉核查。

## 第二步：复杂半合成注入恢复实验

```bash
python3 run_injection_experiment.py --preview-only
python3 run_injection_experiment.py
```

生成器在原始观测空间工作，继承真实曲线的不规则 hour 采样与 HP 残差，并随机加入
时间伸缩、弱峰谷、漂移、相关噪声、局部脉冲、量化平台和缺失采样。标签仅在聚类完成后
计算恢复指标，不参与距离、聚类或参数选择。

`outputs/02_injection/` 保留了第一轮压力测试。它证明直接把复杂曲线交给
average-linkage PFD-DTW 并不能稳定恢复四类，因此没有被删除或隐藏。

## 第三步：最终导数图 + PFD-DTW 融合实验

```bash
MPLCONFIGDIR=/private/tmp/mplconfig LOKY_MAX_CPU_COUNT=8 \
python3 run_derivative_graph_experiment.py \
  --seeds 2026 2027 2028 \
  --ratios 0.1 0.2 0.3 0.4 \
  --pfd-weight 0.15
```

最终方法用16个等时间窗的有序斜率建立10近邻图，并用论文 PFD-DTW 距离对图边
施加15%的惩罚，再做谱聚类。混合数据使用8个微簇，是为了给真实背景保留簇容量，
并不强迫全部真实曲线也属于四种目标形态。

最终结果位于 `outputs/03_derivative_graph/`。详细解释见 `FINAL_REPORT_CN.md`。
