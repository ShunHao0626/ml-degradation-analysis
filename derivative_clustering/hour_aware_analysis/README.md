# 保留实际小时的两层曲线分组

先看 [中文结论](REPORT_CN.md)，以及
[同形态、不同小时尺度示例](outputs/physical_examples/same_shape_different_hours.png)。

本目录是新增实现，未改写 `eg`、原 400 条 synthetic curve 或此前实验。
形态层仍使用相对进程，时间层独立使用原始小时；不是把归一化横坐标改名为 hour。

## 对新数据运行

在 `/Users/shunhao/Desktop/ML/derivative_clustering` 下运行：

```bash
python3 hour_aware_analysis/run_analysis.py \
  --input-curves /绝对路径/新数据曲线目录 \
  --output /绝对路径/新的结果目录 \
  --shape-clusters 4
```

每条曲线一个 CSV，递归读取目录内全部 CSV；请勿把元数据/汇总表放在这个输入目录。
列为 `time_h` 加且仅加一种响应列名 `y`、`y_relative`、`normalized_pce`。
需要至少 11 条曲线，每条至少 5 个不同时间点，响应非全零且所有数值有限。
原始 `time_h` 必须已经是小时；本程序不会猜测秒/分钟/小时单位。
依赖当前项目已有的 numpy、scipy、pandas、scikit-learn、matplotlib、minisom，
以及旧验证目录的 DTW 库；库缺失时旧模块通过 clang++ 编译。

这是整批无监督聚类，不是训练后固定的四分类预测器。`--shape-clusters 4`
是人为指定四组，不是算法证明数据天然只有四类。换批次时组编号可以改变。
默认不自动命名 Bridge/Hill/Slope/Valley；需要看组内曲线后解释。
建议使用新的输出目录；同名输出文件会被当前结果覆盖。

## 关键输出

| 文件 | 用途 |
|---|---|
| `hour_assignments.csv` | 每条曲线的形态组、响应时间组、连续小时参数和不确定标记 |
| `physical_channels.csv` | 原始小时坐标、原始/归一化响应、每小时一阶/二阶导数 |
| `raw_observations.csv` | 输入观测副本，不伸缩时间、不补齐到共同终点 |
| `source_manifest.json` | CLI 曲线编号对应输入路径和 SHA256 |
| `shape_groups_actual_hours.png` | 形态组的原始小时叠图；线性与 symlog 小时视图 |
| `shape_medians_actual_hours.png` | 单独展示各形态组的实际小时逐点 median |
| `shape_medians_actual_hours.csv` | median 数值、每个小时的贡献曲线数与覆盖比例 |
| `response_subgroups_actual_hours.png` | 类内响应时间组；每幅使用自己的线性小时坐标 |
| `subgroup_models.json` | BIC 候选、有效样本数、选出的子组数和小时中位数 |
| `shape_graph_diagnostics.json` | 图连通分量及特征值求解残差 |

`excursion_10_elapsed_h` 指相对估计起点首次持续变化达到 0.10 个最大值归一化单位的时间。
不是“实验总时长”，也不直接等于 T80、半衰期或材料寿命。
`resolved` 仅表示通过预设采样区间宽度规则，不是统计置信保证；
`interval_uncertain` 表示跨越阈值的采样间隔太宽；
`not_observed_censored` 表示观测序列未确认持续跨越，不能排除采样间隙中发生过变化。
`timescale_subgroup=-1` 不确定或样本不足；编号 0 是该形态内较短响应时间组。
只有一个子组意味着当前规则没有分出多个群，不意味着其成员变化速度相同。

## 实验目录

- `outputs/`：最终数值修正后的 6 批、1,312 条曲线实例及物理时间对照。
- `initial_arpack_outputs/`：首轮结果完整保留，便于核对不连通图上的求解器敏感性。
- `cli_smoke/`：通过新数据命令行入口读取原 400 条曲线的验证，不是另一套数据生成。

复现命令：

```bash
python3 hour_aware_analysis/run_analysis.py
python3 hour_aware_analysis/run_analysis.py --input-curves eg_based_synthetic_dataset/curves --output hour_aware_analysis/cli_smoke
python3 hour_aware_analysis/verify_outputs.py
python3 -m pytest -q hour_aware_analysis/test_hour_model.py functional_validation_20260920/test_validation.py eg_dtw_derivative_experiment/test_experiment.py tests/test_pfd_dtw.py
```

大批数据时距离矩阵为平方存储，当前稠密谱求解也不适合直接扩展到数万条以上。

## 分类图中的 median

新增参考图风格的形态趋势图 `shape_groups_median_trends.png`（另存 SVG 和每类单图）。
淡蓝灰背景是原形态模型预处理后的曲线，粗黑线是同类曲线的逐点 median。
这里明确采用 **0–1 相对时间进程**，不是实际小时，也没有把形态 median 换算成小时。
使用已有分类，不重新聚类，不额外拟合预设类别模板或人为调整趋势。
它与下面的实际小时 median 回答不同问题；两套图分别保留。

```bash
python3 hour_aware_analysis/plot_shape_trends.py --folder hour_aware_analysis/outputs/existing_400
```

分类叠图的粗黑线为同一实际小时处的逐点中位数：每条曲线仅在自己的观测区间内
线性插值，等权贡献一个值；不拉伸时间，不外推，不用终点填充。
实线表示至少半数该组曲线仍有观测覆盖，虚线表示不足半数。
随着短曲线结束，参与计算的样本集合会改变，因此它不是时间对齐后的典型形态，
也不是单条材料的真实动力学轨迹。末端少数曲线决定的 median 尤其应谨慎解释。
只重画已有分类（不重新聚类）：

```bash
python3 hour_aware_analysis/run_analysis.py --refresh-galleries hour_aware_analysis/outputs/existing_400
```
