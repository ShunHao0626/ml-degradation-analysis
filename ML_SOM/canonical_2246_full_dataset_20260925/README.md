# 2246 条去重规范曲线：全程数据集

**`dataset/` 才是原始去重数据集：恰好 2246 个 CSV，逐字节复制自原始
2250 文件集合中的规范源文件。**文件名从 `curve_0001.csv` 连续到
`curve_2246.csv`；仅文件名改变，CSV 内的列、行、数值和顺序均未改变。
`dataset_manifest.csv` 逐行对应原来的 F 编号与 SHA-256。

此目录是从项目指定的 `original_curves_2250` 构建的**独立副本**。输入依据
`target_shape_classification/results/canonical_curves.csv`：2250 个原始文件中
4 个是已有规范曲线的重复别名，故数据集有 2246 条不同规范曲线。
原始源文件未修改；其全部行及全时间范围均保留，**没有截断至 200 h**。

## 文件

- `dataset/curve_xxxx.csv`：2246 条规范源 CSV 的**原样副本**。
- `derived/curve_xxxx.csv`：相同 2246 条曲线的可选分析视图；在这里另加
  小时换算和归一化列，不改变 `dataset/` 的任何原始字节。
- `dataset_manifest.csv`：2246 条曲线的新编号、原 F 编号、路径、来源、单位、转换系数、轴状态、
  点数以及两种归一化分母。
- `canonical_curves.csv`：2246 条规范曲线的原始清单副本。
- `provenance/` 另有 2250 个**历史原文件 ID**到 2246 条规范曲线的映射，
  以及原文件清单。这只是来源追溯，不是另外 2250 条数据。
- `audit.json`、`verification.json`：数量、哈希及逐行坐标核验。

## `derived/` 分析视图的列

| 列 | 含义 |
|---|---|
| `source_row` | 原 CSV 的行号，表头为第 1 行 |
| `source_x`, `source_y` | 原文件中的 x、y 文本值，**不能默认把 source_x 当小时** |
| `hour` | 按清单 `time_factor` 从 source_x 换算的小时；无法确认时留空 |
| `normalized_pce_full` | 已确认 PCE 曲线的 source_y ÷ **全程**有效 y 最大值 |
| `normalized_pce_0_200h` | 已确认 PCE 曲线的 source_y ÷ **0–200 h** 有效 y 最大值；全程每行均按同一分母计算，所以 200 h 后可能大于 1 |

两种归一化列都只对 `pce_status=confirmed_pce` 且分母为正的曲线填写。
**`dataset/` 没有这些列，也没有进行缩放、归一化、截窗、插值或平滑。**
若重新做严格 0–200 h 实验，可在明确记录方法后使用 `derived/` 的
`hour` 和 `normalized_pce_0_200h`，并筛选 `0 ≤ hour ≤ 200`。

## 已知坐标边界

- 2245 条曲线可按现有清单转换成小时；1 条（F02216）没有可信的换算系数，
  其原始 x 完整保存，但 `hour` 留空。
- 2221 条是已确认 PCE，均可计算全程最大值归一化；其中 2216 条也有可用的
  0–200 h 正分母。其余 25 条是非 PCE 或派生量，未伪写成 PCE。
- 4 条原始 x 单位为近似月，按原清单的 `730.485 h/月` 换算；如需要精确
  物理时长，应回查论文坐标。负时间原值保留，建模时按协议筛选。

## 复现

```bash
cd ML_SOM/canonical_2246_full_dataset_20260925
python3 build_dataset.py
python3 verify_dataset.py
```

没有复制论文原图；`dataset_manifest.csv` 中保存了它们相对于项目 `data_final/`
的路径。`dataset/` 是仅去重的原始数值数据；`derived/` 只是可复现的派生视图。
