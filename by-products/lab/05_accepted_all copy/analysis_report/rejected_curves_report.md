# 200h数据预处理报告

## 一、数据处理概况

- 原始数据来源: `curves_over_200h/`
- 处理后曲线数: 1852
- 舍弃曲线数: 6
- 舍弃率: 0.3%

## 二、处理参数

- 目标时间范围: 0-200 小时
- 数据点数: 1200 (10分钟间隔)
- 归一化方法: MaxAbsScaler

## 三、按DOI统计舍弃情况

| DOI | 舍弃曲线数 | 舍弃原因 |
|-----|-----------|----------|
| 10.1021_acs.nanolett.9b02142 | 2 | interpolation_failed: 2 |
| unknown | 1 | missing_columns: 1 |
| 10.1016_j.jechem.2020.08.055 | 1 | interpolation_failed: 1 |
| 10.1002_anie.202410069 | 1 | interpolation_failed: 1 |
| 10.1021_acsami.9b00923 | 1 | interpolation_failed: 1 |

## 四、舍弃原因分类统计

| 舍弃原因 | 数量 |
|----------|------|
| interpolation_failed | 5 |
| missing_columns | 1 |

## 五、被舍弃的曲线详细列表

详见 `rejected_curves.csv`
