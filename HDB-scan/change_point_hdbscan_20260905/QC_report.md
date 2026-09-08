# 数据质量审计

共 218 条曲线，99 个来源图片。原始 CSV/元数据已复制至 raw_input，所有源 CSV 均已校验 SHA-256；未覆盖原文件。

| unit | curves | min_points | median_points |
| --- | --- | --- | --- |
| cycles | 17 | 27 | 51.0000 |
| days | 2 | 46 | 46.0000 |
| hours | 157 | 6 | 39.0000 |
| unknown | 42 | 15 | 43.0000 |

非有限值 0；完全重复记录 0；冲突时间戳位置 12（按时刻取中位数，原副本保留）。未因局部偏离或 transient 删除原始点。

不足 12 点的曲线：

| sample_id | figure_id | unit | clean_n |
| --- | --- | --- | --- |
| S0024 | 图片12 | hours | 8 |
| S0094 | 图片190 | hours | 6 |
| S0124 | 图片33 | hours | 10 |

需检查负输出或极端轴跨度的曲线（只标记，不擅自修改）：

| sample_id | figure_id | unit | minimum_output | maximum_output | duration_analysis | qc_flags |
| --- | --- | --- | --- | --- | --- | --- |
| S0048 | 图片140 | hours | 0.9238 | 0.9995 | 42488794.4652 | extreme_axis_span_review;negative_start_retained_shifted |
| S0049 | 图片140 | hours | 0.8330 | 0.9822 | 41308550.1745 | extreme_axis_span_review;negative_start_retained_shifted |
| S0111 | 图片208 | unknown | -142.0000 | -68.0000 | 420.0000 | negative_output |
| S0112 | 图片208 | unknown | -134.0000 | -68.0000 | 410.0000 | negative_output |
| S0113 | 图片208 | unknown | -86.0000 | -65.0000 | 420.0000 | negative_output |
| S0114 | 图片208 | unknown | -105.0000 | -66.0000 | 420.0000 | negative_output |
| S0126 | 图片35 | hours | 101.8945 | 122.7149 | 9939.7570 | absolute_efficiency_gt100 |

完整逐条记录：[qc_summary.csv](qc_summary.csv)。纳入与排除原因：[all_218_sample_status.csv](final/all_218_sample_status.csv)。窗口和稳定性结论：[研究结果与方法报告.md](研究结果与方法报告.md)。
