# PCE 曲线无监督细节搜索（smoothing OFF）

本目录是一个独立的新实验。输入只读自 `lab/data_all`，不会修改 `thesis`、数据集或既有结果目录。

目标是在不固定类别数、不使用 IFO 标签选参、不按目标外观删曲线的条件下，系统比较时间窗口、客观数据质量规则、曲线表征和 SOM 参数。Bridge/Hill/Slope/Valley 名称只在最终模型冻结后用于解释。

运行：

```bash
python3 code/run_pipeline.py
python3 code/verify_outputs.py
```

核心边界：不调用 Savitzky–Golay、移动平均、样条、LOWESS 或任何其他平滑；固定长度向量只通过原始观测点间的分段线性插值构造，且所有代表图均回到原始数字化散点。

结果入口：`FINAL_DECISION_CN.md`、`reports/final_report_cn.md`、`06_final_model/clusters/all_clusters.png`、`07_posthoc_ifo/individual_ifo_raw_point_gallery.png`、`11_extended_window_cohort_sensitivity/extended_summary.json`。
