# IFO 四种曲线形态识别

目标：严格只使用 `lab-v2/som_references/accepted` 中的真实数字化曲线，识别：

- IFO-Bridge：快速上升后缓慢衰减；
- IFO-Hill：快速上升、峰后快速下降，再转慢衰减；
- IFO-Slope：快速下降后慢衰减；
- IFO-Valley：快速下降、功率恢复、随后慢衰减。

流程先系统搜索论文 SOM 的时间窗口、sigma、learning rate 和 iterations，并做多随机种子稳定性检查；只有无法同时得到四个目标拓扑时，才使用相位感知的形态特征分类器。所有候选参数、中间矩阵、替代模型和最终结果均保存在本文件夹。

最终表同时给出四向兼容标签 `final_class` 与保守标签 `strict_class`。只有逐条满足目标相位顺序的曲线才进入严格四类；其余标为 `unresolved_other_shape`，不会为凑四类而污染 Slope。

运行：

```bash
MPLCONFIGDIR=/tmp/mplconfig_ifo_four_shape conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```

完成后先阅读 `REPORT_CN.md`、`00_references/METHOD_DECISION_CN.md`、`06_final_four_classes/classified_curve_assignments.csv` 和 `07_validation/validation_report.json`。
