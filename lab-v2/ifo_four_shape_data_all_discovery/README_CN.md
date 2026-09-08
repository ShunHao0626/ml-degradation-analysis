# data_all 的 IFO 四种曲线形态识别

唯一数值输入目录：`/Users/shunhao/Desktop/ML/lab/data_all`。

目标形态：

- IFO-Bridge：前 200h 快速上升，随后慢衰减；
- IFO-Hill：前 200h 快速上升并快速回落，随后慢衰减；
- IFO-Slope：前 200h 快速下降，随后慢衰减；
- IFO-Valley：前 200h 快速下降，随后恢复，并在恢复峰后再次慢衰减。

程序先复现论文预处理和 2×2 SOM，再扩展时间窗口、sigma、learning rate 与训练迭代数。若普通 SOM 不能产生四个独立目标拓扑，则搜索相位感知的可解释规则。所有小时、天、周、月、年单位统一换算成小时。

最终同时给出：

- `final_class`：与普通四聚类对照用的四向标签；
- `strict_class`：逐条通过目标拓扑的保守标签；
- `unresolved_other_shape`：不强行归入四类的其他形状。

运行：

```bash
MPLCONFIGDIR=/tmp/mplconfig_ifo_four_shape_data_all conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```

完成后首先阅读 `REPORT_CN.md`、`06_final_four_classes/strict_four_class_matches.csv`、`06_final_four_classes/unresolved_or_other_shapes.csv` 与 `07_validation/independent_verification.json`。
