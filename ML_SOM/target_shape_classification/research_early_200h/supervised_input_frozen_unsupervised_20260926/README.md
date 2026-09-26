# 有监督输入整理 + 冻结的无监督四簇拟合

本目录与上一轮[无标签质量筛选结果](../shape_only_input_curation_20260925/README.md)并列保存，不改写原始 CSV 或上一轮输出。**有监督部分只有 `prepare_dataset.py` 生成模型输入名单；聚类阶段的形状编码、参数和 KMeans 拟合函数保持上一轮不变。**这不等于整个端到端流程无监督，也不证明原始全集天然包含四个独立类别。

## 先看结果

- [四簇曲线放大图](results/anonymous_four_groups_zoom.png)（[SVG](results/anonymous_four_groups_zoom.svg)）：彩色淡线为全部成员，黑线为平滑后的逐序位中位数；横轴为相对观测序位，无 hour。
- [统一纵轴图](results/anonymous_four_groups.png)与[逐曲线交互查看器](results/anonymous_cluster_detail_viewer.html)。
- [1842 条匿名簇归属](results/anonymous_assignments.csv)、[完整 2152 条输入决策](full_input_decisions.csv)、[本轮额外暂缓的 55 条](supervised_input_exclusions.csv)。
- [结果、稳定性与限制](RESULTS.md)及[边界核验](results/verification.json)。

## 数据边界

上游 2152 条至少 4 个实测时刻的曲线中，上一轮质量门槛暂缓 255 条，剩余 1897 条。本轮只对这 1897 条做输入整理：用**上一轮冻结的匿名簇**和**旧的、未验证四类自动候选**识别三个小簇中二者不一致的曲线；总起伏不超过 0.01 PCE 的曲线保留，以免丢掉用户关注的约 1% 形状变化。由此额外暂缓 55 条，最后 **1842** 条进入模型；相对于更早的 2135 条方案，额外暂缓 **293** 条，符合约 300 条的取舍范围。

这 55 条不能被称为已证实的“非目标曲线”：旧候选没有独立人工真值，前一轮的簇也不是四类真值。此处是在**监督式整理输入**，让输入中小型形态簇与旧候选更一致。输入 CSV/NPZ 只含曲线 ID、来源、点数及数值轨迹，没有旧类别列；旧类别只存在于[输入决策表](full_input_decisions.csv)和整理脚本中。

## 复现顺序

在工作区根目录运行：

```bash
python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/prepare_dataset.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/run.py
python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/posthoc_audit.py
python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/sensitivity.py
python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/make_viewer.py
python3 target_shape_classification/research_early_200h/supervised_input_frozen_unsupervised_20260926/verify.py
```

[`config.json`](config.json)与上一轮除协议名称和输入文件路径外的全部参数一致。[核验脚本](verify.py)逐函数比较 `original_observations`、`encode`、`fit` 的语法树，并核对参数、输入列、覆盖与有限数值；结果见[verification.json](results/verification.json)。
