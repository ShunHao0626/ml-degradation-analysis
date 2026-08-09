# 36类SOM结果包与复现入口

这是 `11_paper200h_tuned_rise_fall_som` 的统一入口。代码、正式结果、参数说明、软件环境和复现步骤均整理在本文件夹内。

## 1. 文件夹结构

```text
11_paper200h_tuned_rise_fall_som/
├── README_REPRODUCE_CN.md                 # 本入口文件
├── 36_CLASS_SOM_PARAMETERS_AND_WORKFLOW_CN.md
├── REPORT_CN.md                           # 简要正式报告
├── checksums.sha256                       # 文件完整性校验
├── run.log                                # 正式运行日志
├── code/
│   ├── run_paper200h_tuned_rise_fall_som.py
│   ├── reproduce_selected_36_model.py
│   └── verify_paper200h_tuned_rise_fall_som.py
├── environment/
│   ├── requirements-lock.txt
│   └── INPUT_AND_ENVIRONMENT_CN.md
├── 00_fixed_input_and_target/             # 固定输入清单与RTF诊断
├── 01_parameter_screen/                   # 99组参数初筛结果
├── 02_multiseed_validation/               # 9组候选的五种子验证
└── 03_selected_model/                     # 正式36类模型与逐曲线分配
```

正式结果目录不会被复现脚本覆盖。新的复现结果默认写入：

```text
reproduced_selected_model/
reproduced_full_run/
```

## 2. 最终模型摘要

| 参数或结果 | 数值 |
| --- | ---: |
| 输入曲线 | 1,442条 |
| 时间点 | 1,201（0–200 h，10 min间隔） |
| SOM | 6×6，共36节点 |
| `sigma` | 0.3 |
| `learning_rate` | 0.1 |
| 迭代次数 | 50,000 |
| 主随机种子 | 42 |
| RTF目标节点 | 22 |
| 节点成员 | 9条 |
| RTF-core成员 | 8条 |
| 节点纯度 | 88.89% |
| 召回率 | 14.29% |
| QE | 0.5735388 |

完整方法与所有公式见：

```text
36_CLASS_SOM_PARAMETERS_AND_WORKFLOW_CN.md
```

## 3. 第一步：进入结果包并检查正式结果

```bash
cd /Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/11_paper200h_tuned_rise_fall_som
python3 code/verify_paper200h_tuned_rise_fall_som.py
```

预期输出包括：

```json
{
  "status": "PASS",
  "selected_n": 36,
  "selected_sigma": 0.3,
  "selected_learning_rate": 0.1,
  "rise_then_fall_node": 22,
  "node_size": 9,
  "node_core_target_count": 8
}
```

这个检查不会重新训练，只检查正式输出、参数、成员数量和SHA-256。

## 4. 第二步：准备相同软件环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r environment/requirements-lock.txt
```

原始运行环境为Python 3.9.6。建议使用相同Python和依赖版本。固定输入位置及SHA-256见：

```text
environment/INPUT_AND_ENVIRONMENT_CN.md
```

## 5. 推荐复现：只重训最终36类模型

该方式固定使用最终参数，仅重新训练一次 `6×6, sigma=0.3, learning_rate=0.1, seed=42, 50,000 iterations`，并与归档模型逐项比较：

```bash
python3 code/reproduce_selected_36_model.py --strict
```

默认输出到：

```text
reproduced_selected_model/
```

主要生成：

- `reproduced_assignments.csv`：重新训练后的逐曲线节点；
- `reproduced_som_weights.npy`：重新训练后的权重；
- `reproduction_summary.json`：输入哈希、QE、TE、节点指标以及与正式结果的比较。

在相同输入和软件环境下，预期：

```text
status = PASS
labels_exact_match = true
weights_allclose_atol_1e-12 = true
```

`--strict` 会在分类或权重不匹配时返回非零退出码。

## 6. 完整复现：重新运行99组扫描和五种子验证

完整流程包括：

1. 99组参数、种子42、每组20,000次迭代；
2. 选取前8组合格候选，并加入文献4节点基准；
3. 9组候选分别使用5个随机种子进行50,000次训练；
4. 重新计算QE、TE、RTF precision/recall/F1、ARI和Jaccard；
5. 重新选择模型并生成全部图、CSV、模型与报告。

运行：

```bash
python3 code/run_paper200h_tuned_rise_fall_som.py \
  --input-dir ../03_preprocessed/main_high_quality_min10 \
  --output-dir reproduced_full_run
```

这是大量训练任务，耗时会显著长于只重训最终模型。程序默认使用新的 `reproduced_full_run` 文件夹，并拒绝把输出直接写到正式结果根目录。

完整训练结束后检查新结果：

```bash
python3 code/verify_paper200h_tuned_rise_fall_som.py \
  --output-dir reproduced_full_run
```

## 7. 正式结果位置

### 输入与RTF诊断

- `00_fixed_input_and_target/input_manifest.json`
- `00_fixed_input_and_target/rise_then_fall_diagnostics.csv`

### 参数扫描

- `01_parameter_screen/primary_seed_parameter_screen.csv`
- `01_parameter_screen/parameter_screen_precision_recall.png`

### 多随机种子验证

- `02_multiseed_validation/candidate_multiseed_metrics.csv`
- `02_multiseed_validation/candidate_stability.png`

### 最终36类模型

- `03_selected_model/selected_model.json`
- `03_selected_model/final_curve_assignments.csv`
- `03_selected_model/final_cluster_summary.csv`
- `03_selected_model/final_centroid_curves.csv`
- `03_selected_model/final_som_weights.npy`
- `03_selected_model/final_som_model.pkl`
- `03_selected_model/all_som_nodes.png`
- `03_selected_model/rise_then_fall_node_with_raw_points.png`

## 8. 防止误解

- SOM训练本身只使用预处理曲线，不使用RTF标签。
- RTF标签用于训练后的节点评价和参数组合选择。
- 36类不是原文献的主分类数，也不是单独根据QE拐点确定的。
- 节点22具有高纯度但低召回率，只代表一组典型RTF亚型，并未覆盖全部RTF曲线。
- 正式归档结果不要手动覆盖；任何复现都应写入新的输出目录。
