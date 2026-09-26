# 0–200 h 分析代码导航

本目录包含多轮研究，入口之间有依赖关系。以下是当前分析链，命令均从 `ML_SOM/` 运行，且需要本地 `data_final/`。`results/`、`review_inputs/` 在公开仓库中不含曲线或审阅记录；没有这些输入时只能运行纯逻辑测试。

| 阶段 | 主要模块 | 输入 → 输出 |
|---|---|---|
| 来源审计与规范曲线 | `run.py` | `data_final/` 的原始 CSV、轴元数据与经哈希核对的复核覆盖 → `results/` 中的输入清单、规范曲线和形态对照 |
| SOM 特征与训练 | `unsupervised_som.py`、`som_evidence.py` | 规范曲线 → 原始点保留的特征、匿名神经元、稳定性与事后命名证据 |
| 固定 0–200 h 分类 | `classify_200h.py` | 窗口内实测点与 SOM 候选 → 全文件分类、边界 Valley 复核候选及排除原因 |
| 来源复核 | `redigitize.py`、`audit_review_workload.py`、`delivery_coverage.py` | 人工确认的原图和版本化补采 → 审阅任务、覆盖率和可追溯的交付表 |
| 敏感性与诊断 | `input_version_ablation.py`、`recovery_ablation.py`、`hill_heldout_diagnostic.py` 等 | 固定输入/已有结果 → 表示、恢复阶段和来源留出诊断 |
| 旧的早期 200 h 探索 | `research_early_200h/` | 独立配置与脚本；其结果不能与当前入口混用 |

`run.py` 提供输入加载、坐标单位、规范化和形态函数，后续模块直接导入它；`classify_200h.py` 同时使用 `run.py` 与 `unsupervised_som.py`。这使来源审计先于 SOM 和分类。四类名称用于训练后的解释和来源复核，不能把来源复核标签当作无标签拟合输入。

## 最小检查

```bash
cd ML_SOM
python3 -m unittest discover -s target_shape_classification -p 'test_*.py' -q
```

完整重跑的命令与所需输入见 [README](README.md)。运行前先核对 `review_inputs/` 中使用的轴覆盖及其原图哈希；缺少这些本地输入时，不应把新跑出的分类表与研究报告中的数量直接比较。
