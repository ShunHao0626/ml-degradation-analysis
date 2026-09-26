# 200h PvkSOM 论文参数复现

本文件夹针对 `lab-v2/som_references/accepted/samples_test` 的 218 条数字化曲线，按 Hartono 等人 PvkSOM 论文及其公开 notebook 的参数开展独立分析。唯一目标参数改动是把时间窗口从 150h 延长到 200h。

生成结果后优先阅读：

- `REPORT_CN.md`：结果、样本数、聚类和限制；
- `source_references/METHOD_EVIDENCE_CN.md`：原文参数与本次实现逐项映射；
- `data_0_200h/all_218_curves_audit.csv`：218 条曲线逐条纳入/排除记录；
- `data_0_200h/extracted_raw_0_200h_10min.csv`：先提取并统一到前 200h 的数据；
- `results/som_cluster_assignments.csv`：每条曲线的 SOM 归属；
- `verification_report.json` 与 `checksums.sha256`：完整性验证。

复现命令：

```bash
MPLCONFIGDIR=/tmp/mplconfig_som200h_paper_exact conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```

注意：输入是文献图数字化曲线，不是论文作者原始 2min MPPT 数据。无法从输入核验的 N₂、1 sun、温度等实验条件不会被擅自假定为满足。
