#!/usr/bin/env python3
"""Unlabeled diagnostics and historical disagreement, never semantic accuracy."""
from __future__ import annotations

import json
import statistics
from collections import Counter

from early_data import HERE, rows, write_rows


def main():
    summary = json.loads((HERE / "runs/summary.json").read_text(encoding="utf-8"))
    evaluation = HERE / "evaluation"
    evaluation.mkdir(exist_ok=True)
    metrics = []
    for run in summary["runs"]:
        for split in ("train", "validation"):
            metrics.append(dict(run_id=run["run_id"], variant=run["variant"], seed=run["seed"],
                                split=split, **run["metrics"][split]))
    write_rows(evaluation / "unlabeled_metrics.csv", metrics, list(metrics[0]))
    old = {r["file_id"]: r for r in rows(HERE.parent / "results/som_200h/canonical_assignments_200h.csv")}
    current = rows(HERE / "delivery/class_index.csv")
    comparison = []
    for r in current:
        previous = old.get(r["curve_id"], {})
        comparison.append(dict(curve_id=r["curve_id"], source_group=r["source_group"],
                               split=r["split"], new_rule_assisted_candidate=r["candidate_class"],
                               new_som_posthoc_candidate=r["som_posthoc_class"],
                               old_final_class=previous.get("final_class", ""),
                               old_decision_basis=previous.get("decision_basis", ""),
                               same_as_old=int(r["candidate_class"] == previous.get("final_class", "")),
                               note="Historical agreement is not classification accuracy."))
    write_rows(evaluation / "historical_comparison.csv", comparison, list(comparison[0]))
    lines = ["# 无标签结构与历史结果对照", "",
             "E1 和 E2 的距离定义不同，量化误差数值不能直接当作相同尺度的优劣指标。每个实验内的多种子范围如下。",
             "", "| 实验 | 验证 QE 均值 ± SD | 验证拓扑误差均值 ± SD | 占用节点范围 |", "|---|---:|---:|---:|"]
    for variant in ("E1", "E2"):
        subset = [r for r in summary["runs"] if r["variant"] == variant]
        q = [r["metrics"]["validation"]["quantization_error"] for r in subset]
        t = [r["metrics"]["validation"]["topology_error"] for r in subset]
        nodes = [r["metrics"]["validation"]["occupied_nodes"] for r in subset]
        lines.append(f"| {variant} | {statistics.mean(q):.5f} ± {statistics.stdev(q):.5f} | "
                     f"{statistics.mean(t):.4f} ± {statistics.stdev(t):.4f} | {min(nodes)}–{max(nodes)} |")
    equal = sum(r["same_as_old"] for r in comparison)
    lines += ["", f"与旧最终候选一致：{equal}/{len(comparison)}。旧结果含规则、人工覆盖与不同输入版本，"
             "此比例只用于排查差异，不能当作验证准确率。",
             "", "未完成：来源重采样稳定性、地图大小 6×6/10×10 对照、25/100 轮消融、独立四类参考指标。"]
    (evaluation / "structure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(summary["runs"]), "old_agreement": [equal, len(comparison)]}))


if __name__ == "__main__":
    main()
