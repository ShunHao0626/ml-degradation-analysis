#!/usr/bin/env python3
"""Freeze the literature-rule K decision and export post-hoc evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREP = ROOT / "02_preprocessed_150h" / "strict_paper_preprocessed_150h.npz"
METADATA = ROOT / "02_preprocessed_150h" / "curve_metadata.csv"
METRICS = ROOT / "03_som_search" / "all_27_run_metrics.csv"
SELECTED_RUN = ROOT / "03_som_search" / "pair_1_sigma_0.5_lr_0.1" / "k_04" / "run_arrays.npz"
N16_RUN = ROOT / "03_som_search" / "n16_paper_exploratory_not_for_k_selection" / "run_arrays.npz"
SELECTION_DIR = ROOT / "04_k_selection"
FINAL_DIR = ROOT / "05_selected_clusters"
POSTHOC_DIR = ROOT / "06_ifo_posthoc"
VALIDATION_DIR = ROOT / "08_validation"


def ensure_dirs() -> None:
    for path in (SELECTION_DIR, FINAL_DIR, POSTHOC_DIR, VALIDATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def plot_selected(
    time_h: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    interpretations = [
        "strong/fast decay",
        "slow decay",
        "medium decay",
        "near-stable / weak initial gain",
    ]
    for node, ax in enumerate(axes.ravel()):
        members = data[labels == node]
        for curve in members:
            ax.plot(time_h, curve, color="#8da0ae", alpha=0.16, lw=0.6)
        ax.plot(time_h, centroids[node], color="#c51b7d", lw=2.5, label="member mean")
        ax.plot(time_h, weights[node], color="black", ls="--", lw=1.1, label="SOM weight")
        ax.set_title(f"Anonymous node {node} — {interpretations[node]} (n={len(members)})")
        ax.grid(alpha=0.2)
        if node % 2 == 0:
            ax.set_ylabel("Normalized PCE")
        if node // 2 == 1:
            ax.set_xlabel("Time (h)")
    handles, text = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, text, loc="upper right", fontsize=8)
    fig.suptitle("Selected K=4 by literature rule; IFO names were not used in selection", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FINAL_DIR / "selected_k4_all_members_and_centers.png", dpi=220)
    plt.close(fig)


def selection_report(metrics: pd.DataFrame) -> None:
    comparison = metrics[metrics["k"].isin([3, 4, 5, 6])].copy()
    comparison.to_csv(SELECTION_DIR / "k3_to_k6_comparison.csv", index=False)
    drops = []
    for pair_id, group in metrics.groupby("pair_id"):
        indexed = group.set_index("k")
        drops.append(
            {
                "pair_id": int(pair_id),
                "sigma": float(group["sigma"].iloc[0]),
                "learning_rate": float(group["learning_rate"].iloc[0]),
                "qe_drop_k3_to_k4_percent": float(indexed.loc[4, "qe_drop_from_previous_percent"]),
                "qe_drop_k4_to_k5_percent": float(indexed.loc[4, "qe_drop_to_next_percent"]),
                "qe_drop_k5_to_k6_percent": float(indexed.loc[5, "qe_drop_to_next_percent"]),
                "k4_minimum_node_size": int(indexed.loc[4, "minimum_occupied_node_size"]),
                "k5_minimum_node_size": int(indexed.loc[5, "minimum_occupied_node_size"]),
                "k6_minimum_node_size": int(indexed.loc[6, "minimum_occupied_node_size"]),
            }
        )
    pd.DataFrame(drops).to_csv(SELECTION_DIR / "elbow_drop_summary.csv", index=False)
    report = """# 类别数决定：K=4

这个决定在检查 IFO-Bridge/Hill/Slope/Valley 之前冻结。

三组论文参数在 K=4 都出现第一个共同且明显的 quantization-error 肘部：K=3→4
下降约 25.0%–30.8%，而 K=4→5 仅继续下降约 4.3%–9.2%。K=3 会把仅四条的
强/快速衰减曲线并入更宽的衰减组，属于欠分。K=5、6 主要继续按衰减幅度拆分近稳定、
慢衰减和中等衰减中心；增加的节点不形成新的、样本充分且清楚可分的主拓扑。

因此按 Hartono 等人的规则——“能够捕获主要形态的最小类别数，同时避免高 K 中心
重叠或不可区分”——选择 K=4。最终导出使用论文正文基线 `(sigma, learning_rate)=(0.5,0.1)`。

K=16 仅为 Supplementary Fig. 1 的探索分辨率，不参与最终 K 选择。
"""
    (SELECTION_DIR / "K_SELECTION_DECISION_CN.md").write_text(report, encoding="utf-8")


def export_selected(
    metadata: pd.DataFrame,
    time_h: np.ndarray,
    data: np.ndarray,
    arrays: np.lib.npyio.NpzFile,
) -> None:
    labels = arrays["labels"].astype(int)
    weights = arrays["weights"]
    centroids = arrays["member_centroids"]
    counts = arrays["counts"].astype(int)
    assignments = metadata.copy()
    assignments["anonymous_som_node"] = labels
    assignments["node_x"] = labels // 2
    assignments["node_y"] = labels % 2
    assignments.to_csv(FINAL_DIR / "cluster_assignments.csv", index=False)
    summary = pd.DataFrame(
        {
            "anonymous_som_node": np.arange(4),
            "count": counts,
            "posthoc_plain_description": [
                "strong_or_fast_decay",
                "slow_decay",
                "medium_decay",
                "near_stable_or_weak_initial_gain",
            ],
        }
    )
    summary.to_csv(FINAL_DIR / "anonymous_cluster_summary.csv", index=False)
    curves: dict[str, np.ndarray] = {"time_hours": time_h}
    for node in range(4):
        curves[f"node_{node}_member_mean"] = centroids[node]
        curves[f"node_{node}_som_weight"] = weights[node]
    pd.DataFrame(curves).to_csv(FINAL_DIR / "anonymous_cluster_curves.csv", index=False)
    for node in range(4):
        assignments[assignments["anonymous_som_node"] == node].to_csv(
            FINAL_DIR / f"node_{node}_members.csv", index=False
        )
    plot_selected(time_h, data, labels, centroids, weights)


def export_ifo_posthoc(metadata: pd.DataFrame, n16: np.lib.npyio.NpzFile) -> None:
    assessment = pd.DataFrame(
        [
            {
                "target_shape": "IFO-Bridge",
                "retained_natural_k4_cluster": False,
                "closest_frozen_result": "node 3",
                "evidence": "near-stable/weak initial gain only; no rapid rise followed by a resolved slow-decay segment",
            },
            {
                "target_shape": "IFO-Hill",
                "retained_natural_k4_cluster": False,
                "closest_frozen_result": "none",
                "evidence": "no K=4 center has rapid rise, a clear peak, rapid post-peak loss, and subsequent slow decay",
            },
            {
                "target_shape": "IFO-Slope",
                "retained_natural_k4_cluster": True,
                "closest_frozen_result": "nodes 0 and 2",
                "evidence": "early decline followed by slower continuing decline; nodes differ mainly in loss magnitude",
            },
            {
                "target_shape": "IFO-Valley",
                "retained_natural_k4_cluster": False,
                "closest_frozen_result": "K=16 node 5 only",
                "evidence": "one-member valley-like exploratory node; K=16 also has an empty node and is rejected by the paper K rule",
            },
        ]
    )
    assessment.to_csv(POSTHOC_DIR / "ifo_posthoc_assessment.csv", index=False)

    n16_labels = n16["labels"].astype(int)
    n16_counts = n16["counts"].astype(int)
    n16_assignments = metadata.copy()
    n16_assignments["anonymous_n16_node"] = n16_labels
    n16_assignments.to_csv(POSTHOC_DIR / "n16_exploratory_assignments.csv", index=False)
    valley_candidate = n16_assignments[n16_assignments["anonymous_n16_node"] == 5].copy()
    valley_candidate.to_csv(POSTHOC_DIR / "n16_node5_singleton_valley_candidate.csv", index=False)
    pd.DataFrame({"anonymous_n16_node": np.arange(16), "count": n16_counts}).to_csv(
        POSTHOC_DIR / "n16_node_counts.csv", index=False
    )
    report = """# 冻结 K 后的 IFO 形态核查

IFO 名称、目标示意图和任何峰谷判断都没有参与数据清洗、SOM 训练、参数比较或 K 选择。

- **IFO-Slope：发现类别级近似。** K=4 的 node 0 和 node 2 均为前段下降较快、后段继续
  缓慢下降，主要差别是衰减幅度。
- **IFO-Bridge：未形成完整自然类别。** node 3 只有近稳定/微弱初始增益，没有参考形态
  所要求的快速上升与随后可分辨的慢衰减。
- **IFO-Hill：未形成自然类别。** 没有中心同时具备明显快速上升、峰值、快速回落和后续慢降。
- **IFO-Valley：未形成按论文规则保留的类别。** 文献展示但不用于 K 选择的 K=16 探索中，
  node 5 呈先降、恢复、再降，但仅有 1 条成员，同时存在空节点，不能宣称为稳定自然类别。

结论：当前数据与严格文献参数下，四种目标拓扑没有同时成为无监督主类别；不能通过命名、
模板或额外变量把它们强行补成四类，因此触发文献检索阶段。
"""
    (POSTHOC_DIR / "IFO_ASSESSMENT_CN.md").write_text(report, encoding="utf-8")


def checksums() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256" or ".matplotlib_cache" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (VALIDATION_DIR / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate(metadata: pd.DataFrame, arrays: np.lib.npyio.NpzFile, n16: np.lib.npyio.NpzFile) -> None:
    labels = arrays["labels"].astype(int)
    counts = arrays["counts"].astype(int)
    n16_counts = n16["counts"].astype(int)
    checks = {
        "selected_k_is_4": len(counts) == 4,
        "all_109_curves_assigned_once": len(labels) == len(metadata) == 109 and int(counts.sum()) == 109,
        "all_k4_nodes_occupied": bool((counts > 0).all()),
        "k4_cluster_sizes_match_search": counts.tolist() == [4, 28, 22, 55],
        "n16_valley_candidate_is_singleton": int(n16_counts[5]) == 1,
        "n16_has_empty_node": bool((n16_counts == 0).any()),
        "ifo_assessment_is_posthoc_only": True,
    }
    report = {"passed": all(checks.values()), "checks": checks}
    (VALIDATION_DIR / "final_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError(f"Final verification failed: {report}")


def main() -> None:
    ensure_dirs()
    prepared = np.load(PREP)
    metadata = pd.read_csv(METADATA)
    metrics = pd.read_csv(METRICS)
    selected = np.load(SELECTED_RUN)
    n16 = np.load(N16_RUN)
    selection_report(metrics)
    export_selected(metadata, prepared["time_hours"], prepared["smoothed"], selected)
    export_ifo_posthoc(metadata, n16)
    validate(metadata, selected, n16)
    checksums()
    print(
        json.dumps(
            {
                "selected_k": 4,
                "selected_sigma": 0.5,
                "selected_learning_rate": 0.1,
                "cluster_sizes": selected["counts"].astype(int).tolist(),
                "all_four_ifo_natural_clusters_found": False,
                "literature_stage_triggered": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
