#!/usr/bin/env python3
"""Export four real-curve morphology candidates without feeding them back to SOM.

This script is deliberately downstream of the literature-constrained SOM run.  It
does not calculate features, tune parameters, relabel SOM nodes, or change the
selected K.  It only plots four independently audited source CSV files in their
original PCE/efficiency-versus-hour coordinates.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/shunhao/Desktop/ML")
PROJECT = ROOT / "lab-v2/literature_k_som_search"
OUT = PROJECT / "07_posthoc_individual_candidates"

CANDIDATES = [
    {
        "shape": "IFO-Bridge",
        "confidence": "high",
        "record_id": "图片48_8c2ec3e669",
        "path": ROOT
        / "lab-v2/som_references/accepted/samples_test/图片48/accepted"
        / "samples_test__unknown__unknown__图片48__EtCz3EPA_PTAA_BCP__Series_1.csv",
        "y_label": "Efficiency",
        "audit_note": "rapid early increase followed by long slow decay",
    },
    {
        "shape": "IFO-Hill",
        "confidence": "medium; Bridge ambiguity",
        "record_id": "图片105_59069294b1",
        "path": ROOT
        / "lab-v2/som_references/accepted/samples_test/图片105/accepted"
        / "samples_test__unknown__unknown__图片105__Series_2__Series_2.csv",
        "y_label": "Normalized PCE",
        "audit_note": "early rise and post-peak decline, but decline is modest",
    },
    {
        "shape": "IFO-Slope",
        "confidence": "high",
        "record_id": "图片59_f6d9fd27a4",
        "path": ROOT
        / "lab-v2/som_references/accepted/samples_test/图片59/accepted"
        / "samples_test__unknown__unknown__图片59__Control__Series_1.csv",
        "y_label": "Normalized PCE",
        "audit_note": "rapid decay followed by slower long-term decay",
    },
    {
        "shape": "IFO-Valley",
        "confidence": "medium-high",
        "record_id": "图片114_9f1ff62ac8",
        "path": ROOT
        / "lab-v2/som_references/accepted/samples_test/图片114/accepted"
        / "samples_test__unknown__unknown__图片114__Series_2__Series_2.csv",
        "y_label": "Normalized PCE",
        "audit_note": "early decrease, recovery, then slow decay",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curve(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns)[:2] != ["x", "y"]:
        raise ValueError(f"Unexpected schema in {path}: {list(frame.columns)}")
    frame = frame[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.sort_values("x", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No finite points in {path}")
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    colors = ["#4f8c57", "#88a84b", "#e9a83d", "#d8b62c"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    inventory = []

    for ax, item, color in zip(axes.flat, CANDIDATES, colors):
        path = item["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = load_curve(path)
        x = frame["x"].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        xmax = float(np.max(x))
        xmin = float(np.min(x))

        ax.axvspan(max(0.0, xmin), min(200.0, xmax), color="#dcebdc", alpha=0.55)
        if xmax > 200.0:
            ax.axvspan(200.0, xmax, color="#fff0c9", alpha=0.55)
        ax.axvline(200.0, color="#777777", linewidth=0.9, linestyle=":")
        ax.plot(x, y, color=color, linewidth=2.2)
        display_id = item["record_id"].replace("图片", "record ")
        ax.set_title(
            f"{item['shape']} candidate — {display_id}\n"
            f"confidence: {item['confidence']}",
            fontsize=12,
        )
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(item["y_label"])
        ax.grid(alpha=0.22)

        inventory.append(
            {
                "shape_candidate": item["shape"],
                "confidence": item["confidence"],
                "record_id": item["record_id"],
                "source_csv": str(path),
                "source_sha256": sha256(path),
                "n_points": int(len(frame)),
                "x_min_h": xmin,
                "x_max_h": xmax,
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "y_axis": item["y_label"],
                "audit_note": item["audit_note"],
                "role": "posthoc individual candidate only; not a SOM class",
            }
        )

    fig.suptitle(
        "Real PCE/efficiency–hour curves: post-hoc morphology candidates only\n"
        "Not SOM clusters; not used for parameter or K selection",
        fontsize=16,
    )
    figure_path = OUT / "four_real_curve_candidates_not_clusters.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    csv_path = OUT / "candidate_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0].keys()))
        writer.writeheader()
        writer.writerows(inventory)

    manifest = {
        "analysis_role": "independent posthoc visualization only",
        "feedback_into_unsupervised_learning": False,
        "changes_to_som": False,
        "changes_to_features": False,
        "candidate_count": len(inventory),
        "figure": str(figure_path),
        "inventory": str(csv_path),
        "source_files": [row["source_csv"] for row in inventory],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = """# 四条真实曲线的事后形态候选

这里展示的是从原始 `samples_test` CSV 逐条审计后选出的真实 PCE/efficiency–hour 曲线。它们只用于回答“数据中是否存在类似形态的个体曲线”，**不是四个 SOM 类别**，也没有参与参数选择、K 选择、训练或节点命名。

- IFO-Bridge：高可信个体候选。
- IFO-Hill：中等可信，和 Bridge 有一定歧义。
- IFO-Slope：高可信个体候选；同时也是最终 K=4 中唯一有明确类别级近似的目标形态。
- IFO-Valley：中高可信个体候选，但严格聚类中未形成自然类别。

背景色仅标示用户参考图中的 `0–200 h` 与 `>200 h` 时段，不改变原始数据。图上使用原始 CSV 的 x、y；没有额外平滑、特征工程或重归一化。
"""
    (OUT / "README_CN.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
