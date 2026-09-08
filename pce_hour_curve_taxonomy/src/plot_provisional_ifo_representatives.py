#!/usr/bin/env python3
"""Plot four manually audited IFO-like members after unsupervised clustering.

These labels are interpretive annotations only.  They are never passed to SOM
and are not evidence that the four morphologies form four independent clusters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "03_baseline_reproduction/preprocessed/500h/preprocessed_curves.npz"
METADATA = ROOT / "03_baseline_reproduction/preprocessed/500h/curve_metadata.csv"
ASSIGNMENTS = ROOT / "05_cluster_selection/baseline/500h/k05/assignments.csv"
OUTPUT_DIR = ROOT / "06_curve_type_results/provisional_individual_representatives"

REPRESENTATIVES = [
    {
        "morphology": "IFO-Bridge-like",
        "curve_id": "图片105_dccad548c1",
        "reason": "initial rise to a broad maximum followed by gradual decay",
        "color": "#5B8C5A",
    },
    {
        "morphology": "IFO-Hill-like",
        "curve_id": "图片136_d25478ae5a",
        "reason": "pronounced rise to a peak near 100 h followed by sustained decay",
        "color": "#93AA42",
    },
    {
        "morphology": "IFO-Slope-like",
        "curve_id": "图片114_ee5e59416d",
        "reason": "rapid early decay followed by a slower declining tail",
        "color": "#F0B84D",
    },
    {
        "morphology": "IFO-Valley-like",
        "curve_id": "图片71_ce6043959c",
        "reason": "decay to a late minimum followed by partial recovery",
        "color": "#E6C83E",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = np.load(BUNDLE)
    data = bundle["data"]
    time_hours = bundle["time_hours"]
    metadata = read_csv(METADATA)
    assignments = read_csv(ASSIGNMENTS)
    metadata_by_id = {row["curve_id"]: (index, row) for index, row in enumerate(metadata)}
    assignment_by_id = {row["curve_id"]: row for row in assignments}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    provenance: list[dict[str, object]] = []
    for ax, representative in zip(axes.ravel(), REPRESENTATIVES):
        curve_id = representative["curve_id"]
        index, row = metadata_by_id[curve_id]
        assignment = assignment_by_id[curve_id]
        values = data[index]
        ax.plot(time_hours, values, color=representative["color"], linewidth=2.5)
        ax.axvline(200, color="#999999", linewidth=1.0, linestyle="--")
        ax.set_title(
            f"{representative['morphology']} | baseline k=5 cluster {assignment['cluster_display']}",
            fontsize=11,
            weight="bold",
        )
        ax.set_xlabel("Elapsed time (h)")
        ax.set_ylabel("Normalized, smoothed PCE")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.05,
            f"curve_id: {curve_id.replace('图片', 'img')}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#444444",
        )
        provenance.append(
            {
                "morphology": representative["morphology"],
                "curve_id": curve_id,
                "sample": row["sample"],
                "csv_path": row["csv_path"],
                "baseline_k5_cluster": int(assignment["cluster_display"]),
                "selection_stage": "manual_post_clustering_morphology_audit",
                "selection_reason": representative["reason"],
                "model_input_use": "none; label assigned only after clustering",
            }
        )

    fig.suptitle(
        "Provisional IFO-like individual representatives (not four independent SOM clusters)",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Paper-aligned preprocessing; 500 h literature-documented sensitivity window; dashed line = 200 h.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "four_ifo_like_individuals.png", dpi=220)
    plt.close(fig)

    fields = list(provenance[0])
    with (OUTPUT_DIR / "representative_provenance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(provenance)
    with (OUTPUT_DIR / "interpretation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "provisional_individual_matches_not_cluster_labels",
                "training_constraint": "only normalized smoothed PCE trajectory was supplied to SOM",
                "cluster_model": "baseline sigma=0.5, learning_rate=0.1, seed=42, k=5",
                "representatives": provenance,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
