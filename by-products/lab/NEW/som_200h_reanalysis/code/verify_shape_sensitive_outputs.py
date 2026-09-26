#!/usr/bin/env python3
"""Verify the independent shape-sensitive SOM reanalysis outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "10_shape_sensitive_reanalysis"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    qc = pd.read_csv(ROOT / "01_data_qc" / "strict_selection_audit.csv")
    strict = qc[qc["strict_selected"]]
    require(len(qc) == 1442, "QC input count mismatch")
    require(len(strict) > 1400, "Strict dataset unexpectedly small")
    require(
        (strict["max_support_gap_h"] <= 40.0 + 1e-9).all(),
        "Strict dataset contains a support gap >40 h",
    )

    pre = ROOT / "02_preprocessing_comparison"
    matrices = {
        name: np.load(pre / f"X_{name}.npy")
        for name in (
            "akima_normalized",
            "akima_smoothed",
            "pchip_normalized",
            "linear_normalized",
        )
    }
    expected_shape = (len(strict), 1201)
    for name, matrix in matrices.items():
        require(matrix.shape == expected_shape, f"{name} shape mismatch")
        require(np.isfinite(matrix).all(), f"{name} contains nonfinite values")
    require(
        np.max(np.abs(matrices["pchip_normalized"] - matrices["linear_normalized"]))
        < 0.25,
        "PCHIP/linear discrepancy unexpectedly large",
    )

    descriptors = pd.read_csv(ROOT / "03_shape_features" / "shape_descriptors.csv")
    require(len(descriptors) == len(strict), "Descriptor row mismatch")
    require(descriptors["curve_id"].is_unique, "Descriptor curve IDs are not unique")
    require(
        set(descriptors["rule_shape_category"])
        <= {
            "initial_gain",
            "dip_recovery",
            "early_drop",
            "stable",
            "moderate_decay",
            "rapid_decay",
        },
        "Unexpected rule shape category",
    )

    metrics = pd.read_csv(
        ROOT / "04_som_n4_to_n10" / "all_feature_set_som_metrics.csv"
    )
    require(len(metrics) == 4 * 7, "SOM metric row count mismatch")
    require(metrics["feature_set"].nunique() == 4, "Feature-set count mismatch")
    require(set(metrics["n_nodes"]) == set(range(4, 11)), "SOM n range mismatch")

    selected = json.loads(
        (ROOT / "05_selected_model" / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        selected["feature_set"] == "curve_derivative_descriptors",
        "Selected feature set must be the prespecified composite representation",
    )
    require(4 <= int(selected["n_nodes"]) <= 10, "Selected n outside scan")
    assignments = pd.read_csv(
        ROOT / "05_selected_model" / "final_curve_assignments.csv"
    )
    require(len(assignments) == len(strict), "Final assignment row mismatch")
    require(assignments["curve_id"].is_unique, "Final curve IDs are not unique")
    require(
        assignments["cluster_id"].nunique() == int(selected["n_nodes"]),
        "Selected model has empty clusters",
    )
    require(
        assignments["cluster_id"].value_counts().min() / len(assignments) >= 0.01,
        "Selected model contains a <1% cluster",
    )

    checksum_path = ROOT / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        require(digest == expected, f"Checksum mismatch: {relative}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "strict_curves": len(strict),
                "selected_feature_set": selected["feature_set"],
                "selected_n": int(selected["n_nodes"]),
                "cluster_sizes": assignments["cluster_id"].value_counts().sort_index().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
