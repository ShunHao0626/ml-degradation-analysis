#!/usr/bin/env python3
"""Verify the paper-style 200 h tuned rise-then-fall SOM outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
OUTPUT = BUNDLE


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required = [
        "00_fixed_input_and_target/input_manifest.json",
        "00_fixed_input_and_target/rise_then_fall_diagnostics.csv",
        "01_parameter_screen/primary_seed_parameter_screen.csv",
        "02_multiseed_validation/candidate_multiseed_metrics.csv",
        "03_selected_model/selected_model.json",
        "03_selected_model/final_curve_assignments.csv",
        "03_selected_model/final_cluster_summary.csv",
        "03_selected_model/final_som_model.pkl",
        "03_selected_model/rise_then_fall_node_with_raw_points.png",
        "REPORT_CN.md",
        "checksums.sha256",
    ]
    for relative in required:
        require((OUTPUT / relative).is_file(), f"Missing output: {relative}")

    manifest = json.loads(
        (OUTPUT / "00_fixed_input_and_target/input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    require(manifest["n_curves"] == 1442, "Unexpected fixed input curve count")
    require(manifest["n_time_points"] == 1201, "Unexpected time-grid size")
    require(manifest["interpolation"] == "Akima", "Input is not Akima")
    require(manifest["savgol"]["window_length"] == 71, "Wrong smoothing window")
    require(
        manifest["som_tuned"] == ["n_nodes", "sigma", "learning_rate"],
        "Unexpected tuned parameter set",
    )

    diagnostics = pd.read_csv(
        OUTPUT / "00_fixed_input_and_target/rise_then_fall_diagnostics.csv"
    )
    assignments = pd.read_csv(
        OUTPUT / "03_selected_model/final_curve_assignments.csv"
    )
    summary = pd.read_csv(OUTPUT / "03_selected_model/final_cluster_summary.csv")
    selected = json.loads(
        (OUTPUT / "03_selected_model/selected_model.json").read_text(encoding="utf-8")
    )
    require(len(diagnostics) == 1442, "Diagnostic row mismatch")
    require(int(diagnostics["rise_then_fall_core"].sum()) == 56, "Core target drift")
    require(len(assignments) == 1442, "Assignment row mismatch")
    require(assignments["curve_id"].is_unique, "Curve assignments are not unique")
    require(len(summary) == selected["n_nodes"], "Cluster summary node mismatch")
    require(int(summary["n_curves"].sum()) == 1442, "Cluster sizes do not sum")
    require(
        set(assignments["cluster_id"]) == set(summary["cluster_id"]),
        "Assignment and summary nodes differ",
    )
    node = int(selected["rise_then_fall_node"])
    member = assignments["cluster_id"] == node
    target_count = int(assignments.loc[member, "rise_then_fall_core"].sum())
    require(
        target_count == selected["primary_node_metrics"]["target_count"],
        "Selected target count mismatch",
    )
    require(
        int(member.sum()) == selected["primary_node_metrics"]["node_size"],
        "Selected node size mismatch",
    )

    checksum_rows = (OUTPUT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    require(checksum_rows, "Checksum manifest is empty")
    for row in checksum_rows:
        expected, relative = row.split("  ", 1)
        path = OUTPUT / relative
        require(path.is_file(), f"Checksum target missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Checksum mismatch: {relative}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "n_curves": len(assignments),
                "core_rise_then_fall": int(diagnostics.rise_then_fall_core.sum()),
                "selected_n": selected["n_nodes"],
                "selected_sigma": selected["sigma"],
                "selected_learning_rate": selected["learning_rate"],
                "rise_then_fall_node": node,
                "node_size": int(member.sum()),
                "node_core_target_count": target_count,
                "stable_high_purity_core_node": selected[
                    "stable_high_purity_core_node"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the archived result or a newly reproduced output."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLE,
        help="Result directory to verify (default: archived official bundle).",
    )
    args = parser.parse_args()
    OUTPUT = args.output_dir.expanduser().resolve()
    main()
