#!/usr/bin/env python3
"""Independent verification for the pure-unsupervised paper-parameter run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent
WINDOWS = [150, 200, 300, 500, 750, 1000]
SIGMAS = {0.3, 0.4, 0.5}
LEARNING_RATES = {0.1, 0.2, 0.3}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_checks() -> dict:
    config = json.loads((ROOT / "run_config.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(ROOT / "01_data_audit" / "all_input_curves_audit.csv")
    eligible = pd.read_csv(ROOT / "01_data_audit" / "eligible_curve_count_by_window.csv")
    candidates = pd.read_csv(
        ROOT / "03_unsupervised_parameter_search" / "candidates_with_stability_and_final_rank.csv"
    )
    stability_runs = pd.read_csv(ROOT / "04_seed_stability" / "all_seed_run_metrics.csv")
    assignments = pd.read_csv(ROOT / "05_selected_unsupervised_result" / "cluster_assignments.csv")
    metrics = json.loads(
        (ROOT / "05_selected_unsupervised_result" / "selected_run_metrics.json").read_text(encoding="utf-8")
    )
    selected = candidates[candidates["final_unsupervised_rank"] == 1].iloc[0]
    npz = np.load(
        ROOT
        / "02_paper_preprocessing"
        / f"window_{int(selected.window_hours):04d}h"
        / "paper_preprocessed_matrices.npz"
    )
    X = npz["smoothed"]
    labels = assignments["som_node"].to_numpy()

    expected_counts = [130, 124, 116, 104, 87, 55]
    forbidden_columns = {"true_class", "target_class", "bridge", "hill", "slope", "valley"}
    recomputed_silhouette = float(
        silhouette_score(cdist(X, X, metric="euclidean"), labels, metric="precomputed")
    )
    source_text = (ROOT / "run_pipeline.py").read_text(encoding="utf-8").lower()
    checks = {
        "exactly_218_input_csv_audited": len(audit) == 218,
        "exactly_155_axis_valid_curves": int(audit["axis_valid"].sum()) == 155,
        "eligible_counts_match": eligible["eligible_curves"].tolist() == expected_counts,
        "all_six_windows_present": eligible["window_hours"].tolist() == WINDOWS,
        "exactly_54_parameter_candidates": len(candidates) == 54,
        "candidate_sigma_values_limited": set(np.round(candidates["sigma"], 10)) == SIGMAS,
        "candidate_learning_rate_values_limited": set(np.round(candidates["learning_rate"], 10)) == LEARNING_RATES,
        "candidate_windows_limited": set(candidates["window_hours"]) == set(WINDOWS),
        "all_iterations_are_50000": candidates["iterations"].eq(50000).all(),
        "all_maps_are_2x2": candidates["som_x"].eq(2).all() and candidates["som_y"].eq(2).all(),
        "nine_candidates_stability_tested": candidates["stability_evaluated"].sum() == 9,
        "exactly_45_stability_runs": len(stability_runs) == 45,
        "five_fixed_stability_seeds": set(stability_runs["seed"]) == {0, 1, 2, 3, 4},
        "one_automatic_selected_candidate": candidates["final_unsupervised_rank"].eq(1).sum() == 1,
        "selected_assignments_match_eligible_count": len(assignments)
        == int(eligible.loc[eligible["window_hours"] == int(selected.window_hours), "eligible_curves"].iloc[0]),
        "selected_four_nodes_occupied": assignments["som_node"].nunique() == 4,
        "no_training_label_columns": forbidden_columns.isdisjoint({c.lower() for c in assignments.columns}),
        "selected_matrix_full_10min_dimensions": X.shape[1] == int(selected.window_hours) * 6,
        "selected_matrix_all_finite": bool(np.isfinite(X).all()),
        "silhouette_recomputes": abs(recomputed_silhouette - metrics["silhouette"]) < 1e-12,
        "final_random_seed_is_none": metrics["random_seed"] is None,
        "selection_declares_no_target_labels": metrics["selection_used_target_labels"] is False,
        "no_target_archetype_names_in_pipeline": not any(
            word in source_text for word in ["ifo-bridge", "ifo-hill", "ifo-slope", "ifo-valley"]
        ),
        "selected_plot_exists": (
            ROOT / "05_selected_unsupervised_result" / "selected_four_anonymous_som_nodes.png"
        ).exists(),
        "report_exists": (ROOT / "REPORT_CN.md").exists(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "all_passed": all(checks.values()),
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "recomputed": {
            "selected_window_hours": int(selected.window_hours),
            "selected_sigma": float(selected.sigma),
            "selected_learning_rate": float(selected.learning_rate),
            "selected_silhouette": recomputed_silhouette,
            "selected_n_curves": len(assignments),
        },
    }


def write_outputs(result: dict) -> None:
    out = ROOT / "06_validation"
    out.mkdir(exist_ok=True)
    (out / "independent_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    files = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "checksums.sha256"
        and ".matplotlib_cache" not in p.parts
        and "__pycache__" not in p.parts
    )
    with (ROOT / "checksums.sha256").open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(ROOT)}\n")


def verify_checksums() -> bool:
    path = ROOT / "checksums.sha256"
    if not path.exists():
        return False
    ok = True
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        item = ROOT / rel
        ok = ok and item.exists() and sha256(item) == expected
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run_checks()
    if args.write:
        write_outputs(result)
    checksum_ok = verify_checksums() if (ROOT / "checksums.sha256").exists() else None
    print(json.dumps({**result, "checksums_passed": checksum_ok}, indent=2, ensure_ascii=False))
    if not result["all_passed"] or checksum_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
