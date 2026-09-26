#!/usr/bin/env python3
"""Independent, read-only-by-default verification for the synthetic SOM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent
CLASSES = ["IFO-Bridge", "IFO-Hill", "IFO-Slope", "IFO-Valley"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_checks() -> dict:
    truth = pd.read_csv(ROOT / "01_synthetic_data" / "curve_truth_and_generation_parameters.csv")
    qc = pd.read_csv(ROOT / "02_paper_preprocessing" / "preprocessing_qc.csv")
    X = np.load(ROOT / "02_paper_preprocessing" / "preprocessed_matrix_float64.npy")
    assignments = pd.read_csv(ROOT / "03_paper_som_main" / "cluster_assignments.csv")
    node_summary = pd.read_csv(ROOT / "03_paper_som_main" / "node_summary.csv")
    saved_metrics = json.loads((ROOT / "03_paper_som_main" / "main_metrics.json").read_text(encoding="utf-8"))
    saved_cm = pd.read_csv(ROOT / "05_evaluation" / "confusion_matrix.csv", index_col=0).to_numpy()
    seed_metrics = pd.read_csv(ROOT / "04_seed_stability" / "seed_metrics.csv")
    pairwise = pd.read_csv(ROOT / "04_seed_stability" / "pairwise_seed_adjusted_rand_index.csv", index_col=0).to_numpy()

    y_true = assignments["true_class"].to_numpy()
    y_pred = assignments["posthoc_predicted_class"].to_numpy()
    node_ids = assignments["som_node"].to_numpy()
    calc_cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    calc_acc = float(np.mean(y_true == y_pred))
    calc_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="macro"))
    calc_ari = float(adjusted_rand_score(y_true, node_ids))

    checks = {
        "exactly_1000_truth_rows": len(truth) == 1000,
        "four_balanced_truth_classes": truth["true_class"].value_counts().reindex(CLASSES).eq(250).all(),
        "unique_curve_ids": truth["curve_id"].nunique() == 1000,
        "preprocessed_matrix_shape_1000x901": X.shape == (1000, 901),
        "preprocessed_matrix_all_finite": bool(np.isfinite(X).all()),
        "qc_has_1000_rows": len(qc) == 1000,
        "akima_leaves_no_nan": int(qc["remaining_nan_after_akima"].sum()) == 0,
        "maxabs_normalization_correct": bool(np.allclose(qc["normalized_max_abs_before_savgol"], 1.0, atol=1e-12)),
        "assignment_has_1000_rows": len(assignments) == 1000,
        "all_four_nodes_occupied": assignments["som_node"].nunique() == 4,
        "node_counts_sum_to_1000": int(node_summary["count"].sum()) == 1000,
        "posthoc_mapping_is_bijection": set(assignments["posthoc_predicted_class"]) == set(CLASSES),
        "confusion_matrix_recomputes": bool(np.array_equal(calc_cm, saved_cm)),
        "accuracy_recomputes": abs(calc_acc - saved_metrics["accuracy"]) < 1e-12,
        "macro_f1_recomputes": abs(calc_f1 - saved_metrics["macro_f1"]) < 1e-12,
        "ari_recomputes": abs(calc_ari - saved_metrics["adjusted_rand_index_truth_vs_nodes"]) < 1e-12,
        "five_seed_diagnostic_runs": len(seed_metrics) == 5 and seed_metrics["seed"].nunique() == 5,
        "pairwise_seed_matrix_5x5": pairwise.shape == (5, 5),
        "pairwise_seed_ari_diagonal_is_one": bool(np.allclose(np.diag(pairwise), 1.0)),
        "report_exists": (ROOT / "REPORT_CN.md").exists(),
        "main_plot_exists": (ROOT / "03_paper_som_main" / "som_four_nodes.png").exists(),
        "confusion_plot_exists": (ROOT / "05_evaluation" / "confusion_matrix.png").exists(),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {
        "all_passed": all(checks.values()),
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "recomputed_metrics": {
            "accuracy": calc_acc,
            "macro_f1": calc_f1,
            "adjusted_rand_index": calc_ari,
        },
    }


def write_verification_and_checksums(result: dict) -> None:
    validation_dir = ROOT / "06_validation"
    validation_dir.mkdir(exist_ok=True)
    (validation_dir / "independent_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "checksums.sha256"
        and ".matplotlib_cache" not in p.parts
        and "__pycache__" not in p.parts
    )
    with (ROOT / "checksums.sha256").open("w", encoding="utf-8") as f:
        for path in files:
            f.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")


def verify_checksums() -> bool:
    checksum_file = ROOT / "checksums.sha256"
    if not checksum_file.exists():
        return False
    ok = True
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        ok = ok and path.exists() and sha256(path) == expected
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write verification JSON and refresh checksums")
    args = parser.parse_args()
    result = run_checks()
    if args.write:
        write_verification_and_checksums(result)
    checksum_ok = verify_checksums() if (ROOT / "checksums.sha256").exists() else None
    print(json.dumps({**result, "checksums_passed": checksum_ok}, indent=2, ensure_ascii=False))
    if not result["all_passed"] or checksum_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
