#!/usr/bin/env python3
"""Independently verify final four-shape outputs and refresh checksums."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_pipeline import (
    ACCEPTED_ROOT,
    AUDIT_DIR,
    CLASSES,
    CFG,
    FEATURE_DIR,
    FINAL_DIR,
    ROOT,
    VALIDATION_DIR,
    VENDOR,
    centroid_topology_checks,
    write_manifest,
)


def verify() -> dict:
    audit = pd.read_csv(AUDIT_DIR / "all_218_curves_audit.csv")
    classified = pd.read_csv(FINAL_DIR / "classified_curve_assignments.csv")
    all_status = pd.read_csv(FINAL_DIR / "all_218_curves_final_status.csv")
    summary = pd.read_csv(FINAL_DIR / "final_class_summary.csv")
    strict_matches = pd.read_csv(FINAL_DIR / "strict_four_class_matches.csv")
    unresolved = pd.read_csv(FINAL_DIR / "unresolved_or_other_shapes.csv")
    selected = json.loads((FEATURE_DIR / "selected_shape_classifier.json").read_text(encoding="utf-8"))
    matrices = np.load(FINAL_DIR / "final_selected_window_matrices.npz")
    feature_labels = pd.read_csv(FEATURE_DIR / "final_shape_features_and_labels.csv")
    strict_mask = feature_labels["strict_topology_match"].astype(bool).to_numpy()
    final_stub = {
        "features": feature_labels.loc[strict_mask].reset_index(drop=True),
        "labels": feature_labels.loc[strict_mask, "final_class"].to_numpy(str),
    }
    topology = centroid_topology_checks(final_stub)
    counts = classified["final_class"].value_counts()
    strict_counts = strict_matches["final_class"].value_counts()
    checks = {
        "audit_has_218_curves": len(audit) == CFG.expected_input_curves,
        "audit_has_155_axis_valid_curves": int(audit["axis_valid"].sum()) == CFG.expected_axis_valid_curves,
        "all_status_has_218_rows": len(all_status) == CFG.expected_input_curves,
        "all_numeric_sources_are_under_accepted": audit["source_file"].str.startswith(
            "lab-v2/som_references/accepted/"
        ).all(),
        "selected_window_at_least_500h": int(selected["selected_window_hours"]) >= 500,
        "matrix_row_count_matches_classified": matrices["smoothed"].shape[0] == len(classified),
        "matrix_time_ends_at_selected_window": np.isclose(
            matrices["time_hours"][-1], selected["selected_window_hours"]
        ),
        "matrix_spacing_is_10min": np.allclose(np.diff(matrices["time_hours"]), 1 / 6),
        "all_matrices_are_finite": all(
            np.isfinite(matrices[key]).all() for key in ("raw", "normalized", "smoothed", "shape")
        ),
        "maxabs_normalization_recomputes": np.allclose(
            matrices["normalized"],
            matrices["raw"] / np.max(np.abs(matrices["raw"]), axis=1, keepdims=True),
        ),
        "all_four_classes_present": set(counts.index) == set(CLASSES),
        "minimum_three_curves_per_class": int(counts.min()) >= CFG.minimum_final_class_size,
        "all_four_strict_classes_present": set(strict_counts.index) == set(CLASSES),
        "minimum_three_strict_curves_per_class": int(strict_counts.min())
        >= CFG.minimum_final_class_size,
        "strict_and_unresolved_partition_selected": len(strict_matches) + len(unresolved)
        == len(classified),
        "strict_file_ids_match_assignment_flag": strict_matches["curve_id"].tolist()
        == classified.loc[classified["strict_topology_match"], "curve_id"].tolist(),
        "unresolved_file_ids_match_assignment_flag": unresolved["curve_id"].tolist()
        == classified.loc[~classified["strict_topology_match"], "curve_id"].tolist(),
        "assignment_ids_match_matrix_ids": classified["curve_id"].tolist()
        == matrices["curve_ids"].astype(str).tolist(),
        "summary_counts_match_assignments": summary.set_index("class")["n_curves"].astype(int).to_dict()
        == counts.astype(int).to_dict(),
        "summary_strict_counts_match": summary.set_index("class")[
            "n_strict_topology_matches"
        ].astype(int).to_dict()
        == strict_counts.astype(int).to_dict(),
        "topology_checks_pass": topology["passed"],
        "no_source_outside_original_root": ACCEPTED_ROOT.resolve()
        == (ROOT.parents[0] / "som_references" / "accepted").resolve(),
        "vendored_minisom_exists": (VENDOR / "minisom.py").exists(),
    }
    report = {"passed": all(bool(value) for value in checks.values()), "checks": checks, "topology": topology}
    (VALIDATION_DIR / "independent_verification.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = verify()
    write_manifest()
    print("PASS" if result["passed"] else "FAIL")
    for name, passed in result["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    raise SystemExit(0 if result["passed"] else 1)
