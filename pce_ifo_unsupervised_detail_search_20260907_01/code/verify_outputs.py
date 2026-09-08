#!/usr/bin/env python3
"""Independent consistency checks for the expanded curve search."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    checks: dict[str, bool] = {}
    required = [
        "FINAL_RESULT.json", "run_manifest.json", "01_data_audit/input_inventory.csv",
        "02_variants/variant_inventory.csv", "03_kmeans_screen/variant_decisions.csv",
        "04_som_search/setting_decisions.csv", "05_frozen_selection/frozen_selection.json",
        "06_final_model/final_model.json", "06_final_model/assignments.csv",
        "07_posthoc_ifo/cluster_shape_descriptors.csv", "08_stability/bootstrap_stability.csv",
        "reports/final_report_cn.md", "09_figures/01_window_cohort_coverage.png",
        "09_figures/02_kmeans_representation_screen.png", "09_figures/03_som_qe_elbows.png",
        "09_figures/04_bootstrap_stability.png", "06_final_model/clusters/all_clusters.png",
        "07_posthoc_ifo/raw_point_representatives.png",
        "07_posthoc_ifo/individual_ifo_raw_point_gallery.png",
        "07_posthoc_ifo/individual_ifo_candidate_provenance.csv",
        "10_posthoc_grid_audit/audit_summary.json",
        "11_extended_window_cohort_sensitivity/extended_summary.json",
        "11_extended_window_cohort_sensitivity/extended_label_blind_selection.json",
        "11_extended_window_cohort_sensitivity/posthoc_morphology_audit.csv",
    ]
    checks["all_required_outputs_exist"] = all((PROJECT / p).exists() for p in required)
    if not checks["all_required_outputs_exist"]:
        missing = [p for p in required if not (PROJECT / p).exists()]
        raise SystemExit("Missing outputs: " + ", ".join(missing))

    config = json.loads((PROJECT / "config.json").read_text())
    manifest = json.loads((PROJECT / "run_manifest.json").read_text())
    frozen = json.loads((PROJECT / "05_frozen_selection/frozen_selection.json").read_text())
    final = json.loads((PROJECT / "06_final_model/final_model.json").read_text())
    result = json.loads((PROJECT / "FINAL_RESULT.json").read_text())
    assignment = pd.read_csv(PROJECT / "06_final_model/assignments.csv")
    desc = pd.read_csv(PROJECT / "07_posthoc_ifo/cluster_shape_descriptors.csv")
    variants = pd.read_csv(PROJECT / "02_variants/variant_inventory.csv")
    km = pd.read_csv(PROJECT / "03_kmeans_screen/variant_decisions.csv")
    som_runs = pd.read_csv(PROJECT / "04_som_search/all_runs.csv")
    stability = pd.read_csv(PROJECT / "08_stability/bootstrap_stability.csv")
    extended_runs = pd.read_csv(PROJECT / "11_extended_window_cohort_sensitivity/all_runs.csv")
    extended_audit = pd.read_csv(PROJECT / "11_extended_window_cohort_sensitivity/posthoc_morphology_audit.csv")
    extended_summary = json.loads((PROJECT / "11_extended_window_cohort_sensitivity/extended_summary.json").read_text())

    checks["run_complete"] = manifest.get("status") == "complete"
    checks["smoothing_disabled_in_all_manifests"] = not any([
        config.get("smoothing_enabled"), manifest.get("smoothing_enabled"),
        frozen.get("smoothing_enabled"), result.get("smoothing_enabled"),
    ])
    checks["labels_absent_before_freeze"] = (
        config.get("labels_used_for_training_or_selection") is False
        and manifest.get("labels_used_for_training_or_selection") is False
        and frozen.get("selection_used_ifo_names") is False
        and frozen.get("selection_used_target_k4") is False
        and result.get("labels_used_before_freeze") is False
    )
    checks["all_72_variants_created"] = len(variants) == len(config["windows_hours"]) * len(config["cohorts"]) * len(config["representations"])
    checks["one_som_candidate_per_representation"] = (
        km[km["selected_for_som"]]["representation"].nunique() == len(config["representations"])
        and int(km["selected_for_som"].sum()) == config["top_dataset_variants_for_som"]
    )
    expected_som_runs = (
        config["top_dataset_variants_for_som"] * len(config["som_sigmas"]) * len(config["som_learning_rates"])
        * len(config["som_random_order"]) * len(config["candidate_k_som"]) * len(config["screen_seeds"])
    )
    checks["complete_som_grid"] = len(som_runs) == expected_som_runs
    checks["assignment_count_matches_final"] = len(assignment) == final["n"]
    checks["cluster_count_matches_frozen_k"] = assignment["cluster"].nunique() == frozen["selected_k"] == final["selected_k"]
    checks["cluster_sizes_match"] = sorted(assignment["cluster"].value_counts().tolist()) == sorted(final["cluster_sizes"])
    checks["one_descriptor_per_cluster"] = len(desc) == frozen["selected_k"] and desc["cluster"].nunique() == frozen["selected_k"]
    checks["bootstrap_count_matches"] = len(stability) == len(config["bootstrap_seeds"])
    checks["extended_grid_complete"] = len(extended_runs) == 9072 and extended_summary["som_runs"] == 9072
    checks["extended_elbow_audit_complete"] = len(extended_audit) == 432 and extended_summary["elbow_models_audited"] == 432
    checks["extended_exact_four_consistent"] = int(extended_audit["exact_four"].sum()) == extended_summary["exact_four_count"] == 0
    checks["finite_core_metrics"] = np.isfinite(som_runs[["quantization_error", "silhouette", "seed"]].to_numpy()).all()
    checks["frozen_written_before_posthoc"] = (
        (PROJECT / "05_frozen_selection/frozen_selection.json").stat().st_mtime
        <= (PROJECT / "07_posthoc_ifo/cluster_shape_descriptors.csv").stat().st_mtime
    )

    banned_call_fragments = ("savgol", "lowess", "gaussian_filter", "moving_average", "rolling")
    calls = []
    for script in (PROJECT / "code").glob("*.py"):
        tree = ast.parse(script.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls.append(ast.unparse(node.func).lower())
    checks["no_smoothing_function_calls"] = not any(x in call for call in calls for x in banned_call_fragments)

    checks = {key: bool(value) for key, value in checks.items()}
    out = {"all_passed": bool(all(checks.values())), "checks": checks}
    (PROJECT / "verification_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
