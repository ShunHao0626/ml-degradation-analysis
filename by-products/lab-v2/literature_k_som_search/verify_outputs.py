#!/usr/bin/env python3
"""Read-only consistency checks for the strict literature-constrained SOM run.

The script does not train, tune, relabel, or rewrite any model artifact.  Its only
outputs are 05_validation/verification.json and 05_validation/checksums.sha256.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
VALIDATION_DIR = ROOT / "05_validation"
VERIFICATION_PATH = VALIDATION_DIR / "verification.json"
CHECKSUM_PATH = VALIDATION_DIR / "checksums.sha256"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def shape_list(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def main() -> int:
    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        passed: bool,
        expected: Any,
        actual: Any,
        evidence: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "expected": expected,
                "actual": actual,
                "evidence": evidence,
            }
        )

    config_path = ROOT / "run_config.json"
    config = read_json(config_path)

    audit_path = ROOT / "01_data_audit" / "all_218_curves_audit.csv"
    included_path = ROOT / "01_data_audit" / "included_strict_150h.csv"
    audit_rows = read_csv(audit_path)
    included_rows = read_csv(included_path)
    audit_included = [
        row
        for row in audit_rows
        if row["included_strict_150h"].strip().lower() == "true"
    ]
    audit_ids = [row["curve_id"] for row in audit_included]
    included_ids = [row["curve_id"] for row in included_rows]

    check(
        "audit_has_218_rows",
        len(audit_rows) == 218,
        218,
        len(audit_rows),
        str(audit_path.relative_to(ROOT)),
    )
    check(
        "strict_inclusion_has_109_rows",
        len(audit_included) == 109 and len(included_rows) == 109,
        {"audit_true_rows": 109, "included_csv_rows": 109},
        {
            "audit_true_rows": len(audit_included),
            "included_csv_rows": len(included_rows),
        },
        f"{audit_path.relative_to(ROOT)}; {included_path.relative_to(ROOT)}",
    )
    check(
        "strict_inclusion_ids_match_audit",
        audit_ids == included_ids and len(set(included_ids)) == 109,
        "same ordered 109 unique curve_ids",
        {
            "ordered_lists_equal": audit_ids == included_ids,
            "unique_curve_ids": len(set(included_ids)),
        },
        str(included_path.relative_to(ROOT)),
    )
    check(
        "config_strict_input_count_is_109",
        config.get("strict_input_count") == 109,
        109,
        config.get("strict_input_count"),
        str(config_path.relative_to(ROOT)),
    )

    npz_path = (
        ROOT / "02_preprocessed_150h" / "strict_paper_preprocessed_150h.npz"
    )
    metadata_path = ROOT / "02_preprocessed_150h" / "curve_metadata.csv"
    metadata_ids = [row["curve_id"] for row in read_csv(metadata_path)]
    with np.load(npz_path, allow_pickle=False) as arrays:
        required_keys = {"time_hours", "raw", "normalized", "smoothed", "curve_ids"}
        keys = set(arrays.files)
        shapes = {key: shape_list(arrays[key]) for key in sorted(keys)}
        npz_ids = arrays["curve_ids"].astype(str).tolist()
        finite_smoothed = bool(np.isfinite(arrays["smoothed"]).all())
        time_start = float(arrays["time_hours"][0])
        time_end = float(arrays["time_hours"][-1])
    expected_shapes = {
        "time_hours": [900],
        "raw": [109, 900],
        "normalized": [109, 900],
        "smoothed": [109, 900],
        "curve_ids": [109],
    }
    check(
        "preprocessed_array_is_109_by_900",
        keys == required_keys and shapes == expected_shapes and finite_smoothed,
        {"keys": sorted(required_keys), "shapes": expected_shapes, "finite": True},
        {"keys": sorted(keys), "shapes": shapes, "finite": finite_smoothed},
        str(npz_path.relative_to(ROOT)),
    )
    check(
        "preprocessed_ids_match_strict_inclusion",
        set(npz_ids) == set(included_ids)
        and len(npz_ids) == len(set(npz_ids)) == 109
        and npz_ids == metadata_ids,
        "same 109-curve set as audit; NPZ order matches preprocessing metadata",
        {
            "same_curve_id_set_as_audit": set(npz_ids) == set(included_ids),
            "ordered_like_audit_csv": npz_ids == included_ids,
            "ordered_like_curve_metadata": npz_ids == metadata_ids,
            "unique_npz_curve_ids": len(set(npz_ids)),
        },
        f"{npz_path.relative_to(ROOT)}; {metadata_path.relative_to(ROOT)}",
    )
    check(
        "preprocessed_grid_matches_fixed_150h_method",
        config.get("window_hours") == 150
        and config.get("interval_minutes") == 10
        and len(npz_ids) == 109
        and abs(time_start - (1 / 6)) < 1e-12
        and abs(time_end - 150.0) < 1e-10,
        {
            "window_hours": 150,
            "interval_minutes": 10,
            "points": 900,
            "time_start_hours": 1 / 6,
            "time_end_hours": 150.0,
        },
        {
            "window_hours": config.get("window_hours"),
            "interval_minutes": config.get("interval_minutes"),
            "points": 900,
            "time_start_hours": time_start,
            "time_end_hours": time_end,
        },
        f"{config_path.relative_to(ROOT)}; {npz_path.relative_to(ROOT)}",
    )

    expected_pairs = {(1, 0.5, 0.1), (2, 0.3, 0.1), (3, 0.5, 0.3)}
    configured_pairs = {
        (index, float(pair[0]), float(pair[1]))
        for index, pair in enumerate(config.get("parameter_pairs", []), start=1)
    }
    expected_ks = set(range(2, 11))
    configured_ks = set(config.get("n_values", []))
    check(
        "only_three_literature_parameter_pairs_configured",
        configured_pairs == expected_pairs,
        sorted(expected_pairs),
        sorted(configured_pairs),
        str(config_path.relative_to(ROOT)),
    )
    check(
        "configured_k_search_is_exactly_2_through_10",
        configured_ks == expected_ks and len(config.get("n_values", [])) == 9,
        list(range(2, 11)),
        config.get("n_values"),
        str(config_path.relative_to(ROOT)),
    )

    metrics_csv_path = ROOT / "03_k_search_n2_10" / "all_27_run_metrics.csv"
    metric_rows = read_csv(metrics_csv_path)
    metric_keys: list[tuple[int, float, float, int]] = []
    all_metric_invariants = True
    for row in metric_rows:
        key = (
            int(row["parameter_pair_id"]),
            float(row["sigma"]),
            float(row["learning_rate"]),
            int(row["n_clusters"]),
        )
        metric_keys.append(key)
        all_metric_invariants = all_metric_invariants and (
            int(row["iterations"]) == 50000
            and int(row["comparison_seed"]) == 0
            and row["selection_role"] == "paper_K_search_n2_10"
            and int(row["n_input_curves"]) == 109
            and int(row["n_dimensions"]) == 900
        )
    observed_pairs = {(pair_id, sigma, lr) for pair_id, sigma, lr, _ in metric_keys}
    observed_by_pair = {
        pair: sorted(
            k
            for pair_id, sigma, lr, k in metric_keys
            if (pair_id, sigma, lr) == pair
        )
        for pair in sorted(observed_pairs)
    }
    complete_grid = {
        pair: list(range(2, 11)) for pair in sorted(expected_pairs)
    }
    observed_by_pair_json = [
        {"pair": list(pair), "k_values": values}
        for pair, values in sorted(observed_by_pair.items())
    ]
    complete_grid_json = [
        {"pair": list(pair), "k_values": values}
        for pair, values in sorted(complete_grid.items())
    ]
    check(
        "k_search_contains_exactly_27_unique_candidates",
        len(metric_rows) == 27
        and len(set(metric_keys)) == 27
        and observed_pairs == expected_pairs
        and observed_by_pair == complete_grid
        and all_metric_invariants,
        {
            "candidate_count": 27,
            "pairs": sorted(expected_pairs),
            "k_per_pair": complete_grid_json,
            "iterations": 50000,
            "seed": 0,
            "input_shape": [109, 900],
        },
        {
            "candidate_count": len(metric_rows),
            "unique_candidate_keys": len(set(metric_keys)),
            "pairs": sorted(observed_pairs),
            "k_per_pair": observed_by_pair_json,
            "shared_invariants_pass": all_metric_invariants,
        },
        str(metrics_csv_path.relative_to(ROOT)),
    )

    run_metric_paths = sorted(
        (ROOT / "03_k_search_n2_10").glob("pair_*/n_*/metrics.json")
    )
    run_artifacts_complete = all(
        all(
            (path.parent / name).is_file()
            for name in (
                "metrics.json",
                "cluster_assignments.csv",
                "node_counts.csv",
                "som_labels.npy",
                "som_weights.npy",
                "member_mean_centroids.npy",
                "som_model.pkl",
                "anonymous_node_curves.csv",
                "anonymous_nodes_all_members_and_centers.png",
            )
        )
        for path in run_metric_paths
    )
    check(
        "all_27_candidate_artifact_sets_exist",
        len(run_metric_paths) == 27 and run_artifacts_complete,
        {"metric_files": 27, "all_required_artifacts_present": True},
        {
            "metric_files": len(run_metric_paths),
            "all_required_artifacts_present": run_artifacts_complete,
        },
        "03_k_search_n2_10/pair_*/n_*/",
    )

    exact_model_input = [
        "normalized PCE/efficiency values on the 10-minute time grid"
    ]
    prohibited_input_terms = {
        "ifo",
        "template",
        "derivative",
        "peak",
        "valley",
        "material",
        "device",
        "classifier",
    }
    configured_inputs = config.get("model_input_variables", [])
    joined_inputs = " ".join(str(item).lower() for item in configured_inputs)
    terms_found = sorted(term for term in prohibited_input_terms if term in joined_inputs)
    check(
        "model_input_contains_no_added_shape_or_device_variables",
        configured_inputs == exact_model_input and not terms_found,
        exact_model_input,
        {"model_input_variables": configured_inputs, "prohibited_terms_found": terms_found},
        str(config_path.relative_to(ROOT)),
    )

    selected_dir = ROOT / "06_selected_paper_rule_result_k4"
    selected_metrics_path = selected_dir / "metrics.json"
    selected_summary_path = selected_dir / "selection_summary.json"
    selected_metrics = read_json(selected_metrics_path)
    selected_summary = read_json(selected_summary_path)
    selected_weights = np.load(selected_dir / "som_weights.npy", allow_pickle=False)
    selected_labels = np.load(selected_dir / "som_labels.npy", allow_pickle=False)
    selected_centroids = np.load(
        selected_dir / "member_mean_centroids.npy", allow_pickle=False
    )
    selected_assignments = read_csv(selected_dir / "cluster_assignments.csv")
    selected_counts_rows = read_csv(selected_dir / "node_counts.csv")
    selected_counts = [int(row["count"]) for row in selected_counts_rows]
    label_counts = [
        Counter(int(value) for value in selected_labels.tolist()).get(node, 0)
        for node in range(4)
    ]
    selected_pass = (
        selected_summary.get("selected_k") == 4
        and selected_metrics.get("parameter_pair_id") == 1
        and selected_metrics.get("sigma") == 0.5
        and selected_metrics.get("learning_rate") == 0.1
        and selected_metrics.get("n_clusters") == 4
        and selected_metrics.get("layout") == [2, 2]
        and selected_metrics.get("iterations") == 50000
        and selected_metrics.get("comparison_seed") is None
        and selected_metrics.get("selection_role")
        == "final_author_style_seed_none_after_paper_rule_K_selection"
        and selected_metrics.get("n_input_curves") == 109
        and selected_metrics.get("n_dimensions") == 900
        and selected_metrics.get("occupied_nodes") == 4
        and selected_metrics.get("cluster_sizes") == selected_counts
        and selected_counts == label_counts
        and sum(selected_counts) == 109
        and len(selected_assignments) == 109
        and shape_list(selected_weights) == [2, 2, 900]
        and shape_list(selected_labels) == [109]
        and shape_list(selected_centroids) == [4, 900]
    )
    check(
        "final_saved_model_is_selected_k4_author_style_run",
        selected_pass,
        {
            "selected_k": 4,
            "pair": [0.5, 0.1],
            "layout": [2, 2],
            "seed": None,
            "input_shape": [109, 900],
            "occupied_nodes": 4,
        },
        {
            "selected_k": selected_summary.get("selected_k"),
            "pair": [
                selected_metrics.get("sigma"),
                selected_metrics.get("learning_rate"),
            ],
            "layout": selected_metrics.get("layout"),
            "seed": selected_metrics.get("comparison_seed"),
            "input_shape": [
                selected_metrics.get("n_input_curves"),
                selected_metrics.get("n_dimensions"),
            ],
            "occupied_nodes": selected_metrics.get("occupied_nodes"),
            "cluster_sizes_metrics": selected_metrics.get("cluster_sizes"),
            "cluster_sizes_csv": selected_counts,
            "cluster_sizes_labels": label_counts,
            "weights_shape": shape_list(selected_weights),
            "labels_shape": shape_list(selected_labels),
            "centroids_shape": shape_list(selected_centroids),
            "quantization_error": selected_metrics.get("quantization_error"),
        },
        f"{selected_summary_path.relative_to(ROOT)}; {selected_metrics_path.relative_to(ROOT)}",
    )

    exploratory_dir = ROOT / "04_n16_exploratory_not_selected"
    exploratory_metrics_path = exploratory_dir / "metrics.json"
    exploratory_metrics = read_json(exploratory_metrics_path)
    exploratory_weights = np.load(
        exploratory_dir / "som_weights.npy", allow_pickle=False
    )
    exploratory_labels = np.load(
        exploratory_dir / "som_labels.npy", allow_pickle=False
    )
    exploratory_counts_rows = read_csv(exploratory_dir / "node_counts.csv")
    exploratory_counts = [int(row["count"]) for row in exploratory_counts_rows]
    exploratory_pass = (
        exploratory_metrics.get("n_clusters") == 16
        and exploratory_metrics.get("layout") == [4, 4]
        and exploratory_metrics.get("parameter_pair_id") == 1
        and exploratory_metrics.get("sigma") == 0.5
        and exploratory_metrics.get("learning_rate") == 0.1
        and exploratory_metrics.get("comparison_seed") == 0
        and exploratory_metrics.get("selection_role")
        == "exploratory_only_not_part_of_K_selection"
        and exploratory_metrics.get("n_input_curves") == 109
        and exploratory_metrics.get("n_dimensions") == 900
        and exploratory_metrics.get("cluster_sizes") == exploratory_counts
        and sum(exploratory_counts) == 109
        and shape_list(exploratory_weights) == [4, 4, 900]
        and shape_list(exploratory_labels) == [109]
        and config.get("n16_status")
        == "paper-reported exploratory resolution; excluded from K elbow and final K selection"
        and 16 not in configured_ks
    )
    check(
        "n16_is_saved_as_exploratory_only",
        exploratory_pass,
        {
            "n_clusters": 16,
            "layout": [4, 4],
            "selection_role": "exploratory_only_not_part_of_K_selection",
            "excluded_from_configured_k_search": True,
        },
        {
            "n_clusters": exploratory_metrics.get("n_clusters"),
            "layout": exploratory_metrics.get("layout"),
            "selection_role": exploratory_metrics.get("selection_role"),
            "excluded_from_configured_k_search": 16 not in configured_ks,
            "occupied_nodes": exploratory_metrics.get("occupied_nodes"),
            "cluster_sizes": exploratory_counts,
            "weights_shape": shape_list(exploratory_weights),
            "labels_shape": shape_list(exploratory_labels),
        },
        f"{exploratory_metrics_path.relative_to(ROOT)}; {config_path.relative_to(ROOT)}",
    )

    decision_path = ROOT / "K_SELECTION_DECISION_CN.md"
    decision_text = decision_path.read_text(encoding="utf-8")
    check(
        "written_decision_records_k4_before_ifo_review",
        "## 决定：K = 4" in decision_text
        and "该决定严格先于 IFO 标签核查" in decision_text
        and "`n=16`" in decision_text
        and "不是最优 K" in decision_text,
        "K=4 selected before IFO review; n=16 explicitly not optimal K",
        {
            "k4_heading_present": "## 决定：K = 4" in decision_text,
            "pre_ifo_statement_present": "该决定严格先于 IFO 标签核查" in decision_text,
            "n16_not_optimal_statement_present": "不是最优 K" in decision_text,
        },
        str(decision_path.relative_to(ROOT)),
    )

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    report = {
        "overall_status": "PASS" if not failed else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Read-only validation of already-generated artifacts. No SOM training, "
            "parameter search, target-driven relabeling, or model mutation was performed."
        ),
        "summary": {
            "checks_total": len(checks),
            "checks_passed": len(checks) - len(failed),
            "checks_failed": len(failed),
            "failed_check_names": failed,
            "final_k": selected_summary.get("selected_k"),
            "final_quantization_error": selected_metrics.get("quantization_error"),
            "final_cluster_sizes": selected_metrics.get("cluster_sizes"),
        },
        "checks": checks,
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with VERIFICATION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    checksum_lines: list[str] = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        if path == CHECKSUM_PATH:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum_lines.append(f"{digest.hexdigest()}  {path.relative_to(ROOT).as_posix()}")
    CHECKSUM_PATH.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"verification: {VERIFICATION_PATH}")
    print(f"checksums: {CHECKSUM_PATH} ({len(checksum_lines)} files; excludes itself)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
