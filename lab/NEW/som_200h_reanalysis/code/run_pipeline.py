#!/usr/bin/env python3
"""Run the complete, independent 200 h SOM reanalysis."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.data import (
    audit_curves,
    discover_curves,
    export_selected_database,
    save_selection_outputs,
)
from som200h.modeling import (
    choose_cluster_count,
    compare_som_with_kmeans,
    run_kmeans_validation,
    run_som_scan,
    save_final_traceability,
)
from som200h.plotting import (
    plot_all_preprocessed_curves,
    plot_cluster_number_metrics,
    plot_cluster_sizes,
    plot_final_centroids,
    plot_preprocessing_overview,
    plot_selected_raw_curves,
    plot_selected_raw_curves_by_scale,
    plot_selection_flow,
    plot_som_clusters,
)
from som200h.preprocessing import (
    build_preprocessed_dataset,
    save_preprocessed_dataset,
)
from som200h.reporting import (
    write_checksums,
    write_environment,
    write_file_index,
    write_main_report,
    write_si_method_report,
)


def configure_logging() -> logging.Logger:
    log_path = PROJECT / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("som200h")


def main() -> None:
    logger = configure_logging()
    config = AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )
    (PROJECT / "run_config.json").write_text(
        json.dumps(config.to_jsonable(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_environment(config)

    reference_dir = PROJECT / "00_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.work_md, reference_dir / "work.md")
    shutil.copy2(config.supplementary_md, reference_dir / "Supplementary.md")

    logger.info("Discovering curves under %s", config.dataset_root)
    curves = discover_curves(config)
    curves_by_id = {curve.curve_id: curve for curve in curves}
    if len(curves_by_id) != len(curves):
        raise RuntimeError("curve_id values are not unique")
    logger.info("Discovered %d curves", len(curves))

    audit = audit_curves(curves, config)
    save_selection_outputs(audit, config)
    selection_flow = pd.read_csv(
        PROJECT / "01_data_selection" / "selection_flow.csv"
    )
    plot_selection_flow(
        selection_flow,
        PROJECT / "01_data_selection" / "selection_flow.png",
    )
    for selector, label, filename in (
        (
            "main_selected_min10",
            "Main selected raw curves",
            "main_selected_raw_curves_0_200h.png",
        ),
        (
            "paper_selected_min4",
            "Paper-sensitivity selected raw curves",
            "paper_sensitivity_selected_raw_curves_0_200h.png",
        ),
    ):
        plot_selected_raw_curves(
            curves_by_id,
            audit[audit[selector]],
            PROJECT / "01_data_selection" / filename,
            label,
            config.window_hours,
        )
        plot_selected_raw_curves_by_scale(
            curves_by_id,
            audit[audit[selector]],
            PROJECT
            / "01_data_selection"
            / filename.replace(".png", "_by_scale.png"),
            label,
            config.window_hours,
        )
    expected = {
        "all_discovered": 2151,
        "duration_ge_200h": 1862,
        "paper_selected_min4": 1785,
        "main_selected_min10": 1442,
    }
    observed = dict(
        zip(selection_flow["selection_stage"], selection_flow["n_curves"])
    )
    for key, value in expected.items():
        if int(observed[key]) != value:
            raise RuntimeError(
                f"Selection count changed for {key}: expected {value}, got {observed[key]}"
            )

    logger.info("Exporting selected raw-curve databases")
    export_selected_database(
        audit,
        config,
        "main_selected_min10",
        "main_high_quality_min10",
    )
    export_selected_database(
        audit,
        config,
        "paper_selected_min4",
        "paper_aligned_min4",
    )

    datasets = {}
    failures = {}
    for name, selector in (
        ("main_high_quality_min10", "main_selected_min10"),
        ("paper_aligned_min4", "paper_selected_min4"),
    ):
        logger.info("Preprocessing %s", name)
        selected = audit[audit[selector]].copy()
        dataset, failed = build_preprocessed_dataset(
            name, curves_by_id, selected, config
        )
        if not failed.empty:
            raise RuntimeError(
                f"{name}: {len(failed)} selected curves failed preprocessing"
            )
        save_preprocessed_dataset(dataset, failed, config)
        plot_preprocessing_overview(
            dataset,
            PROJECT / "03_preprocessed" / name / "preprocessing_overview.png",
        )
        plot_all_preprocessed_curves(
            dataset.time_grid,
            dataset.x_smoothed,
            PROJECT / "03_preprocessed" / f"{name}_all_preprocessed_curves.png",
            name,
        )
        datasets[name] = dataset
        failures[name] = failed
        logger.info("%s matrix shape=%s", name, dataset.x_smoothed.shape)

    scan_metrics = {}
    runs = {}
    kmeans_metrics = {}
    kmeans_labels = {}
    decisions = {}
    selected_ns = {}
    for name, dataset in datasets.items():
        logger.info("Running n=2..10 multi-seed SOM scan for %s", name)
        metrics, primary_runs = run_som_scan(dataset, config)
        km_metrics, km_labels = run_kmeans_validation(dataset, config)
        selected_n, decision = choose_cluster_count(metrics, config)
        decision_dir = PROJECT / "05_cluster_number_decision" / name
        decision_dir.mkdir(parents=True, exist_ok=True)
        decision.to_csv(
            decision_dir / "cluster_number_decision.csv", index=False
        )
        plot_cluster_number_metrics(
            metrics,
            km_metrics,
            selected_n,
            decision_dir / "cluster_number_decision.png",
        )
        for n, run in primary_runs.items():
            n_dir = PROJECT / "04_som_results" / name / f"n_{n:02d}"
            plot_som_clusters(
                dataset,
                run,
                n_dir / "cluster_curves.png",
                config.plot_max_curves_per_cluster,
            )
        scan_metrics[name] = metrics
        runs[name] = primary_runs
        kmeans_metrics[name] = km_metrics
        kmeans_labels[name] = km_labels
        decisions[name] = decision
        selected_ns[name] = selected_n
        logger.info("%s selected n=%d", name, selected_n)

    main_name = "main_high_quality_min10"
    paper_name = "paper_aligned_min4"
    main_n = selected_ns[main_name]
    paper_n = selected_ns[paper_name]
    main_run = runs[main_name][main_n]
    final_assignments, final_summary = save_final_traceability(
        datasets[main_name],
        main_n,
        main_run,
        kmeans_labels[main_name][main_n],
        config,
    )
    kmeans_comparison = compare_som_with_kmeans(
        main_run.labels, kmeans_labels[main_name][main_n]
    )
    (PROJECT / "06_final_model" / "kmeans_comparison.json").write_text(
        json.dumps(kmeans_comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_cluster_sizes(
        final_summary,
        PROJECT / "06_final_model" / "final_cluster_sizes.png",
    )
    plot_final_centroids(
        datasets[main_name],
        final_assignments,
        final_summary,
        PROJECT / "06_final_model" / "final_cluster_centroids.png",
    )
    plot_som_clusters(
        datasets[main_name],
        main_run,
        PROJECT / "06_final_model" / "final_cluster_curves.png",
        config.plot_max_curves_per_cluster,
    )

    # Cross-dataset comparison on common curves for the selected n values.
    sensitivity_dir = PROJECT / "07_sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    main_map = dict(
        zip(datasets[main_name].metadata["curve_id"], main_run.labels)
    )
    paper_run = runs[paper_name][paper_n]
    paper_map = dict(
        zip(datasets[paper_name].metadata["curve_id"], paper_run.labels)
    )
    common = sorted(set(main_map) & set(paper_map))
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    cross = {
        "main_dataset": main_name,
        "paper_dataset": paper_name,
        "main_selected_n": main_n,
        "paper_selected_n": paper_n,
        "n_common_curves": len(common),
        "ARI_on_common_curves": float(
            adjusted_rand_score(
                [main_map[curve_id] for curve_id in common],
                [paper_map[curve_id] for curve_id in common],
            )
        ),
        "NMI_on_common_curves": float(
            normalized_mutual_info_score(
                [main_map[curve_id] for curve_id in common],
                [paper_map[curve_id] for curve_id in common],
            )
        ),
    }
    (sensitivity_dir / "main_vs_paper_dataset.json").write_text(
        json.dumps(cross, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_si_method_report(
        config,
        scan_metrics[main_name],
        scan_metrics[paper_name],
        decisions[main_name],
        decisions[paper_name],
        main_n,
        paper_n,
    )
    write_main_report(
        config,
        selection_flow,
        main_n,
        paper_n,
        scan_metrics[main_name],
        scan_metrics[paper_name],
        final_summary,
        kmeans_comparison,
    )
    write_file_index(config)
    write_checksums(config)
    logger.info("Complete. Main result: %s", PROJECT / "REPORT.md")


if __name__ == "__main__":
    main()
