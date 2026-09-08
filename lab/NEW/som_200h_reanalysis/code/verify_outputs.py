#!/usr/bin/env python3
"""Independent output-integrity checks for the 200 h SOM reanalysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    flow = pd.read_csv(PROJECT / "01_data_selection" / "selection_flow.csv")
    counts = dict(zip(flow["selection_stage"], flow["n_curves"]))
    require(int(counts["all_discovered"]) == 2151, "raw total must be 2151")
    require(int(counts["duration_ge_200h"]) == 1862, "duration count mismatch")
    require(int(counts["paper_selected_min4"]) == 1785, "paper count mismatch")
    require(int(counts["main_selected_min10"]) == 1442, "main count mismatch")
    for filename in (
        "main_selected_raw_curves_0_200h.png",
        "main_selected_raw_curves_0_200h_by_scale.png",
        "paper_sensitivity_selected_raw_curves_0_200h.png",
        "paper_sensitivity_selected_raw_curves_0_200h_by_scale.png",
    ):
        raw_plot = PROJECT / "01_data_selection" / filename
        require(raw_plot.is_file(), f"missing raw selected-curve plot: {filename}")
        require(raw_plot.stat().st_size > 0, f"empty raw selected-curve plot: {filename}")

    for filename in (
        "main_high_quality_min10_all_preprocessed_curves.png",
        "paper_aligned_min4_all_preprocessed_curves.png",
    ):
        preprocessed_plot = PROJECT / "03_preprocessed" / filename
        require(
            preprocessed_plot.is_file(),
            f"missing all-curve preprocessing plot: {filename}",
        )
        require(
            preprocessed_plot.stat().st_size > 0,
            f"empty all-curve preprocessing plot: {filename}",
        )

    for name, expected in (
        ("main_high_quality_min10", 1442),
        ("paper_aligned_min4", 1785),
    ):
        database = PROJECT / "02_selected_curve_database" / name
        manifest = pd.read_csv(database / "manifest.csv")
        database_curves = [
            path
            for path in database.rglob("*.csv")
            if path.name != "manifest.csv"
        ]
        require(len(manifest) == expected, f"{name} manifest row mismatch")
        require(
            len(database_curves) == expected,
            f"{name} selected database file-count mismatch",
        )
        require(
            all(
                (database / relative).is_file()
                for relative in manifest["source_relative_path"]
            ),
            f"{name} selected database paths do not match manifest",
        )

        pre = PROJECT / "03_preprocessed" / name
        x = np.load(pre / "X_smoothed.npy")
        xn = np.load(pre / "X_normalized.npy")
        meta = pd.read_csv(pre / "curve_metadata.csv")
        time_grid = pd.read_csv(pre / "time_grid.csv")["time_h"].to_numpy()
        require(x.shape == (expected, 1201), f"{name} smoothed shape mismatch")
        require(xn.shape == (expected, 1201), f"{name} normalized shape mismatch")
        require(
            time_grid.shape == (1201,)
            and abs(time_grid[0]) < 1e-12
            and abs(time_grid[-1] - 200.0) < 1e-9,
            f"{name} time grid mismatch",
        )
        require(np.isfinite(x).all(), f"{name} smoothed has nonfinite values")
        require(np.isfinite(xn).all(), f"{name} normalized has nonfinite values")
        require(meta["curve_id"].is_unique, f"{name} curve IDs are not unique")
        require(len(meta) == expected, f"{name} metadata row mismatch")
        require(
            (meta["endpoint_bracket_time_h"] >= 200.0 - 1e-9).all(),
            f"{name} endpoint not bracketed",
        )
        for n in range(2, 11):
            result = (
                PROJECT
                / "04_som_results"
                / name
                / f"n_{n:02d}"
            )
            assignments = pd.read_csv(result / "cluster_assignments.csv")
            summary = pd.read_csv(result / "cluster_summary.csv")
            require(len(assignments) == expected, f"{name} n={n} row mismatch")
            require(
                assignments["curve_id"].is_unique,
                f"{name} n={n} duplicate curve IDs",
            )
            require(
                set(assignments["curve_id"]) == set(meta["curve_id"]),
                f"{name} n={n} assignment IDs do not match preprocessing metadata",
            )
            require(
                int(summary["n_curves"].sum()) == expected,
                f"{name} n={n} cluster total mismatch",
            )
            require((result / "som_model.pkl").is_file(), f"{name} n={n} model missing")
        metrics = pd.read_csv(
            PROJECT
            / "04_som_results"
            / name
            / "som_cluster_number_metrics.csv"
        )
        require(
            metrics["n_nodes"].tolist() == list(range(2, 11)),
            f"{name} cluster-number scan is incomplete",
        )
        require(
            np.isfinite(
                metrics[
                    [
                        "qe_mean_across_seeds",
                        "mean_pairwise_seed_ARI",
                        "min_centroid_rmse",
                        "silhouette_pca20",
                    ]
                ].to_numpy()
            ).all(),
            f"{name} cluster-number metrics contain nonfinite values",
        )

        n16 = PROJECT / "04_som_results" / name / "n_16"
        n16_assignments = pd.read_csv(n16 / "cluster_assignments.csv")
        n16_summary = pd.read_csv(n16 / "cluster_summary.csv")
        n16_metrics = pd.read_csv(n16 / "n16_metrics.csv")
        require(len(n16_assignments) == expected, f"{name} n=16 row mismatch")
        require(
            n16_assignments["curve_id"].is_unique,
            f"{name} n=16 duplicate curve IDs",
        )
        require(
            set(n16_assignments["curve_id"]) == set(meta["curve_id"]),
            f"{name} n=16 assignment IDs do not match preprocessing metadata",
        )
        require(
            int(n16_summary["n_curves"].sum()) == expected
            and len(n16_summary) == 16,
            f"{name} n=16 cluster total/count mismatch",
        )
        require(
            np.isfinite(
                n16_metrics[
                    [
                        "qe_mean_across_seeds",
                        "mean_pairwise_seed_ARI",
                        "min_centroid_rmse",
                        "silhouette_pca20",
                    ]
                ].to_numpy()
            ).all(),
            f"{name} n=16 metrics contain nonfinite values",
        )
        require((n16 / "som_model.pkl").is_file(), f"{name} n=16 model missing")

        for n_nodes in (16, 2, 4, 5, 6):
            si_classes = (
                PROJECT
                / "09_si_exact_replication"
                / name
                / f"n_{n_nodes:02d}"
                / "classes"
            )
            class_dirs = sorted(si_classes.glob("class_*"))
            require(
                len(class_dirs) == n_nodes,
                f"{name} SI n={n_nodes} class-folder count mismatch",
            )
            member_count = 0
            raw_count = 0
            for class_dir in class_dirs:
                members = pd.read_csv(class_dir / "members.csv")
                raw_files = list((class_dir / "raw_curves").rglob("*.csv"))
                require(
                    len(members) == len(raw_files),
                    f"{name} SI n={n_nodes} {class_dir.name} trace count mismatch",
                )
                member_count += len(members)
                raw_count += len(raw_files)
            require(
                member_count == expected and raw_count == expected,
                f"{name} SI n={n_nodes} total trace count mismatch",
            )

    final = pd.read_csv(
        PROJECT / "06_final_model" / "final_curve_classification.csv"
    )
    require(len(final) == 1442, "final traceability row mismatch")
    require(final["curve_id"].is_unique, "final curve IDs are not unique")
    require(final["class_label"].notna().all(), "missing final class label")
    main_meta = pd.read_csv(
        PROJECT
        / "03_preprocessed"
        / "main_high_quality_min10"
        / "curve_metadata.csv"
    )
    require(
        set(final["curve_id"]) == set(main_meta["curve_id"]),
        "final curve IDs do not match the selected main dataset",
    )
    require(
        final["source_absolute_path"].map(lambda value: Path(value).is_file()).all(),
        "some original source paths are missing",
    )
    summary = pd.read_csv(
        PROJECT / "06_final_model" / "final_cluster_summary.csv"
    )
    require(int(summary["n_curves"].sum()) == 1442, "final cluster total mismatch")
    cluster_root = PROJECT / "06_final_model" / "clusters"
    for row in summary.itertuples(index=False):
        class_dir = cluster_root / row.class_label
        members = pd.read_csv(class_dir / "members.csv")
        raw_files = list((class_dir / "raw_curves").rglob("*.csv"))
        require(
            len(members) == int(row.n_curves),
            f"{row.class_label} members.csv count mismatch",
        )
        require(
            len(raw_files) == int(row.n_curves),
            f"{row.class_label} raw curve count mismatch",
        )
        expected_paths = {
            str((class_dir / "raw_curves" / relative).resolve())
            for relative in members["source_relative_path"]
        }
        require(
            expected_paths == {str(path.resolve()) for path in raw_files},
            f"{row.class_label} raw curve trace paths mismatch",
        )
    selected_n = int(
        (PROJECT / "06_final_model" / "selected_n.txt").read_text().strip()
    )
    require(4 <= selected_n <= 6, "selected n must be inside SI range 4-6")
    decision = pd.read_csv(
        PROJECT
        / "05_cluster_number_decision"
        / "main_high_quality_min10"
        / "cluster_number_decision.csv"
    )
    chosen = decision[decision["selected_n"]]
    require(len(chosen) == 1, "main decision must select exactly one n")
    require(
        int(chosen.iloc[0]["n_nodes"]) == selected_n
        and bool(chosen.iloc[0]["passes_all_SI_style_rules"]),
        "selected n does not pass the documented SI-style rules",
    )

    check = {
        "status": "PASS",
        "selection_counts": {key: int(value) for key, value in counts.items()},
        "selected_n": selected_n,
        "final_rows": len(final),
        "class_counts": {
            row.class_label: int(row.n_curves)
            for row in summary.itertuples(index=False)
        },
        "class_raw_files_verified": True,
        "si_exact_counts_verified": [16, 2, 4, 5, 6],
    }
    (PROJECT / "verification_report.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
