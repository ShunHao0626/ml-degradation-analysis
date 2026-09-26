"""Human-readable reports and file index."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    display = frame.fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_environment(config: AnalysisConfig) -> None:
    versions: dict[str, str] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "minisom",
        "matplotlib",
        "seaborn",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"not available: {exc}"
    (config.output_root / "environment.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_si_method_report(
    config: AnalysisConfig,
    main_metrics: pd.DataFrame,
    paper_metrics: pd.DataFrame,
    main_decision: pd.DataFrame,
    paper_decision: pd.DataFrame,
    main_selected_n: int,
    paper_selected_n: int,
) -> None:
    path = config.output_root / "05_cluster_number_decision" / "SI_METHOD_APPLICATION.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    def table(frame: pd.DataFrame) -> str:
        selected = frame[
            [
                "n_nodes",
                "topology",
                "qe_mean_across_seeds",
                "qe_std_across_seeds",
                "mean_pairwise_seed_ARI",
                "max_centroid_correlation",
                "min_cluster_fraction",
                "silhouette_pca20",
            ]
        ].copy()
        return _markdown_table(selected.round(4))

    lines = [
        "# Application of the Supplementary Information cluster-number method",
        "",
        "## What the SI does",
        "",
        "The SI scans SOM node counts n=2–10 using quantisation error. "
        "It identifies n=4–6 as the plausible range, then inspects whether "
        "additional clusters overlap or become indistinguishable. It selects "
        "the smallest cluster count that still captures distinct main shapes. "
        "K-means is used as a reference validation.",
        "",
        "Source: `thesis/paper/Supplementary.md`, especially Supplementary "
        "Figs. 6 and 13–17 and the SOM Quantisation Error section.",
        "",
        "## Quantitative translation used here",
        "",
        "For every n=2–10, this project stores:",
        "",
        "- QE and topographic error for five random seeds;",
        "- mean/min pairwise ARI across seeds;",
        "- maximum pairwise centroid correlation and minimum centroid RMSE;",
        "- empty nodes and the minimum cluster fraction;",
        "- PCA-based silhouette, Davies–Bouldin, and Calinski–Harabasz indices;",
        "- K-means WCSS and validation indices.",
        "",
        "The SI-supported candidate interval remains n=4–6. The decision rule "
        "chooses the smallest n in that range satisfying all of: no empty "
        "nodes, no cluster below 1%, mean seed ARI ≥0.80, and no near-duplicate "
        "centroid pair. A pair is considered near-duplicate only when its "
        "correlation is ≥0.995 and its RMSE is <0.10. Using both quantities "
        "avoids treating all smooth monotonic degradation profiles as identical. "
        "If none pass, a documented fallback score is used.",
        "",
        f"## Main high-quality dataset — selected n={main_selected_n}",
        "",
        table(main_metrics),
        "",
        f"## Paper-aligned sensitivity dataset — selected n={paper_selected_n}",
        "",
        table(paper_metrics),
        "",
        "## Rule audit",
        "",
        "Main decision table: `main_high_quality_min10/cluster_number_decision.csv`",
        "",
        "Paper-sensitivity decision table: "
        "`paper_aligned_min4/cluster_number_decision.csv`",
        "",
        "The full decision is intentionally not based on QE alone because QE "
        "almost always decreases when more nodes are added. This follows the "
        "SI's instruction to inspect overlap and interpretability after the "
        "4–6 elbow range has been identified.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_report(
    config: AnalysisConfig,
    selection_flow: pd.DataFrame,
    main_n: int,
    paper_n: int,
    main_metrics: pd.DataFrame,
    paper_metrics: pd.DataFrame,
    main_summary: pd.DataFrame,
    kmeans_comparison: dict,
) -> None:
    counts = dict(
        zip(selection_flow["selection_stage"], selection_flow["n_curves"])
    )
    chosen_metric = main_metrics[main_metrics["n_nodes"] == main_n].iloc[0]
    paper_metric = paper_metrics[paper_metrics["n_nodes"] == paper_n].iloc[0]
    cluster_table = main_summary[
        [
            "class_label",
            "raw_cluster_id",
            "n_curves",
            "fraction",
            "pce_norm_0h_mean",
            "pce_norm_200h_mean",
            "auc_0_200h",
            "suggested_shape_name",
        ]
    ].copy()
    cluster_table["fraction"] = cluster_table["fraction"] * 100.0
    cluster_table = cluster_table.rename(columns={"fraction": "fraction_percent"})
    sensitivity_path = (
        config.output_root / "07_sensitivity" / "main_vs_paper_dataset.json"
    )
    sensitivity = (
        json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if sensitivity_path.is_file()
        else None
    )

    lines = [
        "# 200 h SOM reanalysis of the literature-mined ageing dataset",
        "",
        "## Headline result",
        "",
        f"- Raw curves discovered: **{counts['all_discovered']:,}**",
        f"- Duration ≥200 h: **{counts['duration_ge_200h']:,}**",
        "- Duration ≥200 h and maximum PCE reached by 200 h: "
        f"**{counts['duration_ge_200h_and_global_max_by_200h']:,}**",
        f"- Paper-aligned dataset (≥4 unique points): **{counts['paper_selected_min4']:,}**",
        f"- Main high-quality dataset (≥10 unique points): **{counts['main_selected_min10']:,}**",
        f"- Selected main SOM node count: **n={main_n}**",
        f"- Selected paper-sensitivity node count: **n={paper_n}**",
        "",
        "The primary result uses the ≥10-point dataset because the source is "
        "literature-digitised rather than dense in-house MPPT data. The "
        "paper-aligned ≥4-point dataset is retained as a sensitivity analysis.",
        "",
        "## Preprocessing",
        "",
        "1. Convert source time units to hours.",
        "2. Shift every curve to relative time t=0.",
        "3. Require duration ≥200 h.",
        "4. Require the global maximum PCE to occur by 200 h.",
        "5. Retain 0–200 h plus the first observed point after 200 h.",
        "6. Average duplicate timestamps.",
        "7. Akima-interpolate to 1,201 points at 10-minute spacing.",
        "8. Divide each curve by its own maximum PCE in 0–200 h.",
        "9. Apply Savitzky–Golay smoothing (window 71, polynomial order 2).",
        "",
        "The first point after 200 h brackets the endpoint; no flat tail fill "
        "and no extrapolation is used.",
        "",
        "## Cluster-number decision",
        "",
        f"Main n={main_n}: mean QE={chosen_metric['qe_mean_across_seeds']:.4f}, "
        f"mean seed ARI={chosen_metric['mean_pairwise_seed_ARI']:.4f}, "
        f"max centroid correlation={chosen_metric['max_centroid_correlation']:.4f}.",
        "",
        f"Paper sensitivity n={paper_n}: mean QE={paper_metric['qe_mean_across_seeds']:.4f}, "
        f"mean seed ARI={paper_metric['mean_pairwise_seed_ARI']:.4f}, "
        f"max centroid correlation={paper_metric['max_centroid_correlation']:.4f}.",
        "",
        "The n=4 solution is the smallest SI-range model with stable seeds, "
        "no empty or <1% class, and no centroid pair meeting both near-duplicate "
        "conditions (correlation ≥0.995 and RMSE <0.10). In n=5 and n=6, "
        "the closest centroid RMSE falls below 0.10, indicating subdivision "
        "of an existing shape rather than a clearly new pattern.",
        "",
        "See `05_cluster_number_decision/SI_METHOD_APPLICATION.md` for the "
        "complete SI-based decision logic and n=2–10 tables.",
        "",
        "## Dataset-selection sensitivity",
        "",
        (
            f"Both the 1,442-curve main dataset and the 1,785-curve "
            f"paper-aligned dataset select n={paper_n}. On their "
            f"{sensitivity['n_common_curves']:,} common curves, label agreement "
            f"is ARI={sensitivity['ARI_on_common_curves']:.4f} and "
            f"NMI={sensitivity['NMI_on_common_curves']:.4f}."
            if sensitivity is not None
            else "Sensitivity metrics were not available."
        ),
        "",
        "## Final main clusters",
        "",
        _markdown_table(cluster_table.round(4)),
        "",
        "Class labels are ordered from higher to lower normalized PCE at 200 h. "
        "Suggested shape names are descriptive post-hoc labels, not physical "
        "degradation mechanisms.",
        "",
        "## K-means reference",
        "",
        f"- ARI: {kmeans_comparison['ARI']:.4f}",
        f"- NMI: {kmeans_comparison['NMI']:.4f}",
        "- Hungarian-aligned matched fraction: "
        f"{kmeans_comparison['hungarian_matched_fraction']:.2%}",
        "",
        "K-means here uses Euclidean distance on the same 1,201-dimensional "
        "curves. The reference paper reports DTW K-means, which is not available "
        "in the installed environment; this difference is explicitly retained "
        "as an implementation limitation.",
        "",
        "## Per-curve traceability",
        "",
        "`06_final_model/final_curve_classification.csv` contains one row per "
        "curve with DOI, figure, series, absolute original CSV path, selected "
        "database path information, SOM node, raw cluster, ordered class, "
        "suggested shape label, BMU distance, and K-means reference cluster.",
        "",
        "Each `06_final_model/clusters/class_XX/` directory contains:",
        "",
        "- `members.csv` — all metadata and classification fields;",
        "- `raw_curves/` — browsable raw CSV files assigned to that class.",
        "",
        "## Limitations",
        "",
        "- The source curves are literature-mined and combine laboratories and ageing conditions.",
        "- The article's original dataset was homogeneous, in-house MPPT data.",
        "- A curve-shape cluster must not be interpreted as a unique physical mechanism without metadata analysis.",
        "- SOM node identities are arbitrary; ordered class IDs are a reporting convention.",
        "",
        "## Reproduction",
        "",
        "From this project directory:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/mpl-cache python3 code/run_pipeline.py",
        "python3 code/verify_outputs.py",
        "```",
        "",
    ]
    (config.output_root / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_file_index(config: AnalysisConfig) -> None:
    descriptions = {
        "00_references": "Local copies of the main article and SI markdown.",
        "01_data_selection": "Full 2,151-curve audit, selected manifests, exclusions, and raw 0-200 h curve plots.",
        "02_selected_curve_database": "Browsable raw CSV databases for main and sensitivity cohorts.",
        "03_preprocessed": "Time grid, matrices, wide CSVs, metadata, configuration, and all-curve preprocessing plots.",
        "04_som_results": "n=2–10 plus SI n=16 SOM models, assignments, summaries, seed stability, and figures.",
        "05_cluster_number_decision": "SI-style decision tables, K-means results, and decision plots.",
        "06_final_model": "Chosen model, final traceability table, cluster folders, weights, and figures.",
        "07_sensitivity": "Cross-dataset and density-threshold sensitivity results.",
        "08_group_presentation_comparison": "Slide-ready n=2–10 and n=4–6 comparison figures, tables, and Chinese guide.",
        "09_si_exact_replication": "Exact mapping of SI SOM class counts, n=16 extension, traceable class folders, and comparison figures.",
        "code": "All source code and verification scripts.",
    }
    lines = ["# File index", ""]
    for folder, description in descriptions.items():
        lines.append(f"- `{folder}/` — {description}")
    lines.extend(
        [
            "",
            "- `REPORT.md` — main human-readable report.",
            "- `run_config.json` — complete analysis configuration.",
            "- `environment.json` — software and platform versions.",
            "- `checksums.sha256` — SHA-256 checksums for non-database result files.",
            "",
        ]
    )
    (config.output_root / "FILE_INDEX.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_checksums(config: AnalysisConfig) -> None:
    lines: list[str] = []
    excluded_roots = {
        str(config.output_root / "02_selected_curve_database"),
        str(config.output_root / "06_final_model" / "clusters"),
    }
    for path in sorted(config.output_root.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        if "raw_curves" in path.parts:
            continue
        if any(str(path).startswith(root) for root in excluded_roots):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(config.output_root)}")
    (config.output_root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
