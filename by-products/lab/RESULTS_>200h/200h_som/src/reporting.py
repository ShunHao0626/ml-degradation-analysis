"""Reporting helpers: run-config JSON, run-log, analysis report."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    """Convert numpy / Path / dataclass to JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return obj


def write_run_config(
    run_config_path: Path,
    *,
    source_csv_files: list[str],
    n_total: int,
    n_included: int,
    n_excluded: int,
    som_model_path: str,
    qe_sweep_path: str,
    analysis_report_path: str,
) -> None:
    payload = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "n_total": n_total,
        "n_included": n_included,
        "n_excluded": n_excluded,
        "window_hours": config.WINDOW_HOURS,
        "n_grid_points": config.N_GRID_POINTS,
        "time_grid_step_h": config.GRID_STEP_HOURS,
        "data_config": config.DATA_CONFIG,
        "qc_config": config.QC_CONFIG,
        "duplicates": config.DUPLICATE_CONFIG,
        "preprocessing_pipeline": config.PREPROCESS_STEPS,
        "savgol": config.SAVGOL_CONFIG,
        "akima": config.AKIMA_CONFIG,
        "som_config": config.SOM_CONFIG,
        "qe_sweep": config.QE_SWEEP,
        "kmeans_config": config.KMEANS_CONFIG,
        "sensitivity": config.SENSITIVITY_CONFIGS,
        "manifest_csv": str(config.MANIFEST_PATH),
        "source_csv_files_count": len(source_csv_files),
        "source_csv_files_first5": source_csv_files[:5],
        "som_model_path": str(som_model_path),
        "qe_sweep_path": str(qe_sweep_path),
        "analysis_report_path": str(analysis_report_path),
    }
    for modname in ("numpy", "pandas", "scipy", "sklearn", "minisom", "matplotlib"):
        try:
            mod = __import__(modname)
            payload[f"{modname}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            payload[f"{modname}_version"] = "not installed"
    payload = _jsonable(payload)
    with open(run_config_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote run-config: %s", run_config_path)


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som")


def write_analysis_report(
    out_path: Path,
    *,
    n_total: int,
    n_included: int,
    n_excluded: int,
    excluded_breakdown: dict,
    som_summary: dict,
    qe_summary: dict,
    kmeans_summary: dict,
    sensitivity_summary: dict,
    cluster_summary_df,
    cluster_shape_df,
    package_versions: dict,
    main_model_info: dict,
    main_cluster_metrics: dict,
    caveats: list[str],
    conclusions: list[str],
    next_steps: list[str],
) -> None:
    lines = ["# PvkSOM 200h Analysis Report\n"]
    lines.append(f"**Generated:** {datetime.utcnow().isoformat(timespec='seconds')} UTC")
    lines.append(f"**Run start:** {package_versions.get('run_start', 'unknown')}")
    lines.append(f"**Output directory:** `{config.OUTPUT_DIR}`")
    lines.append("")
    lines.append("---\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Total curves discovered | {n_total} |")
    lines.append(f"| Included (processed) | {n_included} |")
    lines.append(f"| Excluded (QC failed) | {n_excluded} |")
    lines.append(f"| Time window | 0–{config.WINDOW_HOURS:.0f} h |")
    lines.append(f"| Grid interval | 10 min |")
    lines.append(f"| Grid points per curve | {config.N_GRID_POINTS} |")
    lines.append(f"| Savitzky–Golay window | {config.SAVGOL_CONFIG['window_length']} |")
    lines.append(f"| Savitzky–Golay polyorder | {config.SAVGOL_CONFIG['polyorder']} |")
    lines.append(f"| Per-curve normalization | divide by own max in 0–{config.WINDOW_HOURS:.0f} h |")
    lines.append("")
    lines.append("## 2. Inclusion and Exclusion Criteria\n")
    lines.append("### 2.1 Inclusion\n")
    lines.append("A curve is included if ALL of the following hold:")
    lines.append("- Valid sample_id, time, and PCE values")
    lines.append(f"- ≥ {config.QC_CONFIG['min_unique_points_200h']} unique time points in [0, {config.WINDOW_HOURS:.0f}] h")
    lines.append(f"- Curve spans ≥ {config.QC_CONFIG['require_min_span_h']:.0f} h after relative-time shift")
    lines.append(f"- 0–{config.WINDOW_HOURS:.0f} h max PCE > 0 (finite, positive)")
    lines.append("- Akima interpolation succeeds at exactly 1201 grid points")
    lines.append("- All 1201 values are finite after normalisation + Savitzky–Golay")
    lines.append("")
    lines.append("### 2.2 Exclusion\n")
    if excluded_breakdown:
        lines.append("Top exclusion reasons:")
        for reason, count in excluded_breakdown.items():
            lines.append(f"- {reason}: {count} curves")
        lines.append("")
    lines.append("## 3. Preprocessing\n")
    lines.append("Strict order:")
    for i, step in enumerate(config.PREPROCESS_STEPS, 1):
        lines.append(f"{i}. `{step}`")
    lines.append("")
    lines.append("### 3.1 Resampling & Interpolation\n")
    lines.append("- Time grid: `np.linspace(0, 200, 1201)` (10 min spacing)")
    lines.append("- Interpolation: `scipy.interpolate.Akima1DInterpolator`")
    lines.append("- No extrapolation beyond observed range")
    lines.append("")
    lines.append("### 3.2 Normalisation & Smoothing\n")
    lines.append("- Each curve is divided by its OWN max PCE in [0,200] h")
    lines.append("- Then Savitzky-Golay smoothing (window=71, polyorder=2, mode='interp')")
    lines.append("- Result: every smoothed curve has max ≈ 1.0 (small overshoot allowed)")
    lines.append("")
    lines.append("## 4. SOM Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|------:|")
    for k, v in config.SOM_CONFIG.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if main_model_info:
        lines.append(f"**Main model Quantisation Error:** {main_model_info.get('quantisation_error', 'n/a')}")
        lines.append(f"**Main model Topographic Error:** {main_model_info.get('topographic_error', 'n/a')}")
        lines.append("")
    lines.append("## 5. Cluster Results (2×2 SOM)\n")
    for _, row in cluster_summary_df.iterrows():
        lines.append(f"### Node ({int(row['som_node_x'])}, {int(row['som_node_y'])})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Curves | {int(row['n_samples'])} ({row['fraction']*100:.1f}%) |")
        for key in [
            "pce_norm_0h_mean", "pce_norm_10h_mean", "pce_norm_50h_mean",
            "pce_norm_100h_mean", "pce_norm_150h_mean", "pce_norm_200h_mean",
            "mean_bmu_distance", "median_bmu_distance", "auc_200h",
        ]:
            v = row[key]
            lines.append(f"| {key} | {v:.4f} |")
        lines.append("")
        # Suggested descriptive name (post-hoc)
        match = cluster_shape_df[cluster_shape_df["raw_cluster_id"] == int(row["raw_cluster_id"])]
        if not match.empty:
            sug = match.iloc[0]["suggested_shape_name"]
            basis = match.iloc[0]["naming_basis"]
            lines.append(f"- **Suggested shape name (post-hoc):** `{sug}`")
            lines.append(f"- **Naming basis:** {basis}")
            lines.append("")
    lines.append("## 6. Quantisation Error Sweep\n")
    lines.append(f"Main model (n=4, 2×2) QE: **{main_model_info.get('quantisation_error', 'n/a')}**\n")
    lines.append("| n | topology | QE | source |")
    lines.append("|---:|:--------:|------:|:-------|")
    for _, row in qe_summary.iterrows():
        lines.append(
            f"| {int(row['n_nodes'])} | {row['topology']} | {row['quantisation_error']:.4f} | {row.get('topology_source','-')} |"
        )
    lines.append("")
    lines.append("> Topology for n ≠ 4 is **1×n** in the QE sweep. This is an "
                 "**implementation assumption** because the author code does not "
                 "explicitly document a topology rule for arbitrary node counts.")
    lines.append("")
    lines.append("## 7. Sensitivity Analysis\n")
    lines.append("| Model | sigma | learning_rate | QE |")
    lines.append("|-------|------:|--------------:|------:|")
    for _, row in sensitivity_summary.iterrows():
        lines.append(f"| {row['model']} | {row['sigma']} | {row['learning_rate']} | {row['quantisation_error']:.4f} |")
    lines.append("")
    lines.append("Cross-model ARI / NMI are reported in `sensitivity_summary.csv`. "
                 "Clusters are aligned to the main model by Hungarian matching on the "
                 "codebook vectors.")
    lines.append("")
    lines.append("## 8. K-means Validation (k = 2 … 10)\n")
    lines.append("| k | WCSS (inertia) |")
    lines.append("|---:|---------------:|")
    for _, row in kmeans_summary.iterrows():
        lines.append(f"| {int(row['k'])} | {row['wcss_inertia']:.2f} |")
    lines.append("")
    if main_cluster_metrics.get("ari_to_kmeans") is not None:
        lines.append(f"**SOM(main) vs KMeans(k={main_cluster_metrics.get('k_for_ari','?')}):** "
                     f"ARI = {main_cluster_metrics['ari_to_kmeans']:.4f}, "
                     f"NMI = {main_cluster_metrics['nmi_to_kmeans']:.4f}")
        lines.append("")
    lines.append("## 9. Reproducibility\n")
    lines.append("| Package | Version |")
    lines.append("|---------|---------|")
    for k, v in package_versions.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 10. Research Limitations\n")
    for c in caveats:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 11. Conclusions (Exploratory)\n")
    for c in conclusions:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 12. Suggested Next Steps\n")
    for s in next_steps:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("---\n")
    lines.append("*End of analysis report.*")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote analysis report: %s", out_path)
