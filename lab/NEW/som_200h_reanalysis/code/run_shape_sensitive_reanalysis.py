#!/usr/bin/env python3
"""Run an independent, shape-sensitive 200 h SOM reanalysis.

This workflow never overwrites the existing selection, preprocessing, or SOM
results.  It audits the high-quality cohort, compares interpolation/smoothing
choices, constructs explicitly shape-sensitive feature blocks, scans SOM
sizes n=4..10 across multiple seeds, and validates the selected model against
the original observed points.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minisom import MiniSom
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
OUTPUT = PROJECT / "10_shape_sensitive_reanalysis"
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.data import RawCurve, discover_curves
from som200h.modeling import _davies_bouldin_score_stable, _robust_kmeans
from som200h.preprocessing import (
    _deduplicate_mean,
    _support_through_first_point_after_window,
)


sns.set_theme(style="whitegrid", context="notebook")


NODES = tuple(range(4, 11))
SEEDS = (7, 21, 42, 84, 168)
PRIMARY_SEED = 42
SOM_ITERATIONS = 30_000
MAX_SUPPORT_GAP_H = 40.0
FINE_GRID = np.linspace(0.0, 200.0, 1201)
MODEL_GRID = np.linspace(0.0, 200.0, 201)
TOPOLOGY = {
    4: (2, 2),
    5: (1, 5),
    6: (2, 3),
    7: (1, 7),
    8: (2, 4),
    9: (3, 3),
    10: (2, 5),
}


@dataclass
class PreprocessingBundle:
    metadata: pd.DataFrame
    arrays: dict[str, np.ndarray]
    fidelity: pd.DataFrame


@dataclass
class SOMCandidate:
    feature_set: str
    n_nodes: int
    topology: tuple[int, int]
    labels: np.ndarray
    coords: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    som: MiniSom
    metrics: dict[str, float | int | str]
    cluster_summary: pd.DataFrame
    curve_centroids: np.ndarray


def configure_logging() -> logging.Logger:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("shape_sensitive")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def pce_label_status(name: object, figure_folder: object) -> str:
    text = f"{name or ''} {figure_folder or ''}".lower()
    strong = ("pce", "efficien", "power conversion", "photovoltaic", "stability")
    return "pce_like" if any(token in text for token in strong) else "review_label"


def build_qc_audit(
    selected: pd.DataFrame,
    curves_by_id: dict[str, RawCurve],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for row in selected.itertuples(index=False):
        curve = curves_by_id[row.curve_id]
        x_unique, y_unique, n_duplicates = _deduplicate_mean(
            curve.x_hours_relative, curve.y
        )
        x_support, y_support, bracket_time = _support_through_first_point_after_window(
            x_unique, y_unique, 200.0
        )
        within = (x_unique >= 0.0) & (x_unique <= 200.0 + 1e-9)
        gaps = np.diff(x_support)
        max_gap = float(np.max(gaps)) if gaps.size else math.inf
        reasons: list[str] = []
        if int(np.sum(within)) < 10:
            reasons.append("fewer_than_10_unique_points")
        if max_gap > MAX_SUPPORT_GAP_H:
            reasons.append("max_support_gap_gt_40h")
        if x_support[0] > 1e-9 or x_support[-1] < 200.0 - 1e-9:
            reasons.append("window_not_bracketed")
        if not np.isfinite(y_support).all() or float(np.max(y_support)) <= 0:
            reasons.append("invalid_y_support")
        label_status = pce_label_status(row.y_axis_name, row.figure_folder)
        rows.append(
            {
                **row._asdict(),
                "n_unique_points_total_recomputed": int(x_unique.size),
                "n_unique_points_0_200h_recomputed": int(np.sum(within)),
                "n_duplicate_points_recomputed": n_duplicates,
                "max_support_gap_h": max_gap,
                "endpoint_bracket_time_h_recomputed": float(bracket_time),
                "endpoint_gap_after_200h": float(bracket_time - 200.0),
                "raw_y_min_support": float(np.min(y_support)),
                "raw_y_max_support": float(np.max(y_support)),
                "axis_label_status": label_status,
                "strict_selected": not reasons,
                "strict_selection_reasons": ";".join(reasons) if reasons else "included",
            }
        )
    audit = pd.DataFrame(rows).sort_values("curve_id").reset_index(drop=True)
    strict = audit[audit["strict_selected"]].copy().reset_index(drop=True)
    out = OUTPUT / "01_data_qc"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "strict_selection_audit.csv", index=False)
    strict.to_csv(out / "strict_selected_curves.csv", index=False)
    audit[~audit["strict_selected"]].to_csv(
        out / "strict_excluded_curves.csv", index=False
    )
    audit[audit["axis_label_status"] == "review_label"].to_csv(
        out / "axis_labels_for_review.csv", index=False
    )
    summary = {
        "input_main_selected": int(len(audit)),
        "strict_selected": int(len(strict)),
        "strict_excluded": int((~audit["strict_selected"]).sum()),
        "max_support_gap_threshold_h": MAX_SUPPORT_GAP_H,
        "axis_label_review_count": int(
            (audit["axis_label_status"] == "review_label").sum()
        ),
        "exclusion_reason_counts": audit.loc[
            ~audit["strict_selected"], "strict_selection_reasons"
        ].value_counts().to_dict(),
    }
    (out / "qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(audit["max_support_gap_h"], bins=60, color="#60A5FA", edgecolor="white")
    ax.axvline(
        MAX_SUPPORT_GAP_H,
        color="#DC2626",
        linestyle="--",
        label=f"QC threshold = {MAX_SUPPORT_GAP_H:.0f} h",
    )
    ax.set_xlim(0, min(120, float(audit["max_support_gap_h"].max()) + 5))
    ax.set_xlabel("Maximum gap between interpolation support points (h)")
    ax.set_ylabel("Number of curves")
    ax.set_title("Strict-cohort temporal coverage audit")
    ax.legend()
    save_figure(fig, out / "max_support_gap_distribution.png")
    logger.info(
        "QC complete: %d/%d retained; %d flagged axis labels",
        len(strict),
        len(audit),
        summary["axis_label_review_count"],
    )
    return audit, strict


def normalized_variant(values: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("invalid interpolation maximum")
    return values / maximum, maximum


def preprocess_strict_curves(
    strict: pd.DataFrame,
    curves_by_id: dict[str, RawCurve],
    logger: logging.Logger,
) -> PreprocessingBundle:
    arrays: dict[str, list[np.ndarray]] = {
        "akima_normalized": [],
        "akima_smoothed": [],
        "pchip_normalized": [],
        "linear_normalized": [],
    }
    metadata_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for output_row, row in enumerate(strict.itertuples(index=False)):
        curve = curves_by_id[row.curve_id]
        x_unique, y_unique, n_duplicates = _deduplicate_mean(
            curve.x_hours_relative, curve.y
        )
        x_support, y_support, bracket_time = _support_through_first_point_after_window(
            x_unique, y_unique, 200.0
        )
        akima_interpolator = Akima1DInterpolator(x_support, y_support)
        pchip_interpolator = PchipInterpolator(x_support, y_support, extrapolate=False)
        akima_raw = np.asarray(
            akima_interpolator(FINE_GRID, extrapolate=False), dtype=float
        )
        pchip_raw = np.asarray(pchip_interpolator(FINE_GRID), dtype=float)
        linear_raw = np.interp(FINE_GRID, x_support, y_support)
        if not (
            np.isfinite(akima_raw).all()
            and np.isfinite(pchip_raw).all()
            and np.isfinite(linear_raw).all()
        ):
            raise ValueError(f"Nonfinite interpolation for {row.curve_id}")

        akima_norm, akima_max = normalized_variant(akima_raw)
        pchip_norm, pchip_max = normalized_variant(pchip_raw)
        linear_norm, linear_max = normalized_variant(linear_raw)
        akima_smooth = savgol_filter(
            akima_norm, window_length=71, polyorder=2, mode="interp"
        )
        arrays["akima_normalized"].append(akima_norm)
        arrays["akima_smoothed"].append(akima_smooth)
        arrays["pchip_normalized"].append(pchip_norm)
        arrays["linear_normalized"].append(linear_norm)

        within = (x_unique >= 0.0) & (x_unique <= 200.0 + 1e-9)
        x_raw = x_unique[within]
        y_raw = y_unique[within]
        support_min = float(np.min(y_support))
        support_max = float(np.max(y_support))

        def excursion(y_interp: np.ndarray, scale: float) -> tuple[float, float]:
            below = max(0.0, (support_min - float(np.min(y_interp))) / scale)
            above = max(0.0, (float(np.max(y_interp)) - support_max) / scale)
            return below, above

        a_below, a_above = excursion(akima_raw, akima_max)
        p_below, p_above = excursion(pchip_raw, pchip_max)
        l_below, l_above = excursion(linear_raw, linear_max)

        # Interpolators reproduce the de-duplicated observed knots.  These
        # errors are measured directly at the knots, not via the uniform grid.
        raw_akima = np.asarray(
            akima_interpolator(x_raw, extrapolate=False), dtype=float
        )
        raw_pchip = np.asarray(pchip_interpolator(x_raw), dtype=float)
        raw_linear = np.interp(x_raw, x_support, y_support)
        fidelity_rows.append(
            {
                "curve_id": row.curve_id,
                "array_row": output_row,
                "n_unique_points_0_200h": int(x_raw.size),
                "n_duplicate_points": n_duplicates,
                "max_support_gap_h": float(np.max(np.diff(x_support))),
                "endpoint_bracket_time_h": float(bracket_time),
                "akima_raw_knot_rmse": float(
                    np.sqrt(np.mean((raw_akima - y_raw) ** 2))
                ),
                "pchip_raw_knot_rmse": float(
                    np.sqrt(np.mean((raw_pchip - y_raw) ** 2))
                ),
                "linear_raw_knot_rmse": float(
                    np.sqrt(np.mean((raw_linear - y_raw) ** 2))
                ),
                "akima_below_support_normalized": a_below,
                "akima_above_support_normalized": a_above,
                "pchip_below_support_normalized": p_below,
                "pchip_above_support_normalized": p_above,
                "linear_below_support_normalized": l_below,
                "linear_above_support_normalized": l_above,
                "akima_vs_pchip_rmse": float(
                    np.sqrt(np.mean((akima_norm - pchip_norm) ** 2))
                ),
                "pchip_vs_linear_rmse": float(
                    np.sqrt(np.mean((pchip_norm - linear_norm) ** 2))
                ),
                "akima_smoothing_rmse": float(
                    np.sqrt(np.mean((akima_smooth - akima_norm) ** 2))
                ),
            }
        )
        metadata_rows.append(
            {
                "array_row": output_row,
                "curve_id": row.curve_id,
                "sample_id": row.sample_id,
                "source_absolute_path": row.source_absolute_path,
                "source_relative_path": row.source_relative_path,
                "doi": row.doi,
                "figure_folder": row.figure_folder,
                "series_id": row.series_id,
                "n_unique_points_0_200h": int(x_raw.size),
                "max_support_gap_h": float(np.max(np.diff(x_support))),
                "endpoint_bracket_time_h": float(bracket_time),
                "akima_max_pce": akima_max,
                "pchip_max_pce": pchip_max,
                "linear_max_pce": linear_max,
            }
        )

    stacked = {key: np.stack(value) for key, value in arrays.items()}
    metadata = pd.DataFrame(metadata_rows)
    fidelity = pd.DataFrame(fidelity_rows)
    out = OUTPUT / "02_preprocessing_comparison"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_h": FINE_GRID}).to_csv(out / "time_grid_10min.csv", index=False)
    metadata.to_csv(out / "curve_metadata.csv", index=False)
    fidelity.to_csv(out / "preprocessing_fidelity_metrics.csv", index=False)
    for key, matrix in stacked.items():
        np.save(out / f"X_{key}.npy", matrix)

    summary_rows: list[dict] = []
    for column in (
        "akima_below_support_normalized",
        "akima_above_support_normalized",
        "pchip_below_support_normalized",
        "pchip_above_support_normalized",
        "linear_below_support_normalized",
        "linear_above_support_normalized",
        "akima_vs_pchip_rmse",
        "pchip_vs_linear_rmse",
        "akima_smoothing_rmse",
    ):
        values = fidelity[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": column,
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "maximum": float(np.max(values)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "preprocessing_summary.csv", index=False)
    plot_preprocessing_comparison(stacked, fidelity, metadata, curves_by_id)
    logger.info("Preprocessing comparison complete: %d curves", len(metadata))
    return PreprocessingBundle(metadata=metadata, arrays=stacked, fidelity=fidelity)


def plot_preprocessing_comparison(
    arrays: dict[str, np.ndarray],
    fidelity: pd.DataFrame,
    metadata: pd.DataFrame,
    curves_by_id: dict[str, RawCurve],
) -> None:
    out = OUTPUT / "02_preprocessing_comparison"
    colors = {
        "akima_normalized": "#64748B",
        "akima_smoothed": "#7C3AED",
        "pchip_normalized": "#2563EB",
        "linear_normalized": "#059669",
    }
    titles = {
        "akima_normalized": "Akima, no smoothing",
        "akima_smoothed": "Akima + Savitzky-Golay",
        "pchip_normalized": "PCHIP, no smoothing (recommended)",
        "linear_normalized": "Linear, no smoothing",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, colors):
        for curve in arrays[key]:
            ax.plot(FINE_GRID, curve, color=colors[key], alpha=0.025, linewidth=0.4)
        ax.set_title(f"{titles[key]} (n={arrays[key].shape[0]:,})")
        ax.set_xlabel("Relative ageing time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
    fig.suptitle("All strict-cohort curves across preprocessing variants")
    save_figure(fig, out / "all_preprocessing_variants.png")

    for key in colors:
        fig, ax = plt.subplots(figsize=(10, 6.2))
        for curve in arrays[key]:
            ax.plot(FINE_GRID, curve, color=colors[key], alpha=0.035, linewidth=0.4)
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Relative ageing time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.set_title(f"{titles[key]} — all curves (n={arrays[key].shape[0]:,})")
        save_figure(fig, out / f"all_curves_{key}.png")

    worst = fidelity.nlargest(12, "akima_vs_pchip_rmse")
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, worst.itertuples(index=False)):
        index = int(row.array_row)
        curve = curves_by_id[row.curve_id]
        x, y, _ = _deduplicate_mean(curve.x_hours_relative, curve.y)
        mask = (x >= 0) & (x <= 200 + 1e-9)
        denominator = metadata.iloc[index]["pchip_max_pce"]
        ax.scatter(
            x[mask],
            y[mask] / denominator,
            s=13,
            color="#111827",
            zorder=4,
            label="raw points",
        )
        ax.plot(
            FINE_GRID,
            arrays["akima_normalized"][index],
            color="#7C3AED",
            linewidth=1.2,
            label="Akima",
        )
        ax.plot(
            FINE_GRID,
            arrays["pchip_normalized"][index],
            color="#2563EB",
            linewidth=1.2,
            label="PCHIP",
        )
        ax.plot(
            FINE_GRID,
            arrays["linear_normalized"][index],
            color="#059669",
            linewidth=1.0,
            linestyle="--",
            label="linear",
        )
        ax.set_title(f"row {index}; gap={row.max_support_gap_h:.1f} h")
        ax.set_xlim(0, 200)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Largest Akima–PCHIP differences with original observed points")
    save_figure(fig, out / "largest_interpolation_differences.png")


def value_at(curves: np.ndarray, hour: float) -> np.ndarray:
    index = int(np.argmin(np.abs(MODEL_GRID - hour)))
    return curves[:, index]


def build_shape_descriptors(curves: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    derivative = np.gradient(curves, MODEL_GRID, axis=1)
    peak_index = np.argmax(curves, axis=1)
    minimum_index = np.argmin(curves, axis=1)
    steepest_index = np.argmin(derivative, axis=1)
    rows: list[dict] = []
    for index, curve in enumerate(curves):
        peak_i = int(peak_index[index])
        min_i = int(minimum_index[index])
        later_max = float(np.max(curve[min_i:]))
        p0 = float(curve[0])
        p200 = float(curve[-1])
        row = {
            "array_row": index,
            "curve_id": metadata.iloc[index]["curve_id"],
            "pce_0h": p0,
            "pce_10h": float(value_at(curves[index : index + 1], 10)[0]),
            "pce_20h": float(value_at(curves[index : index + 1], 20)[0]),
            "pce_50h": float(value_at(curves[index : index + 1], 50)[0]),
            "pce_100h": float(value_at(curves[index : index + 1], 100)[0]),
            "pce_150h": float(value_at(curves[index : index + 1], 150)[0]),
            "pce_200h": p200,
            "slope_0_10h": float((curve[10] - curve[0]) / 10.0),
            "slope_10_50h": float((curve[50] - curve[10]) / 40.0),
            "slope_50_100h": float((curve[100] - curve[50]) / 50.0),
            "slope_100_200h": float((curve[200] - curve[100]) / 100.0),
            "peak_time_h": float(MODEL_GRID[peak_i]),
            "peak_value": float(curve[peak_i]),
            "initial_gain": float(curve[peak_i] - p0),
            "minimum_time_h": float(MODEL_GRID[min_i]),
            "minimum_value": float(curve[min_i]),
            "recovery_to_200h": float(max(0.0, p200 - curve[min_i])),
            "maximum_post_min_recovery": float(max(0.0, later_max - curve[min_i])),
            "max_drop_rate_per_h": float(np.min(derivative[index])),
            "max_drop_time_h": float(MODEL_GRID[int(steepest_index[index])]),
            "auc_mean_0_200h": float(np.trapezoid(curve, MODEL_GRID) / 200.0),
            "endpoint_change": float(p200 - p0),
            "mean_abs_derivative": float(np.mean(np.abs(derivative[index]))),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)

    categories: list[str] = []
    for row in frame.itertuples(index=False):
        if row.initial_gain > 0.01 and row.peak_time_h > 5.0:
            category = "initial_gain"
        elif row.maximum_post_min_recovery > 0.03 and row.minimum_time_h < 180.0:
            category = "dip_recovery"
        elif row.slope_0_10h < -0.003:
            category = "early_drop"
        elif row.pce_200h >= 0.90:
            category = "stable"
        elif row.pce_200h >= 0.70:
            category = "moderate_decay"
        else:
            category = "rapid_decay"
        categories.append(category)
    frame["rule_shape_category"] = categories
    return frame


def normalize_block(block: np.ndarray) -> np.ndarray:
    centered = np.asarray(block, dtype=float) - np.mean(block, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot normalize a zero-variance feature block")
    return centered / scale


def build_feature_sets(
    bundle: PreprocessingBundle,
    logger: logging.Logger,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    # The 10-minute PCHIP representation is sampled exactly at integer hours.
    pchip_fine = bundle.arrays["pchip_normalized"]
    model_indices = np.searchsorted(FINE_GRID, MODEL_GRID)
    curves = pchip_fine[:, model_indices]
    descriptors = build_shape_descriptors(curves, bundle.metadata)

    time_weights = np.ones(MODEL_GRID.size, dtype=float)
    time_weights[MODEL_GRID <= 20] = 8.0
    time_weights[(MODEL_GRID > 20) & (MODEL_GRID <= 50)] = 4.0
    time_weights[(MODEL_GRID > 50) & (MODEL_GRID <= 100)] = 2.0
    weighted_curve = normalize_block(curves * np.sqrt(time_weights)[None, :])

    derivative = np.gradient(curves, MODEL_GRID, axis=1)[:, ::5]
    derivative_block = normalize_block(derivative)

    descriptor_columns = [
        "pce_10h",
        "pce_20h",
        "pce_50h",
        "pce_100h",
        "pce_150h",
        "pce_200h",
        "slope_0_10h",
        "slope_10_50h",
        "slope_50_100h",
        "slope_100_200h",
        "peak_time_h",
        "initial_gain",
        "minimum_time_h",
        "minimum_value",
        "recovery_to_200h",
        "maximum_post_min_recovery",
        "max_drop_rate_per_h",
        "max_drop_time_h",
        "auc_mean_0_200h",
        "endpoint_change",
        "mean_abs_derivative",
    ]
    descriptor_values = descriptors[descriptor_columns].to_numpy(dtype=float)
    means = np.mean(descriptor_values, axis=0)
    stds = np.std(descriptor_values, axis=0)
    stds[stds < 1e-12] = 1.0
    descriptor_z = (descriptor_values - means) / stds
    descriptor_block = normalize_block(descriptor_z)

    feature_sets = {
        "time_weighted_curve": weighted_curve,
        "curve_plus_derivative": np.hstack(
            [np.sqrt(0.70) * weighted_curve, np.sqrt(0.30) * derivative_block]
        ),
        "curve_derivative_descriptors": np.hstack(
            [
                np.sqrt(0.50) * weighted_curve,
                np.sqrt(0.30) * derivative_block,
                np.sqrt(0.20) * descriptor_block,
            ]
        ),
        "descriptors_only": descriptor_block,
    }
    out = OUTPUT / "03_shape_features"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_h": MODEL_GRID, "weight": time_weights}).to_csv(
        out / "model_time_grid_and_weights.csv", index=False
    )
    descriptors.to_csv(out / "shape_descriptors.csv", index=False)
    pd.DataFrame(
        {
            "descriptor": descriptor_columns,
            "mean": means,
            "std": stds,
        }
    ).to_csv(out / "descriptor_scaling.csv", index=False)
    np.save(out / "X_pchip_1h.npy", curves)
    for name, matrix in feature_sets.items():
        np.save(out / f"features_{name}.npy", matrix)
    descriptor_counts = (
        descriptors["rule_shape_category"]
        .value_counts()
        .rename_axis("rule_shape_category")
        .reset_index(name="n_curves")
    )
    descriptor_counts.to_csv(out / "rule_shape_category_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(MODEL_GRID, time_weights, color="#7C3AED", linewidth=2)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Squared-distance time weight")
    ax.set_title("Time weighting used by shape-sensitive curve features")
    ax.set_ylim(0, 8.6)
    save_figure(fig, out / "time_weighting.png")
    logger.info(
        "Feature sets built: %s",
        ", ".join(f"{key}={value.shape}" for key, value in feature_sets.items()),
    )
    return feature_sets, descriptors, curves


def train_one_som(
    features: np.ndarray,
    n_nodes: int,
    seed: int,
) -> tuple[MiniSom, np.ndarray, np.ndarray, np.ndarray, float, float]:
    topology = TOPOLOGY[n_nodes]
    np.random.seed(seed)
    som = MiniSom(
        x=topology[0],
        y=topology[1],
        input_len=features.shape[1],
        sigma=0.5,
        learning_rate=0.1,
        topology="rectangular",
        neighborhood_function="gaussian",
        activation_distance="euclidean",
        random_seed=seed,
    )
    som.random_weights_init(features)
    som.train(features, SOM_ITERATIONS, verbose=False)
    coords = np.asarray([som.winner(row) for row in features], dtype=int)
    labels = coords[:, 0] * topology[1] + coords[:, 1]
    flat_weights = som.get_weights().reshape(n_nodes, features.shape[1])
    distances = np.linalg.norm(features - flat_weights[labels], axis=1)
    return (
        som,
        labels,
        coords,
        distances,
        float(som.quantization_error(features)),
        float(som.topographic_error(features)),
    )


def pairwise_seed_ari(labels_by_seed: dict[int, np.ndarray]) -> tuple[float, float]:
    values: list[float] = []
    seeds = sorted(labels_by_seed)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            values.append(
                float(adjusted_rand_score(labels_by_seed[seed_a], labels_by_seed[seed_b]))
            )
    return float(np.mean(values)), float(np.min(values))


def build_candidate_metrics(
    feature_set: str,
    features: np.ndarray,
    descriptors: pd.DataFrame,
    curves: np.ndarray,
    n_nodes: int,
    som: MiniSom,
    labels: np.ndarray,
    coords: np.ndarray,
    distances: np.ndarray,
    qe: float,
    te: float,
    mean_seed_ari: float,
    min_seed_ari: float,
) -> SOMCandidate:
    pca_components = min(20, features.shape[1], features.shape[0] - 1)
    if not np.isfinite(features).all() or float(np.max(np.abs(features))) > 1e6:
        raise ValueError(
            f"Invalid feature matrix for {feature_set}: "
            f"range=({np.min(features)}, {np.max(features)})"
        )
    # Use deterministic full SVD.  The randomized solver can emit spurious
    # overflow warnings with the NumPy/BLAS combination in this environment.
    embedding = PCA(
        n_components=pca_components,
        svd_solver="full",
        random_state=42,
    ).fit_transform(features)
    validation_distances = cdist(embedding, embedding, metric="euclidean")
    unique_labels = np.unique(labels)
    silhouette = (
        float(silhouette_score(validation_distances, labels, metric="precomputed"))
        if unique_labels.size > 1
        else math.nan
    )
    db = (
        _davies_bouldin_score_stable(embedding, labels)
        if unique_labels.size > 1
        else math.nan
    )
    ch = (
        float(calinski_harabasz_score(embedding, labels))
        if unique_labels.size > 1
        else math.nan
    )
    kmeans_labels, _ = _robust_kmeans(
        embedding,
        n_clusters=n_nodes,
        n_init=20,
        base_seed=42,
    )
    kmeans_ari = float(adjusted_rand_score(labels, kmeans_labels))
    category_codes = pd.Categorical(descriptors["rule_shape_category"]).codes
    category_nmi = float(normalized_mutual_info_score(category_codes, labels))

    category_metrics: dict[str, float | int] = {}
    for target in ("initial_gain", "early_drop", "dip_recovery"):
        target_mask = descriptors["rule_shape_category"].to_numpy() == target
        target_total = int(np.sum(target_mask))
        best_f1 = 0.0
        best_precision = {"precision": 0.0, "recall": 0.0, "count": 0}
        for cluster_id in range(n_nodes):
            cluster_mask = labels == cluster_id
            true_count = int(np.sum(target_mask & cluster_mask))
            if true_count == 0:
                continue
            precision = float(true_count / np.sum(cluster_mask))
            recall = float(true_count / target_total) if target_total else 0.0
            f1 = (
                float(2.0 * precision * recall / (precision + recall))
                if precision + recall > 0
                else 0.0
            )
            best_f1 = max(best_f1, f1)
            # Purity gates must not be tied to the cluster with the highest
            # F1: a large mixed cluster can have higher recall/F1 while a
            # smaller, genuinely shape-specific cluster has much higher purity.
            if true_count >= 20 and precision > best_precision["precision"]:
                best_precision = {
                    "precision": precision,
                    "recall": recall,
                    "count": true_count,
                }
        category_metrics[f"{target}_best_f1"] = float(best_f1)
        category_metrics[f"{target}_best_precision"] = float(
            best_precision["precision"]
        )
        category_metrics[f"{target}_best_recall"] = float(best_precision["recall"])
        category_metrics[f"{target}_best_count"] = int(best_precision["count"])

    counts = np.bincount(labels, minlength=n_nodes)
    curve_centroids = np.stack(
        [
            np.mean(curves[labels == cluster_id], axis=0)
            if counts[cluster_id]
            else np.full(curves.shape[1], np.nan)
            for cluster_id in range(n_nodes)
        ]
    )
    valid = counts > 0
    valid_centroids = curve_centroids[valid]
    pairwise_rmse: list[float] = []
    pairwise_corr: list[float] = []
    for i in range(valid_centroids.shape[0]):
        for j in range(i + 1, valid_centroids.shape[0]):
            pairwise_rmse.append(
                float(np.sqrt(np.mean((valid_centroids[i] - valid_centroids[j]) ** 2)))
            )
            pairwise_corr.append(
                float(np.corrcoef(valid_centroids[i], valid_centroids[j])[0, 1])
            )

    cluster_rows: list[dict] = []
    for cluster_id in range(n_nodes):
        mask = labels == cluster_id
        if not np.any(mask):
            cluster_rows.append(
                {"cluster_id": cluster_id, "n_curves": 0, "fraction": 0.0}
            )
            continue
        category_counts = descriptors.loc[mask, "rule_shape_category"].value_counts()
        dominant_category = str(category_counts.index[0])
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "n_curves": int(mask.sum()),
                "fraction": float(mask.mean()),
                "pce_0h_mean": float(np.mean(curves[mask, 0])),
                "pce_20h_mean": float(np.mean(curves[mask, 20])),
                "pce_50h_mean": float(np.mean(curves[mask, 50])),
                "pce_100h_mean": float(np.mean(curves[mask, 100])),
                "pce_200h_mean": float(np.mean(curves[mask, 200])),
                "initial_gain_mean": float(
                    descriptors.loc[mask, "initial_gain"].mean()
                ),
                "peak_time_h_median": float(
                    descriptors.loc[mask, "peak_time_h"].median()
                ),
                "recovery_mean": float(
                    descriptors.loc[mask, "maximum_post_min_recovery"].mean()
                ),
                "dominant_rule_shape": dominant_category,
                "dominant_rule_shape_fraction": float(
                    category_counts.iloc[0] / mask.sum()
                ),
                "mean_bmu_distance": float(np.mean(distances[mask])),
                "median_bmu_distance": float(np.median(distances[mask])),
            }
        )
    summary = pd.DataFrame(cluster_rows)
    metrics: dict[str, float | int | str] = {
        "feature_set": feature_set,
        "n_nodes": n_nodes,
        "topology": f"{TOPOLOGY[n_nodes][0]}x{TOPOLOGY[n_nodes][1]}",
        "n_features": int(features.shape[1]),
        "qe_primary": qe,
        "te_primary": te,
        "mean_pairwise_seed_ARI": mean_seed_ari,
        "min_pairwise_seed_ARI": min_seed_ari,
        "silhouette_pca20": silhouette,
        "davies_bouldin_pca20": db,
        "calinski_harabasz_pca20": ch,
        "kmeans_ARI": kmeans_ari,
        "rule_shape_category_NMI": category_nmi,
        "n_empty_nodes": int(np.sum(counts == 0)),
        "min_cluster_size": int(np.min(counts)),
        "min_cluster_fraction": float(np.min(counts) / len(labels)),
        "max_cluster_size": int(np.max(counts)),
        "max_cluster_fraction": float(np.max(counts) / len(labels)),
        "min_curve_centroid_RMSE": float(np.min(pairwise_rmse))
        if pairwise_rmse
        else math.nan,
        "max_curve_centroid_correlation": float(np.max(pairwise_corr))
        if pairwise_corr
        else math.nan,
        **category_metrics,
    }
    return SOMCandidate(
        feature_set=feature_set,
        n_nodes=n_nodes,
        topology=TOPOLOGY[n_nodes],
        labels=labels,
        coords=coords,
        distances=distances,
        weights=som.get_weights(),
        som=som,
        metrics=metrics,
        cluster_summary=summary,
        curve_centroids=curve_centroids,
    )


def scan_som_models(
    feature_sets: dict[str, np.ndarray],
    descriptors: pd.DataFrame,
    curves: np.ndarray,
    metadata: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[tuple[str, int], SOMCandidate]]:
    root = OUTPUT / "04_som_n4_to_n10"
    root.mkdir(parents=True, exist_ok=True)
    candidates: dict[tuple[str, int], SOMCandidate] = {}
    metrics_rows: list[dict] = []
    for feature_name, features in feature_sets.items():
        feature_dir = root / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        for n_nodes in NODES:
            logger.info("SOM scan %s n=%d", feature_name, n_nodes)
            labels_by_seed: dict[int, np.ndarray] = {}
            primary: tuple[MiniSom, np.ndarray, np.ndarray, np.ndarray, float, float] | None = None
            qe_by_seed: list[float] = []
            te_by_seed: list[float] = []
            for seed in SEEDS:
                result = train_one_som(features, n_nodes, seed)
                labels_by_seed[seed] = result[1]
                qe_by_seed.append(result[4])
                te_by_seed.append(result[5])
                if seed == PRIMARY_SEED:
                    primary = result
            if primary is None:
                raise RuntimeError("Primary SOM seed was not run")
            mean_seed_ari, min_seed_ari = pairwise_seed_ari(labels_by_seed)
            candidate = build_candidate_metrics(
                feature_name,
                features,
                descriptors,
                curves,
                n_nodes,
                *primary,
                mean_seed_ari,
                min_seed_ari,
            )
            candidate.metrics["qe_mean_across_seeds"] = float(np.mean(qe_by_seed))
            candidate.metrics["qe_std_across_seeds"] = float(np.std(qe_by_seed))
            candidate.metrics["te_mean_across_seeds"] = float(np.mean(te_by_seed))
            candidates[(feature_name, n_nodes)] = candidate
            metrics_rows.append(candidate.metrics)

            n_dir = feature_dir / f"n_{n_nodes:02d}"
            n_dir.mkdir(parents=True, exist_ok=True)
            assignments = metadata.copy()
            assignments["cluster_id"] = candidate.labels
            assignments["som_node_x"] = candidate.coords[:, 0]
            assignments["som_node_y"] = candidate.coords[:, 1]
            assignments["bmu_distance"] = candidate.distances
            assignments["rule_shape_category"] = descriptors[
                "rule_shape_category"
            ].to_numpy()
            assignments.to_csv(n_dir / "cluster_assignments.csv", index=False)
            candidate.cluster_summary.to_csv(n_dir / "cluster_summary.csv", index=False)
            pd.DataFrame(
                candidate.curve_centroids,
                columns=[f"t_{int(hour):03d}h" for hour in MODEL_GRID],
            ).assign(cluster_id=np.arange(n_nodes)).to_csv(
                n_dir / "curve_centroids.csv", index=False
            )
            np.save(n_dir / "som_weights.npy", candidate.weights)
            with (n_dir / "som_model.pkl").open("wb") as handle:
                pickle.dump(candidate.som, handle)
            plot_candidate_clusters(candidate, curves, descriptors, n_dir)

        feature_metrics = pd.DataFrame(
            [row for row in metrics_rows if row["feature_set"] == feature_name]
        ).sort_values("n_nodes")
        feature_metrics.to_csv(feature_dir / "som_metrics.csv", index=False)
        plot_feature_scan(feature_metrics, candidates, feature_name, feature_dir)

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["feature_set", "n_nodes"]
    ).reset_index(drop=True)
    metrics.to_csv(root / "all_feature_set_som_metrics.csv", index=False)
    return metrics, candidates


def plot_candidate_clusters(
    candidate: SOMCandidate,
    curves: np.ndarray,
    descriptors: pd.DataFrame,
    out: Path,
) -> None:
    n = candidate.n_nodes
    cols = 2 if n <= 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.2 * cols, 4.0 * rows), sharex=True, sharey=True, squeeze=False
    )
    rng = np.random.RandomState(42)
    for cluster_id, ax in enumerate(axes.flat):
        if cluster_id >= n:
            ax.axis("off")
            continue
        indices = np.flatnonzero(candidate.labels == cluster_id)
        draw = rng.choice(indices, min(300, len(indices)), replace=False)
        for index in draw:
            ax.plot(MODEL_GRID, curves[index], color="#94A3B8", alpha=0.08, linewidth=0.5)
        if indices.size:
            q25 = np.percentile(curves[indices], 25, axis=0)
            q75 = np.percentile(curves[indices], 75, axis=0)
            mean = np.mean(curves[indices], axis=0)
            median = np.median(curves[indices], axis=0)
            ax.fill_between(MODEL_GRID, q25, q75, color="#93C5FD", alpha=0.3)
            ax.plot(MODEL_GRID, mean, color="#1D4ED8", linewidth=2.0, label="mean")
            ax.plot(
                MODEL_GRID,
                median,
                color="#DC2626",
                linewidth=1.2,
                linestyle="--",
                label="median",
            )
            category = descriptors.loc[indices, "rule_shape_category"].value_counts()
            dominant = str(category.index[0])
            purity = float(category.iloc[0] / len(indices))
            ax.set_title(
                f"Cluster {cluster_id}: n={len(indices)}\n{dominant} ({purity:.0%})"
            )
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("PCHIP normalized PCE")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{candidate.feature_set}: SOM n={n}")
    save_figure(fig, out / "cluster_curves.png")


def plot_feature_scan(
    metrics: pd.DataFrame,
    candidates: dict[tuple[str, int], SOMCandidate],
    feature_name: str,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(metrics["n_nodes"], metrics["qe_mean_across_seeds"], marker="o")
    axes[0, 0].set_title("Quantisation error")
    axes[0, 0].set_xlabel("n")
    axes[0, 0].set_ylabel("QE")
    axes[0, 1].plot(
        metrics["n_nodes"], metrics["mean_pairwise_seed_ARI"], marker="o", color="#059669"
    )
    axes[0, 1].axhline(0.8, color="#64748B", linestyle=":")
    axes[0, 1].set_ylim(0, 1.03)
    axes[0, 1].set_title("Seed stability")
    axes[0, 1].set_xlabel("n")
    axes[0, 1].set_ylabel("Mean pairwise ARI")
    axes[1, 0].plot(
        metrics["n_nodes"], metrics["silhouette_pca20"], marker="o", color="#EA580C"
    )
    axes[1, 0].set_title("Silhouette (PCA20)")
    axes[1, 0].set_xlabel("n")
    axes[1, 0].set_ylabel("Silhouette")
    axes[1, 1].plot(
        metrics["n_nodes"], metrics["rule_shape_category_NMI"], marker="o", color="#7C3AED"
    )
    axes[1, 1].set_title("Agreement with diagnostic shape rules")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].set_ylabel("NMI")
    fig.suptitle(feature_name)
    save_figure(fig, out / "som_scan_metrics.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True, sharey=True)
    for ax, n_nodes in zip(axes.flat, NODES):
        candidate = candidates[(feature_name, n_nodes)]
        order = np.argsort(candidate.curve_centroids[:, -1])[::-1]
        for rank, cluster_id in enumerate(order):
            ax.plot(
                MODEL_GRID,
                candidate.curve_centroids[cluster_id],
                linewidth=1.7,
                label=f"C{cluster_id} n={(candidate.labels == cluster_id).sum()}",
            )
        ax.set_title(f"n={n_nodes}")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.legend(fontsize=6)
    for ax in axes.flat[len(NODES) :]:
        ax.axis("off")
    fig.suptitle(f"Centroid overview — {feature_name}")
    save_figure(fig, out / "centroid_overview_n4_to_n10.png")


def add_selection_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["eligible"] = (
        (frame["n_empty_nodes"] == 0)
        & (frame["min_cluster_fraction"] >= 0.02)
        & (frame["mean_pairwise_seed_ARI"] >= 0.90)
        & (frame["initial_gain_best_precision"] >= 0.80)
        & (frame["initial_gain_best_count"] >= 20)
        & (frame["early_drop_best_precision"] >= 0.80)
        & (frame["early_drop_best_count"] >= 20)
    )
    frame["selection_score"] = np.nan
    for feature_name, group in frame.groupby("feature_set"):
        index = group.index

        def minmax(column: str, higher: bool = True) -> np.ndarray:
            values = group[column].to_numpy(dtype=float)
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
            if high - low < 1e-12:
                score = np.ones_like(values)
            else:
                score = (values - low) / (high - low)
            return score if higher else 1.0 - score

        score = (
            0.15 * minmax("silhouette_pca20")
            + 0.20 * minmax("rule_shape_category_NMI")
            + 0.20 * minmax("mean_pairwise_seed_ARI")
            + 0.15 * minmax("kmeans_ARI")
            + 0.10 * minmax("min_curve_centroid_RMSE")
            + 0.15 * minmax("initial_gain_best_f1")
            + 0.05 * minmax("early_drop_best_f1")
        )
        frame.loc[index, "selection_score"] = score
    frame.loc[~frame["eligible"], "selection_score"] = np.nan
    return frame


def select_and_export_model(
    metrics: pd.DataFrame,
    candidates: dict[tuple[str, int], SOMCandidate],
    features: dict[str, np.ndarray],
    descriptors: pd.DataFrame,
    curves: np.ndarray,
    metadata: pd.DataFrame,
    curves_by_id: dict[str, RawCurve],
    logger: logging.Logger,
) -> tuple[str, int, SOMCandidate, pd.DataFrame]:
    scored = add_selection_scores(metrics)
    decision_dir = OUTPUT / "05_selected_model"
    decision_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(decision_dir / "model_selection_scores.csv", index=False)

    preferred = scored[
        (scored["feature_set"] == "curve_derivative_descriptors")
        & scored["eligible"]
    ].sort_values(["selection_score", "n_nodes"], ascending=[False, True])
    if preferred.empty:
        raise RuntimeError("No eligible composite-feature SOM candidate")
    selected_row = preferred.iloc[0]
    feature_name = str(selected_row["feature_set"])
    n_nodes = int(selected_row["n_nodes"])
    candidate = candidates[(feature_name, n_nodes)]

    assignments = metadata.copy()
    assignments["cluster_id"] = candidate.labels
    assignments["som_node_x"] = candidate.coords[:, 0]
    assignments["som_node_y"] = candidate.coords[:, 1]
    assignments["bmu_distance"] = candidate.distances
    assignments = assignments.merge(
        descriptors.drop(columns=["array_row"]), on="curve_id", how="left"
    )
    assignments.to_csv(decision_dir / "final_curve_assignments.csv", index=False)
    candidate.cluster_summary.to_csv(
        decision_dir / "final_cluster_summary.csv", index=False
    )
    np.save(decision_dir / "final_som_weights.npy", candidate.weights)
    np.save(decision_dir / "final_feature_matrix.npy", features[feature_name])
    with (decision_dir / "final_som_model.pkl").open("wb") as handle:
        pickle.dump(candidate.som, handle)
    (decision_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "feature_set": feature_name,
                "n_nodes": n_nodes,
                "topology": list(candidate.topology),
                "primary_seed": PRIMARY_SEED,
                "selection_score": float(selected_row["selection_score"]),
                "selection_rule": (
                    "Within the prespecified composite feature set, require no "
                    "empty node, minimum cluster fraction >=2%, mean seed ARI "
                    ">=0.90, and distinct initial-gain and early-drop clusters "
                    "with >=80% precision and >=20 target curves; then select "
                    "the highest composite score."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_final_validation(candidate, assignments, curves, curves_by_id, decision_dir)
    logger.info("Selected shape-sensitive model: %s n=%d", feature_name, n_nodes)
    return feature_name, n_nodes, candidate, assignments


def plot_final_validation(
    candidate: SOMCandidate,
    assignments: pd.DataFrame,
    curves: np.ndarray,
    curves_by_id: dict[str, RawCurve],
    out: Path,
) -> None:
    n = candidate.n_nodes
    cols = 2 if n <= 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.5 * cols, 4.3 * rows), sharex=True, sharey=True, squeeze=False
    )
    for cluster_id, ax in enumerate(axes.flat):
        if cluster_id >= n:
            ax.axis("off")
            continue
        indices = np.flatnonzero(candidate.labels == cluster_id)
        for index in indices:
            ax.plot(MODEL_GRID, curves[index], color="#94A3B8", alpha=0.05, linewidth=0.45)
            curve = curves_by_id[assignments.iloc[index]["curve_id"]]
            x, y, _ = _deduplicate_mean(curve.x_hours_relative, curve.y)
            mask = (x >= 0) & (x <= 200 + 1e-9)
            denominator = float(assignments.iloc[index]["pchip_max_pce"])
            ax.scatter(
                x[mask],
                y[mask] / denominator,
                s=2.5,
                color="#111827",
                alpha=0.08,
                linewidths=0,
            )
        if indices.size:
            mean = np.mean(curves[indices], axis=0)
            median = np.median(curves[indices], axis=0)
            q25 = np.percentile(curves[indices], 25, axis=0)
            q75 = np.percentile(curves[indices], 75, axis=0)
            ax.fill_between(MODEL_GRID, q25, q75, color="#93C5FD", alpha=0.35)
            ax.plot(MODEL_GRID, mean, color="#1D4ED8", linewidth=2.2, label="mean")
            ax.plot(
                MODEL_GRID,
                median,
                color="#DC2626",
                linewidth=1.4,
                linestyle="--",
                label="median",
            )
        ax.set_title(f"Cluster {cluster_id} — n={len(indices)}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Selected model: PCHIP curves and original observed points")
    save_figure(fig, out / "final_clusters_with_raw_points.png")

    fig, ax = plt.subplots(figsize=(10, 6.2))
    order = np.argsort(candidate.curve_centroids[:, -1])[::-1]
    for cluster_id in order:
        ax.plot(
            MODEL_GRID,
            candidate.curve_centroids[cluster_id],
            linewidth=2.2,
            label=f"Cluster {cluster_id} (n={(candidate.labels == cluster_id).sum()})",
        )
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("PCHIP normalized PCE")
    ax.set_title("Shape-sensitive SOM cluster centroids")
    ax.legend()
    save_figure(fig, out / "final_cluster_centroids.png")

    feature_matrix = np.load(out / "final_feature_matrix.npy")
    flat_weights = candidate.weights.reshape(n, feature_matrix.shape[1])
    representatives: list[dict] = []
    for cluster_id in range(n):
        indices = np.flatnonzero(candidate.labels == cluster_id)
        distances = np.linalg.norm(feature_matrix[indices] - flat_weights[cluster_id], axis=1)
        order_local = indices[np.argsort(distances)]
        for rank, index in enumerate(order_local[:5], start=1):
            representatives.append(
                {
                    "cluster_id": cluster_id,
                    "representative_type": "closest",
                    "rank": rank,
                    "array_row": int(index),
                    "curve_id": assignments.iloc[index]["curve_id"],
                    "distance": float(np.linalg.norm(feature_matrix[index] - flat_weights[cluster_id])),
                }
            )
        for rank, index in enumerate(order_local[-5:][::-1], start=1):
            representatives.append(
                {
                    "cluster_id": cluster_id,
                    "representative_type": "farthest",
                    "rank": rank,
                    "array_row": int(index),
                    "curve_id": assignments.iloc[index]["curve_id"],
                    "distance": float(np.linalg.norm(feature_matrix[index] - flat_weights[cluster_id])),
                }
            )
    pd.DataFrame(representatives).to_csv(
        out / "closest_and_farthest_curves.csv", index=False
    )


def compare_with_baseline(
    assignments: pd.DataFrame,
    selected_feature: str,
    selected_n: int,
) -> dict[str, float | int | str]:
    out = OUTPUT / "07_baseline_comparison"
    out.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(PROJECT / "06_final_model" / "final_curve_classification.csv")
    merged = assignments.merge(
        baseline[["curve_id", "class_label", "raw_cluster_id"]],
        on="curve_id",
        how="left",
        suffixes=("_new", "_baseline"),
    )
    merged.to_csv(out / "old_vs_new_curve_assignments.csv", index=False)
    ari = float(adjusted_rand_score(merged["raw_cluster_id"], merged["cluster_id"]))
    cross = pd.crosstab(
        merged["class_label"], merged["cluster_id"], margins=True
    )
    cross.to_csv(out / "old_vs_new_crosstab.csv")
    summary: dict[str, float | int | str] = {
        "n_common_curves": int(len(merged)),
        "baseline_n": 4,
        "new_n": selected_n,
        "new_feature_set": selected_feature,
        "ARI_old_vs_new": ari,
    }
    (out / "baseline_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_report(
    qc_audit: pd.DataFrame,
    strict: pd.DataFrame,
    bundle: PreprocessingBundle,
    metrics: pd.DataFrame,
    selected_feature: str,
    selected_n: int,
    candidate: SOMCandidate,
    assignments: pd.DataFrame,
    baseline_comparison: dict[str, float | int | str],
) -> None:
    selected_metrics = metrics[
        (metrics["feature_set"] == selected_feature)
        & (metrics["n_nodes"] == selected_n)
    ].iloc[0]
    shape_counts = assignments["rule_shape_category"].value_counts()
    display_summary = candidate.cluster_summary.copy()
    numeric_columns = display_summary.select_dtypes(include=[np.number]).columns
    display_summary[numeric_columns] = display_summary[numeric_columns].round(4)
    headers = [str(column) for column in display_summary.columns]
    cluster_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_summary.itertuples(index=False, name=None):
        cluster_lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    cluster_table = "\n".join(cluster_lines)
    lines = [
        "# 形状敏感 200 h SOM 重分析报告",
        "",
        "## 结论摘要",
        "",
        f"- 原主分析曲线：**{len(qc_audit):,}** 条。",
        f"- 严格时间覆盖 QC 后：**{len(strict):,}** 条。",
        f"- 最终特征表示：`{selected_feature}`。",
        f"- 最终 SOM 节点数：**n={selected_n}**。",
        f"- 多随机种子平均 ARI：**{selected_metrics['mean_pairwise_seed_ARI']:.4f}**。",
        f"- PCA20 silhouette：**{selected_metrics['silhouette_pca20']:.4f}**。",
        f"- 与 K-means 的 ARI：**{selected_metrics['kmeans_ARI']:.4f}**。",
        f"- 与旧 n=4 结果的 ARI：**{baseline_comparison['ARI_old_vs_new']:.4f}**。",
        "",
        "## 数据质量控制",
        "",
        f"采用最大相邻插值支撑间隔 ≤{MAX_SUPPORT_GAP_H:.0f} h。",
        "y 轴标签异常只进入人工复核清单，不因元数据拼写问题自动删除。",
        "",
        "## 预处理对照",
        "",
        "同时保存 Akima+平滑、Akima无平滑、PCHIP无平滑和线性无平滑版本。",
        "最终模型使用 PCHIP 无平滑曲线；PCHIP 在单调区间不制造额外转折，",
        "并通过原始点叠加图核查插值保真度。",
        "",
        "## 形状敏感表示",
        "",
        "最终复合特征由时间加权曲线、每5 h导数采样和21个形状描述参数组成。",
        "三部分距离贡献预设为50%、30%和20%。0–20 h、20–50 h、50–100 h、",
        "100–200 h的曲线距离权重分别为8、4、2和1。",
        "",
        "规则形状类别仅用于诊断聚类是否捕捉到局部行为，不参与监督标签训练：",
        "",
    ]
    for label, count in shape_counts.items():
        lines.append(f"- `{label}`：{count:,} 条")
    lines.extend(
        [
            "",
            "## 最终聚类摘要",
            "",
            cluster_table,
            "",
            "## 模型选择规则",
            "",
            "在复合特征预设下，候选模型必须无空节点、最小类别≥2%、",
            "平均多种子 ARI≥0.90，并且初始增益类与早期骤降类均达到≥80%纯度",
            "且至少包含20条目标曲线；再综合 silhouette、形状规则 NMI、种子稳定性、",
            "K-means 一致性、局部形状 F1 和中心曲线分离度选择。没有强制 n=4。",
            "",
            "## 重要文件",
            "",
            "- `01_data_qc/strict_selection_audit.csv`：严格 QC 全记录。",
            "- `02_preprocessing_comparison/`：四种预处理数组与保真图。",
            "- `03_shape_features/shape_descriptors.csv`：逐曲线形状参数。",
            "- `04_som_n4_to_n10/`：所有特征集及 n=4–10 结果。",
            "- `05_selected_model/final_clusters_with_raw_points.png`：原始点验证。",
            "- `07_baseline_comparison/`：旧模型与新模型逐曲线对照。",
            "",
        ]
    )
    (OUTPUT / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def write_checksums() -> None:
    lines: list[str] = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    logger = configure_logging()
    config = AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )
    run_config = {
        "source_dataset": "main_high_quality_min10",
        "window_hours": 200.0,
        "max_support_gap_h": MAX_SUPPORT_GAP_H,
        "fine_grid_step_h": float(FINE_GRID[1] - FINE_GRID[0]),
        "model_grid_step_h": float(MODEL_GRID[1] - MODEL_GRID[0]),
        "interpolation_variants": ["akima", "pchip", "linear"],
        "selected_preprocessing": "pchip_normalized_no_smoothing",
        "time_weights": {"0_20h": 8, "20_50h": 4, "50_100h": 2, "100_200h": 1},
        "feature_block_weights": {"curve": 0.50, "derivative": 0.30, "descriptors": 0.20},
        "som_nodes": list(NODES),
        "som_seeds": list(SEEDS),
        "som_primary_seed": PRIMARY_SEED,
        "som_iterations": SOM_ITERATIONS,
        "som_sigma": 0.5,
        "som_learning_rate": 0.1,
        "som_distance": "euclidean_on_shape_sensitive_composite_features",
    }
    (OUTPUT / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Discovering raw curves")
    curves = discover_curves(config)
    curves_by_id = {curve.curve_id: curve for curve in curves}
    selected = pd.read_csv(PROJECT / "01_data_selection" / "main_selected_curves.csv")
    qc_audit, strict = build_qc_audit(selected, curves_by_id, logger)
    bundle = preprocess_strict_curves(strict, curves_by_id, logger)
    feature_sets, descriptors, model_curves = build_feature_sets(bundle, logger)
    metrics, candidates = scan_som_models(
        feature_sets,
        descriptors,
        model_curves,
        bundle.metadata,
        logger,
    )
    selected_feature, selected_n, candidate, assignments = select_and_export_model(
        metrics,
        candidates,
        feature_sets,
        descriptors,
        model_curves,
        bundle.metadata,
        curves_by_id,
        logger,
    )
    baseline = compare_with_baseline(assignments, selected_feature, selected_n)
    write_report(
        qc_audit,
        strict,
        bundle,
        metrics,
        selected_feature,
        selected_n,
        candidate,
        assignments,
        baseline,
    )
    logger.info("Shape-sensitive reanalysis complete: %s", OUTPUT)
    write_checksums()


if __name__ == "__main__":
    main()
