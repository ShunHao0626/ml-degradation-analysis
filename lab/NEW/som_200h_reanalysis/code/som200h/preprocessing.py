"""Paper-aligned preprocessing with a non-extrapolated 200 h endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

from .config import AnalysisConfig
from .data import RawCurve


@dataclass
class PreprocessedDataset:
    name: str
    metadata: pd.DataFrame
    time_grid: np.ndarray
    x_interpolated: np.ndarray
    x_normalized: np.ndarray
    x_smoothed: np.ndarray


def _deduplicate_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(x)
    x_sorted = np.asarray(x[order], dtype=float)
    y_sorted = np.asarray(y[order], dtype=float)
    unique, inverse = np.unique(x_sorted, return_inverse=True)
    sums = np.zeros(unique.size, dtype=float)
    counts = np.zeros(unique.size, dtype=float)
    np.add.at(sums, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    averaged = sums / counts
    return unique, averaged, int(x_sorted.size - unique.size)


def _support_through_first_point_after_window(
    x: np.ndarray, y: np.ndarray, limit: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep 0..limit plus the first point after limit for endpoint interpolation."""
    within_indices = np.flatnonzero(x <= limit + 1e-9)
    after_indices = np.flatnonzero(x > limit + 1e-9)
    keep = within_indices.tolist()
    bracket_time = limit
    if after_indices.size:
        first_after = int(after_indices[0])
        keep.append(first_after)
        bracket_time = float(x[first_after])
    support_x = x[keep]
    support_y = y[keep]
    return support_x, support_y, bracket_time


def preprocess_curve(
    curve: RawCurve,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    limit = config.window_hours
    grid = np.linspace(0.0, limit, config.n_grid_points, endpoint=True)
    x_unique, y_unique, n_duplicates = _deduplicate_mean(
        curve.x_hours_relative, curve.y
    )
    x_support, y_support, bracket_time = _support_through_first_point_after_window(
        x_unique, y_unique, limit
    )
    if x_support.size < config.paper_min_unique_points:
        raise ValueError("insufficient_unique_points_for_akima")
    if x_support[0] > 1e-9 or x_support[-1] < limit - 1e-9:
        raise ValueError("window_not_bracketed")

    interpolator = Akima1DInterpolator(x_support, y_support)
    y_interpolated = np.asarray(
        interpolator(grid, extrapolate=False), dtype=float
    )
    if not np.isfinite(y_interpolated).all():
        raise ValueError("nonfinite_after_akima")

    max_pce = float(np.max(y_interpolated))
    if not np.isfinite(max_pce) or max_pce <= 0:
        raise ValueError("invalid_0_200h_max_pce")
    y_normalized = y_interpolated / max_pce
    y_smoothed = savgol_filter(
        y_normalized,
        window_length=config.savgol_window,
        polyorder=config.savgol_polyorder,
        mode=config.savgol_mode,
    )
    if not np.isfinite(y_smoothed).all():
        raise ValueError("nonfinite_after_savgol")

    meta = {
        "curve_id": curve.curve_id,
        "sample_id": curve.sample_id,
        "source_absolute_path": str(curve.source_path),
        "source_relative_path": curve.source_relative_path,
        "doi": curve.doi,
        "figure_folder": curve.figure_folder,
        "series_id": curve.series_id,
        "n_raw_points": int(curve.y.size),
        "n_unique_points_total": int(x_unique.size),
        "n_unique_points_0_200h": int(np.sum(x_unique <= limit + 1e-9)),
        "n_duplicate_points": n_duplicates,
        "duration_h": float(np.max(x_unique) - np.min(x_unique)),
        "endpoint_bracket_time_h": bracket_time,
        "endpoint_uses_point_after_200h": bool(bracket_time > limit + 1e-9),
        "max_pce_0_200h": max_pce,
        "normalized_pce_0h": float(y_normalized[0]),
        "normalized_pce_200h": float(y_normalized[-1]),
    }
    return y_interpolated, y_normalized, y_smoothed, meta


def build_preprocessed_dataset(
    name: str,
    curves_by_id: dict[str, RawCurve],
    selected_audit: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[PreprocessedDataset, pd.DataFrame]:
    interpolated: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    smoothed: list[np.ndarray] = []
    metadata: list[dict] = []
    failures: list[dict] = []

    for row in selected_audit.sort_values("curve_id").itertuples(index=False):
        curve = curves_by_id[row.curve_id]
        try:
            yi, yn, ys, meta = preprocess_curve(curve, config)
        except Exception as exc:
            failures.append(
                {
                    "curve_id": row.curve_id,
                    "sample_id": row.sample_id,
                    "source_absolute_path": row.source_absolute_path,
                    "failure_reason": str(exc),
                }
            )
            continue
        meta["array_row"] = len(metadata)
        interpolated.append(yi)
        normalized.append(yn)
        smoothed.append(ys)
        metadata.append(meta)

    time_grid = np.linspace(
        0.0, config.window_hours, config.n_grid_points, endpoint=True
    )
    dataset = PreprocessedDataset(
        name=name,
        metadata=pd.DataFrame(metadata),
        time_grid=time_grid,
        x_interpolated=np.stack(interpolated),
        x_normalized=np.stack(normalized),
        x_smoothed=np.stack(smoothed),
    )
    return dataset, pd.DataFrame(failures)


def save_preprocessed_dataset(
    dataset: PreprocessedDataset,
    failures: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    out = config.output_root / "03_preprocessed" / dataset.name
    out.mkdir(parents=True, exist_ok=True)
    dataset.metadata.to_csv(out / "curve_metadata.csv", index=False)
    failures.to_csv(out / "preprocessing_failures.csv", index=False)
    pd.DataFrame({"time_h": dataset.time_grid}).to_csv(
        out / "time_grid.csv", index=False
    )
    np.save(out / "X_interpolated.npy", dataset.x_interpolated)
    np.save(out / "X_normalized.npy", dataset.x_normalized)
    np.save(out / "X_smoothed.npy", dataset.x_smoothed)

    columns = [f"t_{i:04d}" for i in range(dataset.time_grid.size)]
    wide = pd.DataFrame(dataset.x_smoothed, columns=columns)
    wide.insert(0, "curve_id", dataset.metadata["curve_id"].to_numpy())
    wide.to_csv(out / "preprocessed_smoothed_curves.csv", index=False)

    normalized_wide = pd.DataFrame(dataset.x_normalized, columns=columns)
    normalized_wide.insert(0, "curve_id", dataset.metadata["curve_id"].to_numpy())
    normalized_wide.to_csv(out / "preprocessed_normalized_curves.csv", index=False)

    details = {
        "dataset_name": dataset.name,
        "n_curves": int(dataset.x_smoothed.shape[0]),
        "n_time_points": int(dataset.x_smoothed.shape[1]),
        "time_window_hours": [0.0, config.window_hours],
        "grid_step_hours": config.grid_step_hours,
        "interpolation": "scipy.interpolate.Akima1DInterpolator",
        "endpoint_policy": (
            "retain the first observed point after 200 h to bracket t=200 h; "
            "no flat tail fill and no extrapolation"
        ),
        "normalization": "divide each curve by its own interpolated max PCE in 0-200 h",
        "savgol": {
            "window_length": config.savgol_window,
            "polyorder": config.savgol_polyorder,
            "mode": config.savgol_mode,
        },
    }
    (out / "preprocessing_config.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
