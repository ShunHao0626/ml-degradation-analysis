"""
src/preprocessing.py
====================
Preprocessing pipeline for MPPT ageing curves.

Strictly follows the order used in the author's notebook
(20230816_degradation_analysis_revision_11_cleaned.ipynb):

  1. Convert time to hours
  2. Relative ageing time (shift to start at t=0)
  3. Keep only 0 ≤ t ≤ 200 h
  4. Sort ascending by time
  5. Resolve duplicate timestamps (mean)
  6. Resample to 10-min uniform grid
  7. Akima interpolation for internal gaps
  8. Per-curve MaxAbsScaler normalisation (divide by own 0–200h max)
  9. Savitzky–Golay smoothing
  10. Assemble into feature matrix

Exclusion rules (aligned with specification Section IV):
  - No valid device ID
  - No valid time or PCE
  - Maximum time < 200 h (curve is too short)
  - Insufficient data to support Akima / resampling
  - Max PCE ≤ 0 in 0–200h window
  - NaN/Inf after interpolation
  - Unique time points < min_unique_points

Per-curve normalisation (Section V):
  Each curve is divided by its OWN maximum PCE within [0, 200h].
  No global or feature-wise scaling is applied.

References:
  - Author notebook cell 33–37: preprocessing and SOM
  - Author notebook cell 14: MaxAbsScaler (row-wise)
  - Author notebook cell 23–24: Savitzky-Golay (window=71, polyorder=2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from sklearn.preprocessing import MaxAbsScaler

from .config import PREPROC_CONFIG, DATA_CONFIG, Y_NORMALIZED_MAX, logger

# =============================================================================
# QC result dataclass
# =============================================================================

@dataclass
class CurveQCResult:
    status: str          # "included" or "excluded"
    reason: str          # human-readable reason
    n_original_points: int = 0
    pce_max_0_200h: float = np.nan
    pce_min: float = np.nan
    max_x_hours: float = np.nan
    n_unique_t: int = 0
    t_coverage_pct: float = np.nan
    # Preprocessed arrays (only if status == "included")
    x_resampled: Optional[np.ndarray] = None
    y_raw_interp: Optional[np.ndarray] = None   # after Akima, before normalisation
    y_normalized: Optional[np.ndarray] = None   # after MaxAbsScaler
    y_smoothed: Optional[np.ndarray] = None     # after Savitzky-Golay


# =============================================================================
# Time grid
# =============================================================================

def build_time_grid() -> np.ndarray:
    """
    Build the uniform 10-min grid for [0, 200] h.

    Returns:
        np.ndarray of shape (1201,) with values [0, 1/6, 2/6, ..., 200].

    1201 = 200 × 6 + 1  (10 min = 1/6 h)
    """
    hour_limit = PREPROC_CONFIG["hour_limit"]
    n_points   = PREPROC_CONFIG["n_time_points"]
    return np.linspace(0.0, hour_limit, n_points, endpoint=True)


TIME_GRID = build_time_grid()   # module-level singleton


# =============================================================================
# Per-curve QC & preprocessing
# =============================================================================

def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill leading and trailing NaN regions.
    Uses the first/last valid value to fill outward.
    """
    arr = arr.copy()
    mask = np.isnan(arr)

    if not mask.any():
        return arr

    # Forward fill from first valid value
    first_valid = np.argmax(~mask)
    if first_valid > 0:
        arr[:first_valid] = arr[first_valid]

    # Backward fill from last valid value
    last_valid = len(arr) - 1 - np.argmax(~mask[::-1])
    if last_valid < len(arr) - 1:
        arr[last_valid + 1:] = arr[last_valid]

    return arr


def _resolve_duplicates(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort by time and average PCE values at duplicate timestamps.

    Returns:
        (sorted_x, averaged_y)
    """
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Group by unique x values
    _, unique_idx, counts = np.unique(xs, return_index=True, return_counts=True)
    if (counts == 1).all():
        return xs, ys

    # Average duplicate y values
    ys_avg = np.zeros_like(ys)
    for i, (start, count) in enumerate(zip(unique_idx, counts)):
        ys_avg[start] = ys[start:start + count].mean()
        if count > 1:
            ys_avg[start + 1:start + count] = np.nan   # mark as duplicate

    # Remove duplicates
    keep = ys_avg != np.nan
    # Use mask approach
    keep = np.ones(len(xs), dtype=bool)
    for i, (start, count) in enumerate(zip(unique_idx, counts)):
        if count > 1:
            keep[start + 1:start + count] = False

    return xs[keep], ys_avg[keep]


def preprocess_curve(
    curve_id: str,
    xs_raw: np.ndarray,
    ys_raw: np.ndarray,
    unit_label: str,
    max_x_hours: float,
    include_short: bool = False,
) -> CurveQCResult:
    """
    Run the full QC + preprocessing pipeline for a single curve.

    Args:
        curve_id:      Unique curve identifier (csv_file path)
        xs_raw:        Raw time values (in whatever original unit)
        ys_raw:        Raw PCE values
        unit_label:    Time unit label for conversion to hours
        max_x_hours:   Maximum time in hours (pre-computed)
        include_short: If False, exclude curves whose max_x_hours < 200h.
                       If True, include short curves and forward-fill
                       missing tail region (for sensitivity analysis).

    Returns:
        CurveQCResult with all metadata and processed arrays.

    Processing order (matches author notebook exactly):
      1. Convert time to hours
      2. Shift to t=0 (relative ageing)
      3. Keep 0 ≤ t ≤ HOUR_LIMIT
      4. Sort ascending
      5. Deduplicate (average)
      6. Resample to 10-min grid (Akima; forward-fill for short curves)
      7. Per-curve MaxAbsScaler normalisation
      8. Savitzky-Golay smoothing
    """
    from .config import PREPROC_CONFIG, UNIT_TO_HOURS

    cfg = PREPROC_CONFIG
    hour_limit = cfg["hour_limit"]

    n_orig = len(xs_raw)
    result = CurveQCResult(
        status="excluded",
        reason="",
        n_original_points=n_orig,
    )

    # ── Step 1: Convert time to hours ─────────────────────────────
    factor = UNIT_TO_HOURS.get(unit_label, 1.0)
    xs_h = xs_raw * factor
    ys = np.array(ys_raw, dtype=float)

    if len(xs_h) == 0 or len(ys) == 0:
        result.reason = "Empty data"
        return result

    # ── Step 2: Relative ageing time (shift to t=0) ─────────────
    t_min = xs_h.min()
    xs_h = xs_h - t_min

    # ── Step 3: Keep only 0 ≤ t ≤ HOUR_LIMIT ─────────────────────
    mask_0_200 = (xs_h >= 0) & (xs_h <= hour_limit)
    xs_win = xs_h[mask_0_200]
    ys_win = ys[mask_0_200]

    if len(xs_win) == 0:
        result.reason = "No data in [0, 200h] window"
        result.max_x_hours = max_x_hours
        return result

    # ── Step 4: Sort ascending by time ──────────────────────────
    order = np.argsort(xs_win)
    xs_win = xs_win[order]
    ys_win = ys_win[order]

    # ── Step 5: Resolve duplicate timestamps ─────────────────────
    xs_win, ys_win = _resolve_duplicates(xs_win, ys_win)

    # ── Quality checks ──────────────────────────────────────────
    n_unique_t = len(np.unique(xs_win))
    t_coverage_pct = (xs_win.max() - xs_win.min()) / hour_limit * 100 \
        if len(xs_win) > 0 else 0.0

    result.n_unique_t = n_unique_t
    result.t_coverage_pct = t_coverage_pct
    result.max_x_hours = xs_win.max()

    min_pts = cfg["min_unique_points"]
    if n_unique_t < min_pts:
        result.reason = (
            f"Too few unique time points ({n_unique_t}) in [0,200]h "
            f"window (data spans {t_coverage_pct:.0f}% of window)"
        )
        return result

    if include_short:
        # Allow short curves through; will forward-fill tail
        pass
    elif max_x_hours < hour_limit:
        result.reason = (
            f"Curve max time {max_x_hours:.1f}h < {hour_limit}h "
            f"(curve too short)"
        )
        return result

    # ── Step 6: Resample to 10-min grid ─────────────────────────
    t_grid = TIME_GRID

    try:
        akima = Akima1DInterpolator(xs_win, ys_win)
        y_interp_raw = akima(t_grid)   # NaN outside xs_win range
    except Exception:
        result.reason = "Akima interpolation failed"
        return result

    # ── Forward-fill NaN tail for short curves ──────────────────
    if include_short or np.any(np.isnan(y_interp_raw)):
        y_interp_raw = _forward_fill_nan(y_interp_raw)

    # ── Post-interpolation QC ────────────────────────────────────
    if np.any(np.isnan(y_interp_raw)) or np.any(np.isinf(y_interp_raw)):
        result.reason = "NaN/Inf after Akima interpolation"
        return result

    # ── Step 8: Per-curve MaxAbsScaler normalisation ────────────
    # Divide each curve by its own maximum PCE in [0, 200h].
    # Author's code: MaxAbsScaler().fit_transform(mySeriesDrop[i])
    # Here we use the confirmed row-wise MaxAbsScaler approach.
    pce_max_200 = float(np.nanmax(y_interp_raw))
    result.pce_max_0_200h = pce_max_200

    if not np.isfinite(pce_max_200) or pce_max_200 <= 0:
        result.reason = f"Invalid max PCE in [0,200h]: {pce_max_200}"
        return result

    y_normalized = y_interp_raw / pce_max_200
    result.pce_min = float(np.nanmin(y_normalized))

    # Sanity check: normalized max should be ~1.0
    if not (0.99 <= y_normalized.max() <= 1.01):
        result.reason = (
            f"Normalisation failed: max(normalized) = {y_normalized.max():.4f} "
            f"(expected ~1.0)"
        )
        return result

    # ── Step 9: Savitzky–Golay smoothing ────────────────────────
    window = cfg["savgol_window"]
    polyorder = cfg["savgol_polyorder"]
    mode = cfg["savgol_mode"]

    if window > len(y_normalized):
        result.reason = (
            f"Savgol window ({window}) > curve length ({len(y_normalized)})"
        )
        return result

    try:
        y_smoothed = savgol_filter(
            y_normalized,
            window_length=window,
            polyorder=polyorder,
            mode=mode,
        )
    except Exception:
        result.reason = "Savitzky-Golay smoothing failed"
        return result

    # ── Store results ────────────────────────────────────────────
    result.status = "included"
    result.reason = "OK"
    result.x_resampled = t_grid
    result.y_raw_interp = y_interp_raw
    result.y_normalized = y_normalized
    result.y_smoothed = y_smoothed

    return result


# =============================================================================
# Batch preprocessing
# =============================================================================

def preprocess_all_curves(
    df: pd.DataFrame,
    include_short: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess all curves from the loaded DataFrame.

    Args:
        df: DataFrame with columns from data_loading.py (csv_file, x, y, unit_label, etc.)
        include_short: If True, include curves < 200h with forward-fill.
                       If False (default), exclude short curves.

    Returns:
        (included_df, excluded_df, qc_report_df)

        included_df: DataFrame with processed curves + columns:
          csv_file, x_resampled (list), y_raw_interp (list),
          y_normalized (list), y_smoothed (list),
          pce_max_0_200h, pce_min_normalized

        excluded_df: DataFrame with curves that failed QC.

        qc_report_df: Full QC report with one row per curve.
    """
    C = DATA_CONFIG

    included_rows: list[dict] = []
    excluded_rows: list[dict] = []
    qc_rows: list[dict] = []

    curve_ids = df[C["col_csv_file"]].unique()
    total = len(curve_ids)
    logger.info("Preprocessing %d curves (include_short=%s)...", total, include_short)

    for i, curve_id in enumerate(curve_ids):
        # Extract curve data
        curve_df = df[df[C["col_csv_file"]] == curve_id].copy()
        if len(curve_df) == 0:
            continue

        # Parse JSON x and y
        raw_x = json.loads(curve_df[C["col_x"]].iloc[0])
        raw_y = json.loads(curve_df[C["col_y"]].iloc[0])

        xs_raw = np.array(raw_x, dtype=float)
        ys_raw = np.array(raw_y, dtype=float)

        unit_label = str(curve_df[C["col_unit"]].iloc[0])
        max_x_hours = float(curve_df[C["col_max_x_h"]].iloc[0])

        # Run preprocessing
        qc = preprocess_curve(
            curve_id=curve_id,
            xs_raw=xs_raw,
            ys_raw=ys_raw,
            unit_label=unit_label,
            max_x_hours=max_x_hours,
            include_short=include_short,
        )

        # Build QC row
        qc_row = {
            "curve_id": curve_id,
            "status": qc.status,
            "reason": qc.reason,
            "n_original_points": qc.n_original_points,
            "pce_max_0_200h": qc.pce_max_0_200h,
            "pce_min_normalized": qc.pce_min,
            "max_x_hours": qc.max_x_hours,
            "n_unique_t": qc.n_unique_t,
            "t_coverage_pct": qc.t_coverage_pct,
            "top_dir": curve_df[C["col_top_dir"]].iloc[0],
            "doi": curve_df[C["col_doi"]].iloc[0],
            "figure_label": curve_df[C["col_fig_label"]].iloc[0],
            "series_id": curve_df[C["col_series"]].iloc[0],
            "unit_label": unit_label,
        }
        qc_rows.append(qc_row)

        if qc.status == "included":
            inc_row = {
                "curve_id": curve_id,
                "top_dir": curve_df[C["col_top_dir"]].iloc[0],
                "doi": curve_df[C["col_doi"]].iloc[0],
                "figure_label": curve_df[C["col_fig_label"]].iloc[0],
                "series_id": curve_df[C["col_series"]].iloc[0],
                "pce_max_0_200h": qc.pce_max_0_200h,
                "pce_min_normalized": qc.pce_min,
                "x_resampled": json.dumps(qc.x_resampled.tolist()),
                "y_raw_interp": json.dumps(qc.y_raw_interp.tolist()),
                "y_normalized": json.dumps(qc.y_normalized.tolist()),
                "y_smoothed": json.dumps(qc.y_smoothed.tolist()),
            }
            included_rows.append(inc_row)
        else:
            excl_row = {
                "curve_id": curve_id,
                "top_dir": curve_df[C["col_top_dir"]].iloc[0],
                "doi": curve_df[C["col_doi"]].iloc[0],
                "figure_label": curve_df[C["col_fig_label"]].iloc[0],
                "series_id": curve_df[C["col_series"]].iloc[0],
                "status": qc.status,
                "reason": qc.reason,
                "n_original_points": qc.n_original_points,
                "pce_max_0_200h": qc.pce_max_0_200h,
                "max_x_hours": qc.max_x_hours,
                "n_unique_t": qc.n_unique_t,
                "t_coverage_pct": qc.t_coverage_pct,
                "unit_label": unit_label,
            }
            excluded_rows.append(excl_row)

        # Progress logging
        if (i + 1) % 200 == 0:
            n_incl = sum(1 for r in included_rows)
            logger.info("  Progress: %d/%d curves processed, %d included",
                        i + 1, total, n_incl)

    included_df = pd.DataFrame(included_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    qc_report_df = pd.DataFrame(qc_rows)

    n_incl = len(included_df)
    n_excl = len(excluded_df)
    logger.info("Preprocessing complete: %d included, %d excluded (total=%d)",
                n_incl, n_excl, n_incl + n_excl)

    return included_df, excluded_df, qc_report_df


def build_feature_matrix(included_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the SOM feature matrix from processed curves.

    Args:
        included_df: DataFrame from preprocess_all_curves().

    Returns:
        (X, curve_ids, pce_maxima)

        X: np.ndarray of shape (n_samples, 1201) — smoothed normalized PCE
        curve_ids: np.ndarray of curve identifiers
        pce_maxima: np.ndarray of the normalisation denominators (pce_max_0_200h)
    """
    curve_ids = included_df["curve_id"].values
    pce_maxima = included_df["pce_max_0_200h"].values

    rows = []
    for _, row in included_df.iterrows():
        y_smoothed = np.array(json.loads(row["y_smoothed"]))
        rows.append(y_smoothed)

    X = np.array(rows, dtype=float)

    # Validate
    assert X.shape[1] == 1201, \
        f"Expected 1201 time points, got {X.shape[1]}"
    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), \
        "X contains NaN or Inf"
    # Savitzky-Golay may produce slight overshoot above 1.0;
    # this is expected and acceptable per the specification.
    # We only assert that the max is in a reasonable range (not wildly wrong).
    maxes = X.max(axis=1)
    assert np.all(maxes < 1.15), \
        f"Some curves have max > 1.15 after smoothing: {maxes[maxes >= 1.15]}"
    assert np.all(maxes > 0.9), \
        f"Some curves have max < 0.9 after smoothing: {maxes[maxes < 0.9]}"

    logger.info("Feature matrix built: X.shape=%s", X.shape)
    return X, curve_ids, pce_maxima


# =============================================================================
# Saved outputs
# =============================================================================

def save_preprocessed_outputs(
    included_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    qc_report_df: pd.DataFrame,
    X: np.ndarray,
    time_grid: np.ndarray,
    curve_ids: np.ndarray,
    out_dir: Path,
) -> None:
    """Save all preprocessing outputs to disk."""
    from .config import OUT_DATA

    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature matrix
    np.save(out_dir / "X_preprocessed.npy", X)
    logger.info("Saved: X_preprocessed.npy  shape=%s", X.shape)

    # Time grid
    pd.DataFrame({"time_h": time_grid}).to_csv(
        out_dir / "time_grid.csv", index=False)
    logger.info("Saved: time_grid.csv")

    # All curves CSV (for traceability)
    preproc_curves = included_df[[
        "curve_id", "doi", "figure_label", "series_id",
        "pce_max_0_200h", "pce_min_normalized"
    ]].copy()
    # Add time columns
    t_df = pd.DataFrame(X, columns=[f"t{i}" for i in range(X.shape[1])])
    t_df.insert(0, "curve_id", included_df["curve_id"].values)
    t_df.to_csv(out_dir / "preprocessed_curves.csv", index=False)
    logger.info("Saved: preprocessed_curves.csv  shape=%s", t_df.shape)

    # QC report
    qc_report_df.to_csv(out_dir / "quality_control_report.csv", index=False)
    logger.info("Saved: quality_control_report.csv  (%d rows)", len(qc_report_df))

    # Included / excluded lists
    if len(included_df):
        included_df.to_csv(out_dir / "included_samples.csv", index=False)
        logger.info("Saved: included_samples.csv  (%d rows)", len(included_df))
    if len(excluded_df):
        excluded_df.to_csv(out_dir / "excluded_samples.csv", index=False)
        logger.info("Saved: excluded_samples.csv  (%d rows)", len(excluded_df))

    # Data overview
    overview = {
        "n_total": len(qc_report_df),
        "n_included": int((qc_report_df["status"] == "included").sum()),
        "n_excluded": int((qc_report_df["status"] == "excluded").sum()),
        "n_features": X.shape[1],
        "time_window_h": 200,
        "time_interval_min": 10,
        "time_grid_points": len(time_grid),
    }
    pd.DataFrame([overview]).to_csv(out_dir / "data_overview.csv", index=False)
    logger.info("Saved: data_overview.csv")
