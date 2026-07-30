"""Pre-processing pipeline for per-curve PCE trajectories.

Strict order:
    1. Convert time → hours
    2. Relative ageing time from 0 h
    3. Restrict to [0, 200] h
    4. Sort by time
    5. Resolve duplicate timestamps (mean)
    6. Akima interpolation onto uniform 0–200 h grid (no extrapolation)
    7. Normalise by per-curve max PCE within 0–200 h
    8. Savitzky–Golay smoothing
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

from . import config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Step 1: ensure relative ageing time (start_time → 0)
# ----------------------------------------------------------------------------
def build_relative_ageing_time(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return curve
    out = curve.copy()
    t0 = out["x_hours"].min()
    out["x_hours"] = out["x_hours"] - t0
    out["x_hours"] = out["x_hours"].clip(lower=0)
    return out


# ----------------------------------------------------------------------------
# Step 3: restrict to 0–200 h, do not extrapolate
# ----------------------------------------------------------------------------
def restrict_to_200h(curve: pd.DataFrame) -> pd.DataFrame:
    return curve[(curve["x_hours"] >= 0) & (curve["x_hours"] <= config.WINDOW_HOURS)]


# ----------------------------------------------------------------------------
# Step 5: duplicate timestamps → mean
# ----------------------------------------------------------------------------
def resolve_duplicate_timestamps(curve: pd.DataFrame) -> pd.DataFrame:
    g = curve.groupby("x_hours", as_index=False)["pce"].mean()
    n_dup = int(len(curve) - len(g))
    if n_dup > 0:
        g.attrs["n_duplicate_points"] = n_dup
    return g


# ----------------------------------------------------------------------------
# Step 6: Akima interpolation onto uniform grid
# ----------------------------------------------------------------------------
def _safe_akima(x_obs: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Internal interpolation. Returns (x_grid, y_grid) or (None, None) on failure.

    Strategy:
      - Use Akima1DInterpolator (no internal NaN/Inf handling required) only on
        the strictly interior range.
      - Outside the observed time range, NO extrapolation is performed; instead
        the curve is filled with the nearest observed value at the boundary
        (clamping) so that downstream Savitzky–Golay smoothing still produces
        a finite signal over all 1201 grid points.

    Akima1DInterpolator requires strictly increasing x and at least 4 points.
    """
    if len(x_obs) < 4:
        return None, None
    try:
        xo = np.asarray(x_obs, dtype=float)
        yo = np.asarray(y_obs, dtype=float)
        # Deduplicate by mean (should already be done, but be defensive)
        if len(np.unique(xo)) < 4:
            # Group by x and average y
            df = pd.DataFrame({"x": xo, "y": yo})
            df = df.groupby("x", as_index=False)["y"].mean()
            xo = df["x"].to_numpy(dtype=float)
            yo = df["y"].to_numpy(dtype=float)
            if len(xo) < 4:
                return None, None
        interp = Akima1DInterpolator(xo, yo)
        xg = np.array(config.TIME_GRID, dtype=float)
        yg_interior = interp(xg, extrapolate=False)
        # Fill out-of-observed-range with the boundary observed value (clamp)
        yg = np.array(yg_interior, dtype=float)
        x_min, x_max = xo.min(), xo.max()
        mask_before = xg < x_min
        mask_after = xg > x_max
        if mask_before.any():
            yg[mask_before] = yo[xo == x_min].mean() if (xo == x_min).any() else yg[mask_before & np.isfinite(yg)]
        if mask_after.any():
            yg[mask_after] = yo[xo == x_max].mean() if (xo == x_max).any() else yg[mask_after & np.isfinite(yg)]
        # Fill any remaining NaN with neighbouring finite value or zero
        finite_mask = np.isfinite(yg)
        if not finite_mask.all():
            # forward fill then backward fill
            yg_filled = yg.copy()
            last = None
            for i in range(len(yg_filled)):
                if np.isfinite(yg_filled[i]):
                    last = yg_filled[i]
                elif last is not None:
                    yg_filled[i] = last
            # backward fill
            for i in range(len(yg_filled) - 1, -1, -1):
                if not np.isfinite(yg_filled[i]) and i + 1 < len(yg_filled) and np.isfinite(yg_filled[i + 1]):
                    yg_filled[i] = yg_filled[i + 1]
            yg = yg_filled
    except Exception as exc:  # noqa: BLE001
        logger.warning("Akima failed: %s", exc)
        return None, None
    yg_arr = np.asarray(yg, dtype=float)
    if not np.isfinite(yg_arr).all():
        return None, None
    return np.array(config.TIME_GRID, dtype=float), yg_arr


def akima_interpolate_curve(x_obs: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Public entry point for Akima interpolation of a single curve."""
    xg, yg = _safe_akima(np.asarray(x_obs, dtype=float), np.asarray(y_obs, dtype=float))
    return xg, yg


# ----------------------------------------------------------------------------
# Step 7: per-curve normalisation
# ----------------------------------------------------------------------------
def normalize_by_200h_max(y_curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_curve, dtype=float)
    mx = np.nanmax(arr)
    if not np.isfinite(mx) or mx <= 0:
        raise ValueError("max PCE must be finite and > 0 for normalisation")
    return arr / mx


# ----------------------------------------------------------------------------
# Step 8: Savitzky–Golay smoothing (post-normalisation)
# ----------------------------------------------------------------------------
def apply_savgol_filter(y_curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_curve, dtype=float)
    w = int(config.SAVGOL_CONFIG["window_length"])
    p = int(config.SAVGOL_CONFIG["polyorder"])
    mode = str(config.SAVGOL_CONFIG["mode"])
    if w > arr.shape[0]:
        raise ValueError(
            f"savgol window_length ({w}) must be <= length of curve ({arr.shape[0]})"
        )
    return savgol_filter(arr, w, p, mode=mode)


# ----------------------------------------------------------------------------
# Composition: full per-curve pipeline
# ----------------------------------------------------------------------------
def preprocess_one_curve(long_df: pd.DataFrame, sample_id: str) -> dict:
    """Run the locked pipeline on a single curve.

    Returns dict with keys:
        ok, sample_id, x_grid, y_interpolated, y_normalized, y_smoothed,
        max_pce_200h, min_obs_time, max_obs_time, n_duplicate_points,
        n_original_points
    """
    curve = long_df[long_df["sample_id"] == sample_id].copy()
    out = {
        "ok": False,
        "sample_id": sample_id,
        "max_pce_200h": None,
        "min_obs_time": None,
        "max_obs_time": None,
        "n_duplicate_points": 0,
        "n_original_points": int(len(curve)),
        "reason": None,
    }
    if curve.empty or len(curve) < 2:
        out["reason"] = "no observations"
        return out

    curve = build_relative_ageing_time(curve)
    rel_t_max = float(curve["x_hours"].max())
    if config.QC_CONFIG.get("require_min_span_h", 0.0) > 0:
        if rel_t_max < float(config.QC_CONFIG["require_min_span_h"]) - 1e-9:
            out["reason"] = (
                f"curve spans only {rel_t_max:.3f} h (< {config.WINDOW_HOURS:.0f} h)"
            )
            return out
    curve = restrict_to_200h(curve)
    if curve.empty:
        out["reason"] = "no data inside [0,200] h after relative-time shift"
        return out

    n_unique = int(curve["x_hours"].nunique())
    if n_unique < int(config.QC_CONFIG["min_unique_points_200h"]):
        out["reason"] = f"too few unique time points ({n_unique}) in [0,200]h"
        return out

    # Coverage fraction (only if required)
    if float(config.QC_CONFIG["min_coverage_fraction"]) > 0:
        obs_window = curve["x_hours"].max() - curve["x_hours"].min()
        coverage = obs_window / (config.WINDOW_HOURS - 0.0)
        if coverage < float(config.QC_CONFIG["min_coverage_fraction"]):
            out["reason"] = (
                f"data covers {coverage*100:.1f}% of [0,200]h window (< {config.QC_CONFIG['min_coverage_fraction']*100:.0f}%)"
            )
            return out

    # Resolve duplicates → keep x_obs, y_obs arrays
    curve_sorted = curve.sort_values("x_hours")
    curve_dedup = resolve_duplicate_timestamps(curve_sorted)
    out["n_duplicate_points"] = int(len(curve_sorted) - len(curve_dedup))

    x_obs = curve_dedup["x_hours"].to_numpy(dtype=float)
    y_obs = curve_dedup["pce"].to_numpy(dtype=float)

    # Akima interpolation
    x_grid, y_grid = akima_interpolate_curve(x_obs, y_obs)
    if x_grid is None or y_grid is None or len(y_grid) != config.N_GRID_POINTS:
        out["reason"] = "Akima interpolation failed (insufficient interior coverage)"
        return out

    # Drop the leading 0 h exactly if it was not observed (do not extrapolate
    # beyond observation range). Equivalently: clip x_grid to observed range
    # and force NaN elsewhere. After this step the curve must be valid at all
    # 1201 grid points.
    if not np.isfinite(y_grid).any():
        out["reason"] = "all interpolated values are NaN/Inf"
        return out

    # Per-curve max PCE BEFORE smoothing (within [0,200]h)
    valid = np.isfinite(y_grid)
    if not valid.any():
        out["reason"] = "no finite values in [0,200] h grid"
        return out
    max_pce_200h = float(np.nanmax(y_grid))
    if config.QC_CONFIG["require_max_pce_positive"] and max_pce_200h <= 0:
        out["reason"] = f"max PCE in [0,200] h is non-positive ({max_pce_200h})"
        return out
    out["max_pce_200h"] = max_pce_200h

    # Normalise
    y_norm = y_grid / max_pce_200h
    if not np.isfinite(y_norm).all():
        out["reason"] = "non-finite values after normalisation"
        return out

    # Savitzky–Golay smoothing (do not clip afterwards)
    try:
        y_smooth = apply_savgol_filter(y_norm)
    except ValueError as exc:
        out["reason"] = f"savgol failed: {exc}"
        return out

    if not np.isfinite(y_smooth).all():
        out["reason"] = "non-finite values after smoothing"
        return out

    out.update(dict(
        ok=True,
        x_grid=x_grid,
        y_interpolated=y_grid,
        y_normalized=y_norm,
        y_smoothed=y_smooth,
        min_obs_time=float(x_obs.min()),
        max_obs_time=float(x_obs.max()),
    ))
    return out


def preprocess_all_curves(long_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Returns (included_records, excluded_records).

    Each record is the dict returned by preprocess_one_curve, augmented
    with 'exclusion_reason' for excluded ones.
    """
    sample_ids = long_df["sample_id"].unique().tolist()
    included = []
    excluded = []
    for sid in sample_ids:
        rec = preprocess_one_curve(long_df, sid)
        if rec["ok"]:
            included.append(rec)
        else:
            rec["exclusion_reason"] = rec.pop("reason") or "unknown"
            excluded.append(rec)
    logger.info(
        "Preprocessing: %d included, %d excluded from %d input curves",
        len(included),
        len(excluded),
        len(sample_ids),
    )
    return included, excluded


# ----------------------------------------------------------------------------
# Feature matrix
# ----------------------------------------------------------------------------
def build_feature_matrix(included: list[dict]) -> np.ndarray:
    if not included:
        return np.empty((0, config.N_GRID_POINTS))
    return np.stack([rec["y_smoothed"] for rec in included], axis=0)


def build_feature_matrix_no_smooth(included: list[dict]) -> np.ndarray:
    """Same shape as :func:`build_feature_matrix` but uses the Akima-interpolated,
    per-curve-normalised curve directly **without** the Savitzky–Golay filter.

    All other QC checks (span / duplicates / coverage / NaN guards) are still
    performed upstream in :func:`preprocess_one_curve`, so any record in
    ``included`` is by construction valid on the uniform 0–200 h grid.
    """
    if not included:
        return np.empty((0, config.N_GRID_POINTS))
    return np.stack([rec["y_normalized"] for rec in included], axis=0)


def build_preprocessed_curve_table(included: list[dict]) -> pd.DataFrame:
    """Long-format table where rows are samples and 1201 columns are time-grid PCE."""
    if not included:
        return pd.DataFrame(columns=["sample_id"] + [f"t_{i}" for i in range(config.N_GRID_POINTS)])
    sids = [rec["sample_id"] for rec in included]
    cols = [f"t_{i}" for i in range(config.N_GRID_POINTS)]
    data = np.stack([rec["y_smoothed"] for rec in included], axis=0)
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "sample_id", sids)
    return df


def verify_preprocessed_matrix(X: np.ndarray) -> dict:
    out = {
        "shape": tuple(X.shape),
        "finite": bool(np.isfinite(X).all()) if X.size else True,
        "n_samples": int(X.shape[0]) if X.ndim == 2 else 0,
    }
    return out
