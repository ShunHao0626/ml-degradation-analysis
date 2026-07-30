#!/usr/bin/env python3
"""
SOM_200h_pipeline.py
====================
Perovskite solar cell MPPT ageing curve clustering using Self-Organizing Maps (2×2 SOM).

Reference: Hartono et al., Nat. Commun. 14, 4869 (2023)
"Stability follows efficiency"
https://doi.org/10.1038/s41467-023-40585-3

Key difference from paper:
  - 200-hour window (paper uses 150 h)
  - Each curve normalized by its own max PCE within 0–200 h

Author: Auto-generated for thesis analysis
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All experiment parameters — edit here, not scattered across functions."""

    # --- Paths ---
    # Set USE_RAW_DATA=True to scan the original x_time_* directories directly.
    # Set False to use the pre-built consolidated CSV.
    USE_RAW_DATA = True

    if USE_RAW_DATA:
        # Path to the 05_accepted_all copy base directory
        RAW_DATA_BASE = Path(
            "/Users/shunhao/Desktop/ML/lab/05_accepted_all copy/"
        )
        # Output: merged DataFrame saved here for reuse
        MERGED_CSV = Path(
            "/Users/shunhao/Desktop/ML/lab/result/merged_all_curves.csv"
        )
    else:
        CONSOLIDATED_CSV = Path(
            "/Users/shunhao/Desktop/ML/lab/05_accepted_all copy/"
            "analysis_report/curves_over_200h_consolidated.csv"
        )

    OUTPUT_DIR = Path("/Users/shunhao/Desktop/ML/lab/result/SOM_200h_output")

    # --- Time window ---
    HOUR_LIMIT = 200          # hours to analyse
    N_TIME_POINTS = int(HOUR_LIMIT * 6 + 1)   # = 1201 (10-min intervals)
    RESAMPLE_INTERVAL_MIN = 10  # minutes

    # --- Preprocessing ---
    SAGGOL_WINDOW = 71         # MUST be odd, > polyorder
    SAGGOL_POLYORDER = 2       # ⚠  Assumed value — NOT confirmed from author code

    # --- SOM ---
    SOM_X = 2
    SOM_Y = 2
    SOM_SIGMA = 0.5
    SOM_LR = 0.1
    SOM_ITER = 50_000
    RANDOM_SEED = 42

    # --- Column names in consolidated / raw DataFrame ---
    COL_CSV_FILE = "csv_file"
    COL_UNIT = "unit_label"
    COL_X = "x"         # time in original unit
    COL_Y = "y"         # PCE value
    COL_MAX_X_H = "max_x_hours"
    COL_DOI = "doi"
    COL_FIG = "figure_label"
    COL_SERIES = "series_id"

    # --- Y-scale threshold ---
    Y_NORMALIZED_MAX = 1.1     # curves with max(y) <= this are treated as normalized [0,1]


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SOM_200h")


# =============================================================================
# 1. LOAD & VALIDATE DATA
# =============================================================================

def load_consolidated_data() -> pd.DataFrame:
    """Load data — either from raw x_time_* directories or from consolidated CSV."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from load_raw_data import load_all_raw_curves

    if Config.USE_RAW_DATA:
        # Try to reuse cached merged CSV if it exists
        if Config.MERGED_CSV.exists() and Config.MERGED_CSV.stat().st_size > 1000:
            logger.info("Loading cached merged CSV: %s", Config.MERGED_CSV)
            df = pd.read_csv(Config.MERGED_CSV)
        else:
            logger.info("Loading raw x_time_* directories: %s", Config.RAW_DATA_BASE)
            df = load_all_raw_curves(Config.RAW_DATA_BASE)
            df.to_csv(Config.MERGED_CSV, index=False)
            logger.info("Cached to: %s", Config.MERGED_CSV)
    else:
        logger.info("Loading consolidated CSV: %s", Config.CONSOLIDATED_CSV)
        df = pd.read_csv(Config.CONSOLIDATED_CSV)

    logger.info(
        "  Loaded %d rows, %d unique curves",
        len(df), df[Config.COL_CSV_FILE].nunique(),
    )
    return df


def build_unit_to_hours_factor_map(df: pd.DataFrame) -> dict[str, float]:
    """Return a hardcoded unit → hours conversion factor map.

    The consolidated CSV stores 'x' in its original unit (hours/days/weeks/...).
    This function provides the standard conversion factors.
    """
    mapping = {
        "hours": 1.0, "h": 1.0,
        "days": 24.0, "day": 24.0,
        "weeks": 168.0,
        "months": 720.0,   # 30 days/month
        "minutes": 1.0 / 60.0, "min": 1.0 / 60.0,
    }
    # Log what we found in the data
    for unit, factor in mapping.items():
        count = (df[Config.COL_UNIT] == unit).sum()
        if count > 0:
            logger.debug("  unit '%s' → factor %.4f (%d curves)", unit, factor, count)
    return mapping


def _default_factor_for_unit(unit: str) -> float:
    defaults = {
        "hours": 1.0, "h": 1.0,
        "days": 24.0, "day": 24.0,
        "weeks": 168.0,
        "months": 720.0,   # 30 days/month approximation
        "minutes": 1 / 60.0, "min": 1 / 60.0,
        "seconds": 1 / 3600.0, "s": 1 / 3600.0,
    }
    return defaults.get(unit, 1.0)


# =============================================================================
# 2. PER-CURVE PREPROCESSING
# =============================================================================

def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values from the last valid value."""
    out = arr.copy()
    last_valid = None
    for i in range(len(out)):
        if np.isnan(out[i]):
            if last_valid is not None:
                out[i] = last_valid
        else:
            last_valid = out[i]
    return out


def preprocess_single_curve(
    curve_df: pd.DataFrame,
    unit_to_hours: dict[str, float],
    curve_id: str,
) -> dict:
    """
    Preprocess one curve through the full pipeline.

    Steps (per paper + your spec):
      1. Convert x to hours
      2. Filter to [0, HOUR_LIMIT]
      3. Shift to start at t=0
      4. Deduplicate x (average y)
      5. Resample to 10-min grid
      6. Akima interpolation
      7. Normalize by 0-200h max PCE
      8. Savitzky-Golay smoothing

    Returns a dict with keys:
      'success', 'curve_id', 'qc_status', 'qc_reason',
      'x_hours', 'y_raw', 'y_norm', 'y_smooth',
      'pce_max', 'n_points_original'
    """
    result = {
        "success": False,
        "curve_id": curve_id,
        "qc_status": "unknown",
        "qc_reason": "",
        "x_hours": None,
        "y_raw": None,
        "y_norm": None,
        "y_smooth": None,
        "pce_max": None,
        "n_points_original": None,
    }

    unit = curve_df[Config.COL_UNIT].iloc[0]
    factor = unit_to_hours.get(unit, 1.0)

    # --- 1. Convert x to hours ---
    x_orig = curve_df[Config.COL_X].values.astype(float) * factor
    y_orig = curve_df[Config.COL_Y].values.astype(float)

    # --- 2. Filter to [0, HOUR_LIMIT] ---
    mask = (x_orig >= 0) & (x_orig <= Config.HOUR_LIMIT)
    x_filt = x_orig[mask]
    y_filt = y_orig[mask]

    if len(x_filt) < 2:
        result["qc_status"] = "excluded"
        result["qc_reason"] = f"Too few points after 0-{Config.HOUR_LIMIT}h filter: {len(x_filt)}"
        return result

    # --- 3. Shift to start at t=0 ---
    x_filt = x_filt - x_filt.min()

    # --- 4. Deduplicate x (average y) ---
    if len(x_filt) != len(np.unique(x_filt)):
        df_dedup = pd.DataFrame({"x": x_filt, "y": y_filt})
        df_dedup = df_dedup.groupby("x", sort=True)["y"].mean().reset_index()
        x_filt = df_dedup["x"].values
        y_filt = df_dedup["y"].values

    # Sort by time
    order = np.argsort(x_filt)
    x_filt = x_filt[order]
    y_filt = y_filt[order]

    result["n_points_original"] = len(x_filt)

    # --- 5. Akima interpolation onto 10-min grid ---
    # NOTE: Akima1DInterpolator in scipy 1.13.x does NOT support
    # bounds_error / fill_value kwargs.  Out-of-range points return NaN.
    # We forward-fill trailing NaN so the curve has exactly N_TIME_POINTS.
    try:
        akima = Akima1DInterpolator(x_filt, y_filt)
        t_grid = np.linspace(0, Config.HOUR_LIMIT, Config.N_TIME_POINTS)
        y_interp = akima(t_grid)
    except Exception:
        result["qc_status"] = "excluded"
        result["qc_reason"] = f"Akima interpolation failed (n_orig={len(x_filt)})"
        return result

    if np.any(np.isinf(y_interp)):
        result["qc_status"] = "excluded"
        result["qc_reason"] = "Inf after Akima interpolation"
        return result

    if np.any(np.isnan(y_interp)):
        y_interp = _forward_fill_nan(y_interp)

    # --- QC: Data density check ---
    # Reject curves with too few unique time points in [0, HOUR_LIMIT].
    # With < 3 unique points Akima produces mostly NaN; even with 3-5 the
    # interpolation quality is too poor for reliable shape clustering.
    min_unique_pts = 10
    frac_coverage = (x_filt.max() - x_filt.min()) / Config.HOUR_LIMIT
    if len(np.unique(x_filt)) < min_unique_pts:
        result["qc_status"] = "excluded"
        result["qc_reason"] = (
            f"Too few unique time points ({len(np.unique(x_filt))}) "
            f"in [0,{Config.HOUR_LIMIT}]h window "
            f"(data spans {frac_coverage:.0%} of window)"
        )
        return result

    x_out = t_grid
    y_raw = y_interp

    # --- 6. Normalize by max PCE in 0-200h ---
    pce_max = np.max(y_raw)
    if pce_max <= 0 or np.isnan(pce_max):
        result["qc_status"] = "excluded"
        result["qc_reason"] = f"Invalid max PCE: {pce_max}"
        return result

    y_norm = y_raw / pce_max

    # Verify normalization
    if not (0.99 <= np.max(y_norm) <= 1.01 + 1e-6):
        result["qc_status"] = "excluded"
        result["qc_reason"] = f"Normalization failed: max={np.max(y_norm):.6f}"
        return result

    # --- 7. Savitzky-Golay smoothing ---
    try:
        y_smooth = savgol_filter(
            y_norm,
            window_length=Config.SAGGOL_WINDOW,
            polyorder=Config.SAGGOL_POLYORDER,
        )
    except Exception:
        result["qc_status"] = "excluded"
        result["qc_reason"] = "Savitzky-Golay failed"
        return result

    result["success"] = True
    result["qc_status"] = "included"
    result["qc_reason"] = ""
    result["x_hours"] = x_out
    result["y_raw"] = y_raw
    result["y_norm"] = y_norm
    result["y_smooth"] = y_smooth
    result["pce_max"] = pce_max

    return result


def preprocess_all_curves(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """
    Run preprocessing on all unique curves.

    Returns:
      results: list of per-curve result dicts
      qc_report: DataFrame of QC info
    """
    unit_to_hours = build_unit_to_hours_factor_map(df)

    curves = df.groupby(Config.COL_CSV_FILE)
    n_total = len(curves)

    results: list[dict] = []
    qc_rows: list[dict] = []

    for i, (curve_id, curve_df) in enumerate(curves):
        if i > 0 and i % 200 == 0:
            included = sum(1 for r in results if r["success"])
            logger.info("  Progress: %d/%d curves processed, %d included",
                        i, n_total, included)

        r = preprocess_single_curve(curve_df, unit_to_hours, curve_id)
        results.append(r)

        qc_rows.append({
            "curve_id": curve_id,
            "qc_status": r["qc_status"],
            "qc_reason": r["qc_reason"],
            "n_points_original": r["n_points_original"],
            "pce_max": r["pce_max"],
            "doi": curve_df[Config.COL_DOI].iloc[0] if Config.COL_DOI in curve_df.columns else "",
            "figure_label": curve_df[Config.COL_FIG].iloc[0] if Config.COL_FIG in curve_df.columns else "",
            "series_id": curve_df[Config.COL_SERIES].iloc[0] if Config.COL_SERIES in curve_df.columns else "",
            "unit_label": curve_df[Config.COL_UNIT].iloc[0] if Config.COL_UNIT in curve_df.columns else "",
        })

    included = sum(1 for r in results if r["success"])
    logger.info(
        "Preprocessing done: %d/%d curves included, %d excluded",
        included, n_total, n_total - included,
    )

    qc_report = pd.DataFrame(qc_rows)
    return results, qc_report


# =============================================================================
# 3. BUILD FEATURE MATRIX
# =============================================================================

def build_feature_matrix(results: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Extract smoothed normalized curves into a (n_curves, 1201) feature matrix."""
    included = [r for r in results if r["success"]]
    matrix = np.stack([r["y_smooth"] for r in included])
    curve_ids = [r["curve_id"] for r in included]
    logger.info("Feature matrix shape: %s", matrix.shape)
    return matrix, curve_ids


# =============================================================================
# 4. TRAIN SOM
# =============================================================================

def train_som(data: np.ndarray) -> tuple[MiniSom, dict]:
    """Train a 2×2 SOM on the feature matrix."""
    logger.info("Training SOM: shape=%s, iterations=%d", data.shape, Config.SOM_ITER)

    som = MiniSom(
        x=Config.SOM_X,
        y=Config.SOM_Y,
        input_len=data.shape[1],
        sigma=Config.SOM_SIGMA,
        learning_rate=Config.SOM_LR,
        random_seed=Config.RANDOM_SEED,
    )

    som.train_random(data, num_iteration=Config.SOM_ITER)

    # Quantization error
    qe = som.quantization_error(data)

    # Topographic error (approximate)
    te = _topographic_error(som, data)

    logger.info(
        "SOM trained: QE=%.4f, TE=%.4f",
        qe, te,
    )

    summary = {
        "som_x": Config.SOM_X,
        "som_y": Config.SOM_Y,
        "sigma": Config.SOM_SIGMA,
        "learning_rate": Config.SOM_LR,
        "iterations": Config.SOM_ITER,
        "random_seed": Config.RANDOM_SEED,
        "n_curves": data.shape[0],
        "n_features": data.shape[1],
        "quantization_error": float(qe),
        "topographic_error": float(te),
    }
    return som, summary


def _topographic_error(som: MiniSom, data: np.ndarray) -> float:
    """Fraction of data points where BMU and 2nd-BMU are not adjacent.
    
    MiniSom already has this built in, so we just call it directly.
    """
    return float(som.topographic_error(data))


# =============================================================================
# 5. ASSIGN CLUSTERS
# =============================================================================

def assign_clusters(
    som: MiniSom,
    data: np.ndarray,
    curve_ids: list[str],
) -> pd.DataFrame:
    """Assign each curve to its SOM node, compute distance to BMU."""
    assignments = []
    for i, (curve_id, x) in enumerate(zip(curve_ids, data)):
        bmu = som.winner(x)
        # Distance to BMU
        bmu_weights = som.get_weights()[bmu[0], bmu[1]].flatten()
        dist_to_bmu = float(np.linalg.norm(x - bmu_weights))

        assignments.append({
            "curve_id": curve_id,
            "som_x": bmu[0],
            "som_y": bmu[1],
            "raw_cluster_id": bmu[0] * Config.SOM_Y + bmu[1],
            "dist_to_bmu": dist_to_bmu,
        })

    df = pd.DataFrame(assignments)
    logger.info("Cluster assignment done.\n%s",
                df.groupby(["som_x", "som_y"])["curve_id"].count().to_string())
    return df


# =============================================================================
# 6. CLUSTER STATISTICS
# =============================================================================

def compute_cluster_statistics(
    results: list[dict],
    assignments: pd.DataFrame,
    data: np.ndarray,
) -> pd.DataFrame:
    """Compute per-cluster statistics from smoothed data."""
    rows = []
    for (sx, sy), grp in assignments.groupby(["som_x", "som_y"]):
        idx = grp.index.tolist()
        cluster_data = data[idx]
        cluster_ids = [results[i]["curve_id"] for i in idx]

        n = len(idx)
        mean_curve = cluster_data.mean(axis=0)
        median_curve = np.median(cluster_data, axis=0)
        std_curve = cluster_data.std(axis=0)
        q25 = np.percentile(cluster_data, 25, axis=0)
        q75 = np.percentile(cluster_data, 75, axis=0)

        # Key time points
        t_grid = np.linspace(0, Config.HOUR_LIMIT, Config.N_TIME_POINTS)
        times_h = [0, 10, 50, 100, 200]
        pt_indices = [int((t / Config.HOUR_LIMIT) * (Config.N_TIME_POINTS - 1)) for t in times_h]
        stats = {
            "n_samples": n,
            "percentage": n / len(data) * 100,
            "mean_pce_t0": float(mean_curve[0]),
            "mean_pce_t10": float(mean_curve[pt_indices[1]]),
            "mean_pce_t50": float(mean_curve[pt_indices[2]]),
            "mean_pce_t100": float(mean_curve[pt_indices[3]]),
            "mean_pce_t200": float(mean_curve[pt_indices[4]]),
            "median_pce_t0": float(median_curve[0]),
            "median_pce_t200": float(median_curve[pt_indices[4]]),
            "std_pce_t0": float(std_curve[0]),
            "std_pce_t200": float(std_curve[pt_indices[4]]),
            "mean_initial_slope": float(
                (mean_curve[pt_indices[1]] - mean_curve[0]) / 10.0
            ),
            "mean_late_slope": float(
                (mean_curve[pt_indices[4]] - mean_curve[pt_indices[3]]) / 100.0
            ),
            "mean_total_change": float(mean_curve[pt_indices[4]] - mean_curve[0]),
            "som_x": sx,
            "som_y": sy,
            "raw_cluster_id": sx * Config.SOM_Y + sy,
        }

        # Time of max PCE (mean)
        t_max_mean = float(
            t_grid[np.argmax(mean_curve)]
        )

        # Classify shape based on curve features
        pce_t0 = stats["mean_pce_t0"]
        pce_t10 = stats["mean_pce_t10"]
        pce_t200 = stats["mean_pce_t200"]

        if pce_t10 > pce_t0 + 0.01:
            shape_hint = "initial_gain"
        elif abs(pce_t10 - pce_t0) <= 0.01:
            shape_hint = "stable_then_decay"
        else:
            shape_hint = "initial_decay"

        if pce_t200 > 0.8:
            decay_hint = "slow_decay"
        elif pce_t200 > 0.5:
            decay_hint = "medium_decay"
        else:
            decay_hint = "fast_decay"

        stats["shape_hint"] = f"{shape_hint}_{decay_hint}"
        stats["t_max_mean"] = t_max_mean

        rows.append(stats)

    return pd.DataFrame(rows)


# =============================================================================
# 7. PLOTS
# =============================================================================

def plot_cluster_curves(
    results: list[dict],
    assignments: pd.DataFrame,
    data: np.ndarray,
    cluster_stats: pd.DataFrame,
    som: MiniSom,
    out_dir: Path,
) -> None:
    """Plot 2×2 subplot of all SOM clusters."""
    t_grid = np.linspace(0, Config.HOUR_LIMIT, Config.N_TIME_POINTS)
    n_time_pts = Config.N_TIME_POINTS

    fig, axes = plt.subplots(
        Config.SOM_X, Config.SOM_Y,
        figsize=(12, 10),
        sharex=True, sharey=True,
    )
    fig.suptitle(
        f"SOM Clusters (2×2) — {len(data)} curves, "
        f"σ={Config.SOM_SIGMA}, lr={Config.SOM_LR}",
        fontsize=13, y=0.98,
    )

    for (sx, sy), ax in zip(
        [(i, j) for i in range(Config.SOM_X) for j in range(Config.SOM_Y)],
        axes.flat,
    ):
        grp = assignments[(assignments["som_x"] == sx) & (assignments["som_y"] == sy)]
        if len(grp) == 0:
            ax.text(0.5, 0.5, "Empty node", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(f"SOM ({sx},{sy}) — 0 curves", fontsize=10)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
            continue

        idx = grp.index.tolist()
        cluster_data = data[idx]

        # All curves (low alpha)
        for row in cluster_data:
            ax.plot(t_grid, row, color="gray", alpha=0.15, linewidth=0.5)

        # Mean curve
        mean_curve = cluster_data.mean(axis=0)
        ax.plot(t_grid, mean_curve, color="black", linewidth=2.5,
                label="Mean", zorder=5)

        # Median curve
        median_curve = np.median(cluster_data, axis=0)
        ax.plot(t_grid, median_curve, color="crimson", linewidth=2,
                linestyle="--", label="Median", zorder=6)

        # IQR shading
        q25 = np.percentile(cluster_data, 25, axis=0)
        q75 = np.percentile(cluster_data, 75, axis=0)
        ax.fill_between(t_grid, q25, q75, color="steelblue", alpha=0.15,
                       label="IQR (25–75%)")

        ax.set_title(
            f"SOM ({sx},{sy})  |  n={len(grp)} ({len(grp)/len(data)*100:.1f}%)",
            fontsize=10, pad=4,
        )
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.set_xlim(0, Config.HOUR_LIMIT)
        ax.set_ylim(0, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "som_cluster_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_codebook_vectors(
    som: MiniSom,
    out_dir: Path,
) -> None:
    """Plot the SOM codebook (weight) vectors as 4 separate curves."""
    t_grid = np.linspace(0, Config.HOUR_LIMIT, Config.N_TIME_POINTS)
    weights = som.get_weights()   # shape: (SOM_X, SOM_Y, n_features)

    fig, axes = plt.subplots(
        Config.SOM_X, Config.SOM_Y,
        figsize=(12, 10),
        sharex=True, sharey=True,
    )
    fig.suptitle("SOM Codebook Vectors (Node Weight Curves)", fontsize=13, y=0.98)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    node_idx = 0

    for i in range(Config.SOM_X):
        for j in range(Config.SOM_Y):
            ax = axes[i, j]
            wv = weights[i, j, :]   # 1-D weight vector for this node
            ax.plot(t_grid, wv, color=colors[node_idx], linewidth=2)
            ax.set_title(f"Node ({i},{j})", fontsize=10)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
            ax.set_xlim(0, Config.HOUR_LIMIT)
            ax.set_ylim(0, 1.08)
            ax.grid(True, alpha=0.3)
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
            node_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "som_codebook_vectors.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_u_matrix(
    som: MiniSom,
    out_dir: Path,
) -> None:
    """Plot U-matrix / distance map of the SOM."""
    umatrix = som.distance_map()   # built-in U-matrix

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # U-matrix
    ax = axes[0]
    im = ax.imshow(
        umatrix.T,
        cmap="coolwarm",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title("U-Matrix (SOM Distance Map)", fontsize=11)
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    for i in range(Config.SOM_X):
        for j in range(Config.SOM_Y):
            ax.text(j, i, f"{umatrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if umatrix[i, j] > umatrix.mean() else "black")
    plt.colorbar(im, ax=ax, label="Distance")

    # Distance map (viridis alternative)
    ax2 = axes[1]
    im2 = ax2.imshow(
        umatrix.T,
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    ax2.set_title("SOM Distance Map (viridis)", fontsize=11)
    ax2.set_xlabel("SOM X")
    ax2.set_ylabel("SOM Y")
    for i in range(Config.SOM_X):
        for j in range(Config.SOM_Y):
            ax2.text(j, i, f"{umatrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=11,
                     color="white")
    plt.colorbar(im2, ax=ax2, label="Distance")

    plt.tight_layout()
    out_path = out_dir / "som_u_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    out_path2 = out_dir / "som_distance_map.png"
    fig2, ax3 = plt.subplots(figsize=(5, 4))
    im3 = ax3.imshow(umatrix.T, cmap="viridis", aspect="equal",
                      interpolation="nearest")
    ax3.set_title("Distance Map", fontsize=11)
    ax3.set_xlabel("SOM X")
    ax3.set_ylabel("SOM Y")
    for i in range(Config.SOM_X):
        for j in range(Config.SOM_Y):
            ax3.text(j, i, f"{umatrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=11, color="white")
    plt.colorbar(im3, ax=ax3)
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s and %s", out_path, out_path2)


def plot_cluster_comparison(
    som: MiniSom,
    data: np.ndarray,
    assignments: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Plot SOM cluster counts vs k-means (k=4) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # SOM counts
    som_counts = (
        assignments.groupby(["som_x", "som_y"])["curve_id"]
        .count()
        .reset_index(name="count")
    )
    som_counts["label"] = som_counts.apply(
        lambda r: f"({r['som_x']},{r['som_y']})", axis=1
    )
    axes[0].bar(som_counts["label"], som_counts["count"], color="steelblue")
    axes[0].set_title("SOM Cluster Distribution", fontsize=11)
    axes[0].set_xlabel("SOM Node")
    axes[0].set_ylabel("Count")
    for i, row in som_counts.iterrows():
        axes[0].text(i, row["count"] + 5, str(row["count"]),
                     ha="center", fontsize=10)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=Config.RANDOM_SEED, n_init=10)
    km_labels = kmeans.fit_predict(data)
    km_counts = pd.Series(km_labels).value_counts().sort_index()
    axes[1].bar(
        [f"K={i}" for i in km_counts.index],
        km_counts.values,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    axes[1].set_title("K-Means (k=4) Cluster Distribution", fontsize=11)
    axes[1].set_xlabel("K-Means Cluster")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(km_counts.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "som_vs_kmeans_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_pca_visualization(
    som: MiniSom,
    data: np.ndarray,
    assignments: pd.DataFrame,
    out_dir: Path,
) -> None:
    """PCA 2D projection colored by SOM cluster."""
    pca = PCA(n_components=2, random_state=Config.RANDOM_SEED)
    proj = pca.fit_transform(data)
    var_explained = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (sx, sy), grp in assignments.groupby(["som_x", "som_y"]):
        idx = grp.index.tolist()
        label = f"SOM ({sx},{sy}), n={len(grp)}"
        ax.scatter(
            proj[idx, 0], proj[idx, 1],
            c=colors[sx * Config.SOM_Y + sy],
            label=label,
            alpha=0.4, s=15,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA Projection (total var: {var_explained*100:.1f}%)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "pca_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s (var explained: %.1f%%)", out_path, var_explained*100)


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

def save_preprocessed_curves(
    results: list[dict],
    out_dir: Path,
) -> None:
    """Save all preprocessed smoothed curves."""
    included = [r for r in results if r["success"]]
    t_grid = np.linspace(0, Config.HOUR_LIMIT, Config.N_TIME_POINTS)

    columns = ["curve_id"] + [f"t_{i:04d}" for i in range(Config.N_TIME_POINTS)]
    rows = []
    for r in included:
        row = [r["curve_id"]] + [float(v) for v in r["y_smooth"]]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    out_path = out_dir / "preprocessed_curves.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved preprocessed curves: %s (%d curves, %d cols)",
                out_path, len(df), len(df.columns))


def save_outputs(
    results: list[dict],
    qc_report: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    som_summary: dict,
    out_dir: Path,
) -> None:
    """Save all required output files."""

    # cluster_assignments.csv
    assignments_out = assignments.copy()
    assignments_out = assignments_out.merge(
        qc_report[["curve_id", "qc_status", "qc_reason"]],
        on="curve_id", how="left",
    )
    assignments_out.to_csv(out_dir / "cluster_assignments.csv", index=False)
    logger.info("Saved: cluster_assignments.csv")

    # cluster_summary.csv
    cluster_stats.to_csv(out_dir / "cluster_summary.csv", index=False)
    logger.info("Saved: cluster_summary.csv")

    # quality_control_report.csv
    qc_report.to_csv(out_dir / "quality_control_report.csv", index=False)
    logger.info("Saved: quality_control_report.csv")

    # run_config.json
    config_dict = {
        "hour_limit": Config.HOUR_LIMIT,
        "n_time_points": Config.N_TIME_POINTS,
        "resample_interval_min": Config.RESAMPLE_INTERVAL_MIN,
        "sagol_window": Config.SAGGOL_WINDOW,
        "sagol_polyorder": Config.SAGGOL_POLYORDER,
        "sagol_polyorder_note": (
            "Assumed value — NOT confirmed from author code. "
            "Requires verification against Zenodo code (10.5281/zenodo.8181602)."
        ),
        "som_x": Config.SOM_X,
        "som_y": Config.SOM_Y,
        "som_sigma": Config.SOM_SIGMA,
        "som_lr": Config.SOM_LR,
        "som_iter": Config.SOM_ITER,
        "random_seed": Config.RANDOM_SEED,
        "normalization": "MaxAbsScaler (per-curve, by max in 0-200h)",
        "interpolation": "Akima1DInterpolator (extrapolate=False)",
        "distance_metric": "Euclidean",
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Saved: run_config.json")

    # som_summary.json
    som_summary_out = {**som_summary}
    som_summary_out["clusters"] = cluster_stats.to_dict("records")
    with open(out_dir / "som_summary.json", "w") as f:
        json.dump(som_summary_out, f, indent=2)
    logger.info("Saved: som_summary.json")


# =============================================================================
# 9. QUALITY CHECKS
# =============================================================================

def run_quality_checks(
    results: list[dict],
    qc_report: pd.DataFrame,
    assignments: pd.DataFrame,
    data: np.ndarray,
    som_summary: dict,
) -> None:
    """Print and log all QC checks."""
    included = sum(1 for r in results if r["success"])
    excluded = len(results) - included

    logger.info("=" * 60)
    logger.info("QUALITY CHECK REPORT")
    logger.info("=" * 60)
    logger.info("Total curves: %d", len(results))
    logger.info("Included (processed): %d", included)
    logger.info("Excluded: %d", excluded)

    # Check 1201 time points per included curve
    wrong_len = sum(
        1 for r in results
        if r["success"] and len(r["y_smooth"]) != Config.N_TIME_POINTS
    )
    logger.info("Curves with wrong length (%d): %d", Config.N_TIME_POINTS, wrong_len)

    # Check NaN/Inf after smoothing
    nan_after_smooth = sum(
        1 for r in results
        if r["success"] and (np.any(np.isnan(r["y_smooth"])) or np.any(np.isinf(r["y_smooth"])))
    )
    logger.info("Curves with NaN/Inf after smoothing: %d", nan_after_smooth)

    # Check max PCE normalization
    bad_norm = sum(
        1 for r in results
        if r["success"] and (np.max(r["y_norm"]) < 0.99 or np.max(r["y_norm"]) > 1.01)
    )
    logger.info("Curves with bad normalization: %d", bad_norm)

    # Check zero/negative max PCE
    invalid_pce = sum(
        1 for r in results
        if r["success"] and (r["pce_max"] <= 0 or np.isnan(r["pce_max"]))
    )
    logger.info("Curves with zero/negative max PCE: %d", invalid_pce)

    # SOM node counts
    logger.info("SOM node distribution:")
    node_counts = assignments.groupby(["som_x", "som_y"])["curve_id"].count()
    for (sx, sy), count in node_counts.items():
        logger.info("  Node (%d,%d): %d curves", sx, sy, count)

    # Empty nodes
    empty_nodes = []
    for i in range(Config.SOM_X):
        for j in range(Config.SOM_Y):
            if (i, j) not in [(r[0], r[1]) for r in node_counts.index]:
                empty_nodes.append((i, j))
    if empty_nodes:
        logger.warning("Empty SOM nodes: %s", empty_nodes)
    else:
        logger.info("No empty SOM nodes.")

    # SOM errors
    logger.info("SOM Quantization Error: %.4f", som_summary["quantization_error"])
    logger.info("SOM Topographic Error: %.4f", som_summary["topographic_error"])

    # Excluded reasons
    logger.info("Exclusion reasons:")
    excl = qc_report[qc_report["qc_status"] == "excluded"]
    reasons = excl["qc_reason"].value_counts()
    for reason, count in reasons.items():
        logger.info("  [%d] %s", count, reason)

    logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    warnings.filterwarnings("ignore")

    out_dir = Path(Config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SOM 200h Pipeline — Starting")
    logger.info("Config: HOUR_LIMIT=%d, SOM=%dx%d, iter=%d, seed=%d",
                Config.HOUR_LIMIT, Config.SOM_X, Config.SOM_Y,
                Config.SOM_ITER, Config.RANDOM_SEED)
    logger.info("=" * 60)

    # 1. Load
    df = load_consolidated_data()

    # 2. Preprocess
    logger.info("Step 2: Preprocessing all curves...")
    results, qc_report = preprocess_all_curves(df)

    # 3. Feature matrix
    logger.info("Step 3: Building feature matrix...")
    data, curve_ids = build_feature_matrix(results)
    included_results = [r for r in results if r["success"]]

    # 4. Train SOM
    logger.info("Step 4: Training SOM...")
    som, som_summary = train_som(data)

    # 5. Assign
    logger.info("Step 5: Assigning clusters...")
    assignments = assign_clusters(som, data, curve_ids)

    # 6. Statistics
    logger.info("Step 6: Computing cluster statistics...")
    cluster_stats = compute_cluster_statistics(included_results, assignments, data)

    # 7. Plots
    logger.info("Step 7: Generating plots...")
    plot_cluster_curves(
        included_results, assignments, data, cluster_stats, som, out_dir,
    )
    plot_codebook_vectors(som, out_dir)
    plot_u_matrix(som, out_dir)
    plot_cluster_comparison(som, data, assignments, out_dir)
    plot_pca_visualization(som, data, assignments, out_dir)

    # 8. Save outputs
    logger.info("Step 8: Saving outputs...")
    save_preprocessed_curves(included_results, out_dir)
    save_outputs(
        included_results, qc_report, assignments,
        cluster_stats, som_summary, out_dir,
    )

    # 9. Quality checks
    run_quality_checks(results, qc_report, assignments, data, som_summary)

    logger.info("=" * 60)
    logger.info("Pipeline COMPLETE. Output: %s", out_dir)
    logger.info("=" * 60)

    # Print cluster summary table
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    print(cluster_stats[[
        "som_x", "som_y", "raw_cluster_id", "n_samples", "percentage",
        "mean_pce_t0", "mean_pce_t200", "shape_hint", "t_max_mean"
    ]].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
