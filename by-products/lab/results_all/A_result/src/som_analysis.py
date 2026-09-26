"""
src/som_analysis.py
====================
SOM training, BMU assignment, cluster metrics, and model persistence.

Author-confirmed parameters (from 20230816_degradation_analysis_revision_11_cleaned.ipynb):
  som_x, som_y = 2, 2
  sigma = 0.5
  learning_rate = 0.1
  iterations = 50_000
  random_seed = 42
  som.random_weights_init(data)
  som.train(data, 50000, verbose=False)

References:
  - Author notebook cell 33: MiniSom initialisation
  - Author notebook cell 35: training loop
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from minisom import MiniSom

from .config import SOM_CONFIG, logger

# =============================================================================
# Cluster metrics dataclass
# =============================================================================

@dataclass
class ClusterMetrics:
    som_x: int
    som_y: int
    n_curves: int
    percentage: float
    # Time-point arrays
    mean_curve: np.ndarray       # (1201,)
    median_curve: np.ndarray     # (1201,)
    std_curve: np.ndarray       # (1201,)
    q25_curve: np.ndarray       # (1201,)
    q75_curve: np.ndarray       # (1201,)
    codebook: np.ndarray        # (1201,)
    # BMU distances
    mean_bmu_dist: float
    median_bmu_dist: float
    # PCE at key time points (indices into the 1201-point grid)
    pce_t0: float
    pce_t10: float
    pce_t50: float
    pce_t100: float
    pce_t150: float
    pce_t200: float
    # Derived metrics
    peak_time_h: float
    change_0_10h: float
    change_0_50h: float
    change_0_200h: float
    slope_0_10h: float     # per hour
    slope_100_200h: float # per hour
    auc: float            # area under curve (trapezoid)
    # Naming (post-hoc)
    suggested_shape_name: str = ""
    naming_basis: str = ""


# =============================================================================
# Key time-point indices
# =============================================================================

# TIME_GRID = linspace(0, 200, 1201, endpoint=True)
# t[i] = i/6  (each index = 10 min = 1/6 h)
_HOUR_TO_IDX = {h: int(h * 6) for h in [0, 10, 50, 100, 150, 200]}


def _pce_at(df_curves: np.ndarray, hour: float) -> np.ndarray:
    """PCE values at a given hour for all curves. hour=0 → idx=0."""
    idx = int(round(hour * 6))
    return df_curves[:, idx]


def _mean_pce_at(df_curves: np.ndarray, hour: float) -> float:
    idx = int(round(hour * 6))
    return float(np.nanmean(df_curves[:, idx]))


def _curve_stats(df_curves: np.ndarray) -> dict:
    """Compute per-cluster statistics from a (n_curves, 1201) array."""
    from scipy.integrate import trapezoid
    idx0  = _HOUR_TO_IDX[0]
    idx10 = _HOUR_TO_IDX[10]
    idx50 = _HOUR_TO_IDX[50]
    idx100 = _HOUR_TO_IDX[100]
    idx150 = _HOUR_TO_IDX[150]
    idx200 = _HOUR_TO_IDX[200]

    mean_curve   = np.nanmean(df_curves, axis=0)
    median_curve = np.nanmedian(df_curves, axis=0)
    std_curve    = np.nanstd(df_curves, axis=0)
    q25_curve    = np.nanpercentile(df_curves, 25, axis=0)
    q75_curve    = np.nanpercentile(df_curves, 75, axis=0)

    # PCE at key time points (mean)
    pce_t0    = float(mean_curve[idx0])
    pce_t10   = float(mean_curve[idx10])
    pce_t50   = float(mean_curve[idx50])
    pce_t100  = float(mean_curve[idx100])
    pce_t150  = float(mean_curve[idx150])
    pce_t200  = float(mean_curve[idx200])

    # Peak time (index of max of mean curve)
    peak_idx = int(np.argmax(mean_curve))
    peak_time_h = peak_idx / 6.0

    # Changes
    change_0_10h  = pce_t10  - pce_t0
    change_0_50h  = pce_t50  - pce_t0
    change_0_200h = pce_t200 - pce_t0

    # Slopes (per hour)
    slope_0_10h   = (mean_curve[idx10]  - mean_curve[idx0])  / 10.0
    slope_100_200h = (mean_curve[idx200]  - mean_curve[idx100]) / 100.0

    # AUC
    time_axis = np.linspace(0, 200, df_curves.shape[1])
    auc = float(trapezoid(mean_curve, time_axis) / 200.0)   # normalised AUC

    return {
        "mean_curve": mean_curve, "median_curve": median_curve,
        "std_curve": std_curve, "q25_curve": q25_curve, "q75_curve": q75_curve,
        "pce_t0": pce_t0, "pce_t10": pce_t10, "pce_t50": pce_t50,
        "pce_t100": pce_t100, "pce_t150": pce_t150, "pce_t200": pce_t200,
        "peak_time_h": peak_time_h,
        "change_0_10h": change_0_10h, "change_0_50h": change_0_50h,
        "change_0_200h": change_0_200h,
        "slope_0_10h": slope_0_10h, "slope_100_200h": slope_100_200h,
        "auc": auc,
    }


# =============================================================================
# Main SOM training
# =============================================================================

def train_som(
    X: np.ndarray,
    shape: list[int, int] | None = None,
    sigma: float | None = None,
    learning_rate: float | None = None,
    iterations: int | None = None,
    random_seed: int | None = None,
    verbose: bool = True,
) -> tuple[MiniSom, dict]:
    """
    Train a MiniSom model.

    Args:
        X:           Feature matrix (n_samples, n_features)
        shape:       SOM grid shape [som_x, som_y]
        sigma:       Neighborhood radius
        learning_rate: Initial learning rate
        iterations:  Number of training iterations
        random_seed: Random seed for reproducibility
        verbose:     Print progress

    Returns:
        (trained_som, summary_dict)
    """
    cfg = SOM_CONFIG
    sx, sy = (shape if shape else cfg["shape"])
    sig  = sigma        if sigma        is not None else cfg["sigma"]
    lr   = learning_rate if learning_rate is not None else cfg["learning_rate"]
    n_iter = iterations  if iterations   is not None else cfg["iterations"]
    seed   = random_seed  if random_seed  is not None else cfg["random_seed"]

    logger.info("Training SOM: shape=%dx%d, sigma=%.1f, lr=%.1f, iter=%d, seed=%d",
                sx, sy, sig, lr, n_iter, seed)

    som = MiniSom(
        x=sx, y=sy,
        input_len=X.shape[1],
        sigma=sig,
        learning_rate=lr,
        activation_distance=cfg["activation_distance"],
        neighborhood_function=cfg["neighborhood_function"],
        random_seed=seed,
    )

    # Author confirmed: random_weights_init (not PCA)
    som.random_weights_init(X)

    # Author confirmed: som.train(data, 50000)
    som.train(X, n_iter, verbose=verbose)

    qe = som.quantization_error(X)
    te = som.topographic_error(X)

    logger.info("  QE=%.4f, TE=%.4f", qe, te)

    summary = {
        "som_x": sx, "som_y": sy,
        "sigma": sig, "lr": lr,
        "iterations": n_iter, "seed": seed,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "quantization_error": float(qe),
        "topographic_error": float(te),
    }
    return som, summary


# =============================================================================
# BMU assignment
# =============================================================================

def assign_bmu(som: MiniSom, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign Best Matching Units to all samples.

    Returns:
        (bmu_x, bmu_y, bmu_distances)
        All arrays of shape (n_samples,)
    """
    winners_x = np.zeros(X.shape[0], dtype=int)
    winners_y = np.zeros(X.shape[0], dtype=int)
    distances = np.zeros(X.shape[0], dtype=float)

    for i, x in enumerate(X):
        wx, wy = som.winner(x)
        winners_x[i] = wx
        winners_y[i] = wy
        # Euclidean distance to codebook vector
        w = som.get_weights()
        distances[i] = float(np.linalg.norm(x - w[wx, wy]))

    return winners_x, winners_y, distances


# =============================================================================
# Cluster metrics
# =============================================================================

def compute_cluster_metrics(
    X: np.ndarray,
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    bmu_distances: np.ndarray,
    shape: tuple[int, int],
    som: Optional["MiniSom"] = None,
) -> tuple[list[ClusterMetrics], dict]:
    """
    Compute detailed metrics for each SOM cluster.

    Args:
        X: Feature matrix (n_curves, n_time_points)
        winners_x, winners_y, bmu_distances: BMU assignments
        shape: SOM grid shape (som_x, som_y)
        som: Optional MiniSom instance for codebook vectors.
             If not provided, codebook field is set to mean_curve.

    Returns:
        (list_of_ClusterMetrics, dict keyed by (sx, sy))
    """
    sx, sy = shape
    metrics_by_node: dict[tuple[int, int], ClusterMetrics] = {}
    total = X.shape[0]

    for wx in range(sx):
        for wy in range(sy):
            mask = (winners_x == wx) & (winners_y == wy)
            n = mask.sum()
            if n == 0:
                logger.warning("  Empty SOM node: (%d, %d)", wx, wy)
                continue

            curves_in_node = X[mask]
            dists_in_node = bmu_distances[mask]

            # Basic statistics
            stats = _curve_stats(curves_in_node)

            # Codebook vector
            if som is not None:
                codebook = som.get_weights()[wx, wy]
            else:
                codebook = stats["mean_curve"]

            cm = ClusterMetrics(
                som_x=wx, som_y=wy,
                n_curves=n,
                percentage=float(n) / total * 100,
                mean_curve=stats["mean_curve"],
                median_curve=stats["median_curve"],
                std_curve=stats["std_curve"],
                q25_curve=stats["q25_curve"],
                q75_curve=stats["q75_curve"],
                codebook=codebook,
                mean_bmu_dist=float(np.mean(dists_in_node)),
                median_bmu_dist=float(np.median(dists_in_node)),
                pce_t0=stats["pce_t0"],
                pce_t10=stats["pce_t10"],
                pce_t50=stats["pce_t50"],
                pce_t100=stats["pce_t100"],
                pce_t150=stats["pce_t150"],
                pce_t200=stats["pce_t200"],
                peak_time_h=stats["peak_time_h"],
                change_0_10h=stats["change_0_10h"],
                change_0_50h=stats["change_0_50h"],
                change_0_200h=stats["change_0_200h"],
                slope_0_10h=stats["slope_0_10h"],
                slope_100_200h=stats["slope_100_200h"],
                auc=stats["auc"],
            )
            metrics_by_node[(wx, wy)] = cm

    logger.info("Cluster sizes:")
    for (wx, wy), cm in sorted(metrics_by_node.items()):
        logger.info("  Node (%d,%d): %d curves (%.1f%%)", wx, wy, cm.n_curves, cm.percentage)

    return list(metrics_by_node.values()), metrics_by_node


def _name_cluster_shape(cm: ClusterMetrics) -> tuple[str, str]:
    """
    Post-hoc descriptive naming for a cluster based on its mean curve metrics.

    Naming is based purely on curve shape, NOT preset physical labels.
    """
    pce200 = cm.pce_t200
    pce10  = cm.pce_t10
    pce0   = cm.pce_t0
    slope0_10  = cm.slope_0_10h
    slope100_200 = cm.slope_100_200h

    # Determine shape category
    if pce200 >= 0.90:
        base = "stable"
    elif pce200 >= 0.75:
        base = "slow_degradation"
    elif pce200 >= 0.50:
        base = "moderate_degradation"
    else:
        base = "rapid_degradation"

    # Initial behaviour
    if slope0_10 < -0.005:
        initial = "initial_drop"
    elif slope0_10 > 0.001:
        initial = "initial_gain"
    else:
        initial = "no_initial_change"

    # Decay rate change
    if slope100_200 < slope0_10 - 0.002:
        decay = "accelerating_loss"
    elif slope100_200 > slope0_10 + 0.002:
        decay = "decelerating_loss"
    else:
        decay = "steady_loss"

    shape_name = f"{initial}_{base}_{decay}"
    basis = (
        f"PCE@200h={pce200:.3f}, "
        f"slope_0_10h={slope0_10:.5f}/h, "
        f"slope_100_200h={slope100_200:.5f}/h, "
        f"Δ200h={cm.change_0_200h:.3f}"
    )
    return shape_name, basis


# =============================================================================
# Save outputs
# =============================================================================

def save_som_outputs(
    som: MiniSom,
    summary: dict,
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    bmu_distances: np.ndarray,
    curve_ids: np.ndarray,
    pce_maxima: np.ndarray,
    X: np.ndarray,
    metrics: list[ClusterMetrics],
    out_dir: Path,
) -> pd.DataFrame:
    """Save SOM model, weights, and assignments to disk."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save SOM model ───────────────────────────────────────────
    model_path = out_dir / "som_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(som, f)
    logger.info("Saved: som_model.pkl")

    # ── Save weights ────────────────────────────────────────────
    weights_path = out_dir / "som_weights.npy"
    np.save(weights_path, som.get_weights())
    logger.info("Saved: som_weights.npy  shape=%s", som.get_weights().shape)

    # ── Save cluster assignments ──────────────────────────────────
    # Build time-point PCE values at key hours
    def _pce_arr(curves: np.ndarray, hour: float) -> np.ndarray:
        idx = int(round(hour * 6))
        return curves[:, idx]

    pce_t0  = _pce_arr(X, 0)
    pce_t10 = _pce_arr(X, 10)
    pce_t50 = _pce_arr(X, 50)
    pce_t100 = _pce_arr(X, 100)
    pce_t150 = _pce_arr(X, 150)
    pce_t200 = _pce_arr(X, 200)
    peak_idx = np.argmax(X, axis=1)
    peak_times = peak_idx / 6.0

    assignments = pd.DataFrame({
        "curve_id": curve_ids,
        "raw_cluster_id": winners_x * 2 + winners_y,
        "som_node_x": winners_x,
        "som_node_y": winners_y,
        "bmu_distance": bmu_distances,
        "max_pce_0_200h": pce_maxima,
        "pce_norm_t0":  pce_t0,
        "pce_norm_t10": pce_t10,
        "pce_norm_t50": pce_t50,
        "pce_norm_t100": pce_t100,
        "pce_norm_t150": pce_t150,
        "pce_norm_t200": pce_t200,
        "peak_time_h": peak_times,
        "qc_status": "included",
    })
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    logger.info("Saved: cluster_assignments.csv")

    # ── Save cluster summary ─────────────────────────────────────
    summary_rows = []
    shape_metrics_rows = []
    for cm in metrics:
        shape_name, basis = _name_cluster_shape(cm)
        cm.suggested_shape_name = shape_name
        cm.naming_basis = basis

        summary_rows.append({
            "som_node_x": cm.som_x,
            "som_node_y": cm.som_y,
            "raw_cluster_id": cm.som_x * 2 + cm.som_y,
            "n_curves": cm.n_curves,
            "percentage": round(cm.percentage, 4),
            "mean_bmu_dist": round(cm.mean_bmu_dist, 6),
            "median_bmu_dist": round(cm.median_bmu_dist, 6),
            "pce_norm_t0": round(cm.pce_t0, 6),
            "pce_norm_t10": round(cm.pce_t10, 6),
            "pce_norm_t50": round(cm.pce_t50, 6),
            "pce_norm_t100": round(cm.pce_t100, 6),
            "pce_norm_t150": round(cm.pce_t150, 6),
            "pce_norm_t200": round(cm.pce_t200, 6),
            "peak_time_h": round(cm.peak_time_h, 3),
            "slope_0_10h": round(cm.slope_0_10h, 6),
            "slope_100_200h": round(cm.slope_100_200h, 6),
            "auc": round(cm.auc, 6),
            "suggested_shape_name": shape_name,
            "naming_basis": basis,
        })

        shape_metrics_rows.append({
            "som_node_x": cm.som_x,
            "som_node_y": cm.som_y,
            "raw_cluster_id": cm.som_x * 2 + cm.som_y,
            "change_0_10h": round(cm.change_0_10h, 6),
            "change_0_50h": round(cm.change_0_50h, 6),
            "change_0_200h": round(cm.change_0_200h, 6),
            "slope_0_10h": round(cm.slope_0_10h, 6),
            "slope_100_200h": round(cm.slope_100_200h, 6),
            "auc": round(cm.auc, 6),
            "suggested_shape_name": shape_name,
            "naming_basis": basis,
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "cluster_summary.csv", index=False)
    pd.DataFrame(shape_metrics_rows).to_csv(out_dir / "cluster_shape_metrics.csv", index=False)
    logger.info("Saved: cluster_summary.csv, cluster_shape_metrics.csv")

    # ── Save run metadata ───────────────────────────────────────
    som_path = out_dir / "som_summary.json"
    with open(som_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: som_summary.json")

    return assignments
