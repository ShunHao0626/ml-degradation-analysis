"""
src/validation.py
=================
Validation analyses:
  1. Quantisation error sweep (n=2..10)
  2. Parameter sensitivity analysis (sigma, learning_rate variants)
  3. K-means auxiliary validation
  4. Cluster matching across runs

Topology rule for QE sweep:
  - n=4: use the author-confirmed 2×2 topology
  - other n: use 1×n (implementation assumption — not confirmed in author code)

References:
  - Author notebook cell 75: QE data for n=2..10
  - Author notebook cell 58: TimeSeriesKMeans(n_clusters=4, metric="dtw")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import SOM_CONFIG, VAL_CONFIG, logger

# =============================================================================
# QE sweep
# =============================================================================

def _topology_for_n(n: int, override: dict[int, tuple[int, int]]) -> tuple[int, int]:
    """Return (som_x, som_y) for a given node count n.

    - n=4: use the author-confirmed 2×2
    - other n: 1×n (implementation assumption)
    """
    if n in override:
        return override[n]
    return (1, n)


def run_qe_sweep(
    X: np.ndarray,
    n_range: list[int] | None = None,
    sigma: float = 0.5,
    learning_rate: float = 0.1,
    iterations: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train SOMs with varying numbers of nodes and compute quantisation error.

    Args:
        X:         Feature matrix (n_samples, n_features)
        n_range:   List of node counts to test (default: 2..10)
        sigma, learning_rate, iterations, seed: SOM hyper-parameters

    Returns:
        DataFrame with columns: n_nodes, som_x, som_y, quantization_error
    """
    if n_range is None:
        n_range = VAL_CONFIG["qe_sweep_n"]

    topo_override = VAL_CONFIG.get("qe_sweep_topology_override", {})
    rows = []

    logger.info("Running QE sweep for n = %s ...", n_range)
    for n in n_range:
        sx, sy = _topology_for_n(n, topo_override)

        som = MiniSom(
            x=sx, y=sy, input_len=X.shape[1],
            sigma=sigma, learning_rate=learning_rate,
            activation_distance=SOM_CONFIG["activation_distance"],
            neighborhood_function=SOM_CONFIG["neighborhood_function"],
            random_seed=seed,
        )
        som.random_weights_init(X)
        som.train(X, iterations, verbose=False)

        qe = som.quantization_error(X)
        rows.append({"n_nodes": n, "som_x": sx, "som_y": sy, "quantization_error": qe})
        logger.info("  n=%2d  topology=%dx%d  QE=%.6f", n, sx, sy, qe)

    df = pd.DataFrame(rows)

    # Identify 2×2 result
    is_main = (df["som_x"] == 2) & (df["som_y"] == 2)
    if is_main.any():
        main_qe = df.loc[is_main, "quantization_error"].values[0]
        logger.info("  Main model (2×2) QE = %.6f", main_qe)

    return df


# =============================================================================
# Sensitivity analysis
# =============================================================================

SENSITIVITY_CONFIGS = {
    "sigma0.3_lr0.1": {"sigma": 0.3, "learning_rate": 0.1},
    "sigma0.5_lr0.3": {"sigma": 0.5, "learning_rate": 0.3},
}


def run_sensitivity_som(
    X: np.ndarray,
    sigma: float,
    learning_rate: float,
    shape: list[int, int] | None = None,
    iterations: int = 50_000,
    seed: int = 42,
) -> tuple[MiniSom, dict]:
    """Train a sensitivity SOM with specified parameters."""
    cfg = SOM_CONFIG
    sx, sy = shape if shape else cfg["shape"]

    som = MiniSom(
        x=sx, y=sy, input_len=X.shape[1],
        sigma=sigma, learning_rate=learning_rate,
        activation_distance=cfg["activation_distance"],
        neighborhood_function=cfg["neighborhood_function"],
        random_seed=seed,
    )
    som.random_weights_init(X)
    som.train(X, iterations, verbose=False)

    qe = som.quantization_error(X)
    summary = {
        "sigma": sigma, "lr": learning_rate,
        "shape": [sx, sy],
        "iterations": iterations, "seed": seed,
        "n_samples": X.shape[0],
        "quantization_error": float(qe),
    }
    return som, summary


def run_all_sensitivity(
    X: np.ndarray,
    out_dir: Path,
) -> dict[str, dict]:
    """
    Run sensitivity SOMs for all SENSITIVITY_CONFIGS.

    Returns:
        Dict keyed by config name, each containing:
          som, summary, winners_x, winners_y, bmu_distances
    """
    results = {}
    cfg = SOM_CONFIG

    for name, params in SENSITIVITY_CONFIGS.items():
        logger.info("Sensitivity run: %s", name)
        som, summary = run_sensitivity_som(
            X,
            sigma=params["sigma"],
            learning_rate=params["learning_rate"],
            shape=cfg["shape"],
            iterations=cfg["iterations"],
            seed=cfg["random_seed"],
        )

        wx, wy, dists = _assign_bmu(som, X)
        summary["cluster_sizes"] = _cluster_sizes(wx, wy)

        # Save SOM
        with open(out_dir / f"som_model_{name}.pkl", "wb") as f:
            pickle.dump(som, f)

        results[name] = {
            "som": som,
            "summary": summary,
            "winners_x": wx,
            "winners_y": wy,
            "bmu_distances": dists,
        }
        logger.info("  QE=%.4f, cluster sizes: %s",
                    summary["quantization_error"], summary["cluster_sizes"])

    return results


# =============================================================================
# K-means
# =============================================================================

def run_kmeans_validation(
    X: np.ndarray,
    k_range: list[int] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Run K-means (sklearn) for k=2..10 and compute WCSS/inertia.

    Also train k=4 for centroid comparison.

    Author's approach: TimeSeriesKMeans(n_clusters=4, metric="dtw")
    Fallback: sklearn.cluster.KMeans (used here for comparison)

    Returns:
        (wcss_df, k4_labels_dict)
    """
    if k_range is None:
        k_range = VAL_CONFIG["kmeans_k_range"]

    wcss_rows = []
    k4_labels: dict[int, np.ndarray] = {}

    logger.info("Running K-means for k = %s ...", k_range)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        wcss = km.inertia_
        wcss_rows.append({"k": k, "wcss": wcss, "n_clusters": k})
        logger.info("  k=%2d  WCSS=%.4f", k, wcss)

        if k == 4:
            k4_labels[4] = labels

    wcss_df = pd.DataFrame(wcss_rows)
    return wcss_df, k4_labels


def kmeans_k4_comparison(
    X: np.ndarray,
    som_wx: np.ndarray,
    som_wy: np.ndarray,
    kmeans_labels: np.ndarray,
    random_state: int = 42,
) -> dict:
    """
    Compare SOM clusters with k=4 k-means clusters using:
      - Adjusted Rand Index (ARI)
      - Normalized Mutual Information (NMI)
    """
    som_labels = som_wx * 2 + som_wy   # flatten SOM coordinates to labels

    ari = adjusted_rand_score(som_labels, kmeans_labels)
    nmi = normalized_mutual_info_score(som_labels, kmeans_labels)

    logger.info("SOM vs K-means(k=4): ARI=%.4f, NMI=%.4f", ari, nmi)

    return {"ari": float(ari), "nmi": float(nmi)}


# =============================================================================
# Cluster matching (Hungarian / label-permutation agreement)
# =============================================================================

def match_clusters_hungarian(
    X: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Find the best label permutation between two clusterings using
    the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Returns:
        permutation array mapping labels_b → matched labels_a.
    """
    from scipy.optimize import linear_sum_assignment

    # Build cost matrix: for each pair of labels, compute
    # sum of Euclidean distances between cluster centroids
    centroids_a = np.array([
        X[labels_a == c].mean(axis=0) for c in range(n_clusters)
    ])
    centroids_b = np.array([
        X[labels_b == c].mean(axis=0) for c in range(n_clusters)
    ])

    # Cost = negative similarity (we minimize)
    cost = np.linalg.norm(centroids_a[:, None, :] - centroids_b[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    permutation = np.full(n_clusters, -1)
    permutation[col_ind] = row_ind
    return permutation


# =============================================================================
# Helpers
# =============================================================================

def _assign_bmu(som: MiniSom, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BMU assignment helper."""
    wx = np.zeros(X.shape[0], dtype=int)
    wy = np.zeros(X.shape[0], dtype=int)
    dists = np.zeros(X.shape[0], dtype=float)
    for i, x in enumerate(X):
        bx, by = som.winner(x)
        wx[i], wy[i] = bx, by
        w = som.get_weights()
        dists[i] = float(np.linalg.norm(x - w[bx, by]))
    return wx, wy, dists


def _cluster_sizes(wx: np.ndarray, wy: np.ndarray) -> dict[str, int]:
    """Return cluster sizes dict."""
    n_clusters = max(wx) + 1
    sizes = {}
    for c in range(n_clusters):
        mask = (wx == c)
        sizes[str(c)] = int(mask.sum())
    return sizes


# =============================================================================
# Save validation outputs
# =============================================================================

def save_validation_outputs(
    qe_df: pd.DataFrame,
    sensitivity_results: dict,
    kmeans_wcss_df: pd.DataFrame,
    k4_comparison: dict,
    out_dir: Path,
) -> None:
    """Save all validation outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    qe_df.to_csv(out_dir / "quantisation_error_by_n.csv", index=False)
    logger.info("Saved: quantisation_error_by_n.csv")

    # Sensitivity summaries
    sens_rows = []
    for name, res in sensitivity_results.items():
        s = res["summary"]
        sizes = _cluster_sizes(res["winners_x"], res["winners_y"])
        for node, n in sizes.items():
            sens_rows.append({
                "sensitivity_config": name,
                "som_sigma": s["sigma"],
                "som_lr": s["lr"],
                "som_node": node,
                "n_curves": n,
                "quantization_error": s["quantization_error"],
            })
    pd.DataFrame(sens_rows).to_csv(out_dir / "sensitivity_summary.csv", index=False)
    logger.info("Saved: sensitivity_summary.csv")

    kmeans_wcss_df.to_csv(out_dir / "kmeans_wcss.csv", index=False)
    logger.info("Saved: kmeans_wcss.csv")

    comp_df = pd.DataFrame([k4_comparison])
    comp_df.to_csv(out_dir / "som_vs_kmeans_comparison.csv", index=False)
    logger.info("Saved: som_vs_kmeans_comparison.csv (ARI=%.4f, NMI=%.4f)",
                k4_comparison["ari"], k4_comparison["nmi"])
