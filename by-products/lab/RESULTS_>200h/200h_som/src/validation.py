"""Validation, QE sweep, parameter sensitivity, k-means helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import config
from .som_analysis import (
    SOMSpec,
    spec_from_config,
    train_som,
    assign_bmu_clusters,
    bmu_to_cluster_id,
    quantization_error,
    match_clusters_across_runs,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# QE sweep over multiple node counts
# ----------------------------------------------------------------------------
def _fallback_topology(n_nodes: int) -> tuple[int, int]:
    """Return a deterministic fallback topology (1 × n)."""
    return (1, n_nodes)


def qe_sweep(
    X: np.ndarray,
    node_counts: List[int] | None = None,
    base_sigma: float | None = None,
    base_lr: float | None = None,
) -> pd.DataFrame:
    """Train a SOM for each requested node count and report QE.

    The fallback topology is 1 × n, clearly marked as an implementation
    assumption, since the author code does not document a strict rule.
    """
    if node_counts is None:
        node_counts = list(config.QE_SWEEP["node_counts"])
    if base_sigma is None:
        base_sigma = float(config.SOM_CONFIG["sigma"])
    if base_lr is None:
        base_lr = float(config.SOM_CONFIG["learning_rate"])

    rows = []
    main_shape = tuple(config.SOM_CONFIG["shape"])
    main_n = main_shape[0] * main_shape[1]
    for n in node_counts:
        # When n matches the main model's node count, use the main topology
        if n == main_n:
            shape = main_shape
            source = "main (2x2)"
        else:
            shape = _fallback_topology(n)
            source = "1xn fallback (implementation assumption)"
        spec = SOMSpec(
            name=f"qe_n{n}",
            som_shape=shape,
            sigma=base_sigma,
            learning_rate=base_lr,
            iterations=int(config.SOM_CONFIG["iterations"]),
            random_seed=int(config.SOM_CONFIG["random_seed"]),
        )
        logger.info("QE sweep training SOM shape=%s, n_nodes=%d", shape, n)
        som = train_som(X, spec)
        qe = quantization_error(som, X)
        rows.append({
            "n_nodes": n,
            "topology_x": shape[0],
            "topology_y": shape[1],
            "topology": f"{shape[0]}x{shape[1]}",
            "quantisation_error": qe,
            "sigma": base_sigma,
            "learning_rate": base_lr,
            "random_seed": spec.random_seed,
            "topology_source": source,
        })

    # ALSO add a 1x4 reference topology for n=4 (the user requested that
    # both the QE-sweep reference topology and the main 2x2 topology be
    # reported for n=4).
    if main_n in node_counts and (1, main_n) != main_shape:
        spec_ref = SOMSpec(
            name=f"qe_n{main_n}_1xN",
            som_shape=(1, main_n),
            sigma=base_sigma,
            learning_rate=base_lr,
            iterations=int(config.SOM_CONFIG["iterations"]),
            random_seed=int(config.SOM_CONFIG["random_seed"]),
        )
        logger.info("QE sweep 1xN reference for n=%d", main_n)
        som_ref = train_som(X, spec_ref)
        qe_ref = quantization_error(som_ref, X)
        rows.append({
            "n_nodes": main_n,
            "topology_x": 1,
            "topology_y": main_n,
            "topology": f"1x{main_n}",
            "quantisation_error": qe_ref,
            "sigma": base_sigma,
            "learning_rate": base_lr,
            "random_seed": spec_ref.random_seed,
            "topology_source": "1xn reference (for n=4 topology comparison)",
        })

    df = pd.DataFrame(rows).sort_values("n_nodes").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Parameter sensitivity
# ----------------------------------------------------------------------------
def run_sensitivity(
    X: np.ndarray, sample_ids: list[str], base_max_pce: list[float]
) -> tuple[pd.DataFrame, list[dict]]:
    """Run sensitivity analysis for the configs in `config.SENSITIVITY_CONFIGS`.

    Returns:
        df_assign: DataFrame of BMU+cluster info for each sensitivity model.
        results: list of dicts with `name`, `som_model`, `assignments`, `summary`.
    """
    df_rows = []
    results = []
    for cfg in config.SENSITIVITY_CONFIGS:
        spec = SOMSpec(
            name=cfg["name"],
            som_shape=tuple(config.SOM_CONFIG["shape"]),
            sigma=float(cfg["sigma"]),
            learning_rate=float(cfg["learning_rate"]),
            iterations=int(config.SOM_CONFIG["iterations"]),
            random_seed=int(config.SOM_CONFIG["random_seed"]),
        )
        logger.info("Training sensitivity SOM %s (sigma=%.2f, lr=%.2f)",
                    cfg["name"], cfg["sigma"], cfg["learning_rate"])
        som = train_som(X, spec)
        coords, dist = assign_bmu_clusters(som, X)
        cids = bmu_to_cluster_id(coords, spec.som_shape)
        qe = quantization_error(som, X)
        df_rows.append({
            "model": cfg["name"],
            "sigma": cfg["sigma"],
            "learning_rate": cfg["learning_rate"],
            "quantisation_error": qe,
        })
        results.append({
            "name": cfg["name"],
            "spec": spec,
            "som_model": som,
            "coords": coords,
            "distances": dist,
            "cluster_ids": cids,
            "quantisation_error": qe,
        })
    df_assign = pd.DataFrame(df_rows)
    return df_assign, results


# ----------------------------------------------------------------------------
# Cross-model ARI/NMI
# ----------------------------------------------------------------------------
def ari_nmi(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    return ari, nmi


# ----------------------------------------------------------------------------
# K-means elbow / validation
# ----------------------------------------------------------------------------
def run_kmeans(X: np.ndarray, k_range: list[int] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    from sklearn.cluster import KMeans
    if k_range is None:
        k_range = list(config.KMEANS_CONFIG["k_range"])

    rows = []
    results = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=int(config.KMEANS_CONFIG["random_state"]),
            n_init=int(config.KMEANS_CONFIG["n_init"]),
            max_iter=int(config.KMEANS_CONFIG["max_iter"]),
        )
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        rows.append({"k": k, "wcss_inertia": inertia})
        centers = km.cluster_centers_
        results.append({"k": k, "labels": labels, "inertia": inertia, "centers": centers})
    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return df, results
