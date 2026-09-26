"""SOM training, SI-style cluster-number assessment, and traceable outputs."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .config import AnalysisConfig
from .preprocessing import PreprocessedDataset

LOGGER = logging.getLogger("som200h")


@dataclass
class SOMRun:
    n_nodes: int
    topology: tuple[int, int]
    seed: int
    som: MiniSom
    labels: np.ndarray
    coords: np.ndarray
    distances: np.ndarray
    qe: float
    te: float


def train_som(
    x: np.ndarray,
    n_nodes: int,
    topology: tuple[int, int],
    seed: int,
    config: AnalysisConfig,
) -> SOMRun:
    np.random.seed(seed)
    som = MiniSom(
        x=topology[0],
        y=topology[1],
        input_len=x.shape[1],
        sigma=config.som_sigma,
        learning_rate=config.som_learning_rate,
        topology="rectangular",
        neighborhood_function="gaussian",
        activation_distance="euclidean",
        random_seed=seed,
    )
    som.random_weights_init(x)
    # Mirrors the author's existing MiniSom.train workflow.
    som.train(x, config.som_iterations, verbose=False)
    coords = np.asarray([som.winner(row) for row in x], dtype=int)
    labels = coords[:, 0] * topology[1] + coords[:, 1]
    weights = som.get_weights()
    distances = np.asarray(
        [
            np.linalg.norm(weights[coord[0], coord[1]] - row)
            for row, coord in zip(x, coords)
        ],
        dtype=float,
    )
    return SOMRun(
        n_nodes=n_nodes,
        topology=topology,
        seed=seed,
        som=som,
        labels=labels,
        coords=coords,
        distances=distances,
        qe=float(som.quantization_error(x)),
        te=float(som.topographic_error(x)),
    )


def _time_index(time_grid: np.ndarray, hour: float) -> int:
    return int(np.argmin(np.abs(time_grid - hour)))


def _shape_name(row: pd.Series) -> str:
    p0 = float(row["pce_norm_0h_mean"])
    p10 = float(row["pce_norm_10h_mean"])
    p200 = float(row["pce_norm_200h_mean"])
    initial_delta = p10 - p0
    total_delta = p200 - p0
    if initial_delta > 0.01:
        initial = "initial_gain"
    elif initial_delta < -0.02:
        initial = "initial_drop"
    else:
        initial = "no_initial_change"

    if total_delta > -0.10:
        magnitude = "stable"
    elif total_delta > -0.40:
        magnitude = "moderate_loss"
    else:
        magnitude = "rapid_loss"

    return f"{initial}_{magnitude}"


def build_cluster_outputs(
    dataset: PreprocessedDataset,
    run: SOMRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = dataset.x_smoothed
    meta = dataset.metadata.reset_index(drop=True)
    time_grid = dataset.time_grid
    weights = run.som.get_weights()
    flat_weights = weights.reshape(run.n_nodes, x.shape[1])

    assignments = meta.copy()
    assignments["raw_cluster_id"] = run.labels
    assignments["som_node_x"] = run.coords[:, 0]
    assignments["som_node_y"] = run.coords[:, 1]
    assignments["node_label"] = [
        f"node_{a}_{b}" for a, b in run.coords
    ]
    assignments["bmu_distance"] = run.distances

    indices = {
        hour: _time_index(time_grid, hour)
        for hour in (0.0, 10.0, 50.0, 100.0, 150.0, 200.0)
    }
    for hour, idx in indices.items():
        assignments[f"pce_norm_{int(hour)}h"] = dataset.x_normalized[:, idx]
    assignments["peak_time_h"] = time_grid[
        np.argmax(dataset.x_normalized, axis=1)
    ]

    summary_rows: list[dict] = []
    centroid_rows: list[dict] = []
    for cluster_id in range(run.n_nodes):
        mask = run.labels == cluster_id
        cluster_x = x[mask]
        node_x = cluster_id // run.topology[1]
        node_y = cluster_id % run.topology[1]
        if cluster_x.shape[0] == 0:
            summary_rows.append(
                {
                    "raw_cluster_id": cluster_id,
                    "som_node_x": node_x,
                    "som_node_y": node_y,
                    "n_curves": 0,
                    "fraction": 0.0,
                }
            )
            continue

        mean_curve = np.mean(cluster_x, axis=0)
        median_curve = np.median(cluster_x, axis=0)
        p_values = {
            hour: float(np.mean(dataset.x_normalized[mask, idx]))
            for hour, idx in indices.items()
        }
        auc = float(np.trapz(mean_curve, time_grid))
        summary_rows.append(
            {
                "raw_cluster_id": cluster_id,
                "som_node_x": node_x,
                "som_node_y": node_y,
                "n_curves": int(mask.sum()),
                "fraction": float(mask.mean()),
                "pce_norm_0h_mean": p_values[0.0],
                "pce_norm_10h_mean": p_values[10.0],
                "pce_norm_50h_mean": p_values[50.0],
                "pce_norm_100h_mean": p_values[100.0],
                "pce_norm_150h_mean": p_values[150.0],
                "pce_norm_200h_mean": p_values[200.0],
                "delta_0_200h_mean": p_values[200.0] - p_values[0.0],
                "slope_0_10h_per_h": (p_values[10.0] - p_values[0.0]) / 10.0,
                "slope_100_200h_per_h": (
                    p_values[200.0] - p_values[100.0]
                )
                / 100.0,
                "auc_0_200h": auc,
                "mean_bmu_distance": float(np.mean(run.distances[mask])),
                "median_bmu_distance": float(np.median(run.distances[mask])),
            }
        )
        centroid_row = {
            "raw_cluster_id": cluster_id,
            "n_curves": int(mask.sum()),
        }
        centroid_row.update(
            {f"t_{i:04d}": value for i, value in enumerate(mean_curve)}
        )
        centroid_rows.append(centroid_row)

    summary = pd.DataFrame(summary_rows)
    nonempty = summary["n_curves"] > 0
    summary.loc[nonempty, "suggested_shape_name"] = summary.loc[
        nonempty
    ].apply(_shape_name, axis=1)

    # Stable ordered class IDs make the final result easier to interpret.
    order = (
        summary[nonempty]
        .sort_values(
            ["pce_norm_200h_mean", "auc_0_200h"],
            ascending=[False, False],
        )["raw_cluster_id"]
        .tolist()
    )
    ordered_map = {raw_id: index + 1 for index, raw_id in enumerate(order)}
    assignments["ordered_class_id"] = assignments["raw_cluster_id"].map(ordered_map)
    summary["ordered_class_id"] = summary["raw_cluster_id"].map(ordered_map)
    assignments = assignments.merge(
        summary[
            ["raw_cluster_id", "ordered_class_id", "suggested_shape_name"]
        ],
        on=["raw_cluster_id", "ordered_class_id"],
        how="left",
    )
    assignments["class_label"] = assignments["ordered_class_id"].map(
        lambda value: f"class_{int(value):02d}"
    )
    summary["class_label"] = summary["ordered_class_id"].map(
        lambda value: f"class_{int(value):02d}" if pd.notna(value) else ""
    )

    centroid_table = pd.DataFrame(centroid_rows)
    if not centroid_table.empty:
        centroid_table["ordered_class_id"] = centroid_table["raw_cluster_id"].map(
            ordered_map
        )
        centroid_table["class_label"] = centroid_table["ordered_class_id"].map(
            lambda value: f"class_{int(value):02d}"
        )
        centroid_table = centroid_table.sort_values("ordered_class_id")

    correlations = np.corrcoef(flat_weights)
    corr_rows: list[dict] = []
    for i, j in itertools.product(range(run.n_nodes), repeat=2):
        rmse = float(np.sqrt(np.mean((flat_weights[i] - flat_weights[j]) ** 2)))
        corr_rows.append(
            {
                "cluster_i": i,
                "cluster_j": j,
                "pearson_correlation": float(correlations[i, j]),
                "centroid_rmse": rmse,
            }
        )
    pairwise = pd.DataFrame(corr_rows)
    return assignments, summary, centroid_table, pairwise


def _pairwise_seed_stability(
    labels_by_seed: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, float, float]:
    rows: list[dict] = []
    aris: list[float] = []
    for seed_a, seed_b in itertools.combinations(sorted(labels_by_seed), 2):
        labels_a = labels_by_seed[seed_a]
        labels_b = labels_by_seed[seed_b]
        ari = float(adjusted_rand_score(labels_a, labels_b))
        nmi = float(normalized_mutual_info_score(labels_a, labels_b))
        rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ARI": ari,
                "NMI": nmi,
            }
        )
        aris.append(ari)
    return pd.DataFrame(rows), float(np.mean(aris)), float(np.min(aris))


def _validation_embedding(x: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    n_components = min(
        config.pca_components_for_validation,
        x.shape[0] - 1,
        x.shape[1],
    )
    # NumPy 2.0 + the randomized sklearn solver can emit overflow warnings on
    # otherwise well-bounded degradation curves.  The deterministic full SVD is
    # fast at this matrix size and keeps all validation indices reproducible.
    return PCA(
        n_components=n_components,
        svd_solver="full",
        random_state=42,
    ).fit_transform(x)


def _davies_bouldin_score_stable(
    x: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the standard DB index without NumPy matmul-based distances."""
    unique = np.unique(labels)
    centroids = np.asarray([x[labels == label].mean(axis=0) for label in unique])
    within = np.asarray(
        [
            np.sqrt(np.sum((x[labels == label] - centroid) ** 2, axis=1)).mean()
            for label, centroid in zip(unique, centroids)
        ]
    )
    between = cdist(centroids, centroids, metric="euclidean")
    ratios = (within[:, None] + within[None, :]) / np.where(
        between > 0.0,
        between,
        np.inf,
    )
    np.fill_diagonal(ratios, -np.inf)
    return float(np.mean(np.max(ratios, axis=1)))


def _kmeans_plus_plus(
    x: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((n_clusters, x.shape[1]), dtype=float)
    centers[0] = x[int(rng.integers(x.shape[0]))]
    closest_sq = cdist(x, centers[:1], metric="sqeuclidean")[:, 0]
    for index in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total > 0.0:
            selected = int(rng.choice(x.shape[0], p=closest_sq / total))
        else:
            selected = int(rng.integers(x.shape[0]))
        centers[index] = x[selected]
        candidate_sq = cdist(
            x,
            centers[index : index + 1],
            metric="sqeuclidean",
        )[:, 0]
        closest_sq = np.minimum(closest_sq, candidate_sq)
    return centers


def _lloyd_kmeans_once(
    x: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iter: int = 500,
) -> tuple[np.ndarray, float]:
    """A small, deterministic Lloyd implementation using stable SciPy distances."""
    centers = _kmeans_plus_plus(x, n_clusters, np.random.default_rng(seed))
    previous_labels: np.ndarray | None = None
    for _ in range(max_iter):
        distances = cdist(x, centers, metric="sqeuclidean")
        labels = np.argmin(distances, axis=1)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        previous_labels = labels.copy()
        nearest_sq = distances[np.arange(x.shape[0]), labels]
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[int(np.argmax(nearest_sq))]

    final_distances = cdist(x, centers, metric="sqeuclidean")
    final_labels = np.argmin(final_distances, axis=1)
    inertia = float(
        final_distances[np.arange(x.shape[0]), final_labels].sum()
    )
    return final_labels, inertia


def _robust_kmeans(
    x: np.ndarray,
    n_clusters: int,
    n_init: int,
    base_seed: int = 42,
) -> tuple[np.ndarray, float]:
    best_labels: np.ndarray | None = None
    best_inertia = math.inf
    for initialization in range(n_init):
        labels, inertia = _lloyd_kmeans_once(
            x,
            n_clusters,
            seed=base_seed + initialization,
        )
        if inertia < best_inertia:
            best_labels = labels
            best_inertia = inertia
    if best_labels is None or not math.isfinite(best_inertia):
        raise RuntimeError(f"K-means failed for k={n_clusters}")
    return best_labels, best_inertia


def run_som_scan(
    dataset: PreprocessedDataset,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[int, SOMRun]]:
    x = dataset.x_smoothed
    embedding = _validation_embedding(x, config)
    # Precompute with SciPy's numerically stable distance kernel.  Passing raw
    # vectors to sklearn's silhouette implementation triggers spurious
    # overflow warnings with NumPy 2.0 on this otherwise bounded dataset.
    validation_distances = cdist(embedding, embedding, metric="euclidean")
    dataset_root = config.output_root / "04_som_results" / dataset.name
    dataset_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    primary_runs: dict[int, SOMRun] = {}

    for n_nodes in config.som_node_counts:
        topology = config.topology_by_n[n_nodes]
        LOGGER.info(
            "%s: training %d-node SOM (%dx%d), %d seeds",
            dataset.name,
            n_nodes,
            topology[0],
            topology[1],
            len(config.som_stability_seeds),
        )
        labels_by_seed: dict[int, np.ndarray] = {}
        seed_rows: list[dict] = []
        primary: SOMRun | None = None
        for seed in config.som_stability_seeds:
            run = train_som(x, n_nodes, topology, seed, config)
            labels_by_seed[seed] = run.labels
            sizes = np.bincount(run.labels, minlength=n_nodes)
            seed_rows.append(
                {
                    "dataset": dataset.name,
                    "n_nodes": n_nodes,
                    "topology": f"{topology[0]}x{topology[1]}",
                    "seed": seed,
                    "quantisation_error": run.qe,
                    "topographic_error": run.te,
                    "n_empty_nodes": int(np.sum(sizes == 0)),
                    "min_cluster_size": int(np.min(sizes)),
                    "max_cluster_size": int(np.max(sizes)),
                }
            )
            if seed == config.som_primary_seed:
                primary = run
        if primary is None:
            raise RuntimeError("Primary seed was not included in stability seeds")
        primary_runs[n_nodes] = primary

        stability, mean_ari, min_ari = _pairwise_seed_stability(labels_by_seed)
        n_dir = dataset_root / f"n_{n_nodes:02d}"
        n_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seed_rows).to_csv(n_dir / "seed_metrics.csv", index=False)
        stability.to_csv(n_dir / "pairwise_seed_stability.csv", index=False)

        assignments, summary, centroids, pairwise = build_cluster_outputs(
            dataset, primary
        )
        assignments.to_csv(n_dir / "cluster_assignments.csv", index=False)
        summary.to_csv(n_dir / "cluster_summary.csv", index=False)
        centroids.to_csv(n_dir / "centroid_curves.csv", index=False)
        pairwise.to_csv(n_dir / "centroid_pairwise_similarity.csv", index=False)
        np.save(n_dir / "som_weights.npy", primary.som.get_weights())
        with (n_dir / "som_model.pkl").open("wb") as handle:
            pickle.dump(primary.som, handle)

        offdiag = pairwise["cluster_i"] != pairwise["cluster_j"]
        max_corr = float(
            pairwise.loc[offdiag, "pearson_correlation"].max()
        )
        min_rmse = float(pairwise.loc[offdiag, "centroid_rmse"].min())
        sizes = np.bincount(primary.labels, minlength=n_nodes)
        unique_labels = np.unique(primary.labels)
        if unique_labels.size >= 2:
            silhouette = float(
                silhouette_score(
                    validation_distances,
                    primary.labels,
                    metric="precomputed",
                )
            )
            db = _davies_bouldin_score_stable(embedding, primary.labels)
            ch = float(calinski_harabasz_score(embedding, primary.labels))
        else:
            silhouette = math.nan
            db = math.nan
            ch = math.nan

        seed_frame = pd.DataFrame(seed_rows)
        metric_rows.append(
            {
                "dataset": dataset.name,
                "n_nodes": n_nodes,
                "topology": f"{topology[0]}x{topology[1]}",
                "primary_seed": config.som_primary_seed,
                "qe_primary_seed": primary.qe,
                "qe_mean_across_seeds": seed_frame[
                    "quantisation_error"
                ].mean(),
                "qe_std_across_seeds": seed_frame[
                    "quantisation_error"
                ].std(ddof=0),
                "te_primary_seed": primary.te,
                "te_mean_across_seeds": seed_frame[
                    "topographic_error"
                ].mean(),
                "mean_pairwise_seed_ARI": mean_ari,
                "min_pairwise_seed_ARI": min_ari,
                "silhouette_pca20": silhouette,
                "davies_bouldin_pca20": db,
                "calinski_harabasz_pca20": ch,
                "max_centroid_correlation": max_corr,
                "min_centroid_rmse": min_rmse,
                "n_empty_nodes": int(np.sum(sizes == 0)),
                "min_cluster_size": int(np.min(sizes)),
                "min_cluster_fraction": float(np.min(sizes) / x.shape[0]),
                "max_cluster_size": int(np.max(sizes)),
                "max_cluster_fraction": float(np.max(sizes) / x.shape[0]),
            }
        )
        LOGGER.info(
            "%s: n=%d complete; QE=%.6f, mean seed ARI=%.4f",
            dataset.name,
            n_nodes,
            primary.qe,
            mean_ari,
        )

    metrics = pd.DataFrame(metric_rows).sort_values("n_nodes").reset_index(drop=True)
    metrics["qe_relative_improvement_from_previous"] = (
        -metrics["qe_mean_across_seeds"].pct_change()
    )
    metrics["qe_relative_improvement_to_next"] = (
        (
            metrics["qe_mean_across_seeds"]
            - metrics["qe_mean_across_seeds"].shift(-1)
        )
        / metrics["qe_mean_across_seeds"]
    )
    metrics.to_csv(dataset_root / "som_cluster_number_metrics.csv", index=False)
    return metrics, primary_runs


def run_kmeans_validation(
    dataset: PreprocessedDataset,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    x = dataset.x_smoothed
    embedding = _validation_embedding(x, config)
    validation_distances = cdist(embedding, embedding, metric="euclidean")
    rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in config.som_node_counts:
        LOGGER.info("%s: KMeans validation k=%d", dataset.name, k)
        labels, inertia = _robust_kmeans(
            x,
            n_clusters=k,
            n_init=config.kmeans_n_init,
        )
        labels_by_k[k] = labels
        rows.append(
            {
                "dataset": dataset.name,
                "k": k,
                "WCSS_full_1201d": inertia,
                "silhouette_pca20": float(
                    silhouette_score(
                        validation_distances,
                        labels,
                        metric="precomputed",
                    )
                ),
                "davies_bouldin_pca20": _davies_bouldin_score_stable(
                    embedding,
                    labels,
                ),
                "calinski_harabasz_pca20": float(
                    calinski_harabasz_score(embedding, labels)
                ),
            }
        )
    results = pd.DataFrame(rows)
    results["relative_WCSS_improvement_from_previous"] = (
        -results["WCSS_full_1201d"].pct_change()
    )
    out = config.output_root / "05_cluster_number_decision" / dataset.name
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "kmeans_validation.csv", index=False)
    return results, labels_by_k


def choose_cluster_count(
    metrics: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[int, pd.DataFrame]:
    decision = metrics.copy()
    decision["in_SI_candidate_range_4_to_6"] = decision["n_nodes"].isin(
        config.si_candidate_nodes
    )
    decision["passes_empty_node_rule"] = decision["n_empty_nodes"] == 0
    decision["passes_min_cluster_fraction_rule"] = (
        decision["min_cluster_fraction"] >= config.min_cluster_fraction
    )
    decision["passes_seed_stability_rule"] = (
        decision["mean_pairwise_seed_ARI"] >= config.min_seed_stability_ari
    )
    # Two degradation profiles can have very high correlation simply because
    # both are smooth and monotonic.  Treat a pair as truly overlapping only
    # when it has both high correlation and small absolute separation.
    decision["passes_centroid_overlap_rule"] = (
        decision["max_centroid_correlation"]
        < config.max_centroid_correlation
    ) | (
        decision["min_centroid_rmse"]
        >= config.min_centroid_rmse_for_distinct
    )
    decision["passes_all_SI_style_rules"] = decision[
        [
            "in_SI_candidate_range_4_to_6",
            "passes_empty_node_rule",
            "passes_min_cluster_fraction_rule",
            "passes_seed_stability_rule",
            "passes_centroid_overlap_rule",
        ]
    ].all(axis=1)

    adequate = decision[decision["passes_all_SI_style_rules"]].sort_values(
        "n_nodes"
    )
    if not adequate.empty:
        chosen = int(adequate.iloc[0]["n_nodes"])
        reason = (
            "smallest n in the SI-supported 4-6 range with no empty node, "
            "no <1% cluster, acceptable multi-seed stability, and no centroid "
            "pair having both correlation >=0.995 and RMSE <0.10"
        )
    else:
        candidates = decision[
            decision["in_SI_candidate_range_4_to_6"]
        ].copy()
        candidates["fallback_score"] = (
            candidates["mean_pairwise_seed_ARI"].rank(pct=True)
            + candidates["silhouette_pca20"].rank(pct=True)
            + candidates["min_centroid_rmse"].rank(pct=True)
            - candidates["max_centroid_correlation"].rank(pct=True)
            - candidates["n_nodes"].rank(pct=True) * 0.25
        )
        chosen = int(
            candidates.sort_values(
                ["fallback_score", "n_nodes"], ascending=[False, True]
            ).iloc[0]["n_nodes"]
        )
        reason = (
            "fallback composite within n=4-6 because no candidate passed every rule"
        )
    decision["selected_n"] = decision["n_nodes"] == chosen
    decision["selection_reason"] = np.where(
        decision["selected_n"], reason, ""
    )
    return chosen, decision


def compare_som_with_kmeans(
    som_labels: np.ndarray,
    kmeans_labels: np.ndarray,
) -> dict:
    n = max(int(np.max(som_labels)), int(np.max(kmeans_labels))) + 1
    contingency = np.zeros((n, n), dtype=int)
    for a, b in zip(som_labels, kmeans_labels):
        contingency[a, b] += 1
    rows, cols = linear_sum_assignment(-contingency)
    return {
        "ARI": float(adjusted_rand_score(som_labels, kmeans_labels)),
        "NMI": float(normalized_mutual_info_score(som_labels, kmeans_labels)),
        "hungarian_matched_fraction": float(
            contingency[rows, cols].sum() / len(som_labels)
        ),
        "mapping_kmeans_to_som": json.dumps(
            {int(k): int(s) for s, k in zip(rows, cols)}, sort_keys=True
        ),
    }


def save_final_traceability(
    dataset: PreprocessedDataset,
    selected_n: int,
    run: SOMRun,
    kmeans_labels: np.ndarray,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments, summary, centroids, pairwise = build_cluster_outputs(
        dataset, run
    )
    assignments["kmeans_cluster_same_n"] = kmeans_labels
    final_root = config.output_root / "06_final_model"
    final_root.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        final_root / "final_curve_classification.csv", index=False
    )
    summary.to_csv(final_root / "final_cluster_summary.csv", index=False)
    centroids.to_csv(final_root / "final_centroid_curves.csv", index=False)
    pairwise.to_csv(
        final_root / "final_centroid_pairwise_similarity.csv", index=False
    )
    np.save(final_root / "final_som_weights.npy", run.som.get_weights())
    with (final_root / "final_som_model.pkl").open("wb") as handle:
        pickle.dump(run.som, handle)
    (final_root / "selected_n.txt").write_text(
        f"{selected_n}\n", encoding="utf-8"
    )

    cluster_root = final_root / "clusters"
    source_db_root = (
        config.output_root
        / "02_selected_curve_database"
        / "main_high_quality_min10"
    )
    for class_label, members in assignments.groupby("class_label"):
        class_dir = cluster_root / class_label
        class_dir.mkdir(parents=True, exist_ok=True)
        members.to_csv(class_dir / "members.csv", index=False)
        raw_dir = class_dir / "raw_curves"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for row in members.itertuples(index=False):
            source = source_db_root / row.source_relative_path
            # Preserve the original hierarchy to prevent identically named
            # files from different papers/directories from colliding.
            destination = raw_dir / row.source_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    return assignments, summary
