#!/usr/bin/env python3
"""Derivative-graph injection recovery experiment.

This keeps the paper's HP + polynomial functional derivatives + DTW distance,
but adds a generic multiscale derivative-change graph.  The graph preserves
the order of rises, plateaus and falls that plain average-linkage PFD-DTW can
erase.  Synthetic target labels are used only after clustering for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from run_injection_experiment import (
    CLASSES,
    COLORS,
    GRID,
    PFD_WEIGHTS,
    dtw_distance,
    extend_matrix,
    generate_synthetic_pool,
    load_real_curves,
    pairwise_matrix,
    selected_synthetic_indices,
    transform_curves,
)
from diagnose_shape_representation import ordered_slope_features


def normalized(matrix: np.ndarray) -> np.ndarray:
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    return matrix / max(scale, 1e-12)


def distance_to_knn_affinity(distance: np.ndarray, neighbors: int) -> np.ndarray:
    """Self-tuned symmetric k-NN affinity from a precomputed distance."""
    n = distance.shape[0]
    k = min(max(2, neighbors), n - 1)
    order = np.argsort(distance, axis=1)
    local = np.take_along_axis(distance, order[:, [k]], axis=1).ravel()
    positive = distance[distance > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    local = np.where(local > 1e-12, local, fallback)
    denom = np.outer(local, local)
    affinity = np.exp(-(distance * distance) / np.maximum(denom, 1e-12))
    mask = np.zeros_like(affinity, dtype=bool)
    for row in range(n):
        mask[row, order[row, 1 : k + 1]] = True
    mask = mask | mask.T
    affinity = np.where(mask, affinity, 0.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def graph_cluster(
    distance: np.ndarray, *, clusters: int, neighbors: int, random_state: int
) -> tuple[np.ndarray, float]:
    affinity = distance_to_knn_affinity(distance, neighbors)
    model = SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=random_state,
    )
    labels = model.fit_predict(affinity)
    score = float(silhouette_score(distance, labels, metric="precomputed"))
    return labels.astype(int), score


def graph_cluster_pfd_regularized(
    shape_distance: np.ndarray,
    pfd_distance: np.ndarray,
    *,
    clusters: int,
    neighbors: int,
    pfd_weight: float,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """Keep the ordered-slope neighbourhoods; let PFD reweight their edges."""
    affinity = distance_to_knn_affinity(shape_distance, neighbors)
    pfd = normalized(pfd_distance)
    affinity *= np.exp(-pfd_weight * pfd * pfd)
    np.fill_diagonal(affinity, 1.0)
    labels = SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=random_state,
    ).fit_predict(affinity)
    reporting_distance = normalized(shape_distance) + pfd_weight * pfd
    score = float(silhouette_score(reporting_distance, labels, metric="precomputed"))
    return labels.astype(int), score


def score_targets(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "synthetic_ari": float(adjusted_rand_score(true, predicted)),
        "synthetic_homogeneity": float(homogeneity_score(true, predicted)),
        "synthetic_completeness": float(completeness_score(true, predicted)),
    }


def cluster_composition(
    true: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, list[str], list[int]]:
    cluster_ids = sorted(int(value) for value in np.unique(predicted))
    table = np.zeros((len(CLASSES), len(cluster_ids)), dtype=int)
    for row, target in enumerate(CLASSES):
        for column, cluster_id in enumerate(cluster_ids):
            table[row, column] = int(np.sum((true == target) & (predicted == cluster_id)))
    return table, list(CLASSES), cluster_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "01_hour_audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "03_derivative_graph",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--pfd-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    real_ids, real_values, raw_hour = load_real_curves(args.audit_dir)
    real_levels, real_pfd, _, derivative_scale = transform_curves(real_values)
    level_distance = lambda a, b: dtw_distance(a, b, np.ones(1))
    pfd_distance = lambda a, b: dtw_distance(a, b, PFD_WEIGHTS)
    print("Computing reusable real-real DTW blocks ...", flush=True)
    real_level_matrix = pairwise_matrix(real_levels, level_distance)
    real_pfd_matrix = pairwise_matrix(real_pfd, pfd_distance)

    maximum_ratio = max(args.ratios)
    maximum_synthetic = math.ceil(len(real_ids) * maximum_ratio / (1.0 - maximum_ratio))
    maximum_per_class = math.ceil(maximum_synthetic / len(CLASSES))
    metrics: list[dict[str, object]] = []
    representative: dict[str, object] | None = None

    for seed in args.seeds:
        print(f"Seed {seed}: generating and transforming curves ...", flush=True)
        pool = generate_synthetic_pool(
            real_ids,
            real_values,
            raw_hour,
            per_class=maximum_per_class,
            seed=seed,
        )
        synthetic_values = np.vstack([item.grid_values for item in pool])
        synthetic_levels, synthetic_pfd, _, _ = transform_curves(
            synthetic_values, derivative_scale=derivative_scale
        )
        print(f"Seed {seed}: extending DTW blocks ...", flush=True)
        full_level = extend_matrix(
            real_levels, synthetic_levels, real_level_matrix, level_distance
        )
        full_pfd = extend_matrix(real_pfd, synthetic_pfd, real_pfd_matrix, pfd_distance)

        # Synthetic-only gate: confirms that four target morphologies are
        # identifiable before testing them against the real background.
        synthetic_features = StandardScaler().fit_transform(
            ordered_slope_features(synthetic_values)
        )
        synthetic_shape = pairwise_distances(synthetic_features)
        synthetic_pfd_distance = full_pfd[len(real_ids) :, len(real_ids) :]
        synthetic_truth = np.asarray([item.target_class for item in pool])
        synthetic_labels, synthetic_silhouette = graph_cluster_pfd_regularized(
            synthetic_shape,
            synthetic_pfd_distance,
            clusters=4,
            neighbors=args.neighbors,
            pfd_weight=args.pfd_weight,
            random_state=seed,
        )
        gate = score_targets(synthetic_truth, synthetic_labels)
        metrics.append(
            {
                "seed": seed,
                "requested_ratio": 1.0,
                "actual_ratio": 1.0,
                "real_curves": 0,
                "synthetic_curves": len(pool),
                "method": "fused_pfd_graph_synthetic_gate",
                "clusters": 4,
                "full_silhouette": synthetic_silhouette,
                **gate,
            }
        )

        for ratio in args.ratios:
            requested_total = math.ceil(len(real_ids) * ratio / (1.0 - ratio))
            per_class = max(1, round(requested_total / len(CLASSES)))
            chosen_pool = selected_synthetic_indices(pool, per_class)
            full_indices = np.asarray(
                list(range(len(real_ids))) + [len(real_ids) + index for index in chosen_pool],
                dtype=int,
            )
            mixed_values = np.vstack(
                [real_values, np.vstack([pool[index].grid_values for index in chosen_pool])]
            )
            true = np.asarray([pool[index].target_class for index in chosen_pool])
            shape_features = StandardScaler().fit_transform(
                ordered_slope_features(mixed_values)
            )
            shape_distance = pairwise_distances(shape_features)
            level_sub = full_level[np.ix_(full_indices, full_indices)]
            pfd_sub = full_pfd[np.ix_(full_indices, full_indices)]
            methods = {
                "level_dtw_graph": level_sub,
                "pfd_dtw_graph": pfd_sub,
                "multiscale_derivative_graph": shape_distance,
            }
            for method, distance in methods.items():
                labels, silhouette = graph_cluster(
                    distance,
                    clusters=args.clusters,
                    neighbors=args.neighbors,
                    random_state=seed,
                )
                synthetic_prediction = labels[len(real_ids) :]
                metrics.append(
                    {
                        "seed": seed,
                        "requested_ratio": ratio,
                        "actual_ratio": len(chosen_pool) / len(labels),
                        "real_curves": len(real_ids),
                        "synthetic_curves": len(chosen_pool),
                        "method": method,
                        "clusters": args.clusters,
                        "full_silhouette": silhouette,
                        **score_targets(true, synthetic_prediction),
                    }
                )
                if (
                    seed == args.seeds[0]
                    and abs(ratio - maximum_ratio) < 1e-9
                    and method == "fused_pfd_graph"
                ):
                    representative = {
                        "pool": pool,
                        "chosen": chosen_pool,
                        "labels": labels,
                        "truth": true,
                        "distance": distance,
                    }
            labels, silhouette = graph_cluster_pfd_regularized(
                shape_distance,
                pfd_sub,
                clusters=args.clusters,
                neighbors=args.neighbors,
                pfd_weight=args.pfd_weight,
                random_state=seed,
            )
            synthetic_prediction = labels[len(real_ids) :]
            metrics.append(
                {
                    "seed": seed,
                    "requested_ratio": ratio,
                    "actual_ratio": len(chosen_pool) / len(labels),
                    "real_curves": len(real_ids),
                    "synthetic_curves": len(chosen_pool),
                    "method": "fused_pfd_graph",
                    "clusters": args.clusters,
                    "full_silhouette": silhouette,
                    **score_targets(true, synthetic_prediction),
                }
            )
            if seed == args.seeds[0] and abs(ratio - maximum_ratio) < 1e-9:
                representative = {
                    "pool": pool,
                    "chosen": chosen_pool,
                    "labels": labels,
                    "truth": true,
                    "distance": normalized(shape_distance) + args.pfd_weight * normalized(pfd_sub),
                }

    with (args.output / "derivative_graph_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    if representative is not None:
        pool = representative["pool"]
        chosen = representative["chosen"]
        labels = np.asarray(representative["labels"])
        truth = np.asarray(representative["truth"])
        synthetic_prediction = labels[len(real_ids) :]
        with (args.output / "representative_assignments.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["curve_id", "origin_or_target", "cluster"]
            )
            writer.writeheader()
            for curve_id, label in zip(real_ids, labels[: len(real_ids)]):
                writer.writerow(
                    {"curve_id": curve_id, "origin_or_target": "real_unlabelled", "cluster": int(label)}
                )
            for pool_index, label in zip(chosen, synthetic_prediction):
                item = pool[pool_index]
                writer.writerow(
                    {"curve_id": item.curve_id, "origin_or_target": item.target_class, "cluster": int(label)}
                )
        table, row_names, column_names = cluster_composition(truth, synthetic_prediction)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
        image = axes[0].imshow(table, cmap="Blues", aspect="auto")
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                axes[0].text(column, row, str(table[row, column]), ha="center", va="center")
        axes[0].set_xticks(range(len(column_names)), [str(value) for value in column_names])
        axes[0].set_yticks(range(len(row_names)), [value.capitalize() for value in row_names])
        axes[0].set_xlabel("unsupervised cluster")
        axes[0].set_title("Injected target composition (labels used only here)")
        fig.colorbar(image, ax=axes[0], fraction=0.046)
        for target in CLASSES:
            for pool_index in chosen:
                item = pool[pool_index]
                if item.target_class == target:
                    axes[1].plot(
                        GRID, item.grid_values, color=COLORS[target], alpha=0.35, linewidth=0.9
                    )
        axes[1].set_xlabel("time (h)")
        axes[1].set_ylabel("relative performance")
        axes[1].set_title("Complex injected curves in representative run")
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output / "representative_recovery.png", dpi=180)
        plt.close(fig)

    methods = [
        "level_dtw_graph",
        "pfd_dtw_graph",
        "multiscale_derivative_graph",
        "fused_pfd_graph",
    ]
    colors = ["#666666", "#d95f02", "#2a9d58", "#2f6db0"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for method, color in zip(methods, colors):
        for metric, axis, title in (
            ("synthetic_ari", axes[0], "Injected-label ARI"),
            ("synthetic_homogeneity", axes[1], "Injected homogeneity"),
            ("synthetic_completeness", axes[2], "Injected completeness"),
        ):
            means = []
            deviations = []
            for ratio in args.ratios:
                values = [
                    float(record[metric])
                    for record in metrics
                    if record["method"] == method and record["requested_ratio"] == ratio
                ]
                means.append(float(np.mean(values)))
                deviations.append(float(np.std(values)))
            axis.errorbar(
                args.ratios,
                means,
                yerr=deviations,
                marker="o",
                capsize=3,
                label=method,
                color=color,
            )
            axis.set_title(title)
            axis.set_xlabel("requested synthetic share")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "recovery_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "real_curves": len(real_ids),
        "seeds": args.seeds,
        "development_seed": args.seeds[0],
        "confirmation_seeds": args.seeds[1:],
        "ratios": args.ratios,
        "mixed_clusters": args.clusters,
        "neighbors": args.neighbors,
        "fusion_weights": {
            "ordered_slope_graph": "defines k-nearest-neighbour topology",
            "paper_pfd_dtw_edge_penalty": args.pfd_weight,
        },
        "note": "Synthetic labels were used only for post-clustering evaluation.",
    }
    (args.output / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
