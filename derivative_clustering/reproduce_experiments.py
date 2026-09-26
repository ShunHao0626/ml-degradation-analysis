#!/usr/bin/env python3
"""Run a transparent reproduction of the paper's UCR subset experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from pfd_dtw import (
    hierarchical_cluster,
    hp_filter,
    pairwise_distance_matrix,
    pfd_dtw_hierarchical_clustering,
    scalar_dtw_distance,
)
from ucr_data import load_ucr_dataset, stratified_sample


PAPER_SUBSETS = {
    "ECG5000": {"classes": 3, "per_class": 2, "split": "train"},
    "Trace": {"classes": 4, "per_class": 2, "split": "both"},
    "Plane": {"classes": 3, "per_class": 3, "split": "both"},
}


def cluster_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Best label-permutation accuracy via the Hungarian assignment."""

    true_values, true_inverse = np.unique(truth, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(prediction, return_inverse=True)
    counts = np.zeros((true_values.size, predicted_values.size), dtype=int)
    np.add.at(counts, (true_inverse, predicted_inverse), 1)
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum() / truth.size)


def run_baselines(
    selected: list[np.ndarray],
    *,
    n_clusters: int,
    linkage_method: str,
    smoothing: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return labels, distances, and linkage for the paper's main baselines."""

    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    euclidean = pairwise_distance_matrix(
        selected, lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    )
    labels, tree = hierarchical_cluster(
        euclidean, n_clusters=n_clusters, linkage_method=linkage_method
    )
    outputs["Euclidean_HC"] = (labels, euclidean, tree)

    dtw = pairwise_distance_matrix(selected, scalar_dtw_distance)
    labels, tree = hierarchical_cluster(dtw, n_clusters=n_clusters, linkage_method=linkage_method)
    outputs["DTW_HC"] = (labels, dtw, tree)

    filtered = [hp_filter(values, smoothing=smoothing) for values in selected]
    filtered_dtw = pairwise_distance_matrix(filtered, scalar_dtw_distance)
    labels, tree = hierarchical_cluster(
        filtered_dtw, n_clusters=n_clusters, linkage_method=linkage_method
    )
    outputs["TFDTW_HC"] = (labels, filtered_dtw, tree)
    return outputs


def plot_selected(series: list[np.ndarray], labels: np.ndarray, output: Path) -> None:
    columns = min(4, len(series))
    rows = int(np.ceil(len(series) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.4 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(series):
            axis.axis("off")
            continue
        axis.plot(series[index], color="#1557d6", linewidth=1.4)
        axis.set_title(f"s{index} (class {labels[index]})")
        axis.set_xlabel("time")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_tree(
    tree: np.ndarray,
    output: Path,
    title: str,
    count: int,
    n_clusters: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    if n_clusters == 1:
        threshold = float("inf")
    else:
        included = count - n_clusters - 1
        next_merge = count - n_clusters
        lower = 0.0 if included < 0 else float(tree[included, 2])
        upper = float(tree[next_merge, 2])
        threshold = (lower + upper) / 2.0
    dendrogram(
        tree,
        labels=[f"s{i}" for i in range(count)],
        color_threshold=threshold,
        above_threshold_color="#333333",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_ylabel("distance")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(PAPER_SUBSETS), default="Plane")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--smoothing", type=float, default=1000.0)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--weights", type=float, nargs=3, default=(3.0, 2.0, 1.0))
    parser.add_argument("--linkage", choices=("single", "complete", "average"), default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = PAPER_SUBSETS[args.dataset]
    values, labels, indices = load_ucr_dataset(
        args.data_dir, args.dataset, split=str(design["split"])
    )
    selected, truth, selected_indices = stratified_sample(
        values,
        labels,
        indices,
        classes=int(design["classes"]),
        per_class=int(design["per_class"]),
        seed=args.seed,
    )
    n_clusters = int(design["classes"])
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)

    pfd = pfd_dtw_hierarchical_clustering(
        selected,
        n_clusters=n_clusters,
        smoothing=args.smoothing,
        degree=args.degree,
        max_degree=args.max_degree,
        weights=args.weights,
        linkage_method=args.linkage,
    )
    methods = run_baselines(
        selected,
        n_clusters=n_clusters,
        linkage_method=args.linkage,
        smoothing=args.smoothing,
    )
    flattened_pfd = [item.features.ravel() for item in pfd.transforms]
    pfd_only = pairwise_distance_matrix(
        flattened_pfd,
        lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b))),
    )
    pfd_only_labels, pfd_only_tree = hierarchical_cluster(
        pfd_only, n_clusters=n_clusters, linkage_method=args.linkage
    )
    methods["PFD_HC"] = (pfd_only_labels, pfd_only, pfd_only_tree)
    methods["PFD_DTW_HC"] = (pfd.labels, pfd.distance_matrix, pfd.linkage_matrix)

    plot_selected(selected, truth, output / "selected_series.png")
    report: dict[str, object] = {
        "dataset": args.dataset,
        "paper_subset_shape": {"classes": design["classes"], "per_class": design["per_class"]},
        "source_indices": selected_indices.tolist(),
        "true_labels": truth.tolist(),
        "parameters": {
            "hp_lambda": args.smoothing,
            "degree": args.degree,
            "max_degree": args.max_degree,
            "weights": list(args.weights),
            "linkage": args.linkage,
            "seed": args.seed,
        },
        "selected_degrees": [item.degree for item in pfd.transforms],
        "selected_r_squared": [item.r_squared for item in pfd.transforms],
        "methods": {},
    }
    matrices: dict[str, np.ndarray] = {}
    for name, (prediction, matrix, tree) in methods.items():
        report["methods"][name] = {
            "cluster_labels": prediction.tolist(),
            "adjusted_rand_index": adjusted_rand_score(truth, prediction),
            "cluster_accuracy": cluster_accuracy(truth, prediction),
        }
        matrices[name] = matrix
        plot_tree(
            tree,
            output / f"{name}_dendrogram.png",
            name,
            len(selected),
            n_clusters,
        )

    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "distance_matrices.npz", **matrices)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to {output}")


if __name__ == "__main__":
    main()
