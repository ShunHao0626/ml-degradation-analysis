#!/usr/bin/env python3
"""Run the paper's SOM over the pre-registered candidate cluster counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom


def grid_shape(count: int) -> tuple[int, int]:
    """Closest rectangular factor pair, matching the paper's displayed layouts."""
    candidates = [(left, count // left) for left in range(1, int(math.sqrt(count)) + 1) if count % left == 0]
    return min(candidates, key=lambda pair: pair[1] - pair[0])


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_preprocessed(data: np.ndarray, time_axis: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in data:
        ax.plot(time_axis, curve, color="#607d8b", alpha=0.18, linewidth=0.7)
    ax.set(xlabel="Time (hours)", ylabel="Normalized PCE", title=f"Paper-aligned input curves (n={len(data)})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_clusters(
    data: np.ndarray,
    time_axis: np.ndarray,
    labels: np.ndarray,
    count: int,
    shape: tuple[int, int],
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ncols = min(4, count)
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    means = np.full((count, data.shape[1]), np.nan)
    counts = np.zeros(count, dtype=int)
    for cluster in range(count):
        ax = axes_array[cluster]
        mask = labels == cluster
        counts[cluster] = int(mask.sum())
        for curve in data[mask]:
            ax.plot(time_axis, curve, color="#9e9e9e", alpha=0.18, linewidth=0.55)
        if counts[cluster]:
            means[cluster] = data[mask].mean(axis=0)
            ax.plot(time_axis, means[cluster], color="black", linewidth=2.1)
        x_coord, y_coord = divmod(cluster, shape[1])
        ax.set_title(f"Cluster {cluster + 1} / node ({x_coord},{y_coord}), n={counts[cluster]}")
        ax.grid(alpha=0.15)
    for ax in axes_array[count:]:
        ax.axis("off")
    for ax in axes_array[:count]:
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Normalized PCE")
    fig.suptitle(f"SOM result: {shape[0]}x{shape[1]} = {count} nodes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return means, counts


def plot_centroid_overlay(means: np.ndarray, time_axis: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, mean in enumerate(means):
        if np.isfinite(mean).all():
            ax.plot(time_axis, mean, linewidth=2, label=f"Cluster {index + 1}")
    ax.set(xlabel="Time (hours)", ylabel="Normalized PCE", title="Cluster means (overlap audit)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sigma", required=True, type=float)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--role", required=True)
    parser.add_argument("--counts", nargs="+", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    all_summaries: list[dict[str, object]] = []
    for window in config["windows_hours"]:
        source_dir = args.preprocessed_root / f"{window}h"
        bundle = np.load(source_dir / "preprocessed_curves.npz")
        data = bundle["data"]
        time_axis = bundle["time_hours"]
        metadata = read_metadata(source_dir / "curve_metadata.csv")
        if len(metadata) != len(data):
            raise ValueError(f"metadata/data mismatch for {window}h")
        window_out = args.output_root / args.role / f"{window}h"
        window_out.mkdir(parents=True, exist_ok=True)
        plot_preprocessed(data, time_axis, window_out / "all_preprocessed_curves.png")

        window_summaries: list[dict[str, object]] = []
        counts_to_run = args.counts or config["cluster_count_candidates"]
        for count in counts_to_run:
            shape = grid_shape(int(count))
            model_out = window_out / f"k{count:02d}"
            model_out.mkdir(parents=True, exist_ok=True)
            som = MiniSom(
                shape[0],
                shape[1],
                input_len=data.shape[1],
                sigma=args.sigma,
                learning_rate=args.learning_rate,
                random_seed=args.seed,
            )
            som.random_weights_init(data)
            started = time.time()
            som.train(data, int(config["som_iterations"]), verbose=False)
            elapsed = time.time() - started
            winners = [som.winner(curve) for curve in data]
            labels = np.array([x_coord * shape[1] + y_coord for x_coord, y_coord in winners], dtype=int)
            means, counts = plot_clusters(
                data, time_axis, labels, int(count), shape, model_out / "clusters.png"
            )
            plot_centroid_overlay(means, time_axis, model_out / "cluster_means_overlay.png")
            np.savez_compressed(
                model_out / "model_arrays.npz",
                som_weights=som.get_weights(),
                cluster_means=means,
                cluster_counts=counts,
                labels=labels,
                time_hours=time_axis,
            )
            assignments: list[dict[str, object]] = []
            for row, label, winner in zip(metadata, labels, winners):
                assignments.append(
                    {
                        "curve_id": row["curve_id"],
                        "sample": row["sample"],
                        "csv_path": row["csv_path"],
                        "cluster_zero_based": int(label),
                        "cluster_display": int(label) + 1,
                        "som_x": int(winner[0]),
                        "som_y": int(winner[1]),
                    }
                )
            write_csv(model_out / "assignments.csv", assignments, list(assignments[0]))
            summary = {
                "window_hours": int(window),
                "n_curves": int(len(data)),
                "n_time_points": int(data.shape[1]),
                "requested_clusters": int(count),
                "occupied_clusters": int(np.count_nonzero(counts)),
                "grid_x": int(shape[0]),
                "grid_y": int(shape[1]),
                "sigma": args.sigma,
                "learning_rate": args.learning_rate,
                "iterations": int(config["som_iterations"]),
                "seed": args.seed,
                "quantization_error": float(som.quantization_error(data)),
                "topographic_error": float(som.topographic_error(data)),
                "cluster_sizes": counts.tolist(),
                "elapsed_seconds": elapsed,
                "selection_note": "IFO morphology not used in fitting or selection",
            }
            (model_out / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            window_summaries.append(summary)
            all_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

        write_csv(window_out / "qe_by_k.csv", window_summaries, list(window_summaries[0]))
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(
            [row["requested_clusters"] for row in window_summaries],
            [row["quantization_error"] for row in window_summaries],
            marker="o",
        )
        ax.set(
            xlabel="Number of SOM nodes",
            ylabel="Quantization error",
            title=f"SOM quantization-error elbow ({window} h)",
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(window_out / "qe_elbow.png", dpi=200)
        plt.close(fig)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / f"{args.role}_all_windows_summary.csv", all_summaries, list(all_summaries[0]))


if __name__ == "__main__":
    main()
