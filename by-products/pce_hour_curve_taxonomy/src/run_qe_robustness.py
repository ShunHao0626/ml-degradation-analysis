#!/usr/bin/env python3
"""Repeat the published SOM candidate-count run over pre-registered seeds."""

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
    pairs = [(left, count // left) for left in range(1, int(math.sqrt(count)) + 1) if count % left == 0]
    return min(pairs, key=lambda pair: pair[1] - pair[0])


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["robustness_seeds"]]
    runs: list[dict[str, object]] = []

    for window in config["windows_hours"]:
        data = np.load(args.preprocessed_root / f"{window}h" / "preprocessed_curves.npz")["data"]
        for count in config["cluster_count_candidates"]:
            shape = grid_shape(int(count))
            for seed in seeds:
                som = MiniSom(
                    shape[0],
                    shape[1],
                    input_len=data.shape[1],
                    sigma=0.5,
                    learning_rate=0.1,
                    random_seed=seed,
                )
                som.random_weights_init(data)
                started = time.time()
                som.train(data, int(config["som_iterations"]), verbose=False)
                winners = [som.winner(curve) for curve in data]
                labels = np.array([x * shape[1] + y for x, y in winners], dtype=int)
                sizes = np.bincount(labels, minlength=int(count))
                row = {
                    "window_hours": int(window),
                    "requested_clusters": int(count),
                    "grid_x": shape[0],
                    "grid_y": shape[1],
                    "seed": seed,
                    "quantization_error": float(som.quantization_error(data)),
                    "topographic_error": float(som.topographic_error(data)),
                    "occupied_clusters": int(np.count_nonzero(sizes)),
                    "minimum_cluster_size": int(np.min(sizes)),
                    "maximum_cluster_size": int(np.max(sizes)),
                    "elapsed_seconds": time.time() - started,
                }
                runs.append(row)
                print(json.dumps(row, ensure_ascii=False))

    write_csv(args.output_root / "qe_seed_runs.csv", runs, list(runs[0]))
    aggregates: list[dict[str, object]] = []
    for window in config["windows_hours"]:
        for count in config["cluster_count_candidates"]:
            subset = [
                row for row in runs
                if row["window_hours"] == int(window) and row["requested_clusters"] == int(count)
            ]
            qes = np.array([row["quantization_error"] for row in subset], dtype=float)
            aggregates.append(
                {
                    "window_hours": int(window),
                    "requested_clusters": int(count),
                    "seeds": len(subset),
                    "qe_mean": float(np.mean(qes)),
                    "qe_std_population": float(np.std(qes)),
                    "qe_min": float(np.min(qes)),
                    "qe_max": float(np.max(qes)),
                    "all_nodes_occupied_in_all_runs": all(
                        row["occupied_clusters"] == int(count) for row in subset
                    ),
                    "minimum_cluster_size_across_runs": min(
                        int(row["minimum_cluster_size"]) for row in subset
                    ),
                }
            )
    write_csv(args.output_root / "qe_seed_aggregates.csv", aggregates, list(aggregates[0]))

    for window in config["windows_hours"]:
        subset = [row for row in aggregates if row["window_hours"] == int(window)]
        x = np.array([row["requested_clusters"] for row in subset], dtype=int)
        mean = np.array([row["qe_mean"] for row in subset], dtype=float)
        std = np.array([row["qe_std_population"] for row in subset], dtype=float)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3)
        ax.set(
            xlabel="Number of SOM nodes",
            ylabel="Mean quantization error (5 fixed seeds)",
            title=f"Seed-robust SOM elbow ({window} h)",
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_root / f"qe_seed_robust_{window}h.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
