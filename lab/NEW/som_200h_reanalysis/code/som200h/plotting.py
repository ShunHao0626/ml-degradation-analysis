"""Publication-style diagnostic plots for selection, preprocessing, and SOM."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import AnalysisConfig
from .modeling import SOMRun
from .preprocessing import PreprocessedDataset


sns.set_theme(style="whitegrid", context="notebook")


def plot_selection_flow(flow: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [
        "All",
        "Duration ≥200 h",
        "+ max PCE ≤200 h",
        "+ ≥4 points",
        "+ ≥10 points",
    ]
    values = flow["n_curves"].to_numpy()
    bars = ax.bar(labels, values, color=["#94A3B8", "#60A5FA", "#34D399", "#FBBF24", "#8B5CF6"])
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    ax.set_ylabel("Number of curves")
    ax.set_title("200 h curve-selection flow")
    ax.set_ylim(0, max(values) * 1.12)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_preprocessing_overview(
    dataset: PreprocessedDataset, out: Path, seed: int = 42
) -> None:
    rng = np.random.RandomState(seed)
    count = min(80, dataset.x_smoothed.shape[0])
    chosen = rng.choice(dataset.x_smoothed.shape[0], size=count, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for index in chosen:
        axes[0].plot(
            dataset.time_grid,
            dataset.x_normalized[index],
            color="#64748B",
            alpha=0.18,
            linewidth=0.8,
        )
        axes[1].plot(
            dataset.time_grid,
            dataset.x_smoothed[index],
            color="#2563EB",
            alpha=0.18,
            linewidth=0.8,
        )
    axes[0].set_title("Akima + per-curve normalization")
    axes[1].set_title("After Savitzky–Golay smoothing")
    for ax in axes:
        ax.set_xlabel("Relative ageing time (h)")
    axes[0].set_ylabel("Normalized PCE")
    fig.suptitle(f"Preprocessing overview — {dataset.name} (random n={count})")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_cluster_number_metrics(
    metrics: pd.DataFrame,
    kmeans: pd.DataFrame,
    selected_n: int,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.errorbar(
        metrics["n_nodes"],
        metrics["qe_mean_across_seeds"],
        yerr=metrics["qe_std_across_seeds"],
        marker="o",
        capsize=3,
        color="#2563EB",
    )
    ax.axvspan(3.5, 6.5, color="#FDE68A", alpha=0.35, label="SI range 4–6")
    ax.axvline(selected_n, color="#DC2626", linestyle="--", label=f"selected n={selected_n}")
    ax.set_title("SOM quantisation error")
    ax.set_xlabel("Number of SOM nodes")
    ax.set_ylabel("Mean QE across seeds")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        metrics["n_nodes"],
        metrics["mean_pairwise_seed_ARI"],
        marker="o",
        color="#059669",
    )
    ax.axhline(0.8, color="#64748B", linestyle=":", label="stability rule")
    ax.axvline(selected_n, color="#DC2626", linestyle="--")
    ax.set_ylim(0, 1.03)
    ax.set_title("Multi-seed assignment stability")
    ax.set_xlabel("Number of SOM nodes")
    ax.set_ylabel("Mean pairwise ARI")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        metrics["n_nodes"],
        metrics["min_centroid_rmse"],
        marker="o",
        color="#7C3AED",
        label="minimum centroid RMSE",
    )
    ax.axhline(
        0.10,
        color="#64748B",
        linestyle=":",
        label="distinctness threshold",
    )
    ax.axvline(selected_n, color="#DC2626", linestyle="--")
    ax.set_title("Closest-centroid separation")
    ax.set_xlabel("Number of SOM nodes")
    ax.set_ylabel("Minimum pairwise RMSE")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        kmeans["k"],
        kmeans["WCSS_full_1201d"],
        marker="o",
        color="#EA580C",
    )
    ax.axvline(selected_n, color="#DC2626", linestyle="--")
    ax.set_title("K-means WCSS reference")
    ax.set_xlabel("k")
    ax.set_ylabel("WCSS")
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_som_clusters(
    dataset: PreprocessedDataset,
    run: SOMRun,
    out: Path,
    max_curves: int,
) -> None:
    n = run.n_nodes
    cols = 4 if n == 16 else (2 if n <= 6 else 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * cols, 4.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rng = np.random.RandomState(42)
    for cluster_id, ax in enumerate(axes.flat):
        if cluster_id >= n:
            ax.axis("off")
            continue
        indices = np.flatnonzero(run.labels == cluster_id)
        draw = (
            rng.choice(indices, size=max_curves, replace=False)
            if indices.size > max_curves
            else indices
        )
        for index in draw:
            ax.plot(
                dataset.time_grid,
                dataset.x_smoothed[index],
                color="#94A3B8",
                alpha=0.10,
                linewidth=0.6,
            )
        if indices.size:
            centroid = np.mean(dataset.x_smoothed[indices], axis=0)
            median = np.median(dataset.x_smoothed[indices], axis=0)
            q25 = np.percentile(dataset.x_smoothed[indices], 25, axis=0)
            q75 = np.percentile(dataset.x_smoothed[indices], 75, axis=0)
            ax.fill_between(
                dataset.time_grid, q25, q75, color="#93C5FD", alpha=0.30
            )
            ax.plot(
                dataset.time_grid,
                centroid,
                color="#1D4ED8",
                linewidth=2.2,
                label="mean",
            )
            ax.plot(
                dataset.time_grid,
                median,
                color="#DC2626",
                linewidth=1.4,
                linestyle="--",
                label="median",
            )
        ax.set_title(f"Raw cluster {cluster_id} — n={indices.size}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
    axes.flat[0].legend(loc="best")
    fig.suptitle(
        f"{dataset.name}: SOM n={n}, topology={run.topology[0]}×{run.topology[1]}, seed={run.seed}",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_sizes(summary: pd.DataFrame, out: Path) -> None:
    ordered = summary.sort_values("ordered_class_id")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        ordered["class_label"],
        ordered["n_curves"],
        color=sns.color_palette("viridis", len(ordered)),
    )
    ax.bar_label(bars, padding=3)
    ax.set_title("Final SOM cluster sizes")
    ax.set_xlabel("Ordered class (most stable → most degraded)")
    ax.set_ylabel("Number of curves")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_final_centroids(
    dataset: PreprocessedDataset,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = sns.color_palette("tab10", len(summary))
    for color, row in zip(palette, summary.sort_values("ordered_class_id").itertuples()):
        mask = assignments["raw_cluster_id"].to_numpy() == row.raw_cluster_id
        centroid = dataset.x_smoothed[mask].mean(axis=0)
        ax.plot(
            dataset.time_grid,
            centroid,
            color=color,
            linewidth=2.3,
            label=(
                f"{row.class_label}: {row.suggested_shape_name} "
                f"(n={row.n_curves})"
            ),
        )
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Normalized PCE")
    ax.set_title("Final selected SOM cluster centroids")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
