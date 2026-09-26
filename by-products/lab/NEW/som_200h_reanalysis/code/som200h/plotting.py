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
from .data import RawCurve
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


def plot_selected_raw_curves(
    curves_by_id: dict[str, RawCurve],
    selected: pd.DataFrame,
    out: Path,
    title: str,
    window_hours: float = 200.0,
) -> None:
    """Plot every selected curve on one axis using raw points in 0--200 h."""
    raw_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for curve_id in selected["curve_id"]:
        curve = curves_by_id.get(curve_id)
        if curve is None:
            raise KeyError(f"Selected curve is missing from discovery: {curve_id}")
        mask = (
            np.isfinite(curve.x_hours_relative)
            & np.isfinite(curve.y)
            & (curve.x_hours_relative >= 0.0)
            & (curve.x_hours_relative <= window_hours + 1e-9)
        )
        x = curve.x_hours_relative[mask]
        y = curve.y[mask]
        if x.size == 0:
            continue
        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]
        raw_curves.append((x, y))

    plotted_count = len(raw_curves)
    if plotted_count != len(selected):
        raise RuntimeError(
            f"Expected to plot {len(selected)} curves, plotted {plotted_count}"
        )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for x, y in raw_curves:
        ax.plot(
            x,
            y,
            color="#2563EB",
            alpha=0.07,
            linewidth=0.45,
            marker=".",
            markersize=0.8,
            markeredgewidth=0,
        )
    ax.set_xlim(0.0, window_hours)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Raw y value as supplied")
    ax.set_title(
        f"{title} (n={plotted_count:,})\n"
        "Observed raw points only; no interpolation, normalization, or smoothing"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_selected_raw_curves_by_scale(
    curves_by_id: dict[str, RawCurve],
    selected: pd.DataFrame,
    out: Path,
    title: str,
    window_hours: float = 200.0,
) -> None:
    """Plot raw selected curves in three panels without changing y values."""
    groups = (
        ("Raw maximum <= 2.5", -np.inf, 2.5, "#2563EB"),
        ("2.5 < raw maximum <= 50", 2.5, 50.0, "#059669"),
        ("Raw maximum > 50", 50.0, np.inf, "#EA580C"),
    )
    grouped_curves: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in groups
    ]

    for curve_id in selected["curve_id"]:
        curve = curves_by_id.get(curve_id)
        if curve is None:
            raise KeyError(f"Selected curve is missing from discovery: {curve_id}")
        mask = (
            np.isfinite(curve.x_hours_relative)
            & np.isfinite(curve.y)
            & (curve.x_hours_relative >= 0.0)
            & (curve.x_hours_relative <= window_hours + 1e-9)
        )
        x = curve.x_hours_relative[mask]
        y = curve.y[mask]
        if x.size == 0:
            continue
        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]
        raw_maximum = float(np.max(y))
        for group_index, (_, lower, upper, _) in enumerate(groups):
            if lower < raw_maximum <= upper:
                grouped_curves[group_index].append((x, y))
                break

    plotted_count = sum(len(curves) for curves in grouped_curves)
    if plotted_count != len(selected):
        raise RuntimeError(
            f"Expected to plot {len(selected)} curves, plotted {plotted_count}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharex=True)
    for ax, group, curves in zip(axes, groups, grouped_curves):
        label, _, _, color = group
        all_y: list[np.ndarray] = []
        for x, y in curves:
            ax.plot(
                x,
                y,
                color=color,
                alpha=0.075,
                linewidth=0.45,
                marker=".",
                markersize=0.8,
                markeredgewidth=0,
            )
            all_y.append(y)
        ax.set_xlim(0.0, window_hours)
        ax.set_title(f"{label} (n={len(curves):,})")
        ax.set_xlabel("Relative ageing time (h)")
        if all_y:
            y_values = np.concatenate(all_y)
            y_min = float(np.min(y_values))
            y_max = float(np.max(y_values))
            padding = max((y_max - y_min) * 0.04, abs(y_max) * 0.01, 1e-6)
            ax.set_ylim(y_min - padding, y_max + padding)
    axes[0].set_ylabel("Raw y value as supplied")
    fig.suptitle(
        f"{title} (n={plotted_count:,})\n"
        "Observed raw points only; no interpolation, normalization, or smoothing"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
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


def plot_all_preprocessed_curves(
    time_grid: np.ndarray,
    curves: np.ndarray,
    out: Path,
    title: str,
) -> None:
    """Plot every final preprocessed curve together on one axis."""
    if curves.ndim != 2:
        raise ValueError(f"Expected a 2D curve matrix, got shape {curves.shape}")
    if curves.shape[1] != time_grid.size:
        raise ValueError(
            "Curve width does not match the time grid: "
            f"{curves.shape[1]} != {time_grid.size}"
        )
    if not np.isfinite(curves).all() or not np.isfinite(time_grid).all():
        raise ValueError("Preprocessed curves and time grid must be finite")

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for curve in curves:
        ax.plot(
            time_grid,
            curve,
            color="#2563EB",
            alpha=0.045,
            linewidth=0.45,
        )
    ax.set_xlim(float(time_grid[0]), float(time_grid[-1]))
    y_min = float(np.min(curves))
    y_max = float(np.max(curves))
    padding = max((y_max - y_min) * 0.03, 1e-6)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Preprocessed normalized PCE")
    ax.set_title(
        f"{title} (n={curves.shape[0]:,})\n"
        "Final SOM input: Akima interpolation, normalization, and Savitzky-Golay smoothing"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
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
