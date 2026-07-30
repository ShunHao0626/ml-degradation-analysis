"""Plotting helpers (matplotlib, 300 dpi PNG)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


def _save(fig, path: Path, dpi: int = DEFAULT_DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Quality control plots
# ----------------------------------------------------------------------------
def plot_curve_duration_distribution(summary: pd.DataFrame, out_dir: Path) -> None:
    s = summary.copy()
    s["duration_h"] = s["end_time_h"] - s["start_time_h"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s["duration_h"], bins=60, color="steelblue", edgecolor="k")
    ax.axvline(config.WINDOW_HOURS, color="red", linestyle="--", label=f"{config.WINDOW_HOURS:.0f} h target")
    ax.set_xlabel("Curve duration (h)")
    ax.set_ylabel("Number of curves")
    ax.set_title("Raw curve duration distribution")
    ax.legend()
    _save(fig, out_dir / "curve_duration_distribution.png")


def plot_curve_overview(
    long_df: pd.DataFrame,
    out_dir: Path,
    title: str = "Raw PCE curves (first 200 h, sample of curves)",
    n_curves: int = 60,
    seed: int = 42,
) -> None:
    rng = np.random.RandomState(seed)
    sids = long_df["sample_id"].unique().tolist()
    rng.shuffle(sids)
    sample_sids = sids[: min(n_curves, len(sids))]
    sub = long_df[long_df["sample_id"].isin(sample_sids)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, g in sub.groupby("sample_id"):
        ax.plot(g["x_hours"], g["pce"], alpha=0.5, lw=0.8)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("PCE (raw)")
    ax.set_title(title)
    ax.axvline(config.WINDOW_HOURS, color="red", linestyle="--", alpha=0.5)
    _save(fig, out_dir / "raw_curve_overview.png")


def plot_normalized_overview(
    X: np.ndarray, sample_ids: List[str], out_dir: Path, n_curves: int = 60, seed: int = 42
) -> None:
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_curves, n), replace=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.array(config.TIME_GRID)
    for i in idx:
        ax.plot(t, X[i], alpha=0.4, lw=0.8)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized PCE (curve / curve.max)")
    ax.set_title("Pre-processed normalized PCE curves (sample)")
    _save(fig, out_dir / "normalized_curve_overview.png")


# ----------------------------------------------------------------------------
# SOM results
# ----------------------------------------------------------------------------
def plot_som_clusters(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    som_shape: tuple[int, int],
    sample_ids: List[str],
    summary: pd.DataFrame,
    out_dir: Path,
    name: str = "main",
) -> None:
    n_x, n_y = som_shape
    t = np.array(config.TIME_GRID)
    fig, axes = plt.subplots(n_x, n_y, figsize=(4.2 * n_y, 3.2 * n_x), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for x in range(n_x):
        for y in range(n_y):
            ax = axes[x, y]
            cid = x * n_y + y
            mask = cluster_ids == cid
            n_curves = int(mask.sum())
            title = f"node ({x},{y})  cluster {cid}  n={n_curves}"
            if n_curves == 0:
                ax.set_title(title + "\n(empty)")
                ax.set_xlabel("Time (h)")
                ax.set_ylabel("Normalized PCE")
                continue
            Xc = X[mask]
            # member curves
            for i in range(Xc.shape[0]):
                ax.plot(t, Xc[i], color="lightgray", alpha=0.3, lw=0.4)
            # 25–75% shading
            q25 = np.percentile(Xc, 25, axis=0)
            q75 = np.percentile(Xc, 75, axis=0)
            ax.fill_between(t, q25, q75, color="C0", alpha=0.2)
            mean_curve = Xc.mean(axis=0)
            median_curve = np.median(Xc, axis=0)
            ax.plot(t, mean_curve, color="black", lw=2.0, label="mean")
            ax.plot(t, median_curve, color="C1", lw=1.4, linestyle="--", label="median")
            ax.set_title(title)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
            if x == 0 and y == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"SOM clusters – {name} (2×2 subplot)", y=1.02)
    _save(fig, out_dir / f"som_cluster_curves_{name}.png")


def plot_codebook_vectors(som: "object", som_shape: tuple[int, int], out_dir: Path) -> None:
    n_x, n_y = som_shape
    t = np.array(config.TIME_GRID)
    weights = som.get_weights()
    fig, axes = plt.subplots(n_x, n_y, figsize=(5 * n_y, 4 * n_x))
    axes = np.atleast_2d(axes)
    for x in range(n_x):
        for y in range(n_y):
            ax = axes[x, y]
            ax.plot(t, weights[x, y], lw=2)
            ax.set_title(f"codebook ({x},{y})")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
            ax.grid(alpha=0.3)
    fig.suptitle("SOM codebook vectors")
    _save(fig, out_dir / "som_codebook_vectors.png")


def plot_u_matrix(som: "object", out_dir: Path) -> None:
    try:
        um = som.distance_map()
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(um, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="distance")
    ax.set_title("SOM U-matrix (distance map)")
    _save(fig, out_dir / "som_u_matrix.png")


def plot_hit_map(
    cluster_ids: np.ndarray, som_shape: tuple[int, int], out_dir: Path, name: str = "main"
) -> None:
    n_x, n_y = som_shape
    hit = np.zeros((n_x, n_y), dtype=int)
    for cid in cluster_ids:
        x = int(cid) // n_y
        y = int(cid) % n_y
        hit[x, y] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(hit, cmap="hot_r", origin="lower")
    for x in range(n_x):
        for y in range(n_y):
            ax.text(y, x, str(int(hit[x, y])), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="count")
    ax.set_title(f"SOM hit map ({name})")
    ax.set_xlabel("som_y")
    ax.set_ylabel("som_x")
    _save(fig, out_dir / f"som_hit_map_{name}.png")


def plot_cluster_size_distribution(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    s = summary.copy()
    s["label"] = s.apply(lambda r: f"({int(r['som_node_x'])},{int(r['som_node_y'])})", axis=1)
    ax.bar(s["label"], s["n_samples"], color="steelblue")
    for i, v in enumerate(s["n_samples"].tolist()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("SOM node (x,y)")
    ax.set_ylabel("Number of curves")
    ax.set_title("Main SOM cluster size distribution")
    _save(fig, out_dir / "cluster_size_distribution.png")


def plot_cluster_size_distribution_n(sizes: list[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"c{i}" for i in range(len(sizes))]
    ax.bar(labels, sizes, color="darkorange")
    for i, v in enumerate(sizes):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of curves")
    ax.set_title(title)
    _save(fig, path)


def plot_bmu_distance_distribution(distances: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(distances, bins=40, color="steelblue", edgecolor="k")
    ax.set_xlabel("BMU Euclidean distance")
    ax.set_ylabel("Number of curves")
    ax.set_title("BMU distance distribution (main model)")
    _save(fig, out_dir / "bmu_distance_distribution.png")


# ----------------------------------------------------------------------------
# QE sweep
# ----------------------------------------------------------------------------
def plot_qe_elbow(qe_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(qe_df["n_nodes"], qe_df["quantisation_error"], "o-", color="steelblue", label="QE")
    main_n = int(config.SOM_CONFIG["shape"][0] * config.SOM_CONFIG["shape"][1])
    if main_n in qe_df["n_nodes"].values:
        row = qe_df[qe_df["n_nodes"] == main_n]
        if not row.empty:
            ax.axvline(main_n, color="red", linestyle="--", alpha=0.6, label=f"n={main_n} (main 2x2)")
            ax.scatter([main_n], [row["quantisation_error"].iloc[0]], s=120, c="red", zorder=5,
                       edgecolors="black", linewidths=1.0)
    # Highlight 1xn rows (except the main one) for reference
    is_1xn = (qe_df["topology_x"] == 1) & (qe_df["n_nodes"] != main_n)
    if is_1xn.any():
        ax.scatter(qe_df.loc[is_1xn, "n_nodes"], qe_df.loc[is_1xn, "quantisation_error"],
                   s=40, c="orange", marker="s", label="1xn fallback (impl. assumption)")
    ax.set_xlabel("Number of SOM nodes / clusters")
    ax.set_ylabel("Quantisation error")
    ax.set_title("Quantisation error vs. number of nodes")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "quantisation_error_elbow.png")


def plot_cluster_curves_n_n(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    n: int,
    out_dir: Path,
    name_suffix: str = "",
) -> None:
    """Generic plot for varying cluster counts."""
    n_x = 1
    n_y = n
    t = np.array(config.TIME_GRID)
    fig, axes = plt.subplots(n_x, n_y, figsize=(3.2 * n_y, 3.2 * n_x), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for k in range(n):
        ax = axes[0, k] if n > 1 else axes[0, 0]
        mask = cluster_ids == k
        n_curves = int(mask.sum())
        title = f"cluster {k}  n={n_curves}"
        if n_curves == 0:
            ax.set_title(title + "\n(empty)")
            continue
        Xc = X[mask]
        for i in range(Xc.shape[0]):
            ax.plot(t, Xc[i], color="lightgray", alpha=0.3, lw=0.4)
        q25 = np.percentile(Xc, 25, axis=0)
        q75 = np.percentile(Xc, 75, axis=0)
        ax.fill_between(t, q25, q75, color="C0", alpha=0.2)
        ax.plot(t, Xc.mean(0), color="black", lw=2.0, label="mean")
        ax.plot(t, np.median(Xc, axis=0), color="C1", lw=1.4, linestyle="--", label="median")
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
    fig.suptitle(f"SOM clusters (n={n})")
    _save(fig, out_dir / f"som_n{n}_curves{name_suffix}.png")


# ----------------------------------------------------------------------------
# Sensitivity / sensitivity comparison
# ----------------------------------------------------------------------------
def plot_sensitivity_comparison(
    results: list[dict],
    sample_ids: List[str],
    out_dir: Path,
    name: str = "sensitivity_cluster_comparison",
) -> None:
    """Side-by-side cluster mean curves from each sensitivity model.

    `results` is the output of `validation.run_sensitivity`.
    """
    import matplotlib.pyplot as plt
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    axes = np.atleast_1d(axes)
    t = np.array(config.TIME_GRID)
    for ax, res in zip(axes, results):
        cids = res["cluster_ids"]
        labels = sorted(set(cids.tolist()))
        for cid in labels:
            mask = cids == cid
            if mask.sum() == 0:
                continue
            mean_curve = X  # noqa
        # build mean curve per cluster id
        # need X passed implicitly; use n_X stored in module level
    _save(fig, out_dir / f"{name}.png")


def plot_sensitivity_comparison_v2(
    results: list[dict],
    X: np.ndarray,
    out_dir: Path,
    name: str = "sensitivity_cluster_comparison",
) -> None:
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    axes = np.atleast_1d(axes)
    t = np.array(config.TIME_GRID)
    for ax, res in zip(axes, results):
        cids = res["cluster_ids"]
        labels = sorted(set(cids.tolist()))
        for cid in labels:
            mask = cids == cid
            if mask.sum() == 0:
                continue
            mean_curve = X[mask].mean(0)
            ax.plot(t, mean_curve, label=f"cluster {cid}  n={mask.sum()}", lw=2)
        ax.set_title(res["name"])
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Mean normalized PCE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Parameter sensitivity: cluster mean curves")
    _save(fig, out_dir / f"{name}.png")


def plot_simple_model(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    n_x: int,
    n_y: int,
    name: str,
    out_dir: Path,
) -> None:
    """Generic 2D subplot grid for sensitivity / alternate param config."""
    t = np.array(config.TIME_GRID)
    fig, axes = plt.subplots(n_x, n_y, figsize=(3.2 * n_y, 3.2 * n_x), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for x in range(n_x):
        for y in range(n_y):
            ax = axes[x, y]
            cid = x * n_y + y
            mask = cluster_ids == cid
            n_curves = int(mask.sum())
            title = f"node ({x},{y})  cluster {cid}  n={n_curves}"
            if n_curves == 0:
                ax.set_title(title + "\n(empty)")
                continue
            Xc = X[mask]
            for i in range(Xc.shape[0]):
                ax.plot(t, Xc[i], color="lightgray", alpha=0.3, lw=0.4)
            q25 = np.percentile(Xc, 25, axis=0)
            q75 = np.percentile(Xc, 75, axis=0)
            ax.fill_between(t, q25, q75, color="C0", alpha=0.2)
            ax.plot(t, Xc.mean(0), color="black", lw=2.0, label="mean")
            ax.plot(t, np.median(Xc, axis=0), color="C1", lw=1.4, linestyle="--", label="median")
            ax.set_title(title)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
    fig.suptitle(f"SOM model – {name}")
    _save(fig, out_dir / f"som_{name}.png")


# ----------------------------------------------------------------------------
# K-means
# ----------------------------------------------------------------------------
def plot_kmeans_elbow(kmeans_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kmeans_df["k"], kmeans_df["wcss_inertia"], "o-", color="darkorange")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("WCSS (inertia)")
    ax.set_title("k-means elbow plot")
    ax.grid(alpha=0.3)
    _save(fig, out_dir / "kmeans_elbow.png")


def plot_kmeans_k4(X: np.ndarray, labels: np.ndarray, out_dir: Path) -> None:
    t = np.array(config.TIME_GRID)
    fig, ax = plt.subplots(figsize=(8, 5))
    k = int(labels.max() + 1)
    palette = plt.cm.tab10(np.linspace(0, 1, max(k, 3)))
    for cid in range(k):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        ax.plot(t, X[mask].mean(0), color=palette[cid], lw=2.0, label=f"cluster {cid}  n={int(mask.sum())}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized PCE (k-means mean)")
    ax.set_title("k-means (k=4) cluster mean curves")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "kmeans_k4_curves.png")
