"""
src/plotting.py
===============
All matplotlib figures for the PvkSOM 200h analysis pipeline.

Figures are saved as 300-dpi PNG (and PDF where suitable) in OUT_FIG.
Consistent fonts, sizes, and colour palettes throughout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from minisom import MiniSom

from .config import OUT_FIG, PREPROC_CONFIG, logger

# ── Matplotlib defaults ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Node colour palette (matches publication quality)
NODE_COLORS = [
    "#2166ac",   # deep blue   — node (0,0)
    "#f4a582",   # salmon      — node (0,1)
    "#a1d99b",   # light green — node (1,0)
    "#9e9ac8",   # lavender    — node (1,1)
]
DARK_COLORS = ["#1a1a2e", "#c0392b", "#27ae60", "#8e44ad"]
MEMBER_ALPHA = 0.07
MEAN_LINEWIDTH = 1.8
MEDIAN_LINEWIDTH = 1.2
SHADE_ALPHA = 0.20

# ── Helpers ────────────────────────────────────────────────────────────────

TIME_GRID = np.linspace(0, 200, 1201)


def _label(coord: tuple[int, int]) -> str:
    return f"({coord[0]},{coord[1]})"


def _node_color(wx: int, wy: int, shape: tuple[int, int]) -> str:
    """Return a consistent colour for each SOM node."""
    idx = wx * shape[1] + wy
    return NODE_COLORS[idx % len(NODE_COLORS)]


def _savefig(fig: plt.Figure, name: str) -> Path:
    out = OUT_FIG / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / name.replace(".png", ".pdf"), bbox_inches="tight")
    logger.info("Saved: %s", out)
    plt.close(fig)
    return out


# =============================================================================
# §1  Data-quality figures
# =============================================================================

def plot_curve_duration_distribution(qc_df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Histogram of max_x_hours for all curves, coloured by inclusion status.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    incl = qc_df[qc_df["status"] == "included"]
    excl = qc_df[qc_df["status"] == "excluded"]

    ax.hist(incl["max_x_hours"].dropna(), bins=40, alpha=0.7, label="Included", color="#2166ac")
    ax.hist(excl["max_x_hours"].dropna(), bins=40, alpha=0.7, label="Excluded", color="#f4a582")
    ax.axvline(200, color="red", linestyle="--", linewidth=1.2, label="200 h")
    ax.set_xlabel("Max time (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Curve duration distribution")
    ax.legend()
    return _savefig(fig, "curve_duration_distribution.png")


def plot_raw_curve_overview(
    sample_curves: np.ndarray,
    n_show: int = 100,
    out_dir: Path | None = None,
) -> Path:
    """
    Overview of a random sample of raw (preprocessed but unsmoothed) curves.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    n = min(n_show, len(sample_curves))
    indices = np.random.choice(len(sample_curves), n, replace=False)
    for idx in indices:
        ax.plot(TIME_GRID, sample_curves[idx], alpha=0.3, linewidth=0.6, color="gray")
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized PCE")
    ax.set_title(f"Raw normalised curves — random sample (n={n})")
    return _savefig(fig, "raw_curve_overview.png")


def plot_normalized_curve_overview(
    X: np.ndarray,
    n_show: int = 100,
    out_dir: Path | None = None,
) -> Path:
    """
    Overview of a random sample of smoothed (final SOM input) curves.
    """
    return plot_raw_curve_overview(X, n_show, out_dir)


# =============================================================================
# §2  Main SOM result figures
# =============================================================================

def plot_som_cluster_curves(
    X: np.ndarray,
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    codebook_vectors: np.ndarray,
    shape: tuple[int, int],
    metrics_list: list,
    out_dir: Path | None = None,
) -> Path:
    """
    2×2 subplot: all SOM cluster members + mean + median + IQR.

    Each subplot = one SOM node:
      - Low-opacity member curves
      - Bold mean curve
      - Dashed median curve
      - Shaded 25–75 percentile band
      - Codebook vector
    """
    sx, sy = shape
    fig, axes = plt.subplots(sx, sy, figsize=(4 * sy, 3.5 * sx),
                              sharex=True, sharey=True, squeeze=False)

    time_axis = TIME_GRID
    n_total = X.shape[0]

    # Build metrics lookup
    m_lookup = {(m.som_x, m.som_y): m for m in metrics_list}

    for wx in range(sx):
        for wy in range(sy):
            ax = axes[wx, wy]
            mask = (winners_x == wx) & (winners_y == wy)
            curves = X[mask]
            n_in = mask.sum()
            col = _node_color(wx, wy, shape)

            # Member curves
            if n_in > 0:
                for curve in curves:
                    ax.plot(time_axis, curve, alpha=MEMBER_ALPHA,
                            linewidth=0.5, color=col)

            # 25–75 percentile band
            if n_in > 4:
                q25 = np.nanpercentile(curves, 25, axis=0)
                q75 = np.nanpercentile(curves, 75, axis=0)
                ax.fill_between(time_axis, q25, q75,
                                alpha=SHADE_ALPHA, color=col)

            # Mean curve
            mean_c = curves.mean(axis=0) if n_in > 0 else np.zeros_like(time_axis)
            ax.plot(time_axis, mean_c,
                    color=col, linewidth=MEAN_LINEWIDTH,
                    label=f"Mean (n={n_in})", zorder=5)

            # Median curve
            median_c = np.nanmedian(curves, axis=0) if n_in > 0 else mean_c
            ax.plot(time_axis, median_c,
                    color="black", linewidth=MEDIAN_LINEWIDTH,
                    linestyle="--", label="Median", zorder=6)

            # Codebook vector
            cb = codebook_vectors[wx, wy]
            ax.plot(time_axis, cb,
                    color="black", linewidth=1.0,
                    linestyle=":", alpha=0.6, label="Codebook", zorder=4)

            ax.set_xlim(0, 200)
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")

            raw_id = wx * sy + wy
            ax.set_title(f"Node ({wx},{wy})  cluster_{raw_id}  n={n_in} ({n_in/n_total*100:.1f}%)")
            ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    return _savefig(fig, "som_cluster_curves.png")


def plot_som_codebook_vectors(
    som: MiniSom,
    shape: tuple[int, int],
    out_dir: Path | None = None,
) -> Path:
    """All 4 codebook vectors overlaid in one panel."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sx, sy = shape
    weights = som.get_weights()

    for wx in range(sx):
        for wy in range(sy):
            col = _node_color(wx, wy, shape)
            cb = weights[wx, wy]
            raw_id = wx * sy + wy
            label = f"Node ({wx},{wy})  cluster_{raw_id}"
            ax.plot(TIME_GRID, cb, color=col, linewidth=1.8, label=label)

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized PCE")
    ax.set_title("SOM codebook vectors")
    ax.legend(fontsize=8)
    return _savefig(fig, "som_codebook_vectors.png")


def plot_som_u_matrix(
    som: MiniSom,
    shape: tuple[int, int],
    out_dir: Path | None = None,
) -> Path:
    """U-matrix (mean distance to neighbours) as heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    sx, sy = shape
    weights = som.get_weights()

    u_matrix = np.zeros((sx, sy))
    for wx in range(sx):
        for wy in range(sy):
            neighbours = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = wx + dx, wy + dy
                    if 0 <= nx < sx and 0 <= ny < sy:
                        d = np.linalg.norm(weights[wx, wy] - weights[nx, ny])
                        neighbours.append(d)
            u_matrix[wx, wy] = np.mean(neighbours) if neighbours else 0

    im = ax.imshow(u_matrix, cmap="coolwarm", aspect="equal")
    ax.set_title("U-matrix (mean neighbour distance)")
    ax.set_xlabel("SOM y")
    ax.set_ylabel("SOM x")
    plt.colorbar(im, ax=ax, label="Distance")
    plt.tight_layout()
    return _savefig(fig, "som_u_matrix.png")


def plot_som_hit_map(
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    shape: tuple[int, int],
    out_dir: Path | None = None,
) -> Path:
    """Hit map (number of samples per node)."""
    sx, sy = shape
    fig, ax = plt.subplots(figsize=(4, 4))
    hits = np.zeros((sx, sy))
    for wx, wy in zip(winners_x, winners_y):
        hits[wx, wy] += 1

    im = ax.imshow(hits, cmap="viridis", aspect="equal")
    ax.set_title("SOM hit map")
    ax.set_xlabel("SOM y")
    ax.set_ylabel("SOM x")
    plt.colorbar(im, ax=ax, label="Count")
    for wx in range(sx):
        for wy in range(sy):
            ax.text(wy, wx, f"{int(hits[wx, wy])}",
                    ha="center", va="center",
                    color="white" if hits[wx, wy] > hits.max() / 2 else "black",
                    fontsize=9)
    plt.tight_layout()
    return _savefig(fig, "som_hit_map.png")


def plot_cluster_size_distribution(
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    shape: tuple[int, int],
    out_dir: Path | None = None,
) -> Path:
    """Bar chart of cluster sizes."""
    sx, sy = shape
    sizes = {}
    for wx, wy in zip(winners_x, winners_y):
        key = f"({wx},{wy})"
        sizes[key] = sizes.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = sorted(sizes.keys())
    vals = [sizes[k] for k in keys]
    colors = [NODE_COLORS[i % len(NODE_COLORS)] for i in range(len(keys))]
    bars = ax.bar(keys, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("SOM node")
    ax.set_ylabel("Number of curves")
    ax.set_title("Cluster size distribution")
    plt.tight_layout()
    return _savefig(fig, "cluster_size_distribution.png")


def plot_bmu_distance_distribution(
    bmu_distances: np.ndarray,
    out_dir: Path | None = None,
) -> Path:
    """Histogram of BMU Euclidean distances."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bmu_distances, bins=50, alpha=0.7, color="#2166ac", edgecolor="white")
    ax.axvline(bmu_distances.mean(), color="red", linestyle="--",
                linewidth=1.2, label=f"Mean={bmu_distances.mean():.3f}")
    ax.set_xlabel("BMU Euclidean distance")
    ax.set_ylabel("Count")
    ax.set_title("BMU distance distribution")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, "bmu_distance_distribution.png")


# =============================================================================
# §3  QE sweep figures
# =============================================================================

def plot_quantisation_error_elbow(
    qe_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """Line+marker plot of QE vs number of SOM nodes."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(qe_df["n_nodes"], qe_df["quantization_error"],
            color="#2166ac", linewidth=2, marker="o", markersize=7)

    # Highlight n=4 (2×2)
    row_4 = qe_df[qe_df["n_nodes"] == 4]
    if not row_4.empty:
        ax.scatter(row_4["n_nodes"], row_4["quantization_error"],
                   color="red", s=120, zorder=5, label="Main model (2×2)", marker="*")

    ax.set_xlabel("Number of SOM nodes")
    ax.set_ylabel("Quantisation error")
    ax.set_title("Quantisation error vs. number of SOM nodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, "quantisation_error_elbow.png")


def _plot_n_curves(
    X: np.ndarray,
    winners_x: np.ndarray,
    winners_y: np.ndarray,
    shape: tuple[int, int],
    out_dir: Path,
    name: str,
) -> Path:
    """Generic n-subplot SOM cluster figure (used for n=2,4,5,6)."""
    sx, sy = shape
    n_total = X.shape[0]

    fig, axes = plt.subplots(sx, sy, figsize=(4 * sy, 3.5 * sx),
                              sharex=True, sharey=True, squeeze=False)
    for wx in range(sx):
        for wy in range(sy):
            ax = axes[wx, wy]
            mask = (winners_x == wx) & (winners_y == wy)
            curves = X[mask]
            n_in = mask.sum()
            col = _node_color(wx, wy, shape)

            if n_in > 0:
                for curve in curves:
                    ax.plot(TIME_GRID, curve, alpha=MEMBER_ALPHA,
                            linewidth=0.5, color=col)
            if n_in > 4:
                q25 = np.nanpercentile(curves, 25, axis=0)
                q75 = np.nanpercentile(curves, 75, axis=0)
                ax.fill_between(TIME_GRID, q25, q75, alpha=SHADE_ALPHA, color=col)

            mean_c = curves.mean(axis=0) if n_in > 0 else np.zeros_like(TIME_GRID)
            ax.plot(TIME_GRID, mean_c, color=col, linewidth=MEAN_LINEWIDTH,
                    label=f"n={n_in}", zorder=5)

            ax.set_xlim(0, 200)
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
            raw_id = wx * sy + wy
            ax.set_title(f"Node ({wx},{wy})  cluster_{raw_id}  n={n_in}")
            ax.legend(fontsize=7)

    plt.tight_layout()
    return _savefig(fig, name)


def plot_som_n_comparison(
    X: np.ndarray,
    som_configs: dict,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Generate som_n2, som_n4, som_n5, som_n6 curves figures.
    som_configs: {name: {"som": MiniSom, "shape": (sx, sy), ...}}
    """
    paths = {}
    for name, cfg in som_configs.items():
        som = cfg["som"]
        shape = cfg["shape"]
        wx, wy, _ = cfg["winners_x"], cfg["winners_y"], cfg.get("bmu_dists", None)
        fname = f"som_{name}_curves.png"
        try:
            p = _plot_n_curves(X, wx, wy, shape, out_dir, fname)
            paths[name] = p
        except Exception as e:
            logger.warning("Could not plot %s: %s", name, e)
    return paths


# =============================================================================
# §4  Sensitivity analysis figures
# =============================================================================

def plot_sensitivity_comparison(
    X: np.ndarray,
    sensitivity_results: dict,
    shape: tuple[int, int],
    out_dir: Path,
) -> dict[str, Path]:
    """Generate sigma/lr sensitivity SOM cluster figures."""
    paths = {}
    for name, res in sensitivity_results.items():
        som = res["som"]
        wx = res["winners_x"]
        wy = res["winners_y"]
        fname = f"som_{name}.png"
        try:
            p = _plot_n_curves(X, wx, wy, shape, out_dir, fname)
            paths[name] = p
        except Exception as e:
            logger.warning("Could not plot sensitivity %s: %s", name, e)
    return paths


def plot_sensitivity_cluster_comparison(
    sensitivity_results: dict,
    X: np.ndarray,
    out_dir: Path,
) -> Path:
    """
    Overlaid mean curves for all sensitivity configurations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, res in sensitivity_results.items():
        som = res["som"]
        wx, wy = res["winners_x"], res["winners_y"]
        # Use public API to get SOM shape
        weights = som.get_weights()
        sx, sy = weights.shape[:2]
        colors_l = list(NODE_COLORS)

        for node_x in range(sx):
            for node_y in range(sy):
                mask = (wx == node_x) & (wy == node_y)
                if mask.sum() == 0:
                    continue
                mean_c = X[mask].mean(axis=0)
                raw_id = node_x * sy + node_y
                label = f"{name} node_{raw_id}"
                col = colors_l[raw_id % len(colors_l)]
                ax.plot(TIME_GRID, mean_c, linewidth=1.5,
                        label=label, color=col, linestyle="--")

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized PCE")
    ax.set_title("Sensitivity analysis: mean curves by configuration")
    ax.legend(fontsize=7, ncol=2)
    return _savefig(fig, "sensitivity_cluster_comparison.png")


# =============================================================================
# §5  K-means figures
# =============================================================================

def plot_kmeans_elbow(
    wcss_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """WCSS elbow plot for K-means."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wcss_df["k"], wcss_df["wcss"], color="#2166ac",
            linewidth=2, marker="o", markersize=7)
    row4 = wcss_df[wcss_df["k"] == 4]
    if not row4.empty:
        ax.scatter(row4["k"], row4["wcss"], color="red", s=120,
                   zorder=5, label="k=4", marker="*")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("WCSS / Inertia")
    ax.set_title("K-means elbow plot")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _savefig(fig, "kmeans_elbow.png")


def plot_kmeans_k4_curves(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    out_dir: Path | None = None,
) -> Path:
    """k=4 k-means cluster curves."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                              sharex=True, sharey=True, squeeze=False)
    n_total = X.shape[0]

    for k in range(4):
        ax = axes.flat[k]
        mask = kmeans_labels == k
        curves = X[mask]
        n_in = mask.sum()
        col = NODE_COLORS[k]

        for curve in curves:
            ax.plot(TIME_GRID, curve, alpha=MEMBER_ALPHA, linewidth=0.5, color=col)
        if n_in > 4:
            q25 = np.nanpercentile(curves, 25, axis=0)
            q75 = np.nanpercentile(curves, 75, axis=0)
            ax.fill_between(TIME_GRID, q25, q75, alpha=SHADE_ALPHA, color=col)

        mean_c = curves.mean(axis=0)
        ax.plot(TIME_GRID, mean_c, color=col, linewidth=MEAN_LINEWIDTH, label=f"n={n_in}")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE")
        ax.set_title(f"K-means cluster {k}  n={n_in} ({n_in/n_total*100:.1f}%)")
        ax.legend(fontsize=7)

    plt.suptitle("K-means (k=4) cluster curves", fontsize=12)
    plt.tight_layout()
    return _savefig(fig, "kmeans_k4_curves.png")
