#!/usr/bin/env python3
"""Unsupervised DTW/derivative ablation on the EG synthetic curve dataset.

Target labels and generator QC flags are loaded only after every distance matrix
and cluster assignment has been computed.  The main representation uses the
relative progress coordinate u in [0, 1], so its derivatives are d/du rather
than physical rates per hour.  Original durations are retained for post-hoc
confounding checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)


CLASSES = ("bridge", "hill", "slope", "valley")
METHODS = (
    "level_dtw",
    "derivative_d12_dtw",
    "shared_level_d12_dtw",
    "separate_level_d12_fusion",
    "derivative_d123_dtw",
    "shared_level_d123_dtw",
)
METHOD_LABELS = {
    "level_dtw": "Level DTW",
    "derivative_d12_dtw": "D1+D2 DTW",
    "shared_level_d12_dtw": "Shared-path level+D1+D2",
    "separate_level_d12_fusion": "Separate level/derivative fusion",
    "derivative_d123_dtw": "D1+D2+D3 DTW",
    "shared_level_d123_dtw": "Shared-path level+D1+D2+D3",
}
COLORS = {
    "bridge": "#0072B2",
    "hill": "#D55E00",
    "slope": "#009E73",
    "valley": "#CC79A7",
}


@dataclass(frozen=True)
class Config:
    grid_points: int = 48
    savgol_window: int = 7
    savgol_degree: int = 3
    dtw_radius: int = 6
    clusters: int = 4
    linkage: str = "average"
    knn_neighbors: int = 10
    separate_level_weight: float = 0.5
    outlier_threshold: float = 5.0


def robust_local_outlier_filter(y: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """Replace only extreme interior spikes using a five-point Hampel rule."""
    clean = np.asarray(y, dtype=float).copy()
    replaced = 0
    if clean.size < 5:
        return clean, replaced
    original = clean.copy()
    for i in range(2, clean.size - 2):
        neighbors = np.r_[original[i - 2 : i], original[i + 1 : i + 3]]
        center = float(np.median(neighbors))
        mad = float(np.median(np.abs(neighbors - center)))
        scale = max(1.4826 * mad, 0.01)
        if abs(original[i] - center) > threshold * scale:
            clean[i] = center
            replaced += 1
    return clean, replaced


def prepare_curve(
    frame: pd.DataFrame, config: Config
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Interpolate one irregular curve on u, smooth, and derive d/du channels."""
    grouped = (
        frame.groupby("time_h", as_index=False)["y_relative"]
        .median()
        .sort_values("time_h")
    )
    t = grouped["time_h"].to_numpy(dtype=float)
    y = grouped["y_relative"].to_numpy(dtype=float)
    if t.size < 5 or not np.all(np.diff(t) > 0) or t[-1] <= t[0]:
        raise ValueError("curve needs at least five strictly ordered time points")
    y, replacements = robust_local_outlier_filter(y, config.outlier_threshold)
    u_observed = (t - t[0]) / (t[-1] - t[0])
    u_grid = np.linspace(0.0, 1.0, config.grid_points)
    level = PchipInterpolator(u_observed, y, extrapolate=False)(u_grid)
    level = savgol_filter(
        level,
        window_length=config.savgol_window,
        polyorder=config.savgol_degree,
        mode="interp",
    )
    delta = 1.0 / (config.grid_points - 1)
    d1 = savgol_filter(
        level,
        window_length=config.savgol_window,
        polyorder=config.savgol_degree,
        deriv=1,
        delta=delta,
        mode="interp",
    )
    d2 = savgol_filter(
        level,
        window_length=config.savgol_window,
        polyorder=config.savgol_degree,
        deriv=2,
        delta=delta,
        mode="interp",
    )
    d3 = savgol_filter(
        level,
        window_length=config.savgol_window,
        polyorder=config.savgol_degree,
        deriv=3,
        delta=delta,
        mode="interp",
    )
    channels = np.column_stack((level, d1, d2, d3))
    max_gap = float(np.max(np.diff(u_observed)))
    return u_grid, channels, replacements, max_gap


def robust_channel_scale(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each channel without using class labels."""
    flat = channels.reshape(-1, channels.shape[-1])
    center = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(flat, axis=0)
    scale = np.where(scale > 1e-10, scale, np.maximum(fallback, 1e-10))
    return (channels - center) / scale, center, scale


def constrained_dtw(x: np.ndarray, y: np.ndarray, weights: np.ndarray, radius: int) -> float:
    """Sakoe-Chiba constrained multichannel DTW with path-length normalization."""
    n, m = x.shape[0], y.shape[0]
    radius = max(int(radius), abs(n - m))
    previous_cost = np.full(m + 1, np.inf)
    previous_len = np.zeros(m + 1, dtype=np.int32)
    previous_cost[0] = 0.0
    for i in range(1, n + 1):
        current_cost = np.full(m + 1, np.inf)
        current_len = np.zeros(m + 1, dtype=np.int32)
        lower, upper = max(1, i - radius), min(m, i + radius)
        for j in range(lower, upper + 1):
            candidates = (previous_cost[j - 1], previous_cost[j], current_cost[j - 1])
            choice = int(np.argmin(candidates))
            delta = x[i - 1] - y[j - 1]
            local = float(np.sqrt(np.dot(weights, delta * delta)))
            if choice == 0:
                current_cost[j] = local + previous_cost[j - 1]
                current_len[j] = 1 + previous_len[j - 1]
            elif choice == 1:
                current_cost[j] = local + previous_cost[j]
                current_len[j] = 1 + previous_len[j]
            else:
                current_cost[j] = local + current_cost[j - 1]
                current_len[j] = 1 + current_len[j - 1]
        previous_cost, previous_len = current_cost, current_len
    return float(previous_cost[m] / max(previous_len[m], 1))


def _distance_row(i: int, channels: np.ndarray, radius: int) -> tuple[int, np.ndarray]:
    n = channels.shape[0]
    values = np.zeros((n - i - 1, 5), dtype=float)
    specifications = (
        ((0,), np.array([1.0])),
        ((1, 2), np.array([0.75, 0.25])),
        ((0, 1, 2), np.array([0.50, 0.375, 0.125])),
        ((1, 2, 3), np.array([0.60, 0.30, 0.10])),
        ((0, 1, 2, 3), np.array([0.50, 0.30, 0.15, 0.05])),
    )
    for offset, j in enumerate(range(i + 1, n)):
        for column, (indices, weights) in enumerate(specifications):
            idx = list(indices)
            values[offset, column] = constrained_dtw(
                channels[i][:, idx], channels[j][:, idx], weights, radius
            )
    return i, values


def pairwise_matrices(channels: np.ndarray, radius: int, jobs: int) -> dict[str, np.ndarray]:
    n = channels.shape[0]
    names = (
        "level_dtw",
        "derivative_d12_dtw",
        "shared_level_d12_dtw",
        "derivative_d123_dtw",
        "shared_level_d123_dtw",
    )
    matrices = {name: np.zeros((n, n), dtype=float) for name in names}
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_distance_row)(i, channels, radius) for i in range(n - 1)
    )
    for i, values in rows:
        for column, name in enumerate(names):
            matrices[name][i, i + 1 :] = values[:, column]
            matrices[name][i + 1 :, i] = values[:, column]
    return matrices


def median_scale_distance(matrix: np.ndarray) -> np.ndarray:
    positive = matrix[np.triu_indices_from(matrix, k=1)]
    positive = positive[positive > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return matrix / max(scale, 1e-12)


def cluster_distance(matrix: np.ndarray, clusters: int, method: str) -> np.ndarray:
    tree = linkage(squareform(matrix, checks=True), method=method)
    return (fcluster(tree, clusters, criterion="maxclust") - 1).astype(int)


def distance_to_knn_affinity(matrix: np.ndarray, neighbors: int) -> np.ndarray:
    """Build a symmetric self-tuned k-NN graph from a distance matrix."""
    n = matrix.shape[0]
    k = min(max(2, int(neighbors)), n - 1)
    order = np.argsort(matrix, axis=1)
    local_scale = np.take_along_axis(matrix, order[:, [k]], axis=1).ravel()
    positive = matrix[matrix > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    local_scale = np.where(local_scale > 1e-12, local_scale, fallback)
    affinity = np.exp(
        -(matrix * matrix) / np.maximum(np.outer(local_scale, local_scale), 1e-12)
    )
    mask = np.zeros_like(affinity, dtype=bool)
    for row in range(n):
        mask[row, order[row, 1 : k + 1]] = True
    affinity = np.where(mask | mask.T, affinity, 0.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def graph_cluster_distance(
    matrix: np.ndarray, clusters: int, neighbors: int, seed: int = 2026
) -> np.ndarray:
    affinity = distance_to_knn_affinity(matrix, neighbors)
    return SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=seed,
    ).fit_predict(affinity).astype(int)


def optimal_accuracy(true: np.ndarray, predicted: np.ndarray) -> tuple[float, dict[int, str]]:
    labels = np.asarray(CLASSES)
    true_index = np.array([int(np.flatnonzero(labels == item)[0]) for item in true])
    pred_ids = np.unique(predicted)
    table = confusion_matrix(true_index, predicted, labels=np.arange(max(4, predicted.max() + 1)))
    rows, cols = linear_sum_assignment(-table[:4, :])
    mapping = {int(col): str(labels[row]) for row, col in zip(rows, cols)}
    correct = sum(int(table[row, col]) for row, col in zip(rows, cols))
    return correct / len(true), mapping


def evaluate(
    true: np.ndarray,
    predicted: np.ndarray,
    distance: np.ndarray,
    duration_regime: np.ndarray,
) -> dict[str, float]:
    accuracy, _ = optimal_accuracy(true, predicted)
    return {
        "ari": float(adjusted_rand_score(true, predicted)),
        "nmi": float(normalized_mutual_info_score(true, predicted)),
        "homogeneity": float(homogeneity_score(true, predicted)),
        "completeness": float(completeness_score(true, predicted)),
        "mapped_accuracy": float(accuracy),
        "silhouette": float(silhouette_score(distance, predicted, metric="precomputed")),
        "duration_regime_ari": float(adjusted_rand_score(duration_regime, predicted)),
    }


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, (clusterer, frame) in zip(axes, metrics.groupby("clusterer", sort=False)):
        ordered = frame.sort_values("ari", ascending=True)
        labels = [METHOD_LABELS[item] for item in ordered["method"]]
        values = ordered["ari"].to_numpy(float)
        ax.barh(labels, values, color="#4C78A8")
        for position, value in enumerate(values):
            ax.text(max(value, 0.0) + 0.012, position, f"{value:.3f}", va="center", fontsize=8)
        ax.set_xlim(-0.03, 1.04)
        ax.set_title(clusterer)
        ax.grid(axis="x", alpha=0.25)
    fig.supxlabel("Adjusted Rand index (labels used only here)")
    fig.suptitle("Distance representation × clustering engine")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_best_confusion(
    true: np.ndarray, predicted: np.ndarray, method: str, clusterer: str, output: Path
) -> None:
    _, mapping = optimal_accuracy(true, predicted)
    mapped = np.asarray([mapping.get(int(item), "unmapped") for item in predicted])
    table = confusion_matrix(true, mapped, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(table, cmap="Blues")
    for row in range(4):
        for col in range(4):
            ax.text(col, row, str(table[row, col]), ha="center", va="center")
    ax.set_xticks(range(4), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Mapped cluster")
    ax.set_ylabel("Generator target (post-hoc only)")
    ax.set_title(f"Best: {METHOD_LABELS[method]} + {clusterer}")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cluster_curves(
    u: np.ndarray, levels: np.ndarray, predicted: np.ndarray, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for cluster_id, ax in enumerate(axes.ravel()):
        subset = levels[predicted == cluster_id]
        for curve in subset:
            ax.plot(u, curve, color="#777777", alpha=0.10, linewidth=0.7)
        if subset.size:
            ax.plot(u, np.median(subset, axis=0), color="#111111", linewidth=2.5)
        ax.set_title(f"Cluster {cluster_id}: n={len(subset)}")
        ax.grid(alpha=0.2)
    fig.supxlabel("Relative progress u (display only; original hour retained in data)")
    fig.supylabel("Smoothed relative PCE")
    fig.suptitle("Unsupervised clusters for the best fixed ablation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    config: Config,
    metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    best_method: str,
    best_clusterer: str,
    replacements: int,
    max_gap_summary: dict[str, float],
    qc_metrics: dict[str, float],
    knn_sensitivity: pd.DataFrame,
) -> None:
    indexed = metrics.set_index(["clusterer", "method"])
    best = indexed.loc[(best_clusterer, best_method)]
    level = indexed.loc[(best_clusterer, "level_dtw")]
    table_sections = []
    for clusterer, frame in metrics.groupby("clusterer", sort=False):
        rows = []
        for _, row in frame.sort_values("ari", ascending=False).iterrows():
            rows.append(
                f"| {METHOD_LABELS[row['method']]} | {row['ari']:.3f} | {row['nmi']:.3f} | "
                f"{row['mapped_accuracy']:.3f} | {row['silhouette']:.3f} | "
                f"{row['duration_regime_ari']:.3f} |"
            )
        table_sections.append(
            f"### {clusterer}\n\n"
            "| 方法 | ARI | NMI | 映射准确率 | silhouette | 聚类-时长层 ARI |\n"
            "|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
        )
    regime_pivot = regime_metrics.loc[
        regime_metrics["clusterer"] == best_clusterer
    ].pivot(index="duration_regime", columns="method", values="ari")
    regime_rows = []
    for regime, row in regime_pivot.iterrows():
        regime_rows.append(
            f"| {regime} | {row['level_dtw']:.3f} | {row[best_method]:.3f} |"
        )
    best_sensitivity = knn_sensitivity.loc[
        knn_sensitivity["method"] == best_method, "ari"
    ]
    level_sensitivity = knn_sensitivity.loc[
        knn_sensitivity["method"] == "level_dtw", "ari"
    ]
    text = f"""# EG 数据集：DTW 与导数融合无监督实验报告

## 结论

本次实验已在 400 条曲线上完整跑通。固定方案中表现最好的是
`{best_method}`（{METHOD_LABELS[best_method]}）与 `{best_clusterer}`，ARI={best['ari']:.3f}、
标签最优映射后的准确率={best['mapped_accuracy']:.3f}。只用幅值的 Level-DTW
在相同聚类器下 ARI={level['ari']:.3f}，二者差值为 {best['ari'] - level['ari']:+.3f}。

这是人工数据上的流程验证，不证明真实实验曲线必然可分，也不证明 Valley 的真实恢复机制
服从生成器所用的幂律。类别标签、形态 QC 和异常点注入标志均未进入距离或聚类，只用于
聚类结束后的评价。

## 固定实验协议

- 使用全部 400 条曲线，预先固定为 4 簇；并列比较 average-linkage 层次聚类与
  自调尺度 {config.knn_neighbors}-NN 图谱聚类；
- 原始 `time_h` 不改写；形态比较内部定义 `u=(t-t_start)/T`；
- PCHIP 只在每条曲线自身观测范围内重采样为 {config.grid_points} 点，随后使用
  Savitzky–Golay（window={config.savgol_window}, degree={config.savgol_degree}）平滑；
- 导数为 `d/du`、`d²/du²`、`d³/du³`，不是每小时导数；
- DTW 使用 Sakoe–Chiba 半径 {config.dtw_radius}，累计代价除以路径长度；
- 各通道用全数据的 median/MAD 无监督缩放；共享路径 D1+D2 权重为
  level/D1/D2 = 0.50/0.375/0.125；分距离融合的 level 权重为
  {config.separate_level_weight:.2f}；
- Hampel 规则仅替换极端局部尖峰，共替换 {replacements} / 16120 个观测点；
- 最大相对采样缺口中位数={max_gap_summary['median']:.3f}，95 分位数=
  {max_gap_summary['p95']:.3f}。稀疏区间内的插值仍是本实验的重要限制。

## 全数据结果

{chr(10).join(table_sections)}

“聚类-时长层 ARI”用于检查时长混杂，越接近 0 表示聚类越不像四个时长档；它不是性能分数。

## 按实际时长层复算

| 时长层 | Level-DTW ARI | 全数据最佳方法 ARI |
|---|---:|---:|
{chr(10).join(regime_rows)}

形态规则 QC 通过的 {int(qc_metrics['n'])} 条曲线上，最佳组合 ARI={qc_metrics['ari']:.3f}。
完整逐方法指标见 `metrics.csv`，逐曲线簇编号见 `assignments.csv`，距离矩阵见
`distance_matrices.npz`。

## 近邻数敏感性

固定距离和 4 簇，只把图的近邻数改为 5、10、15、20、30。最佳融合距离的 ARI 范围为
{best_sensitivity.min():.3f}--{best_sensitivity.max():.3f}，Level-DTW 的范围为
{level_sensitivity.min():.3f}--{level_sensitivity.max():.3f}。完整结果见
`knn_sensitivity.csv`。该检查说明提升不只出现在 k=10，但仍不能替代独立数据验证。

## 解释边界

相对进程对齐回答“阶段顺序是否相似”，不回答“每小时变化速率是否相似”。本数据集中四类
时长分布被刻意配平，因此暂不把 duration 或 `df/dt` 加入主聚类。若转向真实数据，应另建
动力学实验分支，并把 duration、`df/dt` 与观测是否覆盖完整过程明确建模。

average-linkage 下纯导数距离会受少数异常曲线影响而产生近单例簇；图聚类避免由这些全局
离群点主导切树。因此，本结果支持的是“导数融合距离配合局部近邻图”，而不是“任何聚类器
加入导数都会改善”。

方法排名使用了事后标签，因此“最佳方法”是本数据集上的描述性结果，不应直接当成无偏的
泛化估计。下一步应冻结参数，在独立生成种子或真实盲审数据上验证。
"""
    (output / "REPORT_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "eg_based_synthetic_dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=48)
    parser.add_argument("--dtw-radius", type=int, default=6)
    parser.add_argument(
        "--reuse-distances",
        action="store_true",
        help="Reuse distance_matrices.npz in the output directory.",
    )
    args = parser.parse_args()
    config = Config(grid_points=args.grid_points, dtw_radius=args.dtw_radius)
    if config.savgol_window > config.grid_points:
        raise ValueError("Savitzky-Golay window cannot exceed grid size")
    args.output.mkdir(parents=True, exist_ok=True)

    long_data = pd.read_csv(args.data / "curve_data_long.csv")
    # Deliberately avoid target_class and is_injected_outlier during preprocessing.
    ids = sorted(long_data["curve_id"].unique())
    prepared, replacements, gaps = [], 0, []
    u_grid = None
    for curve_id in ids:
        frame = long_data.loc[long_data["curve_id"] == curve_id, ["time_h", "y_relative"]]
        u_grid, channels, count, max_gap = prepare_curve(frame, config)
        prepared.append(channels)
        replacements += count
        gaps.append(max_gap)
    raw_channels = np.stack(prepared)
    scaled_channels, centers, scales = robust_channel_scale(raw_channels)

    distance_file = args.output / "distance_matrices.npz"
    if args.reuse_distances:
        if not distance_file.exists():
            raise FileNotFoundError(f"cannot reuse missing {distance_file}")
        cached_config_file = args.output / "config.json"
        if cached_config_file.exists():
            cached_config = json.loads(cached_config_file.read_text(encoding="utf-8"))
            for key in ("grid_points", "savgol_window", "savgol_degree", "dtw_radius"):
                if cached_config.get(key) != getattr(config, key):
                    raise ValueError(
                        f"cached {key}={cached_config.get(key)!r} does not match "
                        f"requested {getattr(config, key)!r}; recompute without --reuse-distances"
                    )
        print(f"Reusing {distance_file}", flush=True)
        cached = np.load(distance_file)
        matrices = {name: cached[name] for name in cached.files}
    else:
        print(f"Computing five DTW matrices for {len(ids)} curves ...", flush=True)
        matrices = pairwise_matrices(scaled_channels, config.dtw_radius, args.jobs)
    level_scaled = median_scale_distance(matrices["level_dtw"])
    derivative_scaled = median_scale_distance(matrices["derivative_d12_dtw"])
    matrices["separate_level_d12_fusion"] = (
        config.separate_level_weight * level_scaled
        + (1.0 - config.separate_level_weight) * derivative_scaled
    )

    # Labels, generator QC, and duration strata enter only after clustering.
    metadata = pd.read_csv(args.data / "curve_metadata.csv").set_index("curve_id").loc[ids]
    truth = metadata["target_class"].to_numpy(str)
    duration_regime = metadata["duration_regime"].to_numpy(str)
    assignments = pd.DataFrame(
        {
            "curve_id": ids,
            "target_class_posthoc": truth,
            "duration_h": metadata["duration_h"].to_numpy(float),
            "duration_regime": duration_regime,
            "morphology_qc_pass_posthoc": metadata["morphology_qc_pass"].to_numpy(int),
            "max_relative_sampling_gap": gaps,
        }
    )
    clusterers = {
        "average_hierarchical": lambda matrix: cluster_distance(
            matrix, config.clusters, config.linkage
        ),
        "knn10_spectral": lambda matrix: graph_cluster_distance(
            matrix, config.clusters, config.knn_neighbors
        ),
    }
    metric_rows, predictions = [], {}
    for clusterer, cluster_function in clusterers.items():
        for method in METHODS:
            predicted = cluster_function(matrices[method])
            predictions[(clusterer, method)] = predicted
            assignments[f"cluster_{clusterer}_{method}"] = predicted
            metric_rows.append(
                {
                    "clusterer": clusterer,
                    "method": method,
                    **evaluate(truth, predicted, matrices[method], duration_regime),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    best_row = metrics.sort_values(
        ["ari", "clusterer", "method"], ascending=[False, True, True]
    ).iloc[0]
    best_method = str(best_row["method"])
    best_clusterer = str(best_row["clusterer"])

    regime_rows = []
    for regime in sorted(np.unique(duration_regime)):
        mask = duration_regime == regime
        for clusterer, cluster_function in clusterers.items():
            for method in METHODS:
                submatrix = matrices[method][np.ix_(mask, mask)]
                predicted = cluster_function(submatrix)
                regime_rows.append(
                    {
                        "duration_regime": regime,
                        "n": int(mask.sum()),
                        "clusterer": clusterer,
                        "method": method,
                        **evaluate(truth[mask], predicted, submatrix, duration_regime[mask]),
                    }
                )
    regime_metrics = pd.DataFrame(regime_rows)

    sensitivity_rows = []
    for neighbors in (5, 10, 15, 20, 30):
        for method in METHODS:
            predicted = graph_cluster_distance(
                matrices[method], config.clusters, neighbors
            )
            sensitivity_rows.append(
                {
                    "neighbors": neighbors,
                    "method": method,
                    **evaluate(truth, predicted, matrices[method], duration_regime),
                }
            )
    knn_sensitivity = pd.DataFrame(sensitivity_rows)
    qc_mask = metadata["morphology_qc_pass"].to_numpy(int) == 1
    qc_matrix = matrices[best_method][np.ix_(qc_mask, qc_mask)]
    qc_predicted = clusterers[best_clusterer](qc_matrix)
    qc_metrics = {
        "n": int(qc_mask.sum()),
        **evaluate(
            truth[qc_mask], qc_predicted, qc_matrix, duration_regime[qc_mask]
        ),
    }

    assignments.to_csv(args.output / "assignments.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    regime_metrics.to_csv(args.output / "duration_regime_metrics.csv", index=False)
    knn_sensitivity.to_csv(args.output / "knn_sensitivity.csv", index=False)
    np.savez_compressed(distance_file, **matrices)
    configuration = {
        **asdict(config),
        "jobs": args.jobs,
        "data": str(args.data.resolve()),
        "channel_center": centers.tolist(),
        "channel_scale": scales.tolist(),
        "channel_semantics": ["level", "d/du", "d2/du2", "d3/du3"],
        "best_method_posthoc": best_method,
        "best_clusterer_posthoc": best_clusterer,
    }
    (args.output / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_metrics(metrics, args.output / "method_comparison.png")
    plot_best_confusion(
        truth,
        predictions[(best_clusterer, best_method)],
        best_method,
        best_clusterer,
        args.output / "best_confusion.png",
    )
    assert u_grid is not None
    plot_cluster_curves(
        u_grid,
        raw_channels[:, :, 0],
        predictions[(best_clusterer, best_method)],
        args.output / "best_clusters.png",
    )
    write_report(
        args.output,
        config,
        metrics,
        regime_metrics,
        best_method,
        best_clusterer,
        replacements,
        {"median": float(np.median(gaps)), "p95": float(np.quantile(gaps, 0.95))},
        qc_metrics,
        knn_sensitivity,
    )
    print(metrics.sort_values("ari", ascending=False).to_string(index=False), flush=True)
    print(f"Outputs written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
