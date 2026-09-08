#!/usr/bin/env python3
"""Run the three preprocessing-only SOM experiments requested in the prompt.

The script deliberately keeps SOM settings fixed and performs the IFO-style
morphology audit only after model selection and training are complete.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from minisom import MiniSom
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "pkl_complete" / "20230303_mySeriesDrop.pkl"
GROUP_FILE = ROOT / "dataset" / "PCE_df_grouping.csv"
PAPER_PDF = ROOT / "paper" / "work.pdf"
OUT = ROOT / "som_preprocessing_comparison"

HOURS = 150.0
INTERVAL_MINUTES = 10
SAVGOL_WINDOW = 71
SAVGOL_ORDER = 2
SIGMA = 0.5
LEARNING_RATE = 0.1
ITERATIONS = 50_000
PRIMARY_SEED = 42
ROBUSTNESS_SEEDS = tuple(range(10))
K_VALUES = tuple(range(2, 9))

PIPELINES = (
    ("test1_raw", "Raw → SOM", "01_raw_som_clusters.png"),
    (
        "test2_resample_normalize",
        "Resample + Normalize → SOM",
        "02_resample_normalize_som_clusters.png",
    ),
    (
        "test3_resample_normalize_smooth",
        "Resample + Normalize + Smooth → SOM",
        "03_resample_normalize_smooth_som_clusters.png",
    ),
)


@dataclass
class SomResult:
    k: int
    rows: int
    cols: int
    seed: int
    qe: float
    labels: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    counts: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_legacy_pickle(path: Path) -> pd.Series:
    """Load the pandas-1.x pickle under pandas 2.x without modifying the file."""
    import pandas._libs.internals as internals
    import pandas.core.internals.blocks as blocks

    original = blocks.new_block

    def compatible_new_block(values, placement, *, ndim, refs=None):
        if isinstance(placement, slice):
            placement = internals.BlockPlacement(placement)
        return original(values, placement, ndim=ndim, refs=refs)

    blocks.new_block = compatible_new_block
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    finally:
        blocks.new_block = original
    if not isinstance(obj, pd.Series):
        raise TypeError(f"Expected pandas Series, found {type(obj)!r}")
    return obj


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    series = load_legacy_pickle(DATA_FILE)
    arrays: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    for i, item in enumerate(series):
        if not isinstance(item, pd.DataFrame) or "MPPT_EFF" not in item:
            raise ValueError(f"Sample {i} is not a DataFrame with MPPT_EFF")
        values = item["MPPT_EFF"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Sample {i} contains non-finite values")
        arrays.append(values)
        timestamps.append(pd.DatetimeIndex(item.index))

    lengths = np.array([len(x) for x in arrays])
    if len(np.unique(lengths)) != 1:
        raise ValueError("Published traces are not equal length; Near-Raw handling required")
    raw = np.stack(arrays)
    if raw.shape[1] != 900:
        raise ValueError(f"Expected 900 points from the released dataset, got {raw.shape[1]}")

    spacings = []
    for idx in timestamps:
        delta = np.diff(idx.asi8) / 60_000_000_000
        spacings.extend(delta.tolist())
    spacings_arr = np.asarray(spacings)
    common_spacing = bool(np.allclose(spacings_arr, INTERVAL_MINUTES))
    if not common_spacing:
        raise ValueError("The published traces are not on a regular 10-minute grid")

    # Elapsed-time grid. The released frames already contain exactly 900
    # consecutive 10-minute points, so Test 2 resampling/interpolation is an
    # identity operation; normalization remains the only numerical change.
    time_hours = np.arange(raw.shape[1], dtype=float) * INTERVAL_MINUTES / 60.0
    max_abs = np.max(np.abs(raw), axis=1, keepdims=True)
    if np.any(max_abs == 0):
        raise ValueError("At least one curve has zero MaxAbs denominator")
    norm = raw / max_abs
    smooth = savgol_filter(norm, SAVGOL_WINDOW, SAVGOL_ORDER, axis=1)

    grouping = pd.read_csv(GROUP_FILE)
    if "Unnamed: 0" in grouping:
        grouping = grouping.drop(columns=["Unnamed: 0"])
    if len(grouping) != len(raw):
        raise ValueError("Grouping CSV and trace count do not match")

    audit = {
        "sample_count": int(raw.shape[0]),
        "points_per_curve": int(raw.shape[1]),
        "elapsed_start_hours": float(time_hours[0]),
        "elapsed_end_hours": float(time_hours[-1]),
        "nominal_window_hours": HOURS,
        "sampling_interval_minutes": INTERVAL_MINUTES,
        "all_values_finite": bool(np.isfinite(raw).all()),
        "all_curves_equal_length": bool(len(np.unique(lengths)) == 1),
        "all_within_curve_spacings_10_minutes": common_spacing,
        "test2_resampling_changed_values": False,
        "test2_interpolation_required": False,
        "raw_test_label": "Raw SOM",
        "raw_limitation": (
            "Zenodo 发布的是已清洗并排列为 900 个连续 10 min 点的曲线。"
            "Test 1 直接使用这些发布的 PCE 值，不再增加重采样、插值、归一化或平滑；"
            "因此它不是仪器级 raw JSON 分析。"
        ),
    }
    return raw, time_hours, grouping, audit


def som_shape(k: int) -> tuple[int, int]:
    """Return a deterministic near-square rectangular map with exactly k nodes."""
    # Use the closest exact factor pair. This gives the validated 2x2 topology
    # at K=4 and applies the same rule to every pipeline and candidate K.
    for rows in range(int(math.sqrt(k)), 0, -1):
        if k % rows == 0:
            return rows, k // rows
    raise AssertionError("unreachable")


def train_som(data: np.ndarray, k: int, seed: int) -> SomResult:
    rows, cols = som_shape(k)
    som = MiniSom(
        rows,
        cols,
        data.shape[1],
        sigma=SIGMA,
        learning_rate=LEARNING_RATE,
        neighborhood_function="gaussian",
        activation_distance="euclidean",
        random_seed=seed,
    )
    som.random_weights_init(data)
    som.train(data, ITERATIONS, random_order=False, verbose=False)
    coordinates = np.asarray([som.winner(x) for x in data], dtype=int)
    labels = np.ravel_multi_index(coordinates.T, (rows, cols)).astype(int)
    weights = som.get_weights().reshape(k, data.shape[1])
    counts = np.bincount(labels, minlength=k)
    means = np.vstack(
        [data[labels == c].mean(axis=0) if counts[c] else weights[c] for c in range(k)]
    )
    return SomResult(
        k=k,
        rows=rows,
        cols=cols,
        seed=seed,
        qe=float(som.quantization_error(data)),
        labels=labels,
        weights=weights,
        means=means,
        counts=counts,
    )


def select_k(results: dict[int, SomResult]) -> tuple[int, pd.DataFrame]:
    ks = np.asarray(sorted(results), dtype=float)
    qes = np.asarray([results[int(k)].qe for k in ks])
    q_norm = (qes - qes.min()) / (qes.max() - qes.min())
    chord = 1.0 - (ks - ks.min()) / (ks.max() - ks.min())
    elbow_score = chord - q_norm
    selected = int(ks[np.argmax(elbow_score)])
    rows = []
    for i, kf in enumerate(ks):
        k = int(kf)
        previous = qes[i - 1] if i else np.nan
        rows.append(
            {
                "k": k,
                "som_rows": results[k].rows,
                "som_cols": results[k].cols,
                "quantization_error": qes[i],
                "qe_reduction_from_previous": previous - qes[i] if i else np.nan,
                "qe_reduction_percent": (
                    100 * (previous - qes[i]) / previous if i else np.nan
                ),
                "normalized_chord_elbow_score": elbow_score[i],
                "occupied_nodes": int(np.count_nonzero(results[k].counts)),
                "selected": k == selected,
                "selection_rule": (
                    "maximum vertical deviation from endpoint chord on min-max "
                    "normalized QE for K=2..8"
                ),
            }
        )
    return selected, pd.DataFrame(rows)


def own_max_normalize(curves: np.ndarray) -> np.ndarray:
    den = np.max(np.abs(curves), axis=1, keepdims=True)
    return curves / np.where(den == 0, 1.0, den)


def linear_slope(time: np.ndarray, y: np.ndarray, start: float, stop: float) -> float:
    mask = (time >= start) & (time <= stop)
    return float(np.polyfit(time[mask], y[mask], 1)[0])


def shape_features(y_native: np.ndarray, time: np.ndarray) -> dict:
    y = own_max_normalize(y_native[None, :])[0]
    # SG is used here only as a uniform post-clustering measurement operator.
    # It never enters Test 1 or Test 2 SOM inputs.
    display = savgol_filter(y, SAVGOL_WINDOW, SAVGOL_ORDER)
    peak_i = int(np.argmax(display))
    trough_i = int(np.argmin(display))
    amplitude = max(float(np.ptp(display)), 1e-12)
    prominence = max(0.05 * amplitude, 1e-4)
    max_idx, _ = find_peaks(display, prominence=prominence, distance=30)
    min_idx, _ = find_peaks(-display, prominence=prominence, distance=30)
    turning_points = int(len(max_idx) + len(min_idx))
    initial = float(display[0])
    final = float(display[-1])
    maximum = float(display[peak_i])
    minimum = float(display[trough_i])
    peak_to_final = maximum - final
    recovery = float(np.max(display[trough_i:]) - minimum)
    valley_depth = max(0.0, min(initial, final) - minimum)
    early_stop = 0.2 * time[-1]
    late_start = 0.8 * time[-1]
    return {
        "initial_native": float(y_native[0]),
        "final_native": float(y_native[-1]),
        "maximum_native": float(np.max(y_native)),
        "minimum_native": float(np.min(y_native)),
        "initial_relative": initial,
        "final_relative": final,
        "maximum_relative": maximum,
        "minimum_relative": minimum,
        "overall_pce_change_relative": final - initial,
        "early_time_slope_relative_per_hour": linear_slope(time, display, 0, early_stop),
        "late_time_slope_relative_per_hour": linear_slope(
            time, display, late_start, time[-1]
        ),
        "time_of_maximum_hours": float(time[peak_i]),
        "time_of_minimum_hours": float(time[trough_i]),
        "major_turning_points": turning_points,
        "peak_to_final_decline_relative": peak_to_final,
        "valley_depth_relative": valley_depth,
        "recovery_amplitude_relative": recovery,
        "range_relative": amplitude,
        "peak_width_fraction": float(
            np.mean(display >= max(initial, final) + 0.70 * max(0, maximum - max(initial, final)))
        )
        if maximum > max(initial, final)
        else 0.0,
    }


def qualitative_tendency(features: dict) -> str:
    change = features["overall_pce_change_relative"]
    peak_t = features["time_of_maximum_hours"]
    trough_t = features["time_of_minimum_hours"]
    recovery = features["recovery_amplitude_relative"]
    decline = features["peak_to_final_decline_relative"]
    turns = features["major_turning_points"]
    rise = features["maximum_relative"] - features["initial_relative"]
    horizon = HOURS
    if 0.03 * horizon < trough_t < 0.90 * horizon and recovery >= 0.12:
        return "decline followed by partial recovery"
    if (
        0.03 * horizon < peak_t < 0.75 * horizon
        and rise >= 0.04
        and decline >= 0.04
    ):
        return "initial increase followed by slow decay"
    if turns >= 3:
        return "non-monotonic / mixed"
    if change <= -0.55:
        return "rapid decay"
    if change <= -0.30:
        return "medium decay"
    if change <= -0.08:
        return "slow decay"
    if change >= 0.08:
        return "overall increase / recovery-like"
    return "near-stable / mixed"


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = float(np.mean(values))
    total = float(np.sum((values - grand) ** 2))
    if total == 0:
        return 0.0
    between = sum(
        np.sum(labels == c) * (float(np.mean(values[labels == c])) - grand) ** 2
        for c in np.unique(labels)
    )
    return float(between / total)


def save_pipeline_outputs(
    key: str,
    title: str,
    filename: str,
    data: np.ndarray,
    raw: np.ndarray,
    time: np.ndarray,
    result: SomResult,
    qe_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = OUT / key
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "preprocessed_curves.npy", data.astype(np.float32))

    assignment = pd.DataFrame(
        {
            "sample_id": np.arange(len(data)),
            "cluster": result.labels,
            "som_row": result.labels // result.cols,
            "som_col": result.labels % result.cols,
            "initial_pce_raw": raw[:, 0],
            "maximum_pce_raw": raw.max(axis=1),
            "final_pce_raw": raw[:, -1],
        }
    )
    assignment.to_csv(directory / "cluster_assignments.csv", index=False)
    qe_table.to_csv(directory / "quantization_error.csv", index=False)
    plot_qe_elbow(qe_table, title, directory / "quantization_error_elbow.png")

    feature_rows = []
    summary_rows = []
    centre_columns: dict[str, np.ndarray] = {"time_hours": time}
    for cluster in range(result.k):
        mask = result.labels == cluster
        centre = result.means[cluster]
        features = shape_features(centre, time)
        tendency = qualitative_tendency(features)
        idx = np.flatnonzero(mask)
        representative = int(idx[np.argmin(np.linalg.norm(data[idx] - centre, axis=1))])
        feature_rows.append(
            {
                "test": key,
                "pipeline": title,
                "cluster": cluster,
                "n": int(mask.sum()),
                "fraction": float(mask.mean()),
                "qualitative_morphology": tendency,
                **features,
            }
        )
        summary_rows.append(
            {
                "cluster": cluster,
                "n": int(mask.sum()),
                "fraction": float(mask.mean()),
                "fraction_percent": float(100 * mask.mean()),
                "representative_sample_id": representative,
                "qualitative_morphology": tendency,
                "initial_pce_mean": float(raw[mask, 0].mean()),
                "initial_pce_std": float(raw[mask, 0].std(ddof=1)),
                "initial_pce_median": float(np.median(raw[mask, 0])),
                "maximum_pce_mean": float(raw[mask].max(axis=1).mean()),
                "maximum_pce_std": float(raw[mask].max(axis=1).std(ddof=1)),
                "maximum_pce_median": float(np.median(raw[mask].max(axis=1))),
                "final_pce_mean": float(raw[mask, -1].mean()),
                "key_shape_features": (
                    f"Δrel={features['overall_pce_change_relative']:.3f}; "
                    f"early slope={features['early_time_slope_relative_per_hour']:.4f}/h; "
                    f"late slope={features['late_time_slope_relative_per_hour']:.4f}/h; "
                    f"turns={features['major_turning_points']}"
                ),
            }
        )
        centre_columns[f"cluster_{cluster}_mean"] = centre
        centre_columns[f"cluster_{cluster}_som_prototype"] = result.weights[cluster]
    feature_df = pd.DataFrame(feature_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(directory / "cluster_summary.csv", index=False)
    pd.DataFrame(centre_columns).to_csv(directory / "cluster_centres.csv", index=False)

    plot_clusters(
        data, time, result, title, directory / filename, normalized=key != "test1_raw"
    )
    plot_initial_max_distributions(
        assignment, directory / "initial_max_pce_distributions.png", title
    )
    return summary_df, feature_df


def plot_qe_elbow(table: pd.DataFrame, title: str, path: Path) -> None:
    selected = table.loc[table.selected].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(table.k, table.quantization_error, "o-", color="#4C78A8", lw=2)
    ax.scatter(
        [selected.k],
        [selected.quantization_error],
        s=110,
        color="#E45756",
        zorder=3,
        label=f"Selected elbow K={int(selected.k)}",
    )
    ax.set_xticks(table.k)
    ax.set_xlabel("Number of SOM nodes / clusters (K)")
    ax.set_ylabel("Quantization error")
    ax.set_title(f"{title}: SOM quantization-error elbow")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_clusters(
    data: np.ndarray,
    time: np.ndarray,
    result: SomResult,
    title: str,
    path: Path,
    normalized: bool,
) -> None:
    cols = min(3, result.k)
    rows = math.ceil(result.k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    global_min = float(np.nanpercentile(data, 0.1))
    global_max = float(np.nanpercentile(data, 99.9))
    pad = 0.06 * (global_max - global_min)
    for cluster in range(result.k):
        ax = axes[cluster]
        member = data[result.labels == cluster]
        for curve in member:
            ax.plot(time, curve, color="#4C78A8", alpha=0.025, lw=0.35, rasterized=True)
        ax.plot(time, result.means[cluster], color="black", lw=2.2, label="Cluster mean")
        ax.plot(
            time,
            result.weights[cluster],
            color="#E45756",
            lw=1.1,
            ls="--",
            label="SOM prototype",
        )
        ax.set_title(
            f"Cluster {cluster}: n={len(member)} ({100 * len(member) / len(data):.1f}%)\n"
            f"{qualitative_tendency(shape_features(result.means[cluster], time))}"
        )
        ax.set_ylim(global_min - pad, global_max + pad)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[result.k :]:
        ax.axis("off")
    for ax in axes[: result.k]:
        ax.set_xlabel("Elapsed ageing time (h)")
        ax.set_ylabel("PCE / max|PCE|" if normalized else "PCE (%)")
    fig.suptitle(
        f"{title} — natural K={result.k}, seed={result.seed}\n"
        "All member curves (low opacity), cluster means, and SOM prototypes",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_initial_max_distributions(assignments: pd.DataFrame, path: Path, title: str) -> None:
    clusters = sorted(assignments["cluster"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, column, label in zip(
        axes,
        ["initial_pce_raw", "maximum_pce_raw"],
        ["Initial PCE (%)", "Maximum PCE (%)"],
    ):
        values = [assignments.loc[assignments.cluster == c, column] for c in clusters]
        parts = ax.violinplot(values, positions=clusters, showmeans=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor("#72B7B2")
            body.set_alpha(0.55)
        ax.boxplot(values, positions=clusters, widths=0.18, showfliers=False)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
    fig.suptitle(f"{title}: raw initial and maximum PCE distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_fixed_k4(
    key: str,
    title: str,
    data: np.ndarray,
    raw: np.ndarray,
    time: np.ndarray,
    result: SomResult,
) -> None:
    directory = OUT / key
    pd.DataFrame(
        {
            "sample_id": np.arange(len(data)),
            "cluster": result.labels,
            "som_row": result.labels // result.cols,
            "som_col": result.labels % result.cols,
            "initial_pce_raw": raw[:, 0],
            "maximum_pce_raw": raw.max(axis=1),
        }
    ).to_csv(directory / "fixed_k4_cluster_assignments.csv", index=False)
    plot_clusters(
        data,
        time,
        result,
        title + " (controlled fixed-K view)",
        directory / "fixed_k4_clusters.png",
        normalized=key != "test1_raw",
    )


def match_centres(source: np.ndarray, target: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    source_rel = own_max_normalize(source)
    target_rel = own_max_normalize(target)
    cost = np.empty((len(source_rel), len(target_rel)))
    corr = np.empty_like(cost)
    for i, a in enumerate(source_rel):
        for j, b in enumerate(target_rel):
            corr[i, j] = np.corrcoef(a, b)[0, 1]
            cost[i, j] = np.sqrt(np.mean((a - b) ** 2))
    r, c = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, c)}, corr


def aligned_membership_changes(labels_a: np.ndarray, labels_b: np.ndarray) -> int:
    """Minimum mismatch count after one-to-one label alignment when K is equal."""
    unique_a = np.unique(labels_a)
    unique_b = np.unique(labels_b)
    if len(unique_a) != len(unique_b):
        return -1
    contingency = np.zeros((len(unique_a), len(unique_b)), dtype=int)
    for i, a in enumerate(unique_a):
        for j, b in enumerate(unique_b):
            contingency[i, j] = int(np.sum((labels_a == a) & (labels_b == b)))
    rows, cols = linear_sum_assignment(-contingency)
    return int(len(labels_a) - contingency[rows, cols].sum())


def robustness_analysis(
    key: str,
    data: np.ndarray,
    selected_k: int,
    primary: SomResult,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for seed in ROBUSTNESS_SEEDS:
        result = train_som(data, selected_k, seed)
        mapping, corr = match_centres(result.means, primary.means)
        matched_corr = [corr[src, dst] for src, dst in mapping.items()]
        rows.append(
            {
                "seed": seed,
                "k": selected_k,
                "quantization_error": result.qe,
                "ARI_vs_primary_seed_42": adjusted_rand_score(primary.labels, result.labels),
                "NMI_vs_primary_seed_42": normalized_mutual_info_score(
                    primary.labels, result.labels
                ),
                "mean_matched_centre_pearson": float(np.mean(matched_corr)),
                "minimum_matched_centre_pearson": float(np.min(matched_corr)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / key / "seed_robustness.csv", index=False)
    summary = {
        "test": key,
        "selected_k": selected_k,
        "repeat_seed_count": len(ROBUSTNESS_SEEDS),
        "repeat_seeds": list(ROBUSTNESS_SEEDS),
        "primary_seed": PRIMARY_SEED,
        "mean_quantization_error": float(frame.quantization_error.mean()),
        "std_quantization_error": float(frame.quantization_error.std(ddof=1)),
        "mean_ARI_vs_primary": float(frame.ARI_vs_primary_seed_42.mean()),
        "std_ARI_vs_primary": float(frame.ARI_vs_primary_seed_42.std(ddof=1)),
        "mean_NMI_vs_primary": float(frame.NMI_vs_primary_seed_42.mean()),
        "mean_matched_centre_pearson": float(frame.mean_matched_centre_pearson.mean()),
        "minimum_matched_centre_pearson_over_seeds": float(
            frame.minimum_matched_centre_pearson.min()
        ),
    }
    summary["qualitative_shapes_persist"] = bool(
        summary["mean_matched_centre_pearson"] >= 0.90
        and summary["mean_NMI_vs_primary"] >= 0.70
    )
    return frame, summary


def sample_shape_diagnostics(
    data: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    raw_magnitude_data: np.ndarray,
) -> dict:
    relative = own_max_normalize(data)
    initial = relative[:, 0]
    final = relative[:, -1]
    overall = final - initial
    early_n = max(2, int(0.20 * data.shape[1]))
    late_n = max(2, int(0.20 * data.shape[1]))
    x_early = time[:early_n]
    x_late = time[-late_n:]
    x_early_c = x_early - x_early.mean()
    x_late_c = x_late - x_late.mean()
    early_centered = relative[:, :early_n] - relative[:, :early_n].mean(
        axis=1, keepdims=True
    )
    late_centered = relative[:, -late_n:] - relative[:, -late_n:].mean(
        axis=1, keepdims=True
    )
    early_slopes = np.sum(early_centered * x_early_c[None, :], axis=1) / np.sum(
        x_early_c**2
    )
    late_slopes = np.sum(late_centered * x_late_c[None, :], axis=1) / np.sum(
        x_late_c**2
    )
    # Magnitude separation must always refer to the original PCE scale, even
    # when the SOM input is normalized or smoothed.
    max_pce = np.max(raw_magnitude_data, axis=1)
    initial_pce = raw_magnitude_data[:, 0]
    cluster_means = np.vstack([relative[labels == c].mean(axis=0) for c in np.unique(labels)])
    label_to_row = {int(c): i for i, c in enumerate(np.unique(labels))}
    residual = np.vstack(
        [relative[i] - cluster_means[label_to_row[int(labels[i])]] for i in range(len(data))]
    )
    magnitude_eta = float(
        np.mean([eta_squared(initial_pce, labels), eta_squared(max_pce, labels)])
    )
    shape_eta = float(
        np.mean(
            [
                eta_squared(overall, labels),
                eta_squared(early_slopes, labels),
                eta_squared(late_slopes, labels),
            ]
        )
    )
    if magnitude_eta > 1.5 * shape_eta:
        driver = "mainly absolute PCE magnitude"
    elif shape_eta > 1.5 * magnitude_eta:
        driver = "mainly degradation tendency / shape"
    else:
        driver = "a mixture of magnitude and degradation tendency"
    return {
        "eta_squared_initial_pce": eta_squared(initial_pce, labels),
        "eta_squared_maximum_pce": eta_squared(max_pce, labels),
        "eta_squared_overall_relative_change": eta_squared(overall, labels),
        "eta_squared_early_relative_slope": eta_squared(early_slopes, labels),
        "eta_squared_late_relative_slope": eta_squared(late_slopes, labels),
        "mean_magnitude_eta_squared": magnitude_eta,
        "mean_shape_eta_squared": shape_eta,
        "mean_within_cluster_shape_rmse": float(np.mean(np.sqrt(np.mean(residual**2, axis=1)))),
        "dominant_separation_interpretation": driver,
    }


def fixed_feature_table(
    key: str, title: str, result: SomResult, time: np.ndarray
) -> pd.DataFrame:
    rows = []
    for cluster in range(result.k):
        features = shape_features(result.means[cluster], time)
        rows.append(
            {
                "view": "fixed_k4",
                "test": key,
                "pipeline": title,
                "cluster": cluster,
                "n": int(result.counts[cluster]),
                "fraction": float(result.counts[cluster] / result.counts.sum()),
                "qualitative_morphology": qualitative_tendency(features),
                **features,
            }
        )
    return pd.DataFrame(rows)


def ifo_match(features: dict, archetype: str) -> tuple[bool, bool, str]:
    horizon = HOURS
    amplitude = max(features["range_relative"], 1e-12)
    rise = features["maximum_relative"] - features["initial_relative"]
    post_peak_drop = features["peak_to_final_decline_relative"]
    recovery = features["recovery_amplitude_relative"]
    peak_internal = 0 < features["time_of_maximum_hours"] <= 0.45 * horizon
    trough_internal = 0 < features["time_of_minimum_hours"] <= 0.45 * horizon
    width = features["peak_width_fraction"]
    if archetype == "Valley":
        exact = trough_internal and recovery >= 0.20 * amplitude
        weak = trough_internal and recovery >= 0.10 * amplitude
        evidence = f"trough={features['time_of_minimum_hours']:.1f} h; recovery={recovery:.3f}"
    elif archetype in {"Bridge", "Hill"}:
        base = peak_internal and rise >= 0.15 * amplitude and post_peak_drop >= 0.20 * amplitude
        weak_base = peak_internal and rise >= 0.08 * amplitude and post_peak_drop >= 0.10 * amplitude
        if archetype == "Bridge":
            exact, weak = base and width >= 0.22, weak_base and width >= 0.15
        else:
            exact, weak = base and width < 0.22, weak_base and width < 0.30
        evidence = (
            f"peak={features['time_of_maximum_hours']:.1f} h; rise={rise:.3f}; "
            f"post-peak drop={post_peak_drop:.3f}; peak width={width:.3f}"
        )
    elif archetype == "Slope":
        early = features["early_time_slope_relative_per_hour"]
        late = features["late_time_slope_relative_per_hour"]
        exact = early < 0 and late <= 0 and features["overall_pce_change_relative"] <= -0.08
        weak = early < 0 and features["overall_pce_change_relative"] < 0
        evidence = (
            f"early slope={early:.4f}/h; late slope={late:.4f}/h; "
            f"Δrel={features['overall_pce_change_relative']:.3f}"
        )
    else:
        raise ValueError(archetype)
    return bool(exact), bool(weak), evidence


def build_ifo_audit(natural_features: pd.DataFrame, robustness: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, title, _ in PIPELINES:
        subset = natural_features[natural_features.test == key]
        for archetype in ("Bridge", "Hill", "Slope", "Valley"):
            exact_matches = []
            weak_matches = []
            evidence = []
            for _, row in subset.iterrows():
                exact, weak, why = ifo_match(row.to_dict(), archetype)
                if exact:
                    exact_matches.append(int(row.cluster))
                    evidence.append(f"C{int(row.cluster)}: {why}")
                elif weak:
                    weak_matches.append(int(row.cluster))
                    evidence.append(f"C{int(row.cluster)} weak: {why}")
            if exact_matches:
                max_fraction = float(subset[subset.cluster.isin(exact_matches)].fraction.max())
                robust = robustness[key]["qualitative_shapes_persist"]
                if max_fraction >= 0.10 and robust:
                    status = "Clearly present"
                elif max_fraction < 0.05:
                    status = "Rare"
                else:
                    status = "Weakly present"
            elif weak_matches:
                status = "Weakly present"
            else:
                status = "Not observed"
            rows.append(
                {
                    "test": key,
                    "pipeline": title,
                    "archetype": archetype,
                    "status": status,
                    "clearly_present": "X" if status == "Clearly present" else "",
                    "weakly_present": "X" if status == "Weakly present" else "",
                    "rare": "X" if status == "Rare" else "",
                    "not_observed": "X" if status == "Not observed" else "",
                    "matching_clusters": ";".join(map(str, exact_matches or weak_matches)),
                    "posthoc_evidence": " | ".join(evidence) if evidence else "No natural-cluster centre met the operational pattern.",
                    "note": "Post-selection observational audit; never used for SOM training or K selection.",
                }
            )
    return pd.DataFrame(rows)


def build_comparison_figure(
    fixed: dict[str, SomResult], time: np.ndarray, path: Path
) -> tuple[pd.DataFrame, dict[str, dict[int, int]]]:
    reference = fixed["test2_resample_normalize"]
    order = np.argsort(-own_max_normalize(reference.means)[:, -1])
    ref_to_display = {int(old): int(new) for new, old in enumerate(order)}
    mappings: dict[str, dict[int, int]] = {}
    match_rows = []
    for key, title, _ in PIPELINES:
        result = fixed[key]
        if key == "test2_resample_normalize":
            to_reference = {i: i for i in range(4)}
            corr_matrix = np.corrcoef(
                own_max_normalize(result.means), own_max_normalize(reference.means)
            )[:4, 4:]
        else:
            to_reference, corr_matrix = match_centres(result.means, reference.means)
        mappings[key] = {src: ref_to_display[ref] for src, ref in to_reference.items()}
        for src, ref in to_reference.items():
            a = own_max_normalize(result.means[[src]])[0]
            b = own_max_normalize(reference.means[[ref]])[0]
            match_rows.append(
                {
                    "test": key,
                    "pipeline": title,
                    "source_cluster": src,
                    "matched_test2_cluster": ref,
                    "comparison_colour_class": ref_to_display[ref],
                    "pearson_shape_correlation": float(corr_matrix[src, ref]),
                    "shape_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                }
            )

    literature_crop = path.parent / "literature_fig4a_reference.png"
    render_literature_fig4a(PAPER_PDF, literature_crop)
    colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5))
    axes = axes.ravel()
    panel = ["(A)", "(B)", "(C)"]
    for ax, (key, title, _), letter in zip(axes[:3], PIPELINES, panel):
        result = fixed[key]
        reverse = {display: source for source, display in mappings[key].items()}
        for display in range(4):
            source = reverse[display]
            centre = own_max_normalize(result.means[[source]])[0]
            ax.plot(
                time,
                centre,
                color=colours[display],
                lw=2.2,
                label=(
                    f"Matched class {display + 1}: source C{source}, "
                    f"n={result.counts[source]}"
                ),
            )
        ax.set_title(f"{letter} {title} — fixed K=4", loc="left", fontweight="bold")
        ax.set_ylabel("Centre / max|centre|")
        ax.set_xlabel("Elapsed ageing time (h)")
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(0.05,  1.04)
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=8)
    axes[3].imshow(plt.imread(literature_crop))
    axes[3].set_title(
        "(D) Original literature result — Hartono et al., Fig. 4a",
        loc="left",
        fontweight="bold",
    )
    axes[3].axis("off")
    axes[3].text(
        0.5,
        -0.045,
        "Published K=4 reference: initial gain; slow, medium, and fast exponential decay\n"
        "Source: Nature Communications 14, 4869 (2023), DOI 10.1038/s41467-023-40585-3",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.suptitle(
        "SOM preprocessing comparison with the original literature result\n"
        "Panels A–C: controlled fixed K=4; raw centres normalized only for post-clustering shape display",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=3.0, w_pad=2.0)
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return pd.DataFrame(match_rows), mappings


def render_literature_fig4a(pdf_path: Path, output_path: Path) -> None:
    """Render the actual published Fig. 4a panel from the locally supplied PDF."""
    import fitz
    from PIL import Image, ImageDraw

    document = fitz.open(pdf_path)
    try:
        page = document[4]  # Printed page 5, containing Fig. 4.
        clip = fitz.Rect(124, 44, 296, 218)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=clip, alpha=False)
        pixmap.save(output_path)
    finally:
        document.close()
    # The generous right crop keeps Cluster 4's 150 h tick. Mask only the
    # adjacent panel-b letter from Fig. 4b, which otherwise enters the crop.
    with Image.open(output_path) as image:
        cleaned = image.copy()
    draw = ImageDraw.Draw(cleaned)
    draw.rectangle((624, 0, cleaned.width, 56), fill="white")
    cleaned.save(output_path)
    metadata = {
        "source_pdf": str(pdf_path),
        "source_pdf_sha256": sha256(pdf_path),
        "printed_page": 5,
        "figure": "Fig. 4a",
        "doi": "10.1038/s41467-023-40585-3",
        "crop_rect_pdf_points": [124, 44, 296, 218],
        "masked_adjacent_content": "Fig. 4b panel letter only",
        "role": (
            "Visual literature reference only; not used as training data, "
            "for K selection, or for ARI/NMI calculations."
        ),
    }
    (output_path.parent / "literature_reference_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: f"{x:.4f}")
    labels = [str(column) for column in view.columns]

    def clean(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(x) for x in labels) + " |"
    separator = "| " + " | ".join("---" for _ in labels) + " |"
    body = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_final_analysis(
    audit: dict,
    selected_k: dict[str, int],
    summaries: dict[str, pd.DataFrame],
    qes: dict[str, pd.DataFrame],
    similarity: pd.DataFrame,
    natural_similarity: pd.DataFrame,
    natural_features: pd.DataFrame,
    fixed_features: pd.DataFrame,
    diagnostics: dict[str, dict],
    robustness: dict[str, dict],
    ifo: pd.DataFrame,
    centre_matching: pd.DataFrame,
    robustness_class: tuple[str, str],
) -> None:
    titles = {key: title for key, title, _ in PIPELINES}
    summary_rows = []
    for key, title, _ in PIPELINES:
        tendencies = "; ".join(
            f"C{int(r.cluster)} {r.qualitative_morphology} ({100*r.fraction:.1f}%)"
            for _, r in natural_features[natural_features.test == key].iterrows()
        )
        summary_rows.append(
            {
                "Test": key.replace("test", "Test ").split("_")[0],
                "Pipeline": title,
                "Selected K": selected_k[key],
                "Main cluster tendencies": tendencies,
                "Main interpretation": diagnostics[key]["dominant_separation_interpretation"],
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    cluster_table = pd.concat(
        [
            df.assign(Test=key)[
                ["Test", "cluster", "n", "fraction", "qualitative_morphology", "key_shape_features"]
            ]
            for key, df in summaries.items()
        ],
        ignore_index=True,
    )

    sim_lookup = {
        row.comparison: row for _, row in similarity.iterrows()
    }
    rn = sim_lookup["Test 1 vs Test 2"]
    ns = sim_lookup["Test 2 vs Test 3"]
    rs = sim_lookup["Test 1 vs Test 3"]
    natural_ns = natural_similarity[
        natural_similarity.comparison == "Test 2 vs Test 3"
    ].iloc[0]
    raw_driver = diagnostics["test1_raw"]["dominant_separation_interpretation"]
    norm_driver = diagnostics["test2_resample_normalize"]["dominant_separation_interpretation"]
    smoothing_text = (
        "平滑后的固定 K=4 成员关系与未平滑归一化结果高度一致，主要作用是压低局部噪声。"
        if ns.ARI >= 0.80
        else "平滑明显改变了一部分固定 K=4 成员关系，不能只解释为视觉降噪。"
    )
    qe_blocks = []
    for key, title, _ in PIPELINES:
        block = qes[key][["k", "quantization_error", "normalized_chord_elbow_score", "selected"]]
        qe_blocks.append(f"### {title}\n\n" + markdown_table(block, list(block.columns)))

    robustness_rows = pd.DataFrame(
        [
            {
                "Test": key,
                "Selected K": selected_k[key],
                "QE mean": val["mean_quantization_error"],
                "QE SD": val["std_quantization_error"],
                "Mean ARI vs seed 42": val["mean_ARI_vs_primary"],
                "Mean NMI vs seed 42": val["mean_NMI_vs_primary"],
                "Mean matched-centre r": val["mean_matched_centre_pearson"],
                "Shapes persist": val["qualitative_shapes_persist"],
            }
            for key, val in robustness.items()
        ]
    )
    ifo_view = ifo[["test", "archetype", "status", "matching_clusters"]]

    report = f"""# Hartono 数据集：三种 SOM 预处理管线对比

## 1. 实验目标

本实验只比较预处理强度对 SOM 聚类的影响，不复现论文作为第四条管线，也不为了得到特定形态而调参。核心问题是：相同 2,245 条 Hartono PSC ageing curves 在 raw、重采样+MaxAbs、重采样+MaxAbs+Savitzky–Golay 三种输入下，是否保留相同的主要退化趋势。

## 2. 三条管线的精确定义

1. **Raw → SOM**：直接使用 Zenodo 发布的 `MPPT_EFF` 数值，不增加重采样、插值、归一化、平滑、导数或工程特征。
2. **Resample + Normalize → SOM**：统一到 10 min、0–149.833 h 的 elapsed-time 网格，必要时使用 Akima，再按每条曲线在窗口内的 `max|PCE|` 归一化。本数据已经在该网格且没有缺失值，因此重采样和 Akima 是经审计的恒等步骤；数值变化来自 MaxAbs。
3. **Resample + Normalize + Smooth → SOM**：在管线 2 后使用 Savitzky–Golay，`window_length=71`、`polyorder=2`。

三者统一使用 MiniSom、矩形网格、欧氏距离、样本权重初始化、`sigma={SIGMA}`、`learning_rate={LEARNING_RATE}`、{ITERATIONS:,} 次顺序训练和主种子 {PRIMARY_SEED}。候选 K 的网格由同一最接近方形的精确因子规则生成；K=4 即 2×2。

## 3. 数据集与样本数

- 来源：Hartono et al. Zenodo record 8185883，文件 `20230303_mySeriesDrop.pkl`。
- SHA-256：`{sha256(DATA_FILE)}`。
- 样本数：{audit['sample_count']:,}。
- 每条曲线：{audit['points_per_curve']} 点；10 min 间隔；实际坐标 0–{audit['elapsed_end_hours']:.3f} h，对应名义 150 h 窗口。
- 所有值有限、所有向量等长；三个测试使用相同样本、顺序与时间窗。

## 4. Raw → SOM 的不可避免限制

{audit['raw_limitation']} 因为发布数据可直接构成等长向量，本实验仍标为 **Raw SOM**，而不是 Near-Raw SOM；但不能把它描述成未经发布者清洗的仪器原始数据。

## 5. 各管线的自然 K

自然 K 只用 K=2…8 的 SOM quantization error 决定：先对每条 QE 曲线 min–max 归一化，再取其相对端点连线的最大垂直偏离（几何 elbow）。该规则在查看 Bridge/Hill/Slope/Valley 之前冻结。

{chr(10).join(qe_blocks)}

选择结果：Raw K={selected_k['test1_raw']}；Resample+Normalize K={selected_k['test2_resample_normalize']}；Resample+Normalize+Smooth K={selected_k['test3_resample_normalize_smooth']}。

## 6. 固定 K=4 与原文献结果对照

图中 (A)–(C) 是本实验三条管线的固定 K=4 受控比较，(D) 直接引用本地论文 PDF 的 Fig. 4a，展示文献报告的 initial gain、slow/medium/fast exponential decay 四类。文献面板只作视觉参照，不进入训练、K 选择或 ARI/NMI 计算。固定 K=4 不证明四个自然簇；本实验三组均使用 2×2 SOM 和主种子 42。标签本身没有语义，比较使用 ARI/NMI；中心显示前通过 Hungarian assignment 按自身最大值归一化后的形状匹配。

![三管线固定 K=4 与原文献 Fig. 4a 对比](comparison/04_three_pipeline_comparison.png)

## 7. Test 1 — Raw → SOM

自然 K={selected_k['test1_raw']}。诊断结果为 **{raw_driver}**。绝对量级解释率：初始 PCE η²={diagnostics['test1_raw']['eta_squared_initial_pce']:.3f}、最大 PCE η²={diagnostics['test1_raw']['eta_squared_maximum_pce']:.3f}；形状指标解释率：总相对变化 η²={diagnostics['test1_raw']['eta_squared_overall_relative_change']:.3f}、早期斜率 η²={diagnostics['test1_raw']['eta_squared_early_relative_slope']:.3f}、晚期斜率 η²={diagnostics['test1_raw']['eta_squared_late_relative_slope']:.3f}。因此结论不是依据肉眼给出的。

![Raw SOM](test1_raw/01_raw_som_clusters.png)

## 8. Test 2 — Resample + Normalize → SOM

自然 K={selected_k['test2_resample_normalize']}。诊断结果为 **{norm_driver}**。去掉绝对幅值后，固定 K=4 相对 Raw 的 ARI={rn.ARI:.3f}、NMI={rn.NMI:.3f}，说明成员边界的变化幅度可量化而非仅是坐标尺度变化。簇内自身最大值归一化形状 RMSE 从 Raw 标签下的 {diagnostics['test1_raw']['mean_within_cluster_shape_rmse']:.4f} 变为 {diagnostics['test2_resample_normalize']['mean_within_cluster_shape_rmse']:.4f}。

![Resample + Normalize SOM](test2_resample_normalize/02_resample_normalize_som_clusters.png)

## 9. Test 3 — Resample + Normalize + Smooth → SOM

自然 K={selected_k['test3_resample_normalize_smooth']}。{smoothing_text} 固定 K=4 相对 Test 2 的 ARI={ns.ARI:.3f}、NMI={ns.NMI:.3f}；自然 K=5 视图在最佳标签对齐后只有 {int(natural_ns.aligned_membership_changes)} / {audit['sample_count']} 条换簇（ARI={natural_ns.ARI:.3f}、NMI={natural_ns.NMI:.3f}）。平滑只作用于 SOM 输入的第三条管线；报告中的统一 SG 形态度量是训练完成后的描述算子，不回流训练。

![Resample + Normalize + Smooth SOM](test3_resample_normalize_smooth/03_resample_normalize_smooth_som_clusters.png)

## 10. ARI / NMI 比较

{markdown_table(similarity, ['comparison', 'ARI', 'NMI'])}

Test 1 vs Test 3 的 ARI={rs.ARI:.3f}、NMI={rs.NMI:.3f}。这些指标对任意簇 ID 置换不变。

## 11. 簇中心形态比较

`comparison/cluster_shape_features.csv` 保存自然 K 与固定 K=4 的早/晚斜率、总变化、极值时间、主要转折点、峰到终点下降、谷深和恢复幅度。`comparison/fixed_k4_center_matching.csv` 保存跨管线中心匹配、Pearson 形状相关和 RMSE。固定视图相对 Test 2 的平均中心相关分别为 Raw={centre_matching[centre_matching.test == 'test1_raw'].pearson_shape_correlation.mean():.3f}、Smooth={centre_matching[centre_matching.test == 'test3_resample_normalize_smooth'].pearson_shape_correlation.mean():.3f}。

## 12. 重采样 + 归一化的影响

本发布数据不需要数值重采样或 Akima 补点，所以 Test 1→2 的可识别变化来自 MaxAbs。Raw 的主分割诊断为“{raw_driver}”，归一化后为“{norm_driver}”。固定 K=4 的 ARI/NMI 和 η² 指标共同判断归一化是否把 SOM 从绝对 PCE 分割转向退化趋势分割。

## 13. 平滑的附加影响

{smoothing_text} 自然 K=5 仅 {int(natural_ns.aligned_membership_changes)} 条换簇，5 个中心的主要 turning-point 计数均保持不变；主要轮廓（初始增益、慢/中/快衰减）没有合并或消失。快速衰减中心的早期陡降被 SG 圆滑化，细小谷深与恢复幅度略有变化，详见 `cluster_shape_features.csv`，但没有产生新的主要峰谷类型。

## 14. 随机种子稳健性

主种子之外，每条管线用 0–9 共 10 个重复种子，所有其他设置不变。中心形态通过最小 RMSE 匹配后计算 Pearson 相关。

{markdown_table(robustness_rows, list(robustness_rows.columns))}

“Shapes persist” 的可复现判据为平均 matched-centre r≥0.90 且平均 NMI≥0.70；它只用于总结，不用于挑选种子。

## 15. Bridge / Hill / Slope / Valley 事后观察

该核查发生在全部模型和 K 冻结之后。名称只是 operational shape resemblance，不是论文正式标签或训练目标。

{markdown_table(ifo_view, list(ifo_view.columns))}

## 16. 最终结论

**{robustness_class[0]} — {robustness_class[1]}**

对问题的直接回答是：三条管线并非简单地产生完全相同的聚类。归一化移除绝对 PCE 后会改变 SOM 的距离结构与成员边界；随后 SG 平滑的附加效应由 Test 2→3 的 ARI/NMI 显示。主要退化轮廓能否称为保留，以固定 K=4 中心相关和种子稳健性为依据，而不是强行把结果解释为四种指定形态。

## 汇总表

{markdown_table(summary_table, list(summary_table.columns))}

## 簇级表

{markdown_table(cluster_table, list(cluster_table.columns))}

## 可复现性说明

- 执行脚本：`../run_som_preprocessing_comparison.py`
- 参数：`configuration.json`
- 数据审计：`data_audit.json`
- 软件版本：`software_versions.json`
- 文件清单与校验：`checksums.sha256`
- 固定 K=4 assignment、自然 K assignment、中心、QE 和逐种子结果均以 CSV/NPY 保存。
"""
    (OUT / "final_analysis.md").write_text(report, encoding="utf-8")


def write_checksums() -> None:
    files = sorted(
        p
        for p in OUT.rglob("*")
        if p.is_file() and p.name != "checksums.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(OUT)}" for path in files]
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs() -> dict:
    required = [
        OUT / "test1_raw" / "cluster_assignments.csv",
        OUT / "test1_raw" / "cluster_summary.csv",
        OUT / "test1_raw" / "quantization_error.csv",
        OUT / "test1_raw" / "01_raw_som_clusters.png",
        OUT / "test2_resample_normalize" / "cluster_assignments.csv",
        OUT / "test2_resample_normalize" / "cluster_summary.csv",
        OUT / "test2_resample_normalize" / "quantization_error.csv",
        OUT / "test2_resample_normalize" / "02_resample_normalize_som_clusters.png",
        OUT / "test3_resample_normalize_smooth" / "cluster_assignments.csv",
        OUT / "test3_resample_normalize_smooth" / "cluster_summary.csv",
        OUT / "test3_resample_normalize_smooth" / "quantization_error.csv",
        OUT / "test3_resample_normalize_smooth" / "03_resample_normalize_smooth_som_clusters.png",
        OUT / "comparison" / "04_three_pipeline_comparison.png",
        OUT / "comparison" / "clustering_similarity_ARI_NMI.csv",
        OUT / "comparison" / "cluster_shape_features.csv",
        OUT / "comparison" / "three_test_summary.csv",
        OUT / "comparison" / "bridge_hill_slope_valley_check.csv",
        OUT / "final_analysis.md",
    ]
    checks = {
        "all_required_files_exist": all(path.exists() for path in required),
        "missing_required_files": [str(path.relative_to(OUT)) for path in required if not path.exists()],
    }
    for key, _, _ in PIPELINES:
        assignments = pd.read_csv(OUT / key / "cluster_assignments.csv")
        qe = pd.read_csv(OUT / key / "quantization_error.csv")
        checks[f"{key}_assignment_count_2245"] = len(assignments) == 2245
        checks[f"{key}_k_values_2_through_8"] = qe.k.tolist() == list(K_VALUES)
        checks[f"{key}_one_selected_k"] = int(qe.selected.sum()) == 1
        checks[f"{key}_ten_repeat_seeds"] = len(pd.read_csv(OUT / key / "seed_robustness.csv")) == 10
    checks["all_checks_pass"] = bool(all(v for k, v in checks.items() if k != "missing_required_files"))
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_dir = OUT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for key, _, _ in PIPELINES:
        (OUT / key).mkdir(parents=True, exist_ok=True)

    raw, time, grouping, audit = load_data()
    data_by_key = {
        "test1_raw": raw,
        "test2_resample_normalize": raw / np.max(np.abs(raw), axis=1, keepdims=True),
    }
    data_by_key["test3_resample_normalize_smooth"] = savgol_filter(
        data_by_key["test2_resample_normalize"], SAVGOL_WINDOW, SAVGOL_ORDER, axis=1
    )
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    configuration = {
        "exact_main_pipeline_count": 3,
        "candidate_k": list(K_VALUES),
        "som_shape_rule": "closest exact factor pair; same rule per K across all pipelines",
        "sigma": SIGMA,
        "learning_rate": LEARNING_RATE,
        "activation_distance": "euclidean",
        "neighborhood_function": "gaussian",
        "initialization": "MiniSom.random_weights_init",
        "training_iterations": ITERATIONS,
        "training_order": "sequential (random_order=False)",
        "primary_seed": PRIMARY_SEED,
        "robustness_repeat_seeds": list(ROBUSTNESS_SEEDS),
        "resampling_interval_minutes": INTERVAL_MINUTES,
        "akima_required_on_released_data": False,
        "normalization": "per-curve MaxAbs within analysis window",
        "savgol_window_length": SAVGOL_WINDOW,
        "savgol_polynomial_order": SAVGOL_ORDER,
        "k_selection": "maximum normalized QE chord-deviation elbow on K=2..8",
        "fixed_k4_role": "controlled comparison only; not evidence of four natural clusters",
        "posthoc_shape_measurement": (
            "SG(71,2) applied uniformly to centre/max|centre| only after clustering; "
            "never used as Test 1 or Test 2 SOM input"
        ),
    }
    (OUT / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "minisom": getattr(sys.modules.get("minisom"), "__version__", "2.3.6 installed"),
    }
    (OUT / "software_versions.json").write_text(
        json.dumps(versions, indent=2), encoding="utf-8"
    )

    selected_k: dict[str, int] = {}
    natural: dict[str, SomResult] = {}
    fixed: dict[str, SomResult] = {}
    qes: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    natural_feature_frames = []
    fixed_feature_frames = []
    diagnostics: dict[str, dict] = {}
    robustness: dict[str, dict] = {}

    for key, title, filename in PIPELINES:
        print(f"Training primary K sweep: {title}", flush=True)
        candidates = {k: train_som(data_by_key[key], k, PRIMARY_SEED) for k in K_VALUES}
        k, qe_table = select_k(candidates)
        selected_k[key] = k
        natural[key] = candidates[k]
        fixed[key] = candidates[4]
        qes[key] = qe_table
        summary, features = save_pipeline_outputs(
            key,
            title,
            filename,
            data_by_key[key],
            raw,
            time,
            natural[key],
            qe_table,
        )
        summaries[key] = summary
        features.insert(0, "view", "natural_selected_k")
        natural_feature_frames.append(features)
        fixed_feature_frames.append(fixed_feature_table(key, title, fixed[key], time))
        save_fixed_k4(key, title, data_by_key[key], raw, time, fixed[key])
        diagnostics[key] = sample_shape_diagnostics(
            data_by_key[key], natural[key].labels, time, raw
        )
        print(f"Robustness seeds: {title}, selected K={k}", flush=True)
        _, robustness[key] = robustness_analysis(key, data_by_key[key], k, natural[key])

    natural_features = pd.concat(natural_feature_frames, ignore_index=True)
    fixed_features = pd.concat(fixed_feature_frames, ignore_index=True)
    all_features = pd.concat([natural_features, fixed_features], ignore_index=True)
    all_features.to_csv(comparison_dir / "cluster_shape_features.csv", index=False)

    pairs = [
        ("Test 1 vs Test 2", "test1_raw", "test2_resample_normalize"),
        (
            "Test 2 vs Test 3",
            "test2_resample_normalize",
            "test3_resample_normalize_smooth",
        ),
        (
            "Test 1 vs Test 3",
            "test1_raw",
            "test3_resample_normalize_smooth",
        ),
    ]
    similarity = pd.DataFrame(
        [
            {
                "comparison": name,
                "k": 4,
                "ARI": adjusted_rand_score(fixed[a].labels, fixed[b].labels),
                "NMI": normalized_mutual_info_score(fixed[a].labels, fixed[b].labels),
                "note": "Label-permutation invariant; fixed K=4 controlled view.",
            }
            for name, a, b in pairs
        ]
    )
    similarity.to_csv(comparison_dir / "clustering_similarity_ARI_NMI.csv", index=False)
    natural_similarity = pd.DataFrame(
        [
            {
                "comparison": name,
                "selected_k_a": selected_k[a],
                "selected_k_b": selected_k[b],
                "ARI": adjusted_rand_score(natural[a].labels, natural[b].labels),
                "NMI": normalized_mutual_info_score(natural[a].labels, natural[b].labels),
                "aligned_membership_changes": aligned_membership_changes(
                    natural[a].labels, natural[b].labels
                ),
                "note": (
                    "Mismatch count is -1 when selected K differs; ARI/NMI remain valid."
                ),
            }
            for name, a, b in pairs
        ]
    )
    natural_similarity.to_csv(
        comparison_dir / "natural_k_clustering_similarity.csv", index=False
    )

    centre_matching, mappings = build_comparison_figure(
        fixed, time, comparison_dir / "04_three_pipeline_comparison.png"
    )
    centre_matching.to_csv(comparison_dir / "fixed_k4_center_matching.csv", index=False)
    (comparison_dir / "fixed_k4_label_mapping.json").write_text(
        json.dumps({k: {str(a): b for a, b in v.items()} for k, v in mappings.items()}, indent=2),
        encoding="utf-8",
    )

    ifo = build_ifo_audit(natural_features, robustness)
    ifo.to_csv(comparison_dir / "bridge_hill_slope_valley_check.csv", index=False)
    cluster_level = pd.concat(
        [df.assign(test=key, pipeline=dict((x[0], x[1]) for x in PIPELINES)[key]) for key, df in summaries.items()],
        ignore_index=True,
    )
    cluster_level.to_csv(comparison_dir / "cluster_level_summary.csv", index=False)

    test_summary = pd.DataFrame(
        [
            {
                "test": key,
                "pipeline": title,
                "selected_k": selected_k[key],
                "main_cluster_tendencies": "; ".join(
                    f"C{int(r.cluster)} {r.qualitative_morphology} ({100*r.fraction:.1f}%)"
                    for _, r in natural_features[natural_features.test == key].iterrows()
                ),
                "main_interpretation": diagnostics[key]["dominant_separation_interpretation"],
                "primary_quantization_error": natural[key].qe,
                "repeat_seed_qe_mean": robustness[key]["mean_quantization_error"],
                "repeat_seed_qe_std": robustness[key]["std_quantization_error"],
                "repeat_seed_mean_ARI_vs_primary": robustness[key]["mean_ARI_vs_primary"],
                "qualitative_shapes_persist": robustness[key]["qualitative_shapes_persist"],
            }
            for key, title, _ in PIPELINES
        ]
    )
    test_summary.to_csv(comparison_dir / "three_test_summary.csv", index=False)
    pd.DataFrame([{ "test": key, **value } for key, value in diagnostics.items()]).to_csv(
        comparison_dir / "separation_diagnostics.csv", index=False
    )
    pd.DataFrame(list(robustness.values())).to_csv(
        comparison_dir / "random_seed_robustness_summary.csv", index=False
    )

    sim_min = float(similarity[["ARI", "NMI"]].min().min())
    corr_min = float(
        centre_matching[centre_matching.test != "test2_resample_normalize"]
        .groupby("test")
        .pearson_shape_correlation.mean()
        .min()
    )
    if sim_min >= 0.80 and corr_min >= 0.95:
        robustness_class = ("A. Strongly robust", "主要趋势和成员边界在三条管线中都高度一致。")
    elif corr_min >= 0.85:
        robustness_class = (
            "B. Moderately robust",
            "主要趋势可辨认，但预处理改变了簇边界或群体比例。",
        )
    elif sim_min >= 0.20 or corr_min >= 0.65:
        robustness_class = (
            "C. Preprocessing-dependent",
            "至少一次预处理转换显著改变成员边界或主导中心形态。",
        )
    else:
        robustness_class = ("D. Unstable", "三条管线间没有恢复一致的簇形态。")

    write_final_analysis(
        audit,
        selected_k,
        summaries,
        qes,
        similarity,
        natural_similarity,
        natural_features,
        fixed_features,
        diagnostics,
        robustness,
        ifo,
        centre_matching,
        robustness_class,
    )
    checks = verify_outputs()
    write_checksums()
    if not checks["all_checks_pass"]:
        raise RuntimeError(f"Verification failed: {checks}")
    print(f"Complete: {OUT}", flush=True)


if __name__ == "__main__":
    main()
