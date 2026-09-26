#!/usr/bin/env python3
"""Generate four synthetic degradation topologies and recover them with the paper SOM.

The four class labels are used only to generate the benchmark and to evaluate/map
the four SOM nodes after training. They are never passed to MiniSom.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)


CLASS_NAMES = ["IFO-Bridge", "IFO-Hill", "IFO-Slope", "IFO-Valley"]
CLASS_COLORS = {
    "IFO-Bridge": "#5A8F5B",
    "IFO-Hill": "#9BAE55",
    "IFO-Slope": "#F1B84B",
    "IFO-Valley": "#E6C84F",
}


@dataclass(frozen=True)
class Config:
    n_curves: int = 1000
    curves_per_class: int = 250
    duration_hours: int = 150
    raw_nominal_interval_minutes: int = 30
    resample_interval_minutes: int = 10
    savgol_window: int = 71
    savgol_order: int = 2
    som_x: int = 2
    som_y: int = 2
    som_sigma: float = 0.5
    som_learning_rate: float = 0.1
    som_iterations: int = 50000
    synthetic_seed: int = 20260830
    main_som_random_seed: None = None
    stability_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    minisom_version: str = "2.2.9"


CFG = Config()


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "reference": ROOT / "00_parameter_provenance",
        "synthetic": ROOT / "01_synthetic_data",
        "pre": ROOT / "02_paper_preprocessing",
        "som": ROOT / "03_paper_som_main",
        "stability": ROOT / "04_seed_stability",
        "evaluation": ROOT / "05_evaluation",
        "validation": ROOT / "06_validation",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_minisom_class():
    module_path = ROOT / "vendor" / "minisom_2_2_9" / "minisom.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing vendored MiniSom 2.2.9 at {module_path}. "
            "Copy it from the previously audited local project."
        )
    spec = importlib.util.spec_from_file_location("vendored_minisom_2_2_9", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MiniSom from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniSom


MiniSom = load_minisom_class()


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pchip_shape(t: np.ndarray, anchors_t: list[float], anchors_y: list[float]) -> np.ndarray:
    return PchipInterpolator(np.asarray(anchors_t), np.asarray(anchors_y))(t)


def generate_clean_shape(class_name: str, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Generate one topology with randomized but separated within-class parameters."""
    T = float(CFG.duration_hours)
    if class_name == "IFO-Bridge":
        start = rng.uniform(0.54, 0.70)
        rise_t = rng.uniform(18.0, 34.0)
        shoulder_t = rng.uniform(55.0, 82.0)
        high = rng.uniform(0.96, 1.00)
        shoulder = high - rng.uniform(0.005, 0.045)
        end = rng.uniform(0.76, 0.88)
        y = pchip_shape(t, [0.0, rise_t, shoulder_t, T], [start, high, shoulder, end])
        params = {
            "start_norm": start,
            "rise_end_h": rise_t,
            "shoulder_h": shoulder_t,
            "peak_norm": high,
            "shoulder_norm": shoulder,
            "end_norm": end,
        }
    elif class_name == "IFO-Hill":
        start = rng.uniform(0.45, 0.62)
        peak_t = rng.uniform(17.0, 32.0)
        rapid_end_t = rng.uniform(54.0, 75.0)
        peak = rng.uniform(0.97, 1.00)
        rapid_end = rng.uniform(0.39, 0.54)
        end = rng.uniform(0.20, 0.35)
        y = pchip_shape(t, [0.0, peak_t, rapid_end_t, T], [start, peak, rapid_end, end])
        params = {
            "start_norm": start,
            "peak_h": peak_t,
            "rapid_decay_end_h": rapid_end_t,
            "peak_norm": peak,
            "rapid_decay_end_norm": rapid_end,
            "end_norm": end,
        }
    elif class_name == "IFO-Slope":
        start = rng.uniform(0.96, 1.00)
        fast_fraction = rng.uniform(0.48, 0.66)
        tau_fast = rng.uniform(10.0, 20.0)
        tau_slow = rng.uniform(145.0, 230.0)
        floor = rng.uniform(0.08, 0.18)
        decay = fast_fraction * np.exp(-t / tau_fast) + (1.0 - fast_fraction) * np.exp(-t / tau_slow)
        y = floor + (start - floor) * decay
        params = {
            "start_norm": start,
            "fast_fraction": fast_fraction,
            "tau_fast_h": tau_fast,
            "tau_slow_h": tau_slow,
            "floor_norm": floor,
            "end_norm": float(y[-1]),
        }
    elif class_name == "IFO-Valley":
        start = rng.uniform(0.95, 1.00)
        min_t = rng.uniform(24.0, 42.0)
        recovery_t = rng.uniform(78.0, 105.0)
        valley = rng.uniform(0.27, 0.43)
        recovery = rng.uniform(0.78, 0.92)
        end = rng.uniform(0.58, 0.73)
        y = pchip_shape(t, [0.0, min_t, recovery_t, T], [start, valley, recovery, end])
        params = {
            "start_norm": start,
            "minimum_h": min_t,
            "recovery_peak_h": recovery_t,
            "minimum_norm": valley,
            "recovery_peak_norm": recovery,
            "end_norm": end,
        }
    else:
        raise ValueError(class_name)
    return np.clip(y, 0.02, None), params


def ar1_noise(n: int, rng: np.random.Generator, sigma: float, rho: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - rho * rho), n)
    out = np.empty(n, dtype=float)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


def generate_synthetic_dataset(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(CFG.synthetic_seed)
    class_sequence = np.repeat(CLASS_NAMES, CFG.curves_per_class)
    rng.shuffle(class_sequence)
    raw_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for idx, class_name in enumerate(class_sequence, start=1):
        curve_id = f"SYN-{idx:04d}"
        interval_h = CFG.raw_nominal_interval_minutes / 60.0
        base_t = np.arange(0.0, CFG.duration_hours + 1e-9, interval_h)
        jitter = rng.uniform(-0.10, 0.10, len(base_t)) * interval_h
        jitter[[0, -1]] = 0.0
        t = np.clip(base_t + jitter, 0.0, CFG.duration_hours)
        keep = rng.random(len(t)) > rng.uniform(0.015, 0.045)
        keep[[0, -1]] = True
        t = np.unique(np.round(t[keep], 6))

        clean_norm, shape_params = generate_clean_shape(class_name, t, rng)
        amplitude_pce = rng.uniform(8.0, 24.0)
        white_sigma = rng.uniform(0.003, 0.010)
        correlated_sigma = rng.uniform(0.004, 0.014)
        rho = rng.uniform(0.72, 0.92)
        noisy_norm = clean_norm + rng.normal(0.0, white_sigma, len(t))
        noisy_norm += ar1_noise(len(t), rng, correlated_sigma, rho)
        pce = np.clip(amplitude_pce * noisy_norm, 0.02, None)

        raw_parts.append(
            pd.DataFrame(
                {
                    "curve_id": curve_id,
                    "time_hours": t,
                    "pce_percent": pce,
                    "true_class": class_name,
                }
            )
        )
        meta_rows.append(
            {
                "curve_id": curve_id,
                "true_class": class_name,
                "amplitude_pce_percent": amplitude_pce,
                "white_noise_sigma_norm": white_sigma,
                "correlated_noise_sigma_norm": correlated_sigma,
                "noise_ar1_rho": rho,
                "raw_point_count": len(t),
                **shape_params,
            }
        )

    raw = pd.concat(raw_parts, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    raw.to_csv(out_dir / "synthetic_curves_long.csv.gz", index=False, compression="gzip", float_format="%.8f")
    meta.to_csv(out_dir / "curve_truth_and_generation_parameters.csv", index=False, float_format="%.8f")
    return raw, meta


def paper_preprocess(raw: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    grid_minutes = np.arange(0, CFG.duration_hours * 60 + 1, CFG.resample_interval_minutes)
    grid_hours = grid_minutes / 60.0
    grid_td = pd.to_timedelta(grid_minutes, unit="m")
    rows: list[np.ndarray] = []
    qc_rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []

    grouped = {cid: g for cid, g in raw.groupby("curve_id", sort=False)}
    for record in meta.itertuples(index=False):
        g = grouped[record.curve_id].sort_values("time_hours")
        td = pd.to_timedelta(g["time_hours"].to_numpy(), unit="h")
        s = pd.Series(g["pce_percent"].to_numpy(), index=td)
        s = s.groupby(level=0).mean().resample(f"{CFG.resample_interval_minutes}min").mean()
        s = s.reindex(grid_td)
        missing_before = int(s.isna().sum())
        s = s.interpolate(method="akima", limit_direction="both")
        if s.isna().any():
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        arr = s.to_numpy(dtype=float)
        max_abs = float(np.max(np.abs(arr)))
        normalized = arr / max_abs
        smoothed = savgol_filter(normalized, CFG.savgol_window, CFG.savgol_order)
        rows.append(smoothed)
        qc_rows.append(
            {
                "curve_id": record.curve_id,
                "true_class": record.true_class,
                "raw_point_count": len(g),
                "resampled_point_count": len(grid_hours),
                "missing_bins_before_akima": missing_before,
                "remaining_nan_after_akima": int(np.isnan(arr).sum()),
                "pre_normalization_max_abs": max_abs,
                "normalized_max_abs_before_savgol": float(np.max(np.abs(normalized))),
                "smoothed_min": float(smoothed.min()),
                "smoothed_max": float(smoothed.max()),
            }
        )
        long_parts.append(
            pd.DataFrame(
                {
                    "curve_id": record.curve_id,
                    "time_hours": grid_hours,
                    "pce_normalized_smoothed": smoothed,
                    "true_class": record.true_class,
                }
            )
        )

    X = np.asarray(rows, dtype=np.float64)
    qc = pd.DataFrame(qc_rows)
    pd.DataFrame({"time_hours": grid_hours}).to_csv(out_dir / "time_grid_hours.csv", index=False)
    qc.to_csv(out_dir / "preprocessing_qc.csv", index=False, float_format="%.10g")
    np.save(out_dir / "preprocessed_matrix_float64.npy", X)
    pd.concat(long_parts, ignore_index=True).to_csv(
        out_dir / "preprocessed_curves_long.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.8f",
    )
    return X, grid_hours, qc


def train_paper_som(X: np.ndarray, random_seed: int | None):
    kwargs = dict(
        x=CFG.som_x,
        y=CFG.som_y,
        input_len=X.shape[1],
        sigma=CFG.som_sigma,
        learning_rate=CFG.som_learning_rate,
    )
    if random_seed is not None:
        kwargs["random_seed"] = random_seed
    som = MiniSom(**kwargs)
    som.random_weights_init(X)
    som.train(X, CFG.som_iterations, verbose=False)
    coords = np.asarray([som.winner(row) for row in X], dtype=int)
    node_ids = coords[:, 0] * CFG.som_y + coords[:, 1]
    return som, coords, node_ids


def posthoc_map_nodes(node_ids: np.ndarray, truth: np.ndarray) -> tuple[dict[int, str], np.ndarray, np.ndarray]:
    matrix = np.zeros((CFG.som_x * CFG.som_y, len(CLASS_NAMES)), dtype=int)
    for node, label in zip(node_ids, truth):
        matrix[int(node), CLASS_NAMES.index(str(label))] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {int(r): CLASS_NAMES[int(c)] for r, c in zip(row_ind, col_ind)}
    predicted = np.asarray([mapping[int(node)] for node in node_ids], dtype=object)
    return mapping, predicted, matrix


def metric_bundle(truth: np.ndarray, predicted: np.ndarray, node_ids: np.ndarray, som, X: np.ndarray) -> dict:
    return {
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_NAMES, average="macro")),
        "adjusted_rand_index_truth_vs_nodes": float(adjusted_rand_score(truth, node_ids)),
        "normalized_mutual_information_truth_vs_nodes": float(normalized_mutual_info_score(truth, node_ids)),
        "quantization_error": float(som.quantization_error(X)),
        "topographic_error": float(som.topographic_error(X)),
        "occupied_nodes": int(len(np.unique(node_ids))),
    }


def save_main_results(
    X: np.ndarray,
    time_hours: np.ndarray,
    meta: pd.DataFrame,
    som,
    coords: np.ndarray,
    node_ids: np.ndarray,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    truth = meta["true_class"].to_numpy(dtype=object)
    mapping, predicted, contingency = posthoc_map_nodes(node_ids, truth)
    metrics = metric_bundle(truth, predicted, node_ids, som, X)
    metrics["main_random_seed"] = None
    metrics["main_seed_note"] = "The paper did not specify a random seed; MiniSom default random_seed=None was retained."

    assignments = meta[["curve_id", "true_class"]].copy()
    assignments["som_x"] = coords[:, 0]
    assignments["som_y"] = coords[:, 1]
    assignments["som_node"] = node_ids
    assignments["posthoc_predicted_class"] = predicted
    assignments["is_correct"] = assignments["true_class"] == assignments["posthoc_predicted_class"]
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)

    np.save(out_dir / "som_weights.npy", som.get_weights())
    node_curves = {"time_hours": time_hours}
    node_rows = []
    for node in range(CFG.som_x * CFG.som_y):
        x, y = divmod(node, CFG.som_y)
        mask = node_ids == node
        centroid = X[mask].mean(axis=0) if mask.any() else np.full(X.shape[1], np.nan)
        node_curves[f"node_{node}_mean"] = centroid
        node_curves[f"node_{node}_weight"] = som.get_weights()[x, y]
        counts = {name: int(np.sum(truth[mask] == name)) for name in CLASS_NAMES}
        node_rows.append(
            {
                "som_node": node,
                "som_x": x,
                "som_y": y,
                "posthoc_name": mapping[node],
                "count": int(mask.sum()),
                "purity": float(max(counts.values()) / mask.sum()) if mask.any() else np.nan,
                **counts,
            }
        )
    pd.DataFrame(node_curves).to_csv(out_dir / "node_mean_curves_and_weights.csv", index=False, float_format="%.9g")
    pd.DataFrame(node_rows).to_csv(out_dir / "node_summary.csv", index=False, float_format="%.9g")

    contingency_df = pd.DataFrame(contingency, columns=CLASS_NAMES)
    contingency_df.insert(0, "som_node", np.arange(len(contingency_df)))
    contingency_df.to_csv(out_dir / "node_truth_contingency.csv", index=False)
    with open(out_dir / "posthoc_node_name_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "main_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return assignments, metrics, contingency


def run_seed_stability(X: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    truth = meta["true_class"].to_numpy(dtype=object)
    seed_rows = []
    labels_by_seed: dict[int, np.ndarray] = {}
    for seed in CFG.stability_seeds:
        print(f"Stability SOM seed {seed} ...", flush=True)
        som, coords, node_ids = train_paper_som(X, seed)
        mapping, predicted, _ = posthoc_map_nodes(node_ids, truth)
        metrics = metric_bundle(truth, predicted, node_ids, som, X)
        metrics["seed"] = seed
        seed_rows.append(metrics)
        labels_by_seed[seed] = node_ids
        per_curve = meta[["curve_id", "true_class"]].copy()
        per_curve["som_node"] = node_ids
        per_curve["posthoc_predicted_class"] = predicted
        per_curve.to_csv(out_dir / f"assignments_seed_{seed}.csv", index=False)
        with open(out_dir / f"mapping_seed_{seed}.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)

    seed_df = pd.DataFrame(seed_rows).sort_values("seed")
    seed_df.to_csv(out_dir / "seed_metrics.csv", index=False, float_format="%.10g")
    pair = np.zeros((len(CFG.stability_seeds), len(CFG.stability_seeds)), dtype=float)
    for i, s1 in enumerate(CFG.stability_seeds):
        for j, s2 in enumerate(CFG.stability_seeds):
            pair[i, j] = adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2])
    pair_df = pd.DataFrame(pair, index=CFG.stability_seeds, columns=CFG.stability_seeds)
    pair_df.index.name = "seed"
    pair_df.to_csv(out_dir / "pairwise_seed_adjusted_rand_index.csv", float_format="%.10g")
    offdiag = pair[np.triu_indices_from(pair, k=1)]
    summary = {
        "seed_count": len(CFG.stability_seeds),
        "seeds": list(CFG.stability_seeds),
        "mean_accuracy": float(seed_df["accuracy"].mean()),
        "min_accuracy": float(seed_df["accuracy"].min()),
        "max_accuracy": float(seed_df["accuracy"].max()),
        "mean_macro_f1": float(seed_df["macro_f1"].mean()),
        "mean_pairwise_ari_between_seed_runs": float(offdiag.mean()),
        "min_pairwise_ari_between_seed_runs": float(offdiag.min()),
        "note": "Seeds are a supplemental robustness diagnostic; all other paper parameters remain unchanged.",
    }
    with open(out_dir / "stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return seed_df, pair_df, summary


def save_evaluation(assignments: pd.DataFrame, out_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    truth = assignments["true_class"].to_numpy()
    predicted = assignments["posthoc_predicted_class"].to_numpy()
    cm = confusion_matrix(truth, predicted, labels=CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name = "true_class"
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    report = classification_report(
        truth,
        predicted,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(out_dir / "classification_report.csv", float_format="%.10g")
    errors = assignments.loc[~assignments["is_correct"]].copy()
    errors.to_csv(out_dir / "misclassified_curves.csv", index=False)
    return cm, report_df


def plot_raw(raw: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, class_name in zip(axes.ravel(), CLASS_NAMES):
        ids = meta.loc[meta["true_class"] == class_name, "curve_id"].iloc[:35]
        for cid in ids:
            g = raw[raw["curve_id"] == cid]
            y = g["pce_percent"].to_numpy()
            ax.plot(g["time_hours"], y / np.max(y), color=CLASS_COLORS[class_name], alpha=0.18, lw=0.8)
        ax.set_title(class_name)
        ax.set_ylabel("Raw PCE / curve max")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle("Representative synthetic raw curves (35 per true class)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "representative_raw_curves.png", dpi=220)
    plt.close(fig)


def plot_preprocessed(X: np.ndarray, time_hours: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    truth = meta["true_class"].to_numpy()
    for ax, class_name in zip(axes.ravel(), CLASS_NAMES):
        members = X[truth == class_name]
        for curve in members[::10]:
            ax.plot(time_hours, curve, color=CLASS_COLORS[class_name], alpha=0.16, lw=0.7)
        q10, med, q90 = np.quantile(members, [0.1, 0.5, 0.9], axis=0)
        ax.fill_between(time_hours, q10, q90, color=CLASS_COLORS[class_name], alpha=0.25)
        ax.plot(time_hours, med, color=CLASS_COLORS[class_name], lw=2.5)
        ax.set_title(f"{class_name} (n={len(members)})")
        ax.set_ylabel("Normalized, SavGol-smoothed PCE")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle("Paper-preprocessed synthetic classes (truth shown only for audit)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "preprocessed_true_class_topologies.png", dpi=220)
    plt.close(fig)


def plot_som_nodes(X: np.ndarray, time_hours: np.ndarray, assignments: pd.DataFrame, som, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for node, ax in enumerate(axes.ravel()):
        mask = assignments["som_node"].to_numpy() == node
        name = assignments.loc[mask, "posthoc_predicted_class"].iloc[0] if mask.any() else "empty"
        color = CLASS_COLORS.get(name, "#777777")
        members = X[mask]
        for curve in members[:: max(1, len(members) // 45)]:
            ax.plot(time_hours, curve, color=color, alpha=0.12, lw=0.6)
        if len(members):
            mean = members.mean(axis=0)
            x, y = divmod(node, CFG.som_y)
            ax.plot(time_hours, mean, color=color, lw=2.8, label="member mean")
            ax.plot(time_hours, som.get_weights()[x, y], color="black", lw=1.5, ls="--", label="SOM weight")
        ax.set_title(f"Node {node} → {name} (n={len(members)})")
        ax.grid(alpha=0.2)
        ax.set_ylabel("Normalized PCE")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle("Unsupervised 2×2 SOM nodes; names assigned after training", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "som_four_nodes.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(som.distance_map(), cmap="viridis")
    for node in range(4):
        x, y = divmod(node, 2)
        ax.text(y, x, f"Node {node}", ha="center", va="center", color="white", weight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("SOM U-matrix")
    fig.colorbar(im, ax=ax, label="Normalized neighbor distance")
    fig.tight_layout()
    fig.savefig(out_dir / "som_u_matrix.png", dpi=220)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks(range(4), [x.replace("IFO-", "") for x in CLASS_NAMES], rotation=30, ha="right")
    ax.set_yticks(range(4), [x.replace("IFO-", "") for x in CLASS_NAMES])
    ax.set_xlabel("Post-hoc SOM class")
    ax.set_ylabel("Synthetic truth")
    ax.set_title("Confusion matrix (truth used only after SOM training)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_stability(seed_df: pd.DataFrame, pair_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(seed_df["seed"], seed_df["accuracy"], marker="o", label="Accuracy")
    axes[0].plot(seed_df["seed"], seed_df["macro_f1"], marker="s", label="Macro F1")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel("Fixed diagnostic seed")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Recovery across seeds")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    im = axes[1].imshow(pair_df.to_numpy(), vmin=0, vmax=1, cmap="magma")
    axes[1].set_xticks(range(len(pair_df)), pair_df.columns)
    axes[1].set_yticks(range(len(pair_df)), pair_df.index)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Seed")
    axes[1].set_title("Pairwise ARI of raw SOM assignments")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "seed_stability.png", dpi=220)
    plt.close(fig)


def write_parameter_files(dirs: dict[str, Path]) -> None:
    with open(ROOT / "run_config.json", "w", encoding="utf-8") as f:
        payload = asdict(CFG)
        payload["stability_seeds"] = list(CFG.stability_seeds)
        json.dump(payload, f, indent=2, ensure_ascii=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "minisom_runtime_source": str(ROOT / "vendor" / "minisom_2_2_9" / "minisom.py"),
        "minisom_version": CFG.minisom_version,
    }
    with open(ROOT / "environment_actual.json", "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2, ensure_ascii=False)

    evidence = f"""# 原文参数溯源

本实验只改变输入数据：使用人工合成且带已知真值的 1,000 条曲线。训练与预处理参数复刻原文。

| 项目 | 本实验值 | 本地原文/作者代码证据 |
|---|---:|---|
| 分析窗口 | 150 h | `thesis/paper/work.md`：只分析前 150 h |
| 重采样 | 10 min | `thesis/paper/work.md` Methods；作者 notebook |
| 插值 | Akima | `thesis/paper/work.md` Methods；作者 notebook |
| 归一化 | 每条曲线 MaxAbs | `thesis/paper/work.md` Eq. 2 |
| 平滑 | Savitzky–Golay，window=71，order=2 | 正文给 71；作者 notebook 给 `savgol_filter(..., 71, 2)` |
| SOM 实现 | MiniSom {CFG.minisom_version} | `thesis/environment.yml` |
| SOM 网格 | 2×2 | 作者 notebook：`som_x=2`, `som_y=2` |
| sigma | {CFG.som_sigma} | 正文 Fig. 4 / Methods；作者 notebook |
| learning rate | {CFG.som_learning_rate} | 正文 Fig. 4 / Methods；作者 notebook |
| 初始化 | `random_weights_init(data)` | 作者 notebook |
| 训练 | 顺序 `train(data, 50000)` | 作者 notebook；MiniSom 默认 `random_order=False` |
| 随机种子 | 未指定 / `None` | 原文及作者 notebook 均未给 seed；主实验保留该默认值 |

本地证据文件：

- `/Users/shunhao/Desktop/ML/thesis/paper/work.md`
- `/Users/shunhao/Desktop/ML/thesis/paper/Supplementary.md`
- `/Users/shunhao/Desktop/ML/thesis/20230227_degradation_analysis_revision_10_cleaned.ipynb`
- `/Users/shunhao/Desktop/ML/thesis/environment.yml`

## 无监督边界

`true_class` 只用于生成数据和训练后的节点命名/评价。传入 MiniSom 的对象只有 901 维预处理曲线矩阵，训练阶段不读取标签。四个 SOM 节点训练完成后，才用 Hungarian 一一映射将节点命名为 Bridge/Hill/Slope/Valley。
"""
    (dirs["reference"] / "PARAMETER_PROVENANCE_CN.md").write_text(evidence, encoding="utf-8")

    ranges = {
        "purpose": "Separated but noisy synthetic benchmark for four user-defined curve topologies.",
        "balance": {name: CFG.curves_per_class for name in CLASS_NAMES},
        "duration_hours": CFG.duration_hours,
        "raw_sampling": "Nominal 30 min with jitter and 1.5–4.5% random point loss; endpoints retained.",
        "noise": "White plus AR(1) correlated noise; exact per-curve values are in the metadata CSV.",
        "topology_definition": {
            "IFO-Bridge": "rapid increase, broad high shoulder, slow decay",
            "IFO-Hill": "rapid increase, early peak, rapid decay, then slower decay",
            "IFO-Slope": "bi-exponential rapid-to-slow decay",
            "IFO-Valley": "rapid decay, recovery peak, then slow decay",
        },
    }
    with open(dirs["synthetic"] / "generator_design.json", "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2, ensure_ascii=False)


def write_report(main_metrics: dict, stability: dict, assignments: pd.DataFrame) -> None:
    counts = assignments["posthoc_predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    correct = int(assignments["is_correct"].sum())
    count_table = "| 后验类别 | 数量 |\n|---|---:|\n" + "\n".join(
        f"| {name} | {int(counts[name])} |" for name in CLASS_NAMES
    )
    report = f"""# 四类合成曲线 + 原文 SOM 复刻实验报告

## 结论

在这个**人工构造、四类均衡且形态刻意可分**的基准数据上，严格按论文预处理与 SOM 参数运行后，2×2 SOM 成功占用 {main_metrics['occupied_nodes']} 个节点；训练后的一一节点命名得到准确率 **{main_metrics['accuracy']:.2%}**、宏平均 F1 **{main_metrics['macro_f1']:.4f}**、ARI **{main_metrics['adjusted_rand_index_truth_vs_nodes']:.4f}**。共正确分类 {correct}/1,000 条。

固定种子补充复算的平均准确率为 **{stability['mean_accuracy']:.2%}**，最低为 **{stability['min_accuracy']:.2%}**；不同种子原始节点分组的平均两两 ARI 为 **{stability['mean_pairwise_ari_between_seed_runs']:.4f}**。

这证明的是：**当数据确实由这四种清晰拓扑组成时，论文的无监督 SOM 流程有能力把它们恢复出来。** 它不证明真实的两组数据天然存在这四个簇，也不能替代真实数据实验。

## 数据与训练边界

- 数据：1,000 条曲线，Bridge/Hill/Slope/Valley 各 250 条，0–150 h。
- 原始点：约 30 min 间隔，带时间抖动、随机缺点、白噪声和相关噪声。
- 训练输入：只包含预处理后的曲线数值，不包含 `true_class`。
- 真值用途：仅用于训练后的 Hungarian 节点命名和指标计算。
- 主实验随机种子：按论文保持未指定（MiniSom `None`）；固定种子只用于稳健性复算。

## 论文参数

- 10 min 重采样；Akima 插值。
- 逐曲线 MaxAbsScaler 等价归一化。
- Savitzky–Golay：window 71，二阶多项式。
- MiniSom 2.2.9，2×2，sigma=0.5，learning_rate=0.1。
- `random_weights_init`，顺序训练 50,000 次。
- 时间窗口严格使用论文的 150 h。

## 主实验节点规模

{count_table}

## 文件导航

- `01_synthetic_data/synthetic_curves_long.csv.gz`：全部原始合成观测。
- `01_synthetic_data/curve_truth_and_generation_parameters.csv`：真值及每条曲线生成参数。
- `02_paper_preprocessing/preprocessed_curves_long.csv.gz`：完整预处理长表。
- `02_paper_preprocessing/preprocessed_matrix_float64.npy`：实际 SOM 输入矩阵。
- `03_paper_som_main/cluster_assignments.csv`：主实验分类结果。
- `03_paper_som_main/main_metrics.json`：主指标。
- `04_seed_stability/`：五个固定种子的补充稳定性结果与逐曲线分配。
- `05_evaluation/`：混淆矩阵、分类报告及误分类清单。
- `06_validation/`：独立复核结果。

## 解释限制

本实验是“方法可恢复性 / positive-control”测试。因为生成器预先定义了四类，不能将其表述为在真实数据上发现了四种自然簇。严谨写法是：论文 SOM 在受控合成数据上能恢复四类，但在现有真实数据上尚未稳定恢复全部四类。
"""
    (ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    readme = """# ifo_synthetic_four_class_paper_som

这是四类 IFO 拓扑的受控合成数据与论文参数 SOM 可恢复性实验。先阅读 `REPORT_CN.md`，复现执行 `python3 run_pipeline.py`，结果核验执行 `python3 verify_outputs.py`。

注意：合成真值从不进入 SOM 训练，只在训练结束后用于节点命名和评价。
"""
    (ROOT / "README_CN.md").write_text(readme, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", message="The topographic error is not defined")
    dirs = ensure_dirs()
    write_parameter_files(dirs)
    print("Generating 1,000 synthetic curves ...", flush=True)
    raw, meta = generate_synthetic_dataset(dirs["synthetic"])
    plot_raw(raw, meta, dirs["synthetic"])

    print("Applying paper preprocessing ...", flush=True)
    X, time_hours, _ = paper_preprocess(raw, meta, dirs["pre"])
    plot_preprocessed(X, time_hours, meta, dirs["pre"])

    print("Training main paper-parameter SOM (random_seed=None) ...", flush=True)
    som, coords, node_ids = train_paper_som(X, CFG.main_som_random_seed)
    assignments, main_metrics, _ = save_main_results(X, time_hours, meta, som, coords, node_ids, dirs["som"])
    plot_som_nodes(X, time_hours, assignments, som, dirs["som"])

    print("Running fixed-seed stability diagnostics ...", flush=True)
    seed_df, pair_df, stability = run_seed_stability(X, meta, dirs["stability"])
    plot_stability(seed_df, pair_df, dirs["stability"])

    cm, _ = save_evaluation(assignments, dirs["evaluation"])
    plot_confusion(cm, dirs["evaluation"])
    write_report(main_metrics, stability, assignments)
    print(json.dumps({"main_metrics": main_metrics, "stability": stability}, indent=2), flush=True)


if __name__ == "__main__":
    main()
