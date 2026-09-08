#!/usr/bin/env python3
"""Paper-grounded, fully unsupervised PCE curve clustering without smoothing."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")


def load_config() -> dict:
    return json.loads((PROJECT / "config.json").read_text())


def ensure_dirs() -> None:
    for name in (
        "01_data_audit", "02_preprocessed", "03_parameter_search",
        "04_k_selection", "05_final_model/clusters", "06_posthoc_ifo",
        "07_sensitivity", "reports",
    ):
        (PROJECT / name).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize_text(value: object) -> str:
    return str(value or "").strip().lower().replace("μ", "u")


def time_factor(axis: dict, folder: str) -> tuple[float | None, str]:
    text = normalize_text(axis.get("unit")) + " " + normalize_text(axis.get("name"))
    if "min" in text:
        return 1 / 60, "axis_metadata:minute"
    if "hour" in text or re.search(r"\bhrs?\b|\bh\b", text):
        return 1.0, "axis_metadata:hour"
    if "day" in text:
        return 24.0, "axis_metadata:day"
    if "week" in text:
        return 24.0 * 7, "axis_metadata:week"
    if "month" in text:
        return 24.0 * 30, "axis_metadata:month_30d"
    if "year" in text:
        return 24.0 * 365, "axis_metadata:year_365d"
    fallback = {
        "x_time_min": 1 / 60,
        "x_time_h": 1.0,
        "x_time_day": 24.0,
    }.get(folder)
    return fallback, f"folder_fallback:{folder}" if fallback else "unknown"


def is_pce_curve(meta: dict, path: Path) -> bool:
    y_axis = meta.get("axis", {}).get("y", {})
    y_text = normalize_text(y_axis.get("name")) + " " + normalize_text(y_axis.get("unit"))
    record = normalize_text(meta.get("record_id")) + " " + normalize_text(path.parent.parent.name)
    # Delta/loss panels are derived metrics, not the PCE trajectory clustered in
    # the paper. Path names alone are insufficient (a delta-PCE panel still
    # contains the token "pce").
    if any(token in y_text for token in ("δ", "Δ".lower(), "delta", "change", "loss")):
        return False
    explicit = any(token in y_text for token in (
        "pce", "efficien", "power conversion", "power output", "pmax"
    ))
    generic_axis = y_text in ("", "normalized", "normalised", "performance")
    return explicit or (generic_axis and any(token in record for token in ("pce", "efficien")))


def series_legend(meta: dict, path: Path) -> str:
    match = re.search(r"Series[_ ](\d+)", path.stem, re.I)
    wanted = match.group(1) if match else None
    for item in meta.get("series_mappings", []):
        sid = str(item.get("series_id", ""))
        if wanted and re.search(rf"\b{re.escape(wanted)}\b", sid):
            return str(item.get("legend_name") or "")
    return ""


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("x")
    if df.empty:
        return np.array([]), np.array([])
    df = df.groupby("x", as_index=False, sort=True)["y"].mean()
    return df["x"].to_numpy(float), df["y"].to_numpy(float)


def make_curve_id(relpath: str) -> str:
    return "C" + hashlib.sha256(relpath.encode()).hexdigest()[:10]


def build_dataset(cfg: dict, horizon: int, prefix: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    root = Path(cfg["input_root"])
    grid = np.arange(0.0, horizon + cfg["grid_step_hours"], cfg["grid_step_hours"], dtype=float)
    records: list[dict] = []
    arrays: list[np.ndarray] = []
    for path in sorted(root.glob("**/accepted/*.csv")):
        rel = str(path.relative_to(root))
        folder = path.relative_to(root).parts[0]
        validation = path.parent.parent / "validation_result.json"
        try:
            meta = json.loads(validation.read_text(errors="replace"))
        except Exception:
            meta = {}
        axis = meta.get("axis", {})
        fac, fac_source = time_factor(axis.get("x", {}), folder)
        rec = {
            "curve_id": make_curve_id(rel), "included": False, "exclusion_reason": "",
            "source_file": str(path.resolve()), "relative_source": rel,
            "validation_file": str(validation.resolve()), "doi": meta.get("doi", ""),
            "record_id": meta.get("record_id", ""), "legend": series_legend(meta, path),
            "x_axis_name": axis.get("x", {}).get("name", ""),
            "x_axis_unit": axis.get("x", {}).get("unit", ""),
            "y_axis_name": axis.get("y", {}).get("name", ""),
            "y_axis_unit": axis.get("y", {}).get("unit", ""),
            "time_factor_to_hours": fac, "time_factor_source": fac_source,
            "n_original_points": 0, "duration_hours": np.nan,
            "raw_min": np.nan, "raw_max": np.nan, "normalizer": np.nan,
            "smoothing_applied": False,
        }
        if not is_pce_curve(meta, path):
            rec["exclusion_reason"] = "not_identified_as_PCE"
            records.append(rec)
            continue
        if fac is None:
            rec["exclusion_reason"] = "unknown_time_unit"
            records.append(rec)
            continue
        try:
            x, y = read_curve(path)
        except Exception as exc:
            rec["exclusion_reason"] = f"read_error:{type(exc).__name__}"
            records.append(rec)
            continue
        rec["n_original_points"] = len(x)
        if len(x) < cfg["minimum_original_points"]:
            rec["exclusion_reason"] = "too_few_original_points"
            records.append(rec)
            continue
        elapsed = (x - x[0]) * fac
        duration = float(elapsed[-1])
        rec.update(duration_hours=duration, raw_min=float(np.min(y)), raw_max=float(np.max(y)))
        if duration + 1e-9 < horizon:
            rec["exclusion_reason"] = f"shorter_than_{horizon}h"
            records.append(rec)
            continue
        # This is piecewise-linear regridding, not smoothing or spline fitting.
        values = np.interp(grid, elapsed, y)
        scale = float(np.max(np.abs(values)))
        rec["normalizer"] = scale
        if not np.isfinite(scale) or scale <= 0:
            rec["exclusion_reason"] = "zero_or_invalid_scale"
            records.append(rec)
            continue
        values = values / scale
        if float(np.ptp(values)) < 1e-12:
            rec["exclusion_reason"] = "constant_curve"
            records.append(rec)
            continue
        rec["included"] = True
        arrays.append(values)
        records.append(rec)
    inventory = pd.DataFrame(records)
    data = np.vstack(arrays) if arrays else np.empty((0, len(grid)))
    included = inventory[inventory["included"]].reset_index(drop=True)
    out = PROJECT / "02_preprocessed"
    inventory.to_csv(out / f"{prefix}_inventory.csv", index=False)
    included.to_csv(out / f"{prefix}_included_metadata.csv", index=False)
    np.save(out / f"{prefix}_curves.npy", data)
    np.save(out / f"{prefix}_time_hours.npy", grid)
    reasons = inventory["exclusion_reason"].replace("", "included").fillna("included").value_counts().to_dict()
    dump_json(PROJECT / "01_data_audit" / f"{prefix}_preprocessing_summary.json", {
        "horizon_hours": horizon, "grid_step_hours": cfg["grid_step_hours"],
        "grid_points": len(grid), "included": len(included),
        "counts_by_outcome": {str(k): int(v) for k, v in reasons.items()},
        "original_point_count_quantiles_included": {
            str(q): float(included["n_original_points"].quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)
        },
        "smoothing_enabled": False,
    })
    return data, included, grid


def audit(cfg: dict) -> dict:
    root = Path(cfg["input_root"])
    files = sorted(root.glob("**/accepted/*.csv"))
    by_folder = Counter(p.relative_to(root).parts[0] for p in files)
    summary = {
        "input_root": str(root), "accepted_csv_count": len(files),
        "accepted_csv_by_time_folder": dict(sorted(by_folder.items())),
        "labels_used": False, "smoothing_enabled": False,
        "source_tree_modified": False,
    }
    dump_json(PROJECT / "01_data_audit/data_audit.json", summary)
    return summary


def som_layout(k: int) -> tuple[int, int]:
    factors = [(a, k // a) for a in range(1, int(math.sqrt(k)) + 1) if k % a == 0]
    return min(factors, key=lambda ab: abs(ab[0] - ab[1]))


def cluster_metrics(data: np.ndarray, som: MiniSom) -> tuple[np.ndarray, dict]:
    xdim, ydim = som.get_weights().shape[:2]
    winners = np.array([som.winner(row) for row in data], int)
    labels = winners[:, 0] * ydim + winners[:, 1]
    weights = som.get_weights().reshape(xdim * ydim, -1)
    occupied = np.unique(labels)
    counts = np.bincount(labels, minlength=xdim * ydim)
    assigned = weights[labels]
    distances = np.linalg.norm(data - assigned, axis=1)
    center_dist = []
    for a, b in itertools.combinations(occupied, 2):
        center_dist.append(float(np.sqrt(np.mean((weights[a] - weights[b]) ** 2))))
    within_rmse = distances / math.sqrt(data.shape[1])
    sil = float(silhouette_score(data, labels, metric="euclidean", sample_size=min(1000, len(data)), random_state=0)) if len(occupied) > 1 else np.nan
    metric = {
        "quantization_error": float(som.quantization_error(data)),
        "qe_per_sqrt_dimension": float(som.quantization_error(data) / math.sqrt(data.shape[1])),
        "topographic_error": float(som.topographic_error(data)),
        "silhouette": sil, "occupied_nodes": int(len(occupied)),
        "empty_nodes": int(xdim * ydim - len(occupied)),
        "min_cluster_size": int(counts[occupied].min()),
        "min_cluster_fraction": float(counts[occupied].min() / len(data)),
        "median_within_rmse": float(np.median(within_rmse)),
        "minimum_center_rmse": float(min(center_dist)) if center_dist else np.nan,
        "center_overlap_ratio": float(min(center_dist) / (np.median(within_rmse) + 1e-12)) if center_dist else np.nan,
        "cluster_sizes": ";".join(str(int(v)) for v in counts if v),
    }
    return labels, metric


def train_one(data: np.ndarray, k: int, setting: dict, seed: int, iterations: int) -> tuple[MiniSom, np.ndarray, dict]:
    xdim, ydim = som_layout(k)
    som = MiniSom(
        xdim, ydim, data.shape[1], sigma=setting["sigma"],
        learning_rate=setting["learning_rate"], random_seed=seed,
        neighborhood_function="gaussian", activation_distance="euclidean",
    )
    som.random_weights_init(data)
    som.train(data, iterations, random_order=setting["random_order"], verbose=False)
    labels, metric = cluster_metrics(data, som)
    metric.update({
        "setting": setting["name"], "source": setting["source"], "k": k,
        "layout": f"{xdim}x{ydim}", "sigma": setting["sigma"],
        "learning_rate": setting["learning_rate"],
        "random_order": setting["random_order"], "seed": seed,
        "iterations": iterations,
    })
    return som, labels, metric


def pairwise_ari(label_sets: list[np.ndarray]) -> float:
    values = [adjusted_rand_score(a, b) for a, b in itertools.combinations(label_sets, 2)]
    return float(np.mean(values)) if values else 1.0


def elbow_k(summary: pd.DataFrame) -> int:
    s = summary.sort_values("k")
    x = s["k"].to_numpy(float)
    y = s["quantization_error_median"].to_numpy(float)
    if y[0] <= y[-1] or abs(y[0] - y[-1]) < 1e-15:
        return int(x[0])
    improvement = (y[0] - y) / (y[0] - y[-1])
    progress = (x - x[0]) / (x[-1] - x[0])
    return int(x[np.argmax(improvement - progress)])


def aggregate_search(runs: pd.DataFrame, labels_by_key: dict) -> pd.DataFrame:
    rows = []
    for (setting, k), group in runs.groupby(["setting", "k"], sort=False):
        keys = [(setting, int(k), int(seed)) for seed in group["seed"]]
        row = {"setting": setting, "k": int(k), "seed_ari_mean": pairwise_ari([labels_by_key[key] for key in keys])}
        for column in (
            "quantization_error", "qe_per_sqrt_dimension", "topographic_error",
            "silhouette", "occupied_nodes", "empty_nodes", "min_cluster_size",
            "min_cluster_fraction", "minimum_center_rmse", "median_within_rmse",
            "center_overlap_ratio",
        ):
            row[column + "_median"] = float(group[column].median())
        first = group.iloc[0]
        for column in ("source", "sigma", "learning_rate", "random_order", "layout"):
            row[column] = first[column]
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary


def plot_qe(summary: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, group in summary.groupby("setting"):
        g = group.sort_values("k")
        ax.plot(g["k"], g["quantization_error_median"], marker="o", label=name)
    ax.set(xlabel="Number of SOM nodes (K)", ylabel="Median quantization error", title=title)
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_search(cfg: dict, data: np.ndarray, prefix: str, settings: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or cfg["som_settings"]
    rows, labels_by_key = [], {}
    total = len(settings) * len(cfg["candidate_k"]) * len(cfg["screen_seeds"])
    done = 0
    for setting in settings:
        for k in cfg["candidate_k"]:
            for seed in cfg["screen_seeds"]:
                _, labels, metric = train_one(data, k, setting, seed, cfg["screen_iterations"])
                rows.append(metric); labels_by_key[(setting["name"], k, seed)] = labels
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"search {prefix}: {done}/{total}", flush=True)
    runs = pd.DataFrame(rows)
    summary = aggregate_search(runs, labels_by_key)
    out = PROJECT / ("03_parameter_search" if prefix == "primary_500h" else "07_sensitivity")
    runs.to_csv(out / f"{prefix}_all_runs.csv", index=False)
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    plot_qe(summary, out / f"{prefix}_qe_elbows.png", f"No-smoothing SOM QE — {prefix}")
    return runs, summary


def setting_decisions(summary: pd.DataFrame) -> pd.DataFrame:
    decisions = []
    for name, group in summary.groupby("setting", sort=False):
        group = group.sort_values("k").copy()
        e = elbow_k(group)
        group["qe_drop_to_next_fraction"] = (group["quantization_error_median"] - group["quantization_error_median"].shift(-1)) / group["quantization_error_median"]
        r = group[group["k"] == e].iloc[0]
        decisions.append({
            "setting": name, "source": r["source"], "sigma": r["sigma"],
            "learning_rate": r["learning_rate"], "random_order": r["random_order"],
            "elbow_k": e, "qe_at_elbow": r["quantization_error_median"],
            "silhouette_at_elbow": r["silhouette_median"],
            "seed_ari_at_elbow": r["seed_ari_mean"],
            "overlap_ratio_at_elbow": r["center_overlap_ratio_median"],
            "min_cluster_fraction_at_elbow": r["min_cluster_fraction_median"],
            "occupied_nodes_at_elbow": r["occupied_nodes_median"],
            "empty_nodes_at_elbow": r["empty_nodes_median"],
        })
    d = pd.DataFrame(decisions)
    # Unsupervised parameter score. Rank aggregation avoids mixing metric units.
    d["rank_qe"] = d["qe_at_elbow"].rank(ascending=True, method="average")
    d["rank_silhouette"] = d["silhouette_at_elbow"].rank(ascending=False, method="average")
    d["rank_stability"] = d["seed_ari_at_elbow"].rank(ascending=False, method="average")
    d["rank_separation"] = d["overlap_ratio_at_elbow"].rank(ascending=False, method="average")
    d["rank_sum"] = d[["rank_qe", "rank_silhouette", "rank_stability", "rank_separation"]].sum(axis=1)
    # The paper asks for main/distinct shapes rather than singleton outlier
    # nodes. Operationalize that qualitative check before ranking parameters.
    d["passes_main_shape_support"] = (d["min_cluster_fraction_at_elbow"] >= 0.01) & (d["empty_nodes_at_elbow"] == 0)
    return d.sort_values(["passes_main_shape_support", "rank_sum", "elbow_k", "setting"], ascending=[False, True, True, True]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    clean = frame.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
    header = "| " + " | ".join(map(str, clean.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in clean.columns) + " |"
    body = ["| " + " | ".join(row.astype(str)) + " |" for _, row in clean.iterrows()]
    return "\n".join([header, rule] + body)


def kmeans_validation(data: np.ndarray, cfg: dict) -> pd.DataFrame:
    rows = []
    label_sets: dict[int, list[np.ndarray]] = {}
    for k in cfg["candidate_k"]:
        label_sets[k] = []
        for seed in cfg["screen_seeds"]:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(data)
            label_sets[k].append(km.labels_)
            rows.append({
                "k": k, "seed": seed, "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(data, km.labels_, sample_size=min(1000, len(data)), random_state=0)),
            })
    runs = pd.DataFrame(rows)
    agg = runs.groupby("k", as_index=False).agg(inertia_median=("inertia", "median"), silhouette_median=("silhouette", "median"))
    agg["seed_ari_mean"] = [pairwise_ari(label_sets[int(k)]) for k in agg["k"]]
    # Reuse the geometric elbow definition with the inertia curve.
    temp = agg.rename(columns={"inertia_median": "quantization_error_median"})
    agg["elbow_selected"] = agg["k"] == elbow_k(temp)
    return agg


def choose_final(summary: pd.DataFrame) -> tuple[dict, int, pd.DataFrame]:
    decisions = setting_decisions(summary)
    chosen = decisions.iloc[0].to_dict()
    k = int(chosen["elbow_k"])
    return chosen, k, decisions


def final_fit(data: np.ndarray, metadata: pd.DataFrame, grid: np.ndarray, cfg: dict, chosen: dict, k: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict]]:
    setting = next(s for s in cfg["som_settings"] if s["name"] == chosen["setting"])
    models, labels_list, metrics = [], [], []
    for seed in cfg["final_seeds"]:
        som, labels, metric = train_one(data, k, setting, seed, cfg["final_iterations"])
        models.append(som); labels_list.append(labels); metrics.append(metric)
        print(f"final fit seed={seed} QE={metric['quantization_error']:.6f}", flush=True)
    ari_matrix = np.eye(len(models))
    for i, j in itertools.combinations(range(len(models)), 2):
        ari_matrix[i, j] = ari_matrix[j, i] = adjusted_rand_score(labels_list[i], labels_list[j])
    medoid_index = int(np.argmax((ari_matrix.sum(axis=1) - 1) / max(1, len(models) - 1)))
    som = models[medoid_index]; labels = labels_list[medoid_index]
    weights = som.get_weights().reshape(k, -1)
    all_dist = np.linalg.norm(data[:, None, :] - weights[None, :, :], axis=2)
    ordered = np.sort(all_dist, axis=1)
    margin = (ordered[:, 1] - ordered[:, 0]) / (ordered[:, 1] + 1e-12) if k > 1 else np.ones(len(data))
    assignments = metadata.copy()
    assignments["cluster"] = labels
    assignments["bmu_distance"] = all_dist[np.arange(len(data)), labels]
    assignments["assignment_margin"] = margin
    assignments["uncertain_bottom_10pct_margin"] = margin <= np.quantile(margin, .1)
    assignments.to_csv(PROJECT / "05_final_model/assignments.csv", index=False)
    np.save(PROJECT / "05_final_model/weights.npy", weights)
    np.save(PROJECT / "05_final_model/labels.npy", labels)
    pd.DataFrame(ari_matrix, index=cfg["final_seeds"], columns=cfg["final_seeds"]).to_csv(PROJECT / "05_final_model/final_seed_ari_matrix.csv")
    pd.DataFrame(metrics).to_csv(PROJECT / "05_final_model/final_seed_metrics.csv", index=False)
    dump_json(PROJECT / "05_final_model/final_model.json", {
        "setting": setting, "k": k, "layout": som_layout(k),
        "iterations": cfg["final_iterations"], "selected_seed": cfg["final_seeds"][medoid_index],
        "mean_pairwise_seed_ari": pairwise_ari(labels_list), "smoothing_enabled": False,
        "labels_used": False, "selection": chosen,
    })
    plot_clusters(data, labels, weights, grid, assignments, k)
    return labels, weights, assignments, metrics


def plot_clusters(data: np.ndarray, labels: np.ndarray, weights: np.ndarray, grid: np.ndarray, assignments: pd.DataFrame, k: int) -> None:
    cols = min(3, k); rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True, sharey=True, squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        subset = data[labels == c]
        for curve in subset:
            ax.plot(grid, curve, color="#3478a8", alpha=min(.2, max(.015, 8 / max(1, len(subset)))), lw=.45)
        ax.plot(grid, np.median(subset, axis=0), color="black", lw=2, label="median")
        ax.plot(grid, weights[c], color="#d62728", lw=1.5, label="SOM weight")
        ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Cluster {c} (n={len(subset)})"); ax.grid(alpha=.15)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("MaxAbs-normalized PCE")
    fig.suptitle("Final no-smoothing SOM: every interpolated member + median + SOM weight")
    fig.tight_layout(); fig.savefig(PROJECT / "05_final_model/clusters/all_clusters.png", dpi=220); plt.close(fig)
    for c in range(k):
        subset = data[labels == c]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for curve in subset:
            ax.plot(grid, curve, color="#3478a8", alpha=min(.28, max(.02, 12 / max(1, len(subset)))), lw=.6)
        ax.plot(grid, np.median(subset, axis=0), color="black", lw=2.2, label="median")
        ax.plot(grid, weights[c], color="#d62728", lw=1.7, label="SOM weight")
        ax.axvline(200, color="grey", ls="--", label="200 h")
        ax.set(xlabel="Elapsed time (h)", ylabel="MaxAbs-normalized PCE", title=f"Cluster {c}: n={len(subset)}; smoothing OFF")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(PROJECT / f"05_final_model/clusters/cluster_{c:02d}.png", dpi=220); plt.close(fig)
        assignments[assignments["cluster"] == c].to_csv(PROJECT / f"05_final_model/clusters/cluster_{c:02d}_members.csv", index=False)


def linear_slope(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def posthoc_descriptors(weights: np.ndarray, data: np.ndarray, labels: np.ndarray, grid: np.ndarray, output_path: Path | None = None) -> pd.DataFrame:
    rows = []
    for c, w in enumerate(weights):
        peak_i, trough_i = int(np.argmax(w)), int(np.argmin(w))
        early_end = min(100, grid[-1] * .2)
        late_start = min(250, grid[-1] * .5)
        recovery = float(np.max(w[trough_i:]) - w[trough_i]) if trough_i < len(w) - 1 else 0.0
        peak_drop = float(w[peak_i] - w[-1])
        rows.append({
            "cluster": c, "n": int((labels == c).sum()),
            "start": float(w[0]), "at_50h": float(w[np.argmin(abs(grid - 50))]),
            "at_200h": float(w[np.argmin(abs(grid - min(200, grid[-1])))]),
            "end": float(w[-1]), "peak": float(w[peak_i]), "peak_time_h": float(grid[peak_i]),
            "trough": float(w[trough_i]), "trough_time_h": float(grid[trough_i]),
            "early_slope_per_h": linear_slope(grid, w, 0, early_end),
            "late_slope_per_h": linear_slope(grid, w, late_start, grid[-1]),
            "post_trough_recovery": recovery, "post_peak_drop": peak_drop,
            "range": float(np.ptp(w)),
        })
    result = pd.DataFrame(rows)
    # Names are descriptive and assigned only after fitting; they do not affect learning.
    names = []
    for r in result.to_dict("records"):
        if r["trough_time_h"] <= 200 and r["post_trough_recovery"] >= .12 * max(r["range"], 1e-9) and r["trough_time_h"] > 0:
            name = "Valley-like"
        elif r["peak_time_h"] > 0 and r["peak_time_h"] <= 200 and r["early_slope_per_h"] > 0 and r["post_peak_drop"] > .08 * max(r["range"], 1e-9):
            # Broad peaks are Bridge-like; narrow/sharper early peaks are Hill-like.
            w = weights[int(r["cluster"])]
            half = r["end"] + .5 * (r["peak"] - r["end"])
            width = float((w >= half).sum())
            name = "Bridge-like" if width >= .25 * len(w) else "Hill-like"
        elif r["early_slope_per_h"] < 0 and r["late_slope_per_h"] <= 0:
            name = "Slope-like"
        else:
            name = "Other/mixed"
        names.append(name)
    result["posthoc_ifo_name"] = names
    result.to_csv(output_path or PROJECT / "06_posthoc_ifo/cluster_shape_descriptors.csv", index=False)
    return result


def raw_representatives(assignments: pd.DataFrame, data: np.ndarray, labels: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    k = len(weights)
    fig, axes = plt.subplots(math.ceil(k / 3), min(3, k), figsize=(15, 3.8 * math.ceil(k / 3)), squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        idx = np.where(labels == c)[0]
        dist = np.linalg.norm(data[idx] - weights[c], axis=1)
        selected = idx[np.argsort(dist)[:min(10, len(idx))]]
        for rank, i in enumerate(selected, 1):
            rec = assignments.iloc[i]
            x, y = read_curve(Path(rec["source_file"]))
            fac = float(rec["time_factor_to_hours"])
            elapsed = (x - x[0]) * fac
            mask = elapsed <= grid[-1] + 1e-9
            scale = np.max(np.abs(y[mask]))
            ax.plot(elapsed[mask], y[mask] / scale, marker=".", ms=2, lw=.8, alpha=.7)
            rows.append({"cluster": c, "rank": rank, "curve_id": rec["curve_id"], "source_file": rec["source_file"], "legend": rec["legend"], "distance": float(dist[np.argsort(dist)[rank-1]])})
        ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Cluster {c}: nearest raw-point curves")
        ax.grid(alpha=.2)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("Raw-point PCE / in-window MaxAbs")
    fig.suptitle("Raw digitized points (no smoothing; no dense model-grid lines)")
    fig.tight_layout(); fig.savefig(PROJECT / "06_posthoc_ifo/raw_point_representatives.png", dpi=220); plt.close(fig)
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / "06_posthoc_ifo/representative_provenance.csv", index=False)
    return result


def make_report(cfg: dict, audit_result: dict, primary_meta: pd.DataFrame, sensitivity_meta: pd.DataFrame, decisions: pd.DataFrame, chosen: dict, k: int, km: pd.DataFrame, descriptors: pd.DataFrame, final_metrics: list[dict]) -> None:
    paper_rows = decisions[decisions["source"] == "paper"]
    opt_rows = decisions[decisions["source"] == "optimization"]
    km_k = int(km.loc[km["elbow_selected"], "k"].iloc[0])
    shape_counts = descriptors["posthoc_ifo_name"].value_counts().to_dict()
    lines = [
        "# 最终报告：无平滑、全过程无监督的 PCE 曲线分类", "",
        "## 结论", "",
        f"主分析由论文规则选择 **K={k}**，没有预设为四类。最佳无监督参数配置为 "
        f"`{chosen['setting']}`：`sigma={chosen['sigma']}`、`learning_rate={chosen['learning_rate']}`、"
        f"`random_order={chosen['random_order']}`。最终模型使用 50,000 次更新和 10 个随机种子，选择与其他种子分区平均一致性最高的种子作为可复现输出。", "",
        f"K-means 的独立 inertia elbow 为 K={km_k}。它只用于一致性核查，不覆盖 SOM/论文判据。", "",
        f"冻结形态名称完全在模型冻结后生成；节点名称计数为 `{shape_counts}`。它们只是 IFO-like 对照，不是训练标签。", "",
        "## 数据范围", "",
        f"- 扫描 accepted CSV：{audit_result['accepted_csv_count']} 条。",
        f"- 500 h 主模型纳入：{len(primary_meta)} 条；150 h 论文敏感性模型纳入：{len(sensitivity_meta)} 条。",
        "- 只使用 validation metadata/记录名可识别为 PCE/efficiency 的曲线。",
        "- 各曲线从首个观测记为 elapsed time=0；依据轴元数据把 minute/hour/day/week/month/year 转为小时。",
        "- 使用逐曲线 MaxAbs 归一化，与论文一致。",
        "- 固定长度输入通过观测点之间的分段线性插值获得；没有滤波、样条、Savitzky–Golay 或移动平均。",
        "", "## 类别数判定", "",
        "论文规则是：QE elbow 给出候选范围，然后选择能够表达主要形态的最小 K；若更高 K 的中心重叠或不可区分则拒绝。这里用多种子 median QE 计算几何 elbow，并同时输出中心间 RMSE/簇内 RMSE 比、空节点、最小簇比例和 seed ARI。", "",
        "为落实论文的‘主要形态’表述，自动排名前先要求每个节点至少覆盖 1% 样本且无空节点；不满足者仍完整保留在结果表，但不能仅靠离群小簇获选。", "",
        markdown_table(decisions), "",
        "## 冻结后的形态描述", "", markdown_table(descriptors), "",
        "## 解释边界", "",
        "- 这些数据来自文献图像数字化，采样稀疏且不同实验条件混合，不能把簇直接解释成单一物理退化机理。",
        "- 500 h 窗口用于观察 200 h 后行为；它不是原论文的 150 h 窗口，因此同时给出 150 h 敏感性结果。",
        "- 线性插值不会创造新的极值，但密集网格不等于新增测量；`raw_point_representatives.png` 专门展示真实数字化点。",
        "- 若没有四个独立 IFO-like 节点，不能宣称无监督算法发现了四类。",
        "", "## 文件导航", "",
        "- `04_k_selection/setting_and_k_decisions.csv`：每组参数的 elbow 与无监督质量指标。",
        "- `05_final_model/final_model.json`：冻结模型、K 和种子。",
        "- `05_final_model/clusters/all_clusters.png`：所有成员、median 与 SOM 权重。",
        "- `06_posthoc_ifo/raw_point_representatives.png`：未经平滑的原始数字化点。",
        "- `06_posthoc_ifo/representative_provenance.csv`：代表曲线溯源。",
        "- `literature_review.md`：方法文献与取舍。",
    ]
    (PROJECT / "reports/final_report_cn.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "search", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(); ensure_dirs(); audit_result = audit(cfg)
    primary, primary_meta, primary_grid = build_dataset(cfg, cfg["primary_horizon_hours"], "primary_500h")
    sensitivity, sensitivity_meta, sensitivity_grid = build_dataset(cfg, cfg["paper_sensitivity_horizon_hours"], "paper_150h")
    print(f"prepared primary={primary.shape}, sensitivity={sensitivity.shape}")
    if args.stage == "prepare":
        return
    primary_runs, primary_summary = run_search(cfg, primary, "primary_500h")
    # The sensitivity analysis uses only the three parameter settings stated in the paper.
    paper_settings = [s for s in cfg["som_settings"] if s["source"] == "paper"]
    sensitivity_runs, sensitivity_summary = run_search(cfg, sensitivity, "paper_150h", paper_settings)
    chosen, k, decisions = choose_final(primary_summary)
    decisions.to_csv(PROJECT / "04_k_selection/setting_and_k_decisions.csv", index=False)
    km = kmeans_validation(primary, cfg)
    km.to_csv(PROJECT / "04_k_selection/kmeans_validation.csv", index=False)
    dump_json(PROJECT / "04_k_selection/frozen_selection.json", {
        "selected_setting": chosen, "selected_k": k,
        "selection_was_made_before_ifo_naming": True,
        "rule": "paper QE elbow, then centre-overlap/distinctness and multi-seed stability diagnostics",
    })
    labels, weights, assignments, final_metrics = final_fit(primary, primary_meta, primary_grid, cfg, chosen, k)
    descriptors = posthoc_descriptors(weights, primary, labels, primary_grid)
    raw_representatives(assignments, primary, labels, weights, primary_grid)
    sensitivity_decisions = setting_decisions(sensitivity_summary)
    sensitivity_decisions.to_csv(PROJECT / "07_sensitivity/paper_150h_decisions.csv", index=False)
    make_report(cfg, audit_result, primary_meta, sensitivity_meta, decisions, chosen, k, km, descriptors, final_metrics)
    dump_json(PROJECT / "run_manifest.json", {
        "status": "complete", "selected_k": k, "selected_setting": chosen["setting"],
        "primary_n": len(primary), "paper_sensitivity_n": len(sensitivity),
        "smoothing_enabled": False, "labels_used": False,
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
    })
    print(f"complete: K={k}, setting={chosen['setting']}")


if __name__ == "__main__":
    main()
