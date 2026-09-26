#!/usr/bin/env python3
"""Shared utilities for the label-blind, no-smoothing PCE curve search."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score


@dataclass
class Curve:
    curve_id: str
    source_file: str
    relative_source: str
    validation_file: str
    doi: str
    record_id: str
    legend: str
    x_axis_name: str
    x_axis_unit: str
    y_axis_name: str
    y_axis_unit: str
    time_factor_to_hours: float
    time_factor_source: str
    time_hours: np.ndarray
    y: np.ndarray


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    fallback = {"x_time_min": 1 / 60, "x_time_h": 1.0, "x_time_day": 24.0}.get(folder)
    return fallback, f"folder_fallback:{folder}" if fallback else "unknown"


def is_pce_curve(meta: dict, path: Path) -> bool:
    y_axis = meta.get("axis", {}).get("y", {})
    y_text = normalize_text(y_axis.get("name")) + " " + normalize_text(y_axis.get("unit"))
    record = normalize_text(meta.get("record_id")) + " " + normalize_text(path.parent.parent.name)
    if any(token in y_text for token in ("δ", "delta", "change", "loss")):
        return False
    explicit = any(token in y_text for token in ("pce", "efficien", "power conversion", "power output", "pmax"))
    generic = y_text.strip() in ("", "normalized", "normalised", "performance")
    return explicit or (generic and any(token in record for token in ("pce", "efficien")))


def series_legend(meta: dict, path: Path) -> str:
    match = re.search(r"Series[_ ](\d+)", path.stem, re.I)
    wanted = match.group(1) if match else None
    for item in meta.get("series_mappings", []):
        sid = str(item.get("series_id", ""))
        if wanted and re.search(rf"\b{re.escape(wanted)}\b", sid):
            return str(item.get("legend_name") or "")
    return ""


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("x")
    if df.empty:
        return np.array([]), np.array([])
    df = df.groupby("x", as_index=False, sort=True)["y"].mean()
    return df["x"].to_numpy(float), df["y"].to_numpy(float)


def load_curves(input_root: Path) -> tuple[list[Curve], pd.DataFrame]:
    curves: list[Curve] = []
    rows: list[dict] = []
    for path in sorted(input_root.glob("**/accepted/*.csv")):
        rel = str(path.relative_to(input_root))
        validation = path.parent.parent / "validation_result.json"
        try:
            meta = json.loads(validation.read_text(errors="replace"))
        except Exception:
            meta = {}
        axis = meta.get("axis", {})
        fac, fac_source = time_factor(axis.get("x", {}), path.relative_to(input_root).parts[0])
        row = {
            "curve_id": "C" + hashlib.sha256(rel.encode()).hexdigest()[:10],
            "source_file": str(path.resolve()), "relative_source": rel,
            "validation_file": str(validation.resolve()), "doi": meta.get("doi", ""),
            "record_id": meta.get("record_id", ""), "legend": series_legend(meta, path),
            "x_axis_name": axis.get("x", {}).get("name", ""),
            "x_axis_unit": axis.get("x", {}).get("unit", ""),
            "y_axis_name": axis.get("y", {}).get("name", ""),
            "y_axis_unit": axis.get("y", {}).get("unit", ""),
            "time_factor_to_hours": fac, "time_factor_source": fac_source,
            "accepted": False, "exclusion_reason": "", "n_original_points": 0,
            "duration_hours": np.nan, "raw_min": np.nan, "raw_max": np.nan,
        }
        if not is_pce_curve(meta, path):
            row["exclusion_reason"] = "not_identified_as_true_PCE"
            rows.append(row); continue
        if fac is None:
            row["exclusion_reason"] = "unknown_time_unit"
            rows.append(row); continue
        try:
            x, y = read_xy(path)
        except Exception as exc:
            row["exclusion_reason"] = f"read_error:{type(exc).__name__}"
            rows.append(row); continue
        row["n_original_points"] = len(x)
        if len(x) < 4:
            row["exclusion_reason"] = "fewer_than_4_points"
            rows.append(row); continue
        elapsed = (x - x[0]) * fac
        if not np.all(np.diff(elapsed) > 0):
            row["exclusion_reason"] = "non_increasing_time_after_dedup"
            rows.append(row); continue
        row.update(accepted=True, duration_hours=float(elapsed[-1]), raw_min=float(y.min()), raw_max=float(y.max()))
        rows.append(row)
        curves.append(Curve(
            **{k: row[k] for k in (
                "curve_id", "source_file", "relative_source", "validation_file", "doi", "record_id", "legend",
                "x_axis_name", "x_axis_unit", "y_axis_name", "y_axis_unit", "time_factor_to_hours", "time_factor_source"
            )}, time_hours=elapsed, y=y
        ))
    return curves, pd.DataFrame(rows)


def cohort_metrics(curve: Curve, window: float) -> dict:
    t = curve.time_hours
    inside = t <= window + 1e-9
    ti = t[inside]
    if len(ti) == 0 or t[-1] + 1e-9 < window:
        return {"covers_window": False, "n_points_in_window": len(ti), "maximum_relative_gap": np.inf,
                "points_first_quarter": 0, "points_last_quarter": 0}
    # Include the analysis boundary when measuring the last interpolation span.
    nodes = np.unique(np.r_[0.0, ti, window])
    gaps = np.diff(nodes) / window if len(nodes) > 1 else np.array([1.0])
    return {
        "covers_window": True,
        "n_points_in_window": int(len(ti)),
        "maximum_relative_gap": float(gaps.max(initial=0.0)),
        "points_first_quarter": int(np.sum(ti <= 0.25 * window + 1e-9)),
        "points_last_quarter": int(np.sum(ti >= 0.75 * window - 1e-9)),
    }


def cohort_pass(metrics: dict, rule: dict) -> bool:
    return bool(
        metrics["covers_window"]
        and metrics["n_points_in_window"] >= rule["minimum_points_in_window"]
        and metrics["maximum_relative_gap"] <= rule["maximum_relative_gap"] + 1e-12
        and metrics["points_first_quarter"] >= rule["minimum_points_first_quarter"]
        and metrics["points_last_quarter"] >= rule["minimum_points_last_quarter"]
    )


def unit_rows(block: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(block, axis=1, keepdims=True)
    return block / np.maximum(denom, 1e-12)


def make_representation(display: np.ndarray, name: str) -> np.ndarray:
    if name == "maxabs_level":
        return display.copy()
    z = (display - display.mean(axis=1, keepdims=True)) / np.maximum(display.std(axis=1, keepdims=True), 1e-12)
    if name == "z_level":
        return z
    d1 = np.diff(z, axis=1)
    if name == "z_level_d1":
        return np.concatenate([unit_rows(z), unit_rows(d1)], axis=1)
    if name == "z_level_d1_d2":
        d2 = np.diff(d1, axis=1)
        return np.concatenate([unit_rows(z), unit_rows(d1), unit_rows(d2)], axis=1)
    raise ValueError(name)


def build_variant(curves: list[Curve], window: int, cohort: str, rule: dict, representation: str, grid_points: int):
    grid = np.linspace(0.0, float(window), grid_points)
    displays, rows = [], []
    for curve in curves:
        qc = cohort_metrics(curve, window)
        if not cohort_pass(qc, rule):
            continue
        values = np.interp(grid, curve.time_hours, curve.y)
        scale = float(np.max(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0 or np.ptp(values) <= 1e-12:
            continue
        display = values / scale
        displays.append(display)
        rows.append({
            "curve_id": curve.curve_id, "source_file": curve.source_file, "relative_source": curve.relative_source,
            "validation_file": curve.validation_file, "doi": curve.doi, "record_id": curve.record_id,
            "legend": curve.legend, "time_factor_to_hours": curve.time_factor_to_hours,
            "n_original_points": len(curve.time_hours), "duration_hours": float(curve.time_hours[-1]),
            **qc,
        })
    display_array = np.vstack(displays) if displays else np.empty((0, grid_points))
    data = make_representation(display_array, representation) if len(display_array) else display_array
    return data, display_array, pd.DataFrame(rows), grid


def geometric_elbow(k: np.ndarray, values: np.ndarray) -> tuple[int, float]:
    x = np.asarray(k, float); y = np.asarray(values, float)
    if len(x) < 3 or not np.all(np.isfinite(y)) or abs(y[0] - y[-1]) < 1e-15:
        return int(x[0]), 0.0
    progress = (x - x[0]) / (x[-1] - x[0])
    improvement = (y[0] - y) / (y[0] - y[-1])
    distance = improvement - progress
    idx = int(np.argmax(distance))
    return int(x[idx]), float(distance[idx])


def som_layout(k: int) -> tuple[int, int]:
    factors = [(a, k // a) for a in range(1, int(math.sqrt(k)) + 1) if k % a == 0]
    return min(factors, key=lambda ab: abs(ab[0] - ab[1]))


def pairwise_ari(label_sets: list[np.ndarray]) -> float:
    vals = [adjusted_rand_score(a, b) for a, b in itertools.combinations(label_sets, 2)]
    return float(np.mean(vals)) if vals else 1.0


def squared_to_centres(data: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances without BLAS matrix products."""
    out = np.empty((len(data), len(centres)), float)
    for j, centre in enumerate(centres):
        delta = data - centre
        out[:, j] = np.einsum("ij,ij->i", delta, delta)
    return out


def safe_kmeans(data: np.ndarray, k: int, seed: int, n_init: int = 3, max_iter: int = 100):
    """Small auditable Lloyd implementation used around an unstable macOS BLAS."""
    best = None
    for init in range(n_init):
        rng = np.random.default_rng(seed + 1009 * init)
        chosen = [int(rng.integers(len(data)))]
        nearest = squared_to_centres(data, data[chosen]).min(axis=1)
        for _ in range(1, k):
            total = float(nearest.sum())
            if total <= 1e-15:
                candidates = np.setdiff1d(np.arange(len(data)), chosen)
                nxt = int(rng.choice(candidates))
            else:
                nxt = int(rng.choice(len(data), p=nearest / total))
            chosen.append(nxt)
            nearest = np.minimum(nearest, squared_to_centres(data, data[[nxt]])[:, 0])
        centres = data[chosen].copy()
        labels = np.full(len(data), -1, int)
        for _ in range(max_iter):
            dist = squared_to_centres(data, centres)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for c in range(k):
                members = data[labels == c]
                if len(members):
                    centres[c] = members.mean(axis=0)
                else:
                    centres[c] = data[int(np.argmax(dist.min(axis=1)))]
        dist = squared_to_centres(data, centres)
        labels = dist.argmin(axis=1)
        inertia = float(dist[np.arange(len(data)), labels].sum())
        if best is None or inertia < best[0]:
            best = (inertia, centres.copy(), labels.copy())
    assert best is not None
    return best[2], best[1], best[0]


def safe_silhouette(data: np.ndarray, labels: np.ndarray, sample_size: int = 800, seed: int = 0) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(data):
        return float("nan")
    rng = np.random.default_rng(seed)
    sample = np.arange(len(data)) if len(data) <= sample_size else np.sort(rng.choice(len(data), sample_size, replace=False))
    distances = cdist(data[sample], data, metric="euclidean")
    counts = {c: int(np.sum(labels == c)) for c in unique}
    values = []
    for row, idx in enumerate(sample):
        own = labels[idx]
        own_mask = labels == own
        a = float(distances[row, own_mask].sum() / max(1, counts[own] - 1))
        b = min(float(distances[row, labels == c].mean()) for c in unique if c != own)
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


def train_som(data: np.ndarray, k: int, sigma: float, learning_rate: float, random_order: bool, seed: int, iterations: int):
    xdim, ydim = som_layout(k)
    som = MiniSom(xdim, ydim, data.shape[1], sigma=sigma, learning_rate=learning_rate,
                  random_seed=seed, neighborhood_function="gaussian", activation_distance="euclidean")
    som.random_weights_init(data)
    som.train(data, iterations, random_order=random_order, verbose=False)
    weights = som.get_weights().reshape(k, -1)
    winners = np.array([som.winner(row) for row in data], int)
    labels = winners[:, 0] * ydim + winners[:, 1]
    counts = np.bincount(labels, minlength=k)
    occupied = np.flatnonzero(counts)
    assigned = weights[labels]
    within = np.sqrt(np.mean((data - assigned) ** 2, axis=1))
    centre_dist = [float(np.sqrt(np.mean((weights[a] - weights[b]) ** 2))) for a, b in itertools.combinations(occupied, 2)]
    sil = safe_silhouette(data, labels) if len(occupied) > 1 else np.nan
    metrics = {
        "quantization_error": float(som.quantization_error(data)),
        "qe_per_sqrt_dimension": float(som.quantization_error(data) / math.sqrt(data.shape[1])),
        "topographic_error": float(som.topographic_error(data)),
        "silhouette": sil, "occupied_nodes": int(len(occupied)), "empty_nodes": int(k - len(occupied)),
        "min_cluster_size": int(counts[occupied].min()), "min_cluster_fraction": float(counts[occupied].min() / len(data)),
        "median_within_rmse": float(np.median(within)),
        "minimum_center_rmse": float(min(centre_dist)) if centre_dist else np.nan,
        "center_separation_ratio": float(min(centre_dist) / (np.median(within) + 1e-12)) if centre_dist else np.nan,
        "cluster_sizes": ";".join(map(str, counts[counts > 0].astype(int))),
    }
    return som, weights, labels, metrics


def medoid_indices(data: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> list[int]:
    result = []
    for c in range(len(weights)):
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            result.append(-1); continue
        dist = np.linalg.norm(data[idx] - weights[c], axis=1)
        result.append(int(idx[np.argmin(dist)]))
    return result


def slope(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (x >= lo) & (x <= hi)
    return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 2 else np.nan


def describe_ifo(display: np.ndarray, labels: np.ndarray, medoids: list[int], grid: np.ndarray) -> pd.DataFrame:
    """Post-freeze descriptive naming; never called during model selection."""
    rows = []
    horizon = float(grid[-1])
    early_limit = min(200.0, 0.45 * horizon)
    for c, idx in enumerate(medoids):
        if idx < 0:
            continue
        y = display[idx]
        peak_i = int(np.argmax(y)); trough_i = int(np.argmin(y))
        peak, trough, start, end = float(y[peak_i]), float(y[trough_i]), float(y[0]), float(y[-1])
        amplitude = max(float(np.ptp(y)), 1e-12)
        peak_internal = 0 < peak_i < len(y) - 1 and grid[peak_i] <= early_limit
        trough_internal = 0 < trough_i < len(y) - 1 and grid[trough_i] <= early_limit
        rise = peak - start; drop_after_peak = peak - end
        recovery = float(np.max(y[trough_i:]) - trough) if trough_i < len(y) - 1 else 0.0
        base = max(start, end)
        level70 = base + 0.70 * max(0.0, peak - base)
        peak_width = float(np.mean(y >= level70)) if peak > base else 0.0
        early_s = slope(grid, y, 0, early_limit)
        late_s = slope(grid, y, min(250.0, 0.55 * horizon), horizon)
        if trough_internal and recovery >= 0.20 * amplitude:
            name = "Valley-like"
        elif peak_internal and rise >= 0.15 * amplitude and drop_after_peak >= 0.20 * amplitude:
            name = "Bridge-like" if peak_width >= 0.22 else "Hill-like"
        elif early_s < 0 and late_s <= 0:
            name = "Slope-like"
        else:
            name = "Other/mixed"
        rows.append({
            "cluster": c, "n": int(np.sum(labels == c)), "medoid_row": idx, "posthoc_ifo_name": name,
            "start": start, "end": end, "peak": peak, "peak_time_h": float(grid[peak_i]),
            "trough": trough, "trough_time_h": float(grid[trough_i]), "range": amplitude,
            "rise": rise, "post_peak_drop": drop_after_peak, "post_trough_recovery": recovery,
            "peak_width_fraction": peak_width, "early_slope": early_s, "late_slope": late_s,
        })
    return pd.DataFrame(rows)
