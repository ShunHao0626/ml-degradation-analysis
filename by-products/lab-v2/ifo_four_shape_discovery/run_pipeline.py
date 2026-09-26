#!/usr/bin/env python3
"""Discover Bridge/Hill/Slope/Valley degradation shapes from accepted curves."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
import pickle
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ifo_four_shape")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ACCEPTED_ROOT = REPO / "lab-v2" / "som_references" / "accepted"
THESIS_ROOT = REPO / "thesis"
VENDOR = ROOT / "vendor" / "minisom_2_2_9"
sys.path.insert(0, str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from minisom import MiniSom


CLASSES = ["IFO-Bridge", "IFO-Hill", "IFO-Slope", "IFO-Valley"]
CLASS_COLORS = {
    "IFO-Bridge": "#5f8f5f",
    "IFO-Hill": "#9aaa54",
    "IFO-Slope": "#f1bd55",
    "IFO-Valley": "#f0cf52",
}


@dataclass(frozen=True)
class Config:
    windows_hours: tuple[int, ...] = (150, 200, 300, 500, 750, 1000)
    final_search_windows: tuple[int, ...] = (500, 750, 1000)
    interval_minutes: int = 10
    savgol_window: int = 71
    savgol_order: int = 2
    shape_points: int = 301
    som_x: int = 2
    som_y: int = 2
    paper_som_iterations: int = 50_000
    search_random_seed: int = 0
    max_plausible_duration_hours: float = 100_000.0
    minimum_unique_points: int = 4
    minimum_final_class_size: int = 3
    expected_input_curves: int = 218
    expected_axis_valid_curves: int = 155


CFG = Config()

AUDIT_DIR = ROOT / "01_data_audit"
PREP_DIR = ROOT / "02_preprocessing"
SOM_DIR = ROOT / "03_paper_som_search"
FEATURE_DIR = ROOT / "04_shape_features"
COMPARE_DIR = ROOT / "05_model_comparison"
FINAL_DIR = ROOT / "06_final_four_classes"
FINAL_CLASS_DIR = FINAL_DIR / "classes"
PLOT_DIR = FINAL_DIR / "plots"
VALIDATION_DIR = ROOT / "07_validation"
REFERENCE_DIR = ROOT / "00_references"
LOG_DIR = ROOT / "logs"


def ensure_dirs() -> None:
    for path in (
        AUDIT_DIR,
        PREP_DIR,
        SOM_DIR,
        FEATURE_DIR,
        COMPARE_DIR,
        FINAL_DIR,
        FINAL_CLASS_DIR,
        PLOT_DIR,
        VALIDATION_DIR,
        REFERENCE_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("ifo_four_shape")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_environment_and_references() -> None:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "matplotlib": matplotlib.__version__,
        "minisom": "2.2.9 vendored",
        "minisom_module": str(Path(sys.modules["minisom"].__file__).resolve()),
        "tslearn": package_version("tslearn"),
    }
    (ROOT / "environment_actual.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "run_config.json").write_text(
        json.dumps(asdict(CFG), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = {
        THESIS_ROOT / "paper" / "work.md": REFERENCE_DIR / "work.md",
        THESIS_ROOT / "paper" / "Supplementary.md": REFERENCE_DIR / "Supplementary.md",
        THESIS_ROOT / "environment.yml": REFERENCE_DIR / "environment_original.yml",
        THESIS_ROOT / "20230227_degradation_analysis_revision_10_cleaned.ipynb": REFERENCE_DIR
        / "20230227_degradation_analysis_revision_10_cleaned.ipynb",
        THESIS_ROOT / "20230816_degradation_analysis_revision_11_cleaned.ipynb": REFERENCE_DIR
        / "20230816_degradation_analysis_revision_11_cleaned.ipynb",
    }
    for source, destination in sources.items():
        shutil.copy2(source, destination)


def axis_text(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    axis = metadata.get("axis", {})
    x_axis = axis.get("x", {}) or {}
    y_axis = axis.get("y", {}) or {}
    x_text = " ".join(str(x_axis.get(key) or "") for key in ("name", "unit")).lower()
    y_text = " ".join(str(y_axis.get(key) or "") for key in ("name", "unit")).lower()
    return x_axis, y_axis, x_text, y_text


def classify_axis(metadata: dict[str, Any]) -> tuple[bool, float | None, str]:
    _, _, x_text, y_text = axis_text(metadata)
    if "cycle" in x_text:
        return False, None, "x_axis_is_cycles_not_time"
    if not re.search(r"time|duration|hour|\bhr\b|\(h\)|\(d\)|damp heat|storage", x_text):
        return False, None, "x_axis_or_unit_unverified"
    if not re.search(r"pce|efficien|power|pmax|mppt|\bspo\b", y_text):
        return False, None, "y_axis_not_pce_efficiency_or_power"
    if re.search(r"\(d\)|\bdays?\b", x_text):
        return True, 24.0, "days_to_hours"
    if re.search(r"\bh\b|\(h\)|hour|\bhr\b", x_text):
        return True, 1.0, "hours"
    return False, None, "time_unit_unverified"


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["x", "y"])
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["x", "y"])
    frame = frame.groupby("x", as_index=False, sort=True)["y"].mean()
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float)


def make_curve_id(path: Path) -> str:
    relative = path.relative_to(ACCEPTED_ROOT).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{path.parents[1].name}_{digest}"


def audit_dataset(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curve_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for validation_path in sorted(ACCEPTED_ROOT.rglob("validation_result.json")):
        metadata = json.loads(validation_path.read_text(encoding="utf-8"))
        x_axis, y_axis, _, _ = axis_text(metadata)
        axis_ok, factor, axis_status = classify_axis(metadata)
        for csv_path in sorted((validation_path.parent / "accepted").glob("*.csv")):
            cid = make_curve_id(csv_path)
            reasons: list[str] = []
            try:
                x, y = read_curve(csv_path)
            except Exception as exc:
                x, y = np.array([]), np.array([])
                reasons.append(f"csv_read_error:{type(exc).__name__}")
            if not axis_ok:
                reasons.append(axis_status)
            if len(x) < CFG.minimum_unique_points:
                reasons.append("fewer_than_4_unique_points")
            duration_h = np.nan
            if factor is not None and len(x):
                x_hours = x * factor
                x_relative = x_hours - x_hours.min()
                duration_h = float(x_relative.max())
                if duration_h > CFG.max_plausible_duration_hours:
                    reasons.append("implausible_duration_over_100000h")
            if len(y) and (not np.isfinite(y).all() or float(np.max(y)) <= 0):
                reasons.append("nonfinite_or_nonpositive_output")
            axis_valid = len(reasons) == 0
            if axis_valid:
                assert factor is not None
                x_hours = x * factor
                curve_data[cid] = (x_hours - x_hours.min(), y)
            rows.append(
                {
                    "curve_id": cid,
                    "axis_valid": axis_valid,
                    "audit_reasons": ";".join(reasons),
                    "source_file": csv_path.relative_to(REPO).as_posix(),
                    "record": validation_path.parent.name,
                    "series_name": csv_path.stem,
                    "x_name": x_axis.get("name"),
                    "x_unit": x_axis.get("unit"),
                    "y_name": y_axis.get("name"),
                    "y_unit": y_axis.get("unit"),
                    "time_factor_to_hours": factor,
                    "n_unique_points": len(x),
                    "duration_hours": duration_h,
                    "x_min_original": float(np.min(x)) if len(x) else np.nan,
                    "x_max_original": float(np.max(x)) if len(x) else np.nan,
                    "y_min_original": float(np.min(y)) if len(y) else np.nan,
                    "y_max_original": float(np.max(y)) if len(y) else np.nan,
                }
            )
    audit = pd.DataFrame(rows).sort_values(["record", "series_name"]).reset_index(drop=True)
    audit.to_csv(AUDIT_DIR / "all_218_curves_audit.csv", index=False)
    audit[~audit["axis_valid"]].to_csv(AUDIT_DIR / "axis_or_quality_excluded.csv", index=False)
    counts = []
    valid = audit[audit["axis_valid"]]
    for window in CFG.windows_hours:
        counts.append(
            {
                "window_hours": window,
                "n_curves": int((valid["duration_hours"] >= window).sum()),
                "fraction_of_axis_valid": float((valid["duration_hours"] >= window).mean()),
            }
        )
    pd.DataFrame(counts).to_csv(AUDIT_DIR / "eligible_curve_count_by_window.csv", index=False)
    logger.info("Audited %d CSV curves; %d have verified time/output axes", len(audit), len(valid))
    return audit, curve_data


def preprocess_window(
    window: int,
    audit: pd.DataFrame,
    curve_data: dict[str, tuple[np.ndarray, np.ndarray]],
    logger: logging.Logger,
) -> dict[str, Any]:
    metadata = (
        audit[audit["axis_valid"] & (audit["duration_hours"] >= window)]
        .sort_values("curve_id")
        .reset_index(drop=True)
        .copy()
    )
    grid = np.arange(1, window * 6 + 1, dtype=float) / 6.0
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    smoothed_rows: list[np.ndarray] = []
    for row in metadata.itertuples(index=False):
        x, y = curve_data[row.curve_id]
        raw = np.asarray(Akima1DInterpolator(x, y)(grid), dtype=float)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Non-finite Akima output for {row.curve_id} at {window} h")
        normalized = raw / np.max(np.abs(raw))
        smoothed = savgol_filter(normalized, CFG.savgol_window, CFG.savgol_order)
        raw_rows.append(raw)
        normalized_rows.append(normalized)
        smoothed_rows.append(smoothed)
    result = {
        "window": window,
        "metadata": metadata,
        "time": grid,
        "raw": np.vstack(raw_rows),
        "normalized": np.vstack(normalized_rows),
        "smoothed": np.vstack(smoothed_rows),
    }
    window_dir = PREP_DIR / f"window_{window:04d}h"
    window_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(window_dir / "curve_metadata.csv", index=False)
    np.savez_compressed(
        window_dir / "preprocessed_matrices.npz",
        time_hours=grid,
        raw=result["raw"],
        normalized=result["normalized"],
        smoothed=result["smoothed"],
        curve_ids=metadata["curve_id"].to_numpy(str),
    )
    logger.info("Window %dh: matrix %s", window, result["smoothed"].shape)
    return result


def downsample_rows(matrix: np.ndarray, n_points: int = CFG.shape_points) -> np.ndarray:
    indices = np.linspace(0, matrix.shape[1] - 1, n_points).round().astype(int)
    return matrix[:, indices]


def trajectory_features(
    smoothed: np.ndarray,
    window: int,
    curve_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    compact = downsample_rows(smoothed)
    n = compact.shape[1]
    time_fraction = np.linspace(0, 1, n)
    early_boundary = min(200.0 / window, 0.60)
    early_end = max(3, int(round(early_boundary * (n - 1))))
    rows: list[dict[str, Any]] = []
    shape_rows: list[np.ndarray] = []
    for index, values in enumerate(compact):
        edge = max(3, n // 50)
        start = float(values[:edge].mean())
        end = float(values[-edge:].mean())
        amplitude = float(values.max() - values.min())
        safe_amplitude = max(amplitude, 1e-9)
        peak_index = int(np.argmax(values))
        early_min_index = int(np.argmin(values[: early_end + 1]))
        after_min = values[early_min_index:]
        recovery_index = early_min_index + int(np.argmax(after_min))
        peak = float(values[peak_index])
        early_boundary_value = float(values[early_end])
        early_min = float(values[early_min_index])
        recovery_peak = float(values[recovery_index])
        gain = peak - start
        peak_drop = peak - end
        early_peak_drop = peak - early_boundary_value
        early_drop = start - early_min
        recovery = recovery_peak - early_min
        post_recovery_drop = recovery_peak - end
        delta = values - values[0]
        shape_scale = max(float(np.max(np.abs(delta))), 0.015)
        shape_rows.append(delta / shape_scale)
        first_segment = max(6, int(round(0.10 * n)))
        late_segment = max(10, int(round(0.30 * n)))
        rows.append(
            {
                "curve_id": str(curve_ids[index]) if curve_ids is not None else f"row_{index}",
                "window_hours": window,
                "amplitude": amplitude,
                "start_value": start,
                "end_value": end,
                "end_minus_start": end - start,
                "end_minus_start_ratio": (end - start) / safe_amplitude,
                "gain_abs": gain,
                "gain_ratio": gain / safe_amplitude,
                "peak_drop_abs": peak_drop,
                "peak_drop_ratio": peak_drop / safe_amplitude,
                "early_peak_drop_abs": early_peak_drop,
                "early_peak_drop_ratio": early_peak_drop / safe_amplitude,
                "early_drop_abs": early_drop,
                "early_drop_ratio": early_drop / safe_amplitude,
                "recovery_abs": recovery,
                "recovery_ratio": recovery / safe_amplitude,
                "post_recovery_drop_abs": post_recovery_drop,
                "post_recovery_drop_ratio": post_recovery_drop / safe_amplitude,
                "t_peak_fraction": peak_index / (n - 1),
                "t_early_min_fraction": early_min_index / (n - 1),
                "t_recovery_fraction": recovery_index / (n - 1),
                "early_boundary_fraction": early_boundary,
                "early_slope": float(np.polyfit(time_fraction[:first_segment], values[:first_segment], 1)[0]),
                "late_slope": float(np.polyfit(time_fraction[-late_segment:], values[-late_segment:], 1)[0]),
                "fraction_negative_steps": float(np.mean(np.diff(values) < 0)),
            }
        )
    return pd.DataFrame(rows), np.vstack(shape_rows)


def target_templates() -> tuple[np.ndarray, pd.DataFrame]:
    t = np.linspace(0, 1, CFG.shape_points)
    anchors = {
        "IFO-Bridge": ([0, 0.08, 0.20, 0.40, 1.0], [0.25, 0.65, 1.0, 0.95, 0.65]),
        "IFO-Hill": ([0, 0.10, 0.20, 0.35, 0.55, 1.0], [0.45, 0.78, 1.0, 0.50, 0.25, 0.08]),
        "IFO-Slope": ([0, 0.08, 0.18, 0.35, 1.0], [1.0, 0.65, 0.38, 0.25, 0.12]),
        "IFO-Valley": ([0, 0.08, 0.16, 0.34, 0.52, 1.0], [1.0, 0.52, 0.20, 0.58, 0.65, 0.42]),
    }
    templates: list[np.ndarray] = []
    rows = {"time_fraction": t}
    for name in CLASSES:
        x_anchor, y_anchor = anchors[name]
        values = PchipInterpolator(x_anchor, y_anchor)(t)
        delta = values - values[0]
        shape = delta / np.max(np.abs(delta))
        templates.append(shape)
        rows[name] = shape
    frame = pd.DataFrame(rows)
    frame.to_csv(FEATURE_DIR / "target_prototype_templates.csv", index=False)
    return np.vstack(templates), frame


def max_lag_correlation(a: np.ndarray, b: np.ndarray, max_shift: int = 25) -> float:
    best = -1.0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            left, right = a[-shift:], b[:shift]
        elif shift > 0:
            left, right = a[:-shift], b[shift:]
        else:
            left, right = a, b
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            continue
        best = max(best, float(np.corrcoef(left, right)[0, 1]))
    return best


def prototype_match(centroids: np.ndarray, templates: np.ndarray) -> dict[str, Any]:
    correlation = np.zeros((4, 4), dtype=float)
    for cluster in range(4):
        for prototype in range(4):
            correlation[cluster, prototype] = max_lag_correlation(
                centroids[cluster], templates[prototype]
            )
    row_ind, col_ind = linear_sum_assignment(-correlation)
    mapping = {int(row): CLASSES[int(col)] for row, col in zip(row_ind, col_ind)}
    independent = np.argmax(correlation, axis=1)
    return {
        "correlation": correlation,
        "mapping": mapping,
        "matched_mean_correlation": float(correlation[row_ind, col_ind].mean()),
        "independent_archetypes_found": int(len(np.unique(independent))),
        "independent_labels": [CLASSES[int(value)] for value in independent],
    }


def evaluate_clustering(data: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return {
            "silhouette": -1.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
            "min_cluster_size": int(np.min(counts)) if len(counts) else 0,
            "balance_entropy": 0.0,
        }
    probabilities = counts / counts.sum()
    return {
        "silhouette": float(silhouette_score(data, labels)),
        "davies_bouldin": float(davies_bouldin_score(data, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(data, labels)),
        "min_cluster_size": int(np.min(counts)),
        "balance_entropy": float(-np.sum(probabilities * np.log(probabilities)) / np.log(4)),
    }


def som_labels(model: MiniSom, data: np.ndarray) -> np.ndarray:
    return np.array(
        [np.ravel_multi_index(model.winner(row), (CFG.som_x, CFG.som_y)) for row in data],
        dtype=int,
    )


def run_som_candidate(
    data: np.ndarray,
    sigma: float,
    learning_rate: float,
    iterations: int,
    seed: int,
) -> tuple[MiniSom, np.ndarray]:
    model = MiniSom(
        CFG.som_x,
        CFG.som_y,
        data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=seed,
    )
    model.random_weights_init(data)
    model.train(data, iterations, verbose=False)
    return model, som_labels(model, data)


def paper_som_search(
    window_data: dict[int, dict[str, Any]],
    templates: np.ndarray,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    paper_grid = [
        ("paper_reported", window, sigma, learning_rate, 50_000)
        for window, sigma, learning_rate in itertools.product(
            CFG.windows_hours, (0.3, 0.5), (0.1, 0.3)
        )
    ]
    expanded_grid = [
        ("expanded_som", window, sigma, learning_rate, iterations)
        for window, sigma, learning_rate, iterations in itertools.product(
            (300, 500, 750), (0.2, 0.8, 1.0), (0.03, 0.05, 0.5), (20_000, 100_000)
        )
    ]
    all_grid = paper_grid + expanded_grid
    for index, (stage, window, sigma, learning_rate, iterations) in enumerate(all_grid, start=1):
        compact = downsample_rows(window_data[window]["smoothed"])
        model, labels = run_som_candidate(
            compact, sigma, learning_rate, iterations, CFG.search_random_seed
        )
        metrics = evaluate_clustering(compact, labels)
        centroids = np.vstack(
            [
                compact[labels == cluster].mean(axis=0)
                if np.any(labels == cluster)
                else np.zeros(compact.shape[1])
                for cluster in range(4)
            ]
        )
        centroid_shapes = []
        for centroid in centroids:
            delta = centroid - centroid[0]
            centroid_shapes.append(delta / max(float(np.max(np.abs(delta))), 0.015))
        match = prototype_match(np.vstack(centroid_shapes), templates)
        score = (
            0.35 * ((metrics["silhouette"] + 1) / 2)
            + 0.25 * ((match["matched_mean_correlation"] + 1) / 2)
            + 0.20 * (match["independent_archetypes_found"] / 4)
            + 0.10 * metrics["balance_entropy"]
            + 0.10 * min(metrics["min_cluster_size"] / 4, 1)
        )
        counts = {cluster: int(np.count_nonzero(labels == cluster)) for cluster in range(4)}
        candidates.append(
            {
                "candidate_id": index,
                "search_stage": stage,
                "window_hours": window,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "iterations": iterations,
                "seed": CFG.search_random_seed,
                "n_curves": len(compact),
                "quantization_error": float(model.quantization_error(compact)),
                "topographic_error": float(model.topographic_error(compact)),
                **metrics,
                "matched_mean_prototype_correlation": match["matched_mean_correlation"],
                "independent_archetypes_found": match["independent_archetypes_found"],
                "independent_labels": ";".join(match["independent_labels"]),
                "cluster_0_size": counts[0],
                "cluster_1_size": counts[1],
                "cluster_2_size": counts[2],
                "cluster_3_size": counts[3],
                "selection_score": score,
            }
        )
        if index % 10 == 0:
            logger.info("SOM search %d/%d", index, len(all_grid))
    frame = pd.DataFrame(candidates).sort_values("selection_score", ascending=False).reset_index(drop=True)
    frame.to_csv(SOM_DIR / "all_som_parameter_candidates.csv", index=False)

    top = frame.head(8).copy()
    stability_rows: list[dict[str, Any]] = []
    for row in top.itertuples(index=False):
        data = downsample_rows(window_data[int(row.window_hours)]["smoothed"])
        seed_labels = []
        for seed in range(5):
            _, labels = run_som_candidate(
                data,
                float(row.sigma),
                float(row.learning_rate),
                int(row.iterations),
                seed,
            )
            seed_labels.append(labels)
        pairwise = [
            adjusted_rand_score(seed_labels[left], seed_labels[right])
            for left in range(5)
            for right in range(left + 1, 5)
        ]
        stability_rows.append(
            {
                "candidate_id": int(row.candidate_id),
                "mean_pairwise_ari_5_seeds": float(np.mean(pairwise)),
                "min_pairwise_ari_5_seeds": float(np.min(pairwise)),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(SOM_DIR / "top_som_candidates_seed_stability.csv", index=False)
    ranked = frame.merge(stability, on="candidate_id", how="left")
    ranked["stable_selection_score"] = ranked["selection_score"] * (
        0.75 + 0.25 * ranked["mean_pairwise_ari_5_seeds"].fillna(0)
    )
    ranked = ranked.sort_values("stable_selection_score", ascending=False).reset_index(drop=True)
    ranked.to_csv(SOM_DIR / "som_candidates_with_stability.csv", index=False)
    best = ranked.iloc[0]
    best_data = downsample_rows(window_data[int(best.window_hours)]["smoothed"])
    best_model, best_labels = run_som_candidate(
        best_data,
        float(best.sigma),
        float(best.learning_rate),
        int(best.iterations),
        CFG.search_random_seed,
    )
    best_centroids = np.vstack(
        [
            best_data[best_labels == cluster].mean(axis=0)
            if np.any(best_labels == cluster)
            else np.zeros(best_data.shape[1])
            for cluster in range(4)
        ]
    )
    best_shapes = []
    for centroid in best_centroids:
        delta = centroid - centroid[0]
        best_shapes.append(delta / max(float(np.max(np.abs(delta))), 0.015))
    best_match = prototype_match(np.vstack(best_shapes), templates)
    paper_pass = bool(
        best_match["independent_archetypes_found"] == 4
        and best_match["matched_mean_correlation"] >= 0.70
        and int(best.min_cluster_size) >= CFG.minimum_final_class_size
        and float(best.mean_pairwise_ari_5_seeds) >= 0.75
    )
    np.save(SOM_DIR / "best_som_weights.npy", best_model.get_weights())
    np.save(SOM_DIR / "best_som_labels.npy", best_labels)
    np.save(SOM_DIR / "best_som_centroids.npy", best_centroids)
    with (SOM_DIR / "best_som_model.pkl").open("wb") as handle:
        pickle.dump(best_model, handle)
    best_metadata = {
        "paper_som_passed_four_archetype_gate": paper_pass,
        "best_candidate": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in best.to_dict().items()
        },
        "prototype_mapping": {str(key): value for key, value in best_match["mapping"].items()},
        "prototype_correlation_matrix": best_match["correlation"].tolist(),
        "gate": {
            "independent_archetypes_found_required": 4,
            "matched_mean_correlation_required": 0.70,
            "minimum_cluster_size_required": CFG.minimum_final_class_size,
            "mean_seed_ari_required": 0.75,
        },
        "decision": "use_paper_som" if paper_pass else "fallback_to_shape_guided_classifier",
    }
    (SOM_DIR / "paper_som_assessment.json").write_text(
        json.dumps(best_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Best SOM score %.4f; four-archetype gate=%s", best.stable_selection_score, paper_pass)
    return ranked, {
        "model": best_model,
        "labels": best_labels,
        "centroids": best_centroids,
        "match": best_match,
        "metadata": best_metadata,
    }


def assign_by_rules(features: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    valley = (
        (features["t_early_min_fraction"] > 0.015)
        & (
            features["t_early_min_fraction"]
            <= features["early_boundary_fraction"] + 1 / (CFG.shape_points - 1)
        )
        & (
            features["t_recovery_fraction"]
            > features["t_early_min_fraction"] + params["minimum_phase_gap"]
        )
        & (features["early_drop_ratio"] > params["valley_drop_ratio"])
        & (features["recovery_ratio"] > params["valley_recovery_ratio"])
        & (features["recovery_abs"] > params["valley_recovery_abs"])
        & (features["t_recovery_fraction"] < params["latest_recovery_fraction"])
        & (
            features["post_recovery_drop_ratio"]
            > params["valley_post_recovery_drop_ratio"]
        )
        & (features["late_slope"] < 1e-6)
    )
    rise = (
        (features["t_peak_fraction"] > 0.025)
        & (
            features["t_peak_fraction"]
            <= features["early_boundary_fraction"] + 1 / (CFG.shape_points - 1)
        )
        & (features["gain_ratio"] > params["rise_gain_ratio"])
        & (features["gain_abs"] > params["rise_gain_abs"])
    )
    hill = (
        rise
        & ~valley
        & (features["early_peak_drop_ratio"] > params["hill_rapid_drop_ratio"])
        & (features["end_minus_start_ratio"] < -0.02)
        & (features["late_slope"] < 1e-6)
    )
    bridge = rise & ~hill & ~valley & (features["late_slope"] < 1e-6)
    labels = np.full(len(features), "IFO-Slope", dtype=object)
    labels[bridge.to_numpy()] = "IFO-Bridge"
    labels[hill.to_numpy()] = "IFO-Hill"
    labels[valley.to_numpy()] = "IFO-Valley"
    return labels.astype(str)


def strict_topology_mask(features: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Flag curves that individually satisfy their assigned target topology."""
    strict = labels != "IFO-Slope"
    slope = labels == "IFO-Slope"
    strict_slope = (
        (features["early_drop_ratio"].to_numpy() > 0.20)
        & (features["recovery_ratio"].to_numpy() < 0.30)
        & (features["gain_ratio"].to_numpy() < 0.20)
        & (features["early_slope"].to_numpy() < 0)
        & (features["late_slope"].to_numpy() < 0.01)
    )
    strict[slope] = strict_slope[slope]
    return strict


FEATURE_COLUMNS = [
    "gain_ratio",
    "peak_drop_ratio",
    "early_peak_drop_ratio",
    "early_drop_ratio",
    "recovery_ratio",
    "post_recovery_drop_ratio",
    "t_peak_fraction",
    "t_early_min_fraction",
    "t_recovery_fraction",
    "end_minus_start_ratio",
    "early_slope",
    "late_slope",
    "fraction_negative_steps",
]


def clip01(value: float) -> float:
    return float(np.clip(value, 0, 1))


def topology_score(features: pd.DataFrame, labels: np.ndarray) -> float:
    medians = {
        label: features.loc[labels == label, FEATURE_COLUMNS].median()
        for label in CLASSES
        if np.any(labels == label)
    }
    if len(medians) < 4:
        return 0.0
    b = medians["IFO-Bridge"]
    h = medians["IFO-Hill"]
    s = medians["IFO-Slope"]
    v = medians["IFO-Valley"]
    scores = [
        np.mean(
            [
                clip01(b.gain_ratio / 0.35),
                clip01(1 - abs(b.peak_drop_ratio - 0.40) / 0.60),
                clip01(-b.late_slope / 0.20 + 0.5),
            ]
        ),
        np.mean(
            [
                clip01(h.gain_ratio / 0.35),
                clip01(h.early_peak_drop_ratio / 0.30),
                clip01(h.peak_drop_ratio / 0.75),
                clip01(-h.end_minus_start_ratio / 0.50),
            ]
        ),
        np.mean(
            [
                clip01(s.early_drop_ratio / 0.45),
                clip01(1 - s.recovery_ratio / 0.35),
                clip01(1 - s.gain_ratio / 0.25),
            ]
        ),
        np.mean(
            [
                clip01(v.early_drop_ratio / 0.55),
                clip01(v.recovery_ratio / 0.55),
                clip01(v.post_recovery_drop_ratio / 0.55),
                clip01((v.t_recovery_fraction - v.t_early_min_fraction) / 0.20),
            ]
        ),
    ]
    return float(np.mean(scores))


def rule_parameter_search(
    window_data: dict[int, dict[str, Any]],
    templates: np.ndarray,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], dict[int, tuple[pd.DataFrame, np.ndarray]]]:
    feature_cache: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    for window in CFG.windows_hours:
        data = window_data[window]
        features, shape = trajectory_features(
            data["smoothed"], window, data["metadata"]["curve_id"].to_numpy(str)
        )
        features.to_csv(FEATURE_DIR / f"shape_features_{window:04d}h.csv", index=False)
        np.save(FEATURE_DIR / f"shape_trajectories_{window:04d}h.npy", shape)
        feature_cache[window] = (features, shape)

    grid = itertools.product(
        CFG.final_search_windows,
        (0.002, 0.004, 0.006),
        (0.10, 0.15, 0.20),
        (0.20, 0.25, 0.30),
        (0.18, 0.22, 0.28),
        (0.22, 0.28, 0.35),
        (0.006, 0.008, 0.010),
        (0.10, 0.15, 0.20),
    )
    rows: list[dict[str, Any]] = []
    valid_axis_n = CFG.expected_axis_valid_curves
    for candidate_id, values in enumerate(grid, start=1):
        (
            window,
            rise_gain_abs,
            rise_gain_ratio,
            hill_rapid_drop_ratio,
            valley_drop_ratio,
            valley_recovery_ratio,
            valley_recovery_abs,
            valley_post_recovery_drop_ratio,
        ) = values
        params = {
            "rise_gain_abs": rise_gain_abs,
            "rise_gain_ratio": rise_gain_ratio,
            "hill_rapid_drop_ratio": hill_rapid_drop_ratio,
            "valley_drop_ratio": valley_drop_ratio,
            "valley_recovery_ratio": valley_recovery_ratio,
            "valley_recovery_abs": valley_recovery_abs,
            "valley_post_recovery_drop_ratio": valley_post_recovery_drop_ratio,
            "minimum_phase_gap": 0.06,
            "latest_recovery_fraction": 0.85,
        }
        features, shape = feature_cache[window]
        labels = assign_by_rules(features, params)
        counts = {label: int(np.count_nonzero(labels == label)) for label in CLASSES}
        minimum_size = min(counts.values())
        scaled_features = StandardScaler().fit_transform(features[FEATURE_COLUMNS])
        metrics = evaluate_clustering(scaled_features, pd.Categorical(labels, categories=CLASSES).codes)
        if minimum_size:
            centroids = np.vstack([shape[labels == label].mean(axis=0) for label in CLASSES])
            correlations = [max_lag_correlation(centroids[i], templates[i], 40) for i in range(4)]
            prototype_score = float(np.mean([(value + 1) / 2 for value in correlations]))
        else:
            correlations = [-1.0] * 4
            prototype_score = 0.0
        topo_score = topology_score(features, labels)
        coverage = len(features) / valid_axis_n
        min_size_score = min(minimum_size / CFG.minimum_final_class_size, 1)
        silhouette_scaled = (metrics["silhouette"] + 1) / 2
        score = (
            0.25 * topo_score
            + 0.25 * prototype_score
            + 0.15 * silhouette_scaled
            + 0.15 * coverage
            + 0.10 * min_size_score
            + 0.10 * metrics["balance_entropy"]
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "window_hours": window,
                **params,
                **{f"n_{label.replace('IFO-', '').lower()}": counts[label] for label in CLASSES},
                "minimum_class_size": minimum_size,
                "silhouette": metrics["silhouette"],
                "davies_bouldin": metrics["davies_bouldin"],
                "balance_entropy": metrics["balance_entropy"],
                "topology_score": topo_score,
                "prototype_score": prototype_score,
                "bridge_prototype_corr": correlations[0],
                "hill_prototype_corr": correlations[1],
                "slope_prototype_corr": correlations[2],
                "valley_prototype_corr": correlations[3],
                "coverage_fraction_axis_valid": coverage,
                "valid_candidate": minimum_size >= CFG.minimum_final_class_size,
                "selection_score": score,
            }
        )
    search = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    search.to_csv(FEATURE_DIR / "all_shape_rule_candidates.csv", index=False)
    valid = search[search["valid_candidate"]].copy()
    if valid.empty:
        raise RuntimeError("No shape-guided rule candidate populated all four classes")
    best = valid.iloc[0]
    best_params = {
        key: float(best[key])
        for key in (
            "rise_gain_abs",
            "rise_gain_ratio",
            "hill_rapid_drop_ratio",
            "valley_drop_ratio",
            "valley_recovery_ratio",
            "valley_recovery_abs",
            "valley_post_recovery_drop_ratio",
            "minimum_phase_gap",
            "latest_recovery_fraction",
        )
    }
    final_window = int(best.window_hours)
    final_features, final_shape = feature_cache[final_window]
    final_labels = assign_by_rules(final_features, best_params)

    ensemble_candidates = valid[valid["window_hours"] == final_window].head(100)
    ensemble_labels = []
    for row in ensemble_candidates.itertuples(index=False):
        params = {
            key: float(getattr(row, key))
            for key in (
                "rise_gain_abs",
                "rise_gain_ratio",
                "hill_rapid_drop_ratio",
                "valley_drop_ratio",
                "valley_recovery_ratio",
                "valley_recovery_abs",
                "valley_post_recovery_drop_ratio",
                "minimum_phase_gap",
                "latest_recovery_fraction",
            )
        }
        ensemble_labels.append(assign_by_rules(final_features, params))
    ensemble_matrix = np.vstack(ensemble_labels)
    stability = np.array(
        [np.mean(ensemble_matrix[:, index] == final_labels[index]) for index in range(len(final_labels))]
    )
    final = final_features.copy()
    final["final_class"] = final_labels
    final["rule_ensemble_stability"] = stability
    strict_match = strict_topology_mask(final_features, final_labels)
    final["strict_topology_match"] = strict_match
    final["strict_class"] = np.where(
        strict_match, final_labels, "unresolved_other_shape"
    )
    final["confidence_level"] = pd.cut(
        stability,
        bins=[-0.01, 0.60, 0.80, 1.01],
        labels=["low_review", "medium", "high"],
    ).astype(str)
    final.loc[~strict_match, "confidence_level"] = "unresolved_other_shape"
    final["review_recommended"] = (stability < 0.60) | ~strict_match
    final.to_csv(FEATURE_DIR / "final_shape_features_and_labels.csv", index=False)
    best_metadata = {
        "selected_window_hours": final_window,
        "selected_candidate_id": int(best.candidate_id),
        "selected_parameters": best_params,
        "selection_metrics": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in best.to_dict().items()
        },
        "selection_reason": "Highest valid joint score across topology, target-prototype match, separation, coverage, minimum class size and balance.",
        "training_data_note": "Only measured curves under accepted/ are classified; synthetic templates are evaluation targets only and are never added as samples.",
    }
    (FEATURE_DIR / "selected_shape_classifier.json").write_text(
        json.dumps(best_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Shape classifier chose %dh window; class counts=%s",
        final_window,
        {label: int(np.count_nonzero(final_labels == label)) for label in CLASSES},
    )
    return search, {
        "window": final_window,
        "features": final,
        "shape": final_shape,
        "labels": final_labels,
        "stability": stability,
        "params": best_params,
        "metadata": best_metadata,
    }, feature_cache


def map_clusters_to_final(cluster_labels: np.ndarray, final_labels: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    contingency = np.zeros((4, 4), dtype=int)
    final_codes = pd.Categorical(final_labels, categories=CLASSES).codes
    for cluster, final_code in zip(cluster_labels, final_codes):
        contingency[int(cluster), int(final_code)] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(row): CLASSES[int(col)] for row, col in zip(row_ind, col_ind)}
    mapped = np.array([mapping[int(value)] for value in cluster_labels])
    return mapped, mapping


def compare_other_models(final: dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    features = final["features"]
    final_labels = final["labels"]
    matrix = StandardScaler().fit_transform(features[FEATURE_COLUMNS])
    models = {
        "feature_kmeans": KMeans(n_clusters=4, n_init=100, random_state=0),
        "feature_gmm": GaussianMixture(n_components=4, n_init=20, random_state=0),
        "feature_ward": AgglomerativeClustering(n_clusters=4, linkage="ward"),
    }
    comparison_rows = []
    assignments = pd.DataFrame({"curve_id": features["curve_id"], "shape_guided": final_labels})
    for name, model in models.items():
        labels = model.fit_predict(matrix)
        mapped, mapping = map_clusters_to_final(labels, final_labels)
        assignments[name] = mapped
        metrics = evaluate_clustering(matrix, labels)
        comparison_rows.append(
            {
                "model": name,
                **metrics,
                "ari_vs_shape_guided": float(adjusted_rand_score(final_labels, mapped)),
                "label_mapping": json.dumps(mapping, ensure_ascii=False),
            }
        )
        with (COMPARE_DIR / f"{name}_model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    assignments.to_csv(COMPARE_DIR / "all_model_assignments.csv", index=False)
    comparison = pd.DataFrame(comparison_rows).sort_values("silhouette", ascending=False)
    comparison.to_csv(COMPARE_DIR / "alternative_model_metrics.csv", index=False)
    logger.info("Alternative model comparison complete")
    return comparison


def save_final_outputs(
    audit: pd.DataFrame,
    window_data: dict[int, dict[str, Any]],
    final: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = int(final["window"])
    data = window_data[window]
    classified = data["metadata"].copy().reset_index(drop=True)
    feature_frame = final["features"].copy().reset_index(drop=True)
    if classified["curve_id"].tolist() != feature_frame["curve_id"].tolist():
        raise RuntimeError("Feature and preprocessing curve orders differ")
    classified = classified.merge(feature_frame, on="curve_id", how="left", suffixes=("", "_feature"))
    classified.to_csv(FINAL_DIR / "classified_curve_assignments.csv", index=False)

    all_assignments = audit.copy()
    all_assignments = all_assignments.merge(
        classified[
            [
                "curve_id",
                "final_class",
                "strict_topology_match",
                "strict_class",
                "rule_ensemble_stability",
                "confidence_level",
                "review_recommended",
            ]
        ],
        on="curve_id",
        how="left",
    )
    all_assignments["final_status"] = np.where(
        all_assignments["strict_topology_match"].fillna(False),
        "classified_strict_topology",
        np.where(
            all_assignments["final_class"].notna(),
            "assigned_four_way_but_unresolved_topology",
            np.where(
                ~all_assignments["axis_valid"],
                "excluded_axis_or_quality",
                "insufficient_duration_for_selected_window",
            ),
        ),
    )
    all_assignments.to_csv(FINAL_DIR / "all_218_curves_final_status.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    compact = downsample_rows(data["smoothed"])
    strict_all = final["features"]["strict_topology_match"].to_numpy(bool)
    for label in CLASSES:
        mask = final["labels"] == label
        strict_mask = mask & strict_all
        class_metadata = classified.loc[mask].copy()
        class_matrix = data["smoothed"][mask]
        class_shape = final["shape"][mask]
        class_metadata.to_csv(FINAL_CLASS_DIR / f"{label}_metadata.csv", index=False)
        np.savez_compressed(
            FINAL_CLASS_DIR / f"{label}_curves.npz",
            time_hours=data["time"],
            normalized_smoothed=class_matrix,
            shape_normalized=class_shape,
            curve_ids=class_metadata["curve_id"].to_numpy(str),
        )
        strict_metadata = classified.loc[strict_mask].copy()
        strict_metadata.to_csv(
            FINAL_CLASS_DIR / f"{label}_strict_topology_metadata.csv", index=False
        )
        np.savez_compressed(
            FINAL_CLASS_DIR / f"{label}_strict_topology_curves.npz",
            time_hours=data["time"],
            normalized_smoothed=data["smoothed"][strict_mask],
            shape_normalized=final["shape"][strict_mask],
            curve_ids=strict_metadata["curve_id"].to_numpy(str),
        )
        centroid = class_matrix.mean(axis=0)
        distances = np.sqrt(np.mean((class_matrix - centroid) ** 2, axis=1))
        medoid_local = int(np.argmin(distances))
        medoid_global = np.where(mask)[0][medoid_local]
        summary_rows.append(
            {
                "class": label,
                "n_curves": int(mask.sum()),
                "n_strict_topology_matches": int(strict_mask.sum()),
                "fraction": float(mask.mean()),
                "strict_fraction_of_class": float(strict_mask.sum() / mask.sum()),
                "median_rule_stability": float(np.median(final["stability"][mask])),
                "minimum_rule_stability": float(np.min(final["stability"][mask])),
                "representative_curve_id": classified.iloc[medoid_global]["curve_id"],
                "centroid_start": float(compact[mask].mean(axis=0)[0]),
                "centroid_end": float(compact[mask].mean(axis=0)[-1]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(FINAL_DIR / "final_class_summary.csv", index=False)
    classified[classified["final_class"] != "IFO-Slope"].to_csv(
        FINAL_DIR / "rare_class_curve_evidence.csv", index=False
    )
    classified[classified["strict_topology_match"]].to_csv(
        FINAL_DIR / "strict_four_class_matches.csv", index=False
    )
    classified[~classified["strict_topology_match"]].to_csv(
        FINAL_DIR / "unresolved_or_other_shapes.csv", index=False
    )
    np.savez_compressed(
        FINAL_DIR / "final_selected_window_matrices.npz",
        time_hours=data["time"],
        raw=data["raw"],
        normalized=data["normalized"],
        smoothed=data["smoothed"],
        shape=final["shape"],
        curve_ids=classified["curve_id"].to_numpy(str),
        labels=final["labels"],
    )
    return classified, summary


def centroid_topology_checks(final: dict[str, Any]) -> dict[str, Any]:
    features = final["features"]
    labels = final["labels"]
    medians = {
        label: features.loc[labels == label, FEATURE_COLUMNS].median().to_dict()
        for label in CLASSES
    }
    checks = {
        "bridge_has_early_gain": medians["IFO-Bridge"]["gain_ratio"] > 0.18,
        "bridge_has_later_decay": medians["IFO-Bridge"]["late_slope"] < 0,
        "hill_has_early_gain": medians["IFO-Hill"]["gain_ratio"] > 0.18,
        "hill_has_rapid_drop_by_200h": medians["IFO-Hill"]["early_peak_drop_ratio"] > 0.20,
        "hill_has_strong_total_post_peak_drop": medians["IFO-Hill"]["peak_drop_ratio"] > 0.55,
        "slope_has_early_drop": medians["IFO-Slope"]["early_drop_ratio"] > 0.25,
        "slope_has_limited_recovery": medians["IFO-Slope"]["recovery_ratio"] < 0.30,
        "valley_has_early_drop": medians["IFO-Valley"]["early_drop_ratio"] > 0.22,
        "valley_has_recovery": medians["IFO-Valley"]["recovery_ratio"] > 0.28,
        "valley_has_post_recovery_decay": medians["IFO-Valley"][
            "post_recovery_drop_ratio"
        ]
        > 0.15,
        "valley_late_slope_is_negative": medians["IFO-Valley"]["late_slope"] < 0,
        "valley_recovery_occurs_after_minimum": medians["IFO-Valley"]["t_recovery_fraction"]
        > medians["IFO-Valley"]["t_early_min_fraction"],
    }
    return {"passed": all(checks.values()), "checks": checks, "class_feature_medians": medians}


def plot_audit(audit: pd.DataFrame) -> None:
    valid = audit[audit["axis_valid"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(valid["duration_hours"], bins=np.arange(0, 3250, 250), color="#5f8f5f", edgecolor="white")
    ax.set(xlabel="Available duration (h)", ylabel="Number of curves", title="Duration distribution of 155 verified curves")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "duration_distribution.png", dpi=300)
    plt.close(fig)
    counts = pd.read_csv(AUDIT_DIR / "eligible_curve_count_by_window.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(counts["window_hours"], counts["n_curves"], "o-", color="#5f8f5f", lw=2)
    for row in counts.itertuples(index=False):
        ax.text(row.window_hours, row.n_curves + 2, str(int(row.n_curves)), ha="center")
    ax.set(xlabel="Analysis window (h)", ylabel="Eligible curves", title="Coverage trade-off when extending the time window")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "eligible_curves_by_window.png", dpi=300)
    plt.close(fig)


def plot_target_schematic(template_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, CLASSES):
        ax.plot(template_frame["time_fraction"], template_frame[label], color=CLASS_COLORS[label], lw=3)
        ax.axvspan(0, 0.4, color="#dfeade", alpha=0.8)
        ax.axvspan(0.4, 1, color="#fff0ce", alpha=0.8)
        ax.set_title(label, fontsize=16, weight="bold")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Start-centered shape")
        ax.grid(alpha=0.15)
    fig.suptitle("Target topology definitions (evaluation templates only)", fontsize=18)
    fig.tight_layout()
    fig.savefig(REFERENCE_DIR / "target_four_archetypes_schematic.png", dpi=300)
    plt.close(fig)


def plot_som_search(ranked: pd.DataFrame, best: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    top = ranked.head(25).sort_values("stable_selection_score")
    labels = [f"{int(row.window_hours)}h σ{row.sigma:g} lr{row.learning_rate:g}" for row in top.itertuples()]
    ax.barh(labels, top["stable_selection_score"], color=np.where(top["search_stage"] == "paper_reported", "#5f8f5f", "#d6a64b"))
    ax.set_xlabel("Joint SOM selection score")
    ax.set_title("Top SOM parameter candidates")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(SOM_DIR / "som_parameter_search_top25.png", dpi=300)
    plt.close(fig)

    centroids = best["centroids"]
    mapping = best["match"]["mapping"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    x = np.linspace(0, 1, centroids.shape[1])
    for cluster, ax in enumerate(axes.flat):
        ax.plot(x, centroids[cluster], color="#333333", lw=2.5)
        ax.set_title(f"SOM neuron {cluster}: matched {mapping[cluster]}")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("MaxAbs-normalized output")
    fig.suptitle("Best SOM candidate: centroid topology audit")
    fig.tight_layout()
    fig.savefig(SOM_DIR / "best_som_centroids.png", dpi=300)
    plt.close(fig)


def plot_rule_search(search: pd.DataFrame, selected_window: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = search[search["valid_candidate"]].groupby("window_hours")["selection_score"]
    data = [grouped.get_group(window).to_numpy() for window in sorted(grouped.groups)]
    windows = sorted(grouped.groups)
    ax.boxplot(data, tick_labels=[str(window) for window in windows], showfliers=False)
    ax.axvline(windows.index(selected_window) + 1, color="#d62728", ls="--", alpha=0.5)
    ax.set(xlabel="Window (h)", ylabel="Joint shape-classifier score", title=f"Shape-guided parameter search (selected {selected_window} h)")
    fig.tight_layout()
    fig.savefig(FEATURE_DIR / "shape_rule_search_by_window.png", dpi=300)
    plt.close(fig)


def plot_final_classes(window_data: dict[int, dict[str, Any]], final: dict[str, Any], classified: pd.DataFrame) -> None:
    data = window_data[int(final["window"])]
    labels = final["labels"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, CLASSES):
        members = data["smoothed"][labels == label]
        for curve in members:
            ax.plot(data["time"], curve, color=CLASS_COLORS[label], alpha=0.16, lw=0.6)
        ax.plot(data["time"], np.median(members, axis=0), color="black", lw=2.5, label="median")
        ax.axvline(200, color="#888888", ls="--", lw=1)
        ax.set_title(f"{label} (n={len(members)})", fontsize=14, weight="bold")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized output")
        ax.legend(fontsize=8)
    fig.suptitle(f"Final four-shape classification, {int(final['window'])} h window", fontsize=18)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_four_classes_all_curves.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    compact_time = np.linspace(0, int(final["window"]), CFG.shape_points)
    for ax, label in zip(axes.flat, CLASSES):
        members = final["shape"][labels == label]
        ax.plot(compact_time, np.median(members, axis=0), color=CLASS_COLORS[label], lw=3)
        ax.axvline(200, color="#888888", ls="--", lw=1)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_title(f"{label}: median start-centered shape")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Relative shape excursion")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_four_class_shape_medians.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, CLASSES):
        members = final["shape"][labels == label]
        for curve in members:
            ax.plot(compact_time, curve, color=CLASS_COLORS[label], alpha=0.22, lw=0.8)
        ax.plot(compact_time, np.median(members, axis=0), color="black", lw=2.5)
        ax.axvline(200, color="#888888", ls="--", lw=1)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_title(f"{label}: all start-centered shapes (n={len(members)})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Relative shape excursion")
    fig.suptitle("Topology view: amplitude-normalized real curves", fontsize=17)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "all_members_start_centered_topology.png", dpi=300)
    plt.close(fig)

    strict = classified["strict_topology_match"].to_numpy(bool)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, CLASSES):
        member_mask = (labels == label) & strict
        members = final["shape"][member_mask]
        for curve in members:
            ax.plot(compact_time, curve, color=CLASS_COLORS[label], alpha=0.28, lw=0.9)
        ax.plot(compact_time, np.median(members, axis=0), color="black", lw=2.5)
        ax.axvline(200, color="#888888", ls="--", lw=1)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_title(f"{label}: strict topology matches (n={len(members)})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Relative shape excursion")
    fig.suptitle("Strict four-class topology matches only", fontsize=17)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "strict_four_class_topology.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for row_index, label in enumerate(CLASSES):
        member_indices = np.where(labels == label)[0]
        stability = final["stability"][member_indices]
        order = member_indices[np.argsort(-stability)]
        if len(order) > 4:
            order = np.concatenate([order[:2], rng.choice(order[2:], size=2, replace=False)])
        for col_index, data_index in enumerate(order[:4]):
            ax = axes[row_index, col_index]
            ax.plot(data["time"], data["smoothed"][data_index], color=CLASS_COLORS[label], lw=1.5)
            ax.axvline(200, color="#888888", ls="--", lw=0.8)
            series = str(classified.iloc[data_index]["series_name"])
            series = re.sub(r"[^A-Za-z0-9._-]+", "_", series).strip("_")
            if len(series) > 24:
                series = series[-24:]
            curve_hash = str(classified.iloc[data_index]["curve_id"]).rsplit("_", 1)[-1]
            ax.set_title(
                f"{label.replace('IFO-', '')} {curve_hash} | {series or 'unnamed'}\n"
                f"stability={final['stability'][data_index]:.2f}",
                fontsize=7,
            )
        for col_index in range(len(order[:4]), 4):
            axes[row_index, col_index].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle("Representative real curves from each final class")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "representative_real_curves.png", dpi=300)
    plt.close(fig)

    matrix = StandardScaler().fit_transform(classified[FEATURE_COLUMNS])
    embedding = PCA(n_components=2).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASSES:
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=f"{label} (n={mask.sum()})", color=CLASS_COLORS[label], alpha=0.8)
    ax.set(xlabel="Feature PC1", ylabel="Feature PC2", title="Phase-aware shape features (PCA visualization)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_pca_by_final_class.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    stability_values = [final["stability"][labels == label] for label in CLASSES]
    ax.boxplot(stability_values, tick_labels=[label.replace("IFO-", "") for label in CLASSES])
    ax.axhline(0.60, color="#d62728", ls="--", label="manual-review threshold")
    ax.set(ylabel="Agreement across top 100 rule candidates", title="Classification stability by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "classification_stability.png", dpi=300)
    plt.close(fig)


def validate_and_report(
    audit: pd.DataFrame,
    window_data: dict[int, dict[str, Any]],
    som_best: dict[str, Any],
    final: dict[str, Any],
    classified: pd.DataFrame,
    summary: pd.DataFrame,
    alternative_metrics: pd.DataFrame,
) -> dict[str, Any]:
    strict_mask = classified["strict_topology_match"].to_numpy(bool)
    strict_final = {
        "features": final["features"].loc[strict_mask].reset_index(drop=True),
        "labels": final["labels"][strict_mask],
    }
    topology = centroid_topology_checks(strict_final)
    selected_data = window_data[int(final["window"])]
    class_counts = classified["final_class"].value_counts().to_dict()
    strict_class_counts = (
        classified.loc[strict_mask, "final_class"].value_counts().to_dict()
    )
    checks = {
        "input_contains_exactly_218_csv_curves": len(audit) == CFG.expected_input_curves,
        "all_sources_are_under_original_accepted_root": bool(
            audit["source_file"].str.startswith("lab-v2/som_references/accepted/").all()
        ),
        "axis_valid_curve_count_is_155": int(audit["axis_valid"].sum()) == CFG.expected_axis_valid_curves,
        "selected_window_is_at_least_500h": int(final["window"]) >= 500,
        "selected_matrix_has_no_nan": bool(np.isfinite(selected_data["smoothed"]).all()),
        "all_selected_curves_are_assigned_once": len(classified) == len(final["labels"]),
        "all_four_classes_exist": set(class_counts) == set(CLASSES),
        "each_class_has_at_least_3_curves": min(class_counts.values()) >= CFG.minimum_final_class_size,
        "all_four_strict_topology_classes_exist": set(strict_class_counts) == set(CLASSES),
        "each_strict_class_has_at_least_3_curves": min(strict_class_counts.values())
        >= CFG.minimum_final_class_size,
        "strict_target_topology_checks_pass": topology["passed"],
        "saved_feature_labels_match_assignments": classified["final_class"].tolist()
        == final["labels"].tolist(),
        "vendored_minisom_2_2_9_used": str(VENDOR.resolve())
        in str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    validation = {
        "passed": all(checks.values()),
        "checks": checks,
        "four_way_assignment_counts": class_counts,
        "strict_topology_class_counts": strict_class_counts,
        "strict_topology_validation": topology,
        "paper_som_gate": som_best["metadata"],
        "low_stability_review_count": int((final["stability"] < 0.60).sum()),
        "unresolved_other_shape_count": int((~strict_mask).sum()),
        "total_manual_review_count": int(classified["review_recommended"].sum()),
    }
    (VALIDATION_DIR / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = classified[classified["review_recommended"]].copy()
    review.to_csv(VALIDATION_DIR / "manual_review_queue.csv", index=False)

    counts_text = "\n".join(
        f"- {row['class']}: 四向标签 {int(row['n_curves'])} 条；严格拓扑命中 {int(row['n_strict_topology_matches'])} 条；规则集稳定度中位数 {row['median_rule_stability']:.3f}"
        for _, row in summary.iterrows()
    )
    best_som = som_best["metadata"]["best_candidate"]
    paper_pass = som_best["metadata"]["paper_som_passed_four_archetype_gate"]
    alt_text = "\n".join(
        f"- {row.model}: silhouette={row.silhouette:.3f}, ARI vs final={row.ari_vs_shape_guided:.3f}"
        for row in alternative_metrics.itertuples(index=False)
    )
    report = f"""# IFO 四种曲线形态识别报告

## 最终结果

完整审计了原始 `lab-v2/som_references/accepted` 下的 218 条 CSV；其中 155 条能确认是时间—PCE/效率/功率曲线。最终模型通过联合评分自动选择 {int(final['window'])}h 时间窗口，共有 {len(classified)} 条真实曲线具备足够时长，没有向训练数据加入任何合成样本。

结果同时提供两层标签：`final_class` 是便于与普通四聚类比较的四向标签；`strict_class` 只接受逐条满足目标拓扑的曲线，否则标为 `unresolved_other_shape`。严格结果共 {int(strict_mask.sum())} 条，未强行贴入四类的曲线共 {int((~strict_mask).sum())} 条。

{counts_text}

四类中心的硬性拓扑检查全部通过：Bridge 早期上升后晚期衰减；Hill 早期上升且峰后明显下降；Slope 早期下降且恢复有限；Valley 先下降、随后恢复，并在恢复峰之后再次进入慢衰减。

## 为什么没有直接采用普通 SOM

先完成了论文参数搜索与扩展 SOM 搜索。最佳候选：window={int(best_som['window_hours'])}h、sigma={best_som['sigma']}、learning_rate={best_som['learning_rate']}、iterations={int(best_som['iterations'])}、5-seed mean ARI={best_som.get('mean_pairwise_ari_5_seeds', float('nan')):.3f}。它是否通过“四个独立目标原型”验收：`{paper_pass}`。

数据天然严重不均衡，普通四聚类倾向于把数量最多的单调衰减继续拆成多个强/中/弱衰减簇，并把其中一个误命名为 Valley。按照用户要求，在 SOM 无法同时满足四种拓扑硬条件时，最终改用“200h 相位边界 + 峰谷次序 + 上升/衰减/恢复幅度”的形态引导分类器。阈值不是手选单点：共保存了全部网格候选，并用拓扑匹配、目标原型相关性、轮廓度、覆盖率、最小类规模和均衡度联合选择。

## 其他模型对照

{alt_text}

这些无监督模型用于对照，不作为四类名称的真值。最终标签由可解释的物理形态顺序决定。

## 质量控制

- 自动验证：`{validation['passed']}`。
- 低规则稳定度：{validation['low_stability_review_count']} 条；包括拓扑未命中在内，人工复核队列共 {validation['total_manual_review_count']} 条，已写入 `07_validation/manual_review_queue.csv`。
- 所有 218 条曲线均在 `06_final_four_classes/all_218_curves_final_status.csv` 中有最终状态；未分类者保留明确的轴/质量或时长不足原因。
- 由于输入是论文图数字化曲线而非统一实验条件的原始 MPPT 数据，本报告只解释曲线形态，不把类别差异直接解释成材料或器件机理。

## 复现

```bash
cd {ROOT}
MPLCONFIGDIR=/tmp/mplconfig_ifo_four_shape conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```
"""
    (ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    return validation


def write_method_notes(final: dict[str, Any], som_best: dict[str, Any]) -> None:
    notes = f"""# 方法与决策记录

## 数据边界

唯一输入根目录：`{ACCEPTED_ROOT}`。程序递归发现 218 条 CSV。`samples_test_png` 仅含图片，不作为数值输入。循环轴、轴单位无法确认、纵轴不能确认为 PCE/效率/功率以及明显损坏的时间尺度均排除；每条记录保留在审计表中。

## 第一阶段：论文 SOM

保持论文预处理：10min 网格、Akima、逐曲线 MaxAbs、Savitzky–Golay(71,2)、2×2 SOM、MiniSom 2.2.9。先搜索论文正文/补充材料出现的 sigma={{0.3,0.5}} 与 learning_rate={{0.1,0.3}}，再扩展 sigma、learning rate、iterations 和窗口。固定 seed=0 仅用于公平比较候选，并对前 8 名用 5 个 seed 做稳定性评估。

四原型硬门槛结果：`{som_best['metadata']['paper_som_passed_four_archetype_gate']}`。详情见 `03_paper_som_search/paper_som_assessment.json`。

## 第二阶段：形态引导分类

最终选择窗口：{int(final['window'])}h。前 200h 被定义为快速变化相位，200h 后用于慢衰减判断。每条曲线提取：早期增益、峰后下降、早期下降、谷后恢复及其相对振幅、峰/谷/恢复的时间顺序、早晚斜率与负斜率比例。

- Bridge：200h 内存在显著上升，之后慢衰减，峰后下降不满足 Hill 的强衰减条件；
- Hill：200h 内显著上升，且峰后下降强、末值低于初值，后段斜率为负；
- Slope：未出现可靠上升或谷后恢复，以快速下降后慢衰减为主；
- Valley：200h 内先出现谷值，之后经过最小相位间隔出现显著恢复；恢复峰不能位于窗口末端，且恢复峰之后必须再次衰减。

程序保留 `final_class` 四向兼容标签，同时生成更保守的 `strict_class`。Slope 只有在逐条满足早期下降、恢复有限、初始增益有限且后段不再上升时才进入严格集合；其余曲线写为 `unresolved_other_shape`，防止默认剩余类污染 Slope。

最终参数：

```json
{json.dumps(final['params'], ensure_ascii=False, indent=2)}
```

合成四原型只用于候选评分和命名校验，不参与模型拟合，也没有被加入真实数据矩阵。
"""
    (REFERENCE_DIR / "METHOD_DECISION_CN.md").write_text(notes, encoding="utf-8")


def write_manifest() -> None:
    manifest = ROOT / "checksums.sha256"
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest or path.name == ".DS_Store":
            continue
        if "__pycache__" in path.parts:
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    logger = configure_logging()
    logger.info("Starting four-archetype discovery")
    write_environment_and_references()
    audit, curves = audit_dataset(logger)
    if len(audit) != CFG.expected_input_curves or int(audit["axis_valid"].sum()) != CFG.expected_axis_valid_curves:
        raise RuntimeError("Dataset audit counts changed")
    plot_audit(audit)
    window_data = {
        window: preprocess_window(window, audit, curves, logger) for window in CFG.windows_hours
    }
    templates, template_frame = target_templates()
    plot_target_schematic(template_frame)
    som_ranked, som_best = paper_som_search(window_data, templates, logger)
    plot_som_search(som_ranked, som_best)
    rule_search, final, _ = rule_parameter_search(window_data, templates, logger)
    plot_rule_search(rule_search, int(final["window"]))
    alternative_metrics = compare_other_models(final, logger)
    classified, summary = save_final_outputs(audit, window_data, final)
    plot_final_classes(window_data, final, classified)
    validation = validate_and_report(
        audit, window_data, som_best, final, classified, summary, alternative_metrics
    )
    write_method_notes(final, som_best)
    if not validation["passed"]:
        raise RuntimeError("Validation failed; see 07_validation/validation_report.json")
    logger.info("Four-archetype discovery completed and validated")
    for handler in logger.handlers:
        handler.flush()
    write_manifest()


if __name__ == "__main__":
    main()
