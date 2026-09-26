#!/usr/bin/env python3
"""Paper-aligned 200 h PvkSOM analysis for the digitized samples_test data."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_som200h_paper_exact")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATASET_ROOT = REPO / "lab-v2" / "som_references" / "accepted" / "samples_test"
THESIS_ROOT = REPO / "thesis"
VENDOR = ROOT / "vendor" / "minisom_2_2_9"
sys.path.insert(0, str(VENDOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, mean_squared_error

from minisom import MiniSom


@dataclass(frozen=True)
class Config:
    window_hours: int = 200
    interval_minutes: int = 10
    savgol_window: int = 71
    savgol_order: int = 2
    som_x: int = 2
    som_y: int = 2
    som_sigma: float = 0.5
    som_learning_rate: float = 0.1
    som_iterations: int = 50_000
    som_random_seed: None = None
    max_plausible_duration_hours: float = 100_000.0
    minimum_points_for_akima: int = 4
    expected_input_curves: int = 218
    expected_selected_curves: int = 114

    @property
    def n_points(self) -> int:
        return self.window_hours * 60 // self.interval_minutes

    @property
    def time_grid(self) -> np.ndarray:
        # The source notebook drops the first resampling row, leaving 10 min..cutoff.
        return np.arange(1, self.n_points + 1, dtype=float) / 6.0


CFG = Config()

DATA_DIR = ROOT / "data_0_200h"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
REFERENCES_DIR = ROOT / "source_references"
LOGS_DIR = ROOT / "logs"


def ensure_directories() -> None:
    for path in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, CLUSTERS_DIR, REFERENCES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("som200h")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / "run.log", mode="w", encoding="utf-8")
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


def write_environment() -> None:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "matplotlib": matplotlib.__version__,
        "minisom": "2.2.9 (vendored and imported from output folder)",
        "tslearn": package_version("tslearn"),
        "minisom_module": str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    (ROOT / "environment_actual.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def copy_reference_files() -> None:
    sources = {
        THESIS_ROOT / "paper" / "work.md": REFERENCES_DIR / "work.md",
        THESIS_ROOT / "paper" / "Supplementary.md": REFERENCES_DIR / "Supplementary.md",
        THESIS_ROOT / "environment.yml": REFERENCES_DIR / "environment_original.yml",
        THESIS_ROOT / "20230227_degradation_analysis_revision_10_cleaned.ipynb": REFERENCES_DIR
        / "20230227_degradation_analysis_revision_10_cleaned.ipynb",
        THESIS_ROOT / "20230816_degradation_analysis_revision_11_cleaned.ipynb": REFERENCES_DIR
        / "20230816_degradation_analysis_revision_11_cleaned.ipynb",
    }
    for source, destination in sources.items():
        shutil.copy2(source, destination)


def axis_strings(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    axis = metadata.get("axis", {})
    x_axis = axis.get("x", {}) or {}
    y_axis = axis.get("y", {}) or {}
    x_text = " ".join(str(x_axis.get(key) or "") for key in ("name", "unit")).lower()
    y_text = " ".join(str(y_axis.get(key) or "") for key in ("name", "unit")).lower()
    return x_axis, y_axis, x_text, y_text


def classify_axes(metadata: dict[str, Any]) -> tuple[bool, float | None, str, bool]:
    _, _, x_text, y_text = axis_strings(metadata)
    if re.search(r"cycle", x_text):
        return False, None, "x_axis_is_cycles_not_time", False
    if not re.search(r"time|duration|hour|\bhr\b|\(h\)|\(d\)|damp heat|storage", x_text):
        return False, None, "x_axis_or_unit_unverified", False
    if not re.search(r"pce|efficien|power|pmax|mppt|\bspo\b", y_text):
        return False, None, "y_axis_not_verified_as_pce_efficiency_or_power", False
    if re.search(r"\(d\)|\bdays?\b", x_text):
        factor = 24.0
    elif re.search(r"\bh\b|\(h\)|hour|\bhr\b", x_text):
        factor = 1.0
    else:
        return False, None, "time_unit_unverified", False
    absolute_percent = bool(
        re.search(r"%|percent", y_text)
        and not re.search(r"norm|relative|a\.u", y_text)
    )
    return True, factor, "axis_verified", absolute_percent


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["x", "y"])
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["x", "y"])
    frame = frame.groupby("x", as_index=False, sort=True)["y"].mean()
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float)


def curve_id(path: Path) -> str:
    relative = path.relative_to(DATASET_ROOT).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{path.parents[1].name}_{digest}"


def audit_curves(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    validation_files = sorted(DATASET_ROOT.glob("*/validation_result.json"))
    for validation_path in validation_files:
        metadata = json.loads(validation_path.read_text(encoding="utf-8"))
        x_axis, y_axis, _, _ = axis_strings(metadata)
        axis_ok, factor, axis_reason, absolute_percent = classify_axes(metadata)
        for path in sorted((validation_path.parent / "accepted").glob("*.csv")):
            cid = curve_id(path)
            reasons: list[str] = []
            try:
                x, y = read_curve(path)
            except Exception as exc:
                x, y = np.array([]), np.array([])
                reasons.append(f"csv_read_error:{type(exc).__name__}")
            if not axis_ok:
                reasons.append(axis_reason)
            if len(x) < CFG.minimum_points_for_akima:
                reasons.append("fewer_than_4_unique_points_for_akima")

            duration_h = np.nan
            max_time_h = np.nan
            top10_after_cutoff = np.nan
            x_min = float(np.min(x)) if len(x) else np.nan
            x_max = float(np.max(x)) if len(x) else np.nan
            y_min = float(np.min(y)) if len(y) else np.nan
            y_max = float(np.max(y)) if len(y) else np.nan
            if factor is not None and len(x):
                x_hours = x * factor
                x_relative = x_hours - x_hours.min()
                duration_h = float(x_relative.max())
                if duration_h < CFG.window_hours:
                    reasons.append("duration_shorter_than_200h")
                if duration_h > CFG.max_plausible_duration_hours:
                    reasons.append("implausible_duration_over_100000h")
                max_index = int(np.argmax(y))
                max_time_h = float(x_relative[max_index])
                if max_time_h > CFG.window_hours:
                    reasons.append("global_maximum_pce_after_200h")
                n_top = min(10, len(y))
                top_indices = np.argsort(y)[-n_top:]
                top10_after_cutoff = int(np.count_nonzero(x_relative[top_indices] > CFG.window_hours))
            if len(y) and (not np.isfinite(y).all() or y_max <= 0):
                reasons.append("nonfinite_or_nonpositive_pce")

            selected = len(reasons) == 0
            if selected:
                assert factor is not None
                x_hours = x * factor
                x_relative = x_hours - x_hours.min()
                curves[cid] = (x_relative, y)

            rows.append(
                {
                    "curve_id": cid,
                    "selected": selected,
                    "exclusion_reasons": ";".join(reasons),
                    "source_file": path.relative_to(REPO).as_posix(),
                    "record": validation_path.parent.name,
                    "series_name": path.stem,
                    "x_name": x_axis.get("name"),
                    "x_unit": x_axis.get("unit"),
                    "y_name": y_axis.get("name"),
                    "y_unit": y_axis.get("unit"),
                    "time_factor_to_hours": factor,
                    "is_absolute_pce_percent": absolute_percent,
                    "n_unique_points": len(x),
                    "x_min_original": x_min,
                    "x_max_original": x_max,
                    "duration_hours": duration_h,
                    "y_min_original": y_min,
                    "y_max_original": y_max,
                    "time_of_global_max_hours": max_time_h,
                    "top10_points_after_200h": top10_after_cutoff,
                }
            )
    audit = pd.DataFrame(rows).sort_values(["record", "series_name"]).reset_index(drop=True)
    logger.info("Audited %d curves; selected %d", len(audit), int(audit["selected"].sum()))
    return audit, curves


def extract_and_preprocess(
    audit: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    selected = audit[audit["selected"]].copy().sort_values("curve_id").reset_index(drop=True)
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    smoothed_rows: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    grid = CFG.time_grid
    for row in selected.itertuples(index=False):
        x, y = curves[row.curve_id]
        try:
            # SciPy 1.13 (the local PvkSOM environment) does not expose the
            # newer extrapolate= constructor argument. The target grid is
            # guaranteed to lie inside [x.min(), x.max()] by selection.
            interpolator = Akima1DInterpolator(x, y)
            raw = np.asarray(interpolator(grid), dtype=float)
            if not np.isfinite(raw).all():
                raise ValueError("Akima interpolation produced non-finite values")
            max_abs = float(np.max(np.abs(raw)))
            if not np.isfinite(max_abs) or max_abs == 0:
                raise ValueError("MaxAbs scale is zero or non-finite")
            normalized = raw / max_abs
            smoothed = savgol_filter(
                normalized,
                window_length=CFG.savgol_window,
                polyorder=CFG.savgol_order,
            )
            raw_rows.append(raw)
            normalized_rows.append(normalized)
            smoothed_rows.append(smoothed)
        except Exception as exc:
            failures.append({"curve_id": row.curve_id, "reason": str(exc)})

    if failures:
        pd.DataFrame(failures).to_csv(DATA_DIR / "preprocessing_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} selected curves failed preprocessing")
    (DATA_DIR / "preprocessing_failures.csv").unlink(missing_ok=True)

    raw_matrix = np.vstack(raw_rows)
    normalized_matrix = np.vstack(normalized_rows)
    smoothed_matrix = np.vstack(smoothed_rows)
    selected["matrix_row"] = np.arange(len(selected))
    logger.info("Preprocessed matrix shape: %s", smoothed_matrix.shape)
    return selected, raw_matrix, normalized_matrix, smoothed_matrix


def matrix_to_csv(matrix: np.ndarray, selected: pd.DataFrame, path: Path) -> None:
    frame = pd.DataFrame(matrix, columns=[f"t_{t:.6f}h" for t in CFG.time_grid])
    frame.insert(0, "curve_id", selected["curve_id"].to_numpy())
    frame.to_csv(path, index=False, float_format="%.10g")


def save_data_outputs(
    audit: pd.DataFrame,
    selected: pd.DataFrame,
    raw: np.ndarray,
    normalized: np.ndarray,
    smoothed: np.ndarray,
) -> None:
    audit.to_csv(DATA_DIR / "all_218_curves_audit.csv", index=False)
    audit[~audit["selected"]].to_csv(DATA_DIR / "excluded_curves.csv", index=False)
    selected.to_csv(DATA_DIR / "selected_curve_metadata.csv", index=False)
    pd.DataFrame({"time_hours": CFG.time_grid}).to_csv(DATA_DIR / "time_grid_hours.csv", index=False)
    matrix_to_csv(raw, selected, DATA_DIR / "extracted_raw_0_200h_10min.csv")
    matrix_to_csv(normalized, selected, DATA_DIR / "maxabs_normalized_0_200h.csv")
    matrix_to_csv(smoothed, selected, DATA_DIR / "savgol_smoothed_0_200h.csv")
    np.savez_compressed(
        DATA_DIR / "analysis_matrices.npz",
        time_hours=CFG.time_grid,
        raw=raw,
        normalized=normalized,
        smoothed=smoothed,
        curve_ids=selected["curve_id"].to_numpy(str),
    )
    reason_counts = (
        audit.assign(reason=audit["exclusion_reasons"].replace("", "selected"))
        .groupby("reason", dropna=False)
        .size()
        .rename("n_curves")
        .reset_index()
        .sort_values("n_curves", ascending=False)
    )
    reason_counts.to_csv(DATA_DIR / "selection_reason_counts.csv", index=False)


def train_som(data: np.ndarray, logger: logging.Logger) -> tuple[MiniSom, np.ndarray]:
    # This call deliberately matches the paper notebook, including seed=None and
    # MiniSom's default sequential order (random_order=False).
    som = MiniSom(
        CFG.som_x,
        CFG.som_y,
        data.shape[1],
        sigma=CFG.som_sigma,
        learning_rate=CFG.som_learning_rate,
        random_seed=CFG.som_random_seed,
    )
    som.random_weights_init(data)
    initial_weights = som.get_weights().copy()
    som.train(data, CFG.som_iterations, verbose=False)
    labels = np.array(
        [np.ravel_multi_index(som.winner(row), (CFG.som_x, CFG.som_y)) for row in data],
        dtype=int,
    )
    np.save(RESULTS_DIR / "som_initial_weights.npy", initial_weights)
    np.save(RESULTS_DIR / "som_final_weights.npy", som.get_weights())
    with (RESULTS_DIR / "som_model_minisom_2_2_9.pkl").open("wb") as handle:
        pickle.dump(som, handle)
    logger.info(
        "SOM trained: QE=%.6f TE=%.6f", som.quantization_error(data), som.topographic_error(data)
    )
    return som, labels


def semantic_cluster_mapping(data: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, dict[int, str]]:
    rows: list[dict[str, Any]] = []
    centroids: dict[int, np.ndarray] = {}
    for cluster in range(CFG.som_x * CFG.som_y):
        members = data[labels == cluster]
        if len(members) == 0:
            centroid = np.full(data.shape[1], np.nan)
        else:
            centroid = members.mean(axis=0)
        centroids[cluster] = centroid
        early_limit = min(240, len(centroid))  # first 40 h at 10-min intervals
        early_gain = float(np.nanmax(centroid[:early_limit]) - centroid[0])
        endpoint = float(centroid[-1])
        rows.append(
            {
                "neuron_id": cluster,
                "neuron_x": cluster // CFG.som_y,
                "neuron_y": cluster % CFG.som_y,
                "n_curves": int(np.count_nonzero(labels == cluster)),
                "centroid_start": float(centroid[0]),
                "centroid_peak": float(np.nanmax(centroid)),
                "centroid_end_200h": endpoint,
                "early_gain_first_40h": early_gain,
            }
        )
    summary = pd.DataFrame(rows)
    populated = summary[summary["n_curves"] > 0]
    gain_cluster = int(populated.sort_values("early_gain_first_40h", ascending=False).iloc[0]["neuron_id"])
    remaining = [int(value) for value in populated["neuron_id"] if int(value) != gain_cluster]
    remaining = sorted(remaining, key=lambda key: centroids[key][-1], reverse=True)
    mapping = {gain_cluster: "initial_gain"}
    decay_names = ["slow_exponential_decay", "medium_exponential_decay", "fast_exponential_decay"]
    for cluster, name in zip(remaining, decay_names):
        mapping[cluster] = name
    for cluster in range(CFG.som_x * CFG.som_y):
        mapping.setdefault(cluster, "empty")
    summary["paper_shape_label"] = summary["neuron_id"].map(mapping)
    return summary, mapping


def save_som_outputs(
    som: MiniSom,
    data: np.ndarray,
    selected: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    # Semantic names can change when the paper-faithful unseeded SOM is rerun.
    # Remove only prior generated cluster CSVs so stale labels cannot survive.
    for stale_path in CLUSTERS_DIR.glob("som_neuron_*.csv"):
        stale_path.unlink()
    summary, mapping = semantic_cluster_mapping(data, labels)
    summary.to_csv(RESULTS_DIR / "som_cluster_summary.csv", index=False)
    assignments = selected.copy()
    assignments["som_neuron_id"] = labels
    assignments["som_neuron_x"] = labels // CFG.som_y
    assignments["som_neuron_y"] = labels % CFG.som_y
    assignments["paper_shape_label"] = [mapping[int(label)] for label in labels]
    assignments.to_csv(RESULTS_DIR / "som_cluster_assignments.csv", index=False)

    centroid_rows = {"time_hours": CFG.time_grid}
    for cluster in range(CFG.som_x * CFG.som_y):
        members = data[labels == cluster]
        centroid_rows[f"neuron_{cluster}_{mapping[cluster]}"] = (
            members.mean(axis=0) if len(members) else np.full(data.shape[1], np.nan)
        )
        cluster_frame = pd.DataFrame(data[labels == cluster], columns=[f"t_{t:.6f}h" for t in CFG.time_grid])
        cluster_frame.insert(0, "curve_id", assignments.loc[labels == cluster, "curve_id"].to_numpy())
        cluster_frame.to_csv(CLUSTERS_DIR / f"som_neuron_{cluster}_{mapping[cluster]}.csv", index=False)
    pd.DataFrame(centroid_rows).to_csv(RESULTS_DIR / "som_centroid_curves.csv", index=False)

    metrics = {
        "parameters": asdict(CFG),
        "n_series": int(len(data)),
        "series_length": int(data.shape[1]),
        "quantization_error": float(som.quantization_error(data)),
        "topographic_error": float(som.topographic_error(data)),
        "cluster_sizes": {str(int(row.neuron_id)): int(row.n_curves) for row in summary.itertuples()},
        "semantic_mapping": {str(key): value for key, value in mapping.items()},
        "random_seed_note": "None, exactly as the source notebook; trained weights are saved for this run.",
    }
    (RESULTS_DIR / "som_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return assignments, summary, mapping


def pce_statistics(
    raw: np.ndarray,
    selected: pd.DataFrame,
    labels: np.ndarray,
    mapping: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    absolute_mask = selected["is_absolute_pce_percent"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    for index in np.where(absolute_mask)[0]:
        values = raw[index]
        max_pce_top3_mean = float(np.mean(np.sort(values)[-3:]))
        pce_200h = float(values[-1])
        relative_loss = float((max_pce_top3_mean - pce_200h) * 100.0 / max_pce_top3_mean)
        rows.append(
            {
                "curve_id": selected.iloc[index]["curve_id"],
                "matrix_row": int(index),
                "max_pce_top3_mean_percent": max_pce_top3_mean,
                "pce_at_200h_percent": pce_200h,
                "relative_pce_loss_200h_percent": relative_loss,
                "som_neuron_id": int(labels[index]),
                "paper_shape_label": mapping[int(labels[index])],
            }
        )
    pce = pd.DataFrame(rows).sort_values("max_pce_top3_mean_percent").reset_index(drop=True)
    n_groups = 5
    length_per_group = int(np.round(len(pce) / n_groups))
    group_ids = np.zeros(len(pce), dtype=int)
    start = 0
    for group in range(1, n_groups + 1):
        stop = len(pce) if group == n_groups else min(group * length_per_group, len(pce))
        group_ids[start:stop] = group
        start = stop
    pce["pce_group"] = group_ids
    group_summary = (
        pce.groupby("pce_group")
        .agg(
            n_curves=("curve_id", "size"),
            max_pce_mean=("max_pce_top3_mean_percent", "mean"),
            max_pce_median=("max_pce_top3_mean_percent", "median"),
            max_pce_q25=("max_pce_top3_mean_percent", lambda value: value.quantile(0.25)),
            max_pce_q75=("max_pce_top3_mean_percent", lambda value: value.quantile(0.75)),
            relative_loss_mean=("relative_pce_loss_200h_percent", "mean"),
            relative_loss_median=("relative_pce_loss_200h_percent", "median"),
            relative_loss_q25=("relative_pce_loss_200h_percent", lambda value: value.quantile(0.25)),
            relative_loss_q75=("relative_pce_loss_200h_percent", lambda value: value.quantile(0.75)),
        )
        .reset_index()
    )
    model = LinearRegression().fit(
        group_summary[["max_pce_mean"]], group_summary["relative_loss_mean"]
    )
    predicted = model.predict(group_summary[["max_pce_mean"]])
    regression = {
        "n_absolute_pce_curves": int(len(pce)),
        "n_groups": n_groups,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(group_summary[["max_pce_mean"]], group_summary["relative_loss_mean"])),
        "mse_on_group_means": float(mean_squared_error(group_summary["relative_loss_mean"], predicted)),
        "warning": "Exploratory subset only: atmosphere, irradiance and temperature metadata are unavailable.",
    }
    pce.to_csv(RESULTS_DIR / "absolute_pce_subset_200h.csv", index=False)
    group_summary.to_csv(RESULTS_DIR / "pce_group_statistics_200h.csv", index=False)
    (RESULTS_DIR / "pce_group_regression_200h.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fractions = (
        pce.groupby(["pce_group", "paper_shape_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    fractions["fraction_within_pce_group"] = fractions["count"] / fractions.groupby("pce_group")["count"].transform("sum")
    fractions.to_csv(RESULTS_DIR / "som_shape_fraction_by_pce_group.csv", index=False)
    return pce, group_summary, regression


def run_dtw_kmeans(data: np.ndarray, logger: logging.Logger) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    try:
        from tslearn.clustering import TimeSeriesKMeans

        model = TimeSeriesKMeans(n_clusters=4, metric="dtw")
        labels = model.fit_predict(data)
        centers = np.asarray(model.cluster_centers_).squeeze(-1)
        with (RESULTS_DIR / "dtw_kmeans_model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        np.save(RESULTS_DIR / "dtw_kmeans_labels.npy", labels)
        np.save(RESULTS_DIR / "dtw_kmeans_centers.npy", centers)
        metrics = {
            "status": "completed",
            "parameters": {
                "n_clusters": 4,
                "metric": "dtw",
                "max_iter": 50,
                "tol": 1e-6,
                "n_init": 1,
                "max_iter_barycenter": 100,
                "metric_params": None,
                "n_jobs": None,
                "dtw_inertia": False,
                "verbose": 0,
                "random_state": None,
                "init": "k-means++",
            },
            "defaults_note": "The source notebook specified only n_clusters and metric. All defaults above match tslearn 0.5.2 exactly.",
            "tslearn_version": package_version("tslearn"),
            "cluster_sizes": {str(cluster): int(np.count_nonzero(labels == cluster)) for cluster in range(4)},
            "inertia": float(model.inertia_),
        }
        logger.info("DTW k-means completed; inertia=%.6f", model.inertia_)
        return labels.astype(int), centers, metrics
    except Exception as exc:
        metrics = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        logger.exception("DTW k-means failed")
        return None, None, metrics


def match_dtw_to_som(som_labels: np.ndarray, dtw_labels: np.ndarray | None, metrics: dict[str, Any]) -> None:
    if dtw_labels is None:
        (RESULTS_DIR / "dtw_kmeans_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    contingency = np.zeros((4, 4), dtype=int)
    for som_label, dtw_label in zip(som_labels, dtw_labels):
        contingency[int(som_label), int(dtw_label)] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(dtw): int(som) for som, dtw in zip(row_ind, col_ind)}
    metrics["adjusted_rand_index_vs_som"] = float(adjusted_rand_score(som_labels, dtw_labels))
    metrics["dtw_to_som_optimal_label_mapping"] = {str(key): value for key, value in mapping.items()}
    metrics["contingency_som_rows_dtw_columns"] = contingency.tolist()
    (RESULTS_DIR / "dtw_kmeans_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_selection(audit: pd.DataFrame) -> None:
    counts = pd.Series(
        {
            "All digitized curves": len(audit),
            "Verified time/PCE axes": int(
                (~audit["exclusion_reasons"].str.contains("axis|cycles", case=False, regex=True)).sum()
            ),
            "Final 200 h set": int(audit["selected"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#9aa0a6", "#5f9ea0", "#1f77b4"])
    ax.set_ylabel("Number of curves")
    ax.set_title("Auditable selection for the 200 h analysis")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_selection_flow.png", dpi=300)
    plt.close(fig)


def plot_preprocessing(raw: np.ndarray, normalized: np.ndarray, smoothed: np.ndarray, selected: pd.DataFrame) -> None:
    n_show = min(12, len(raw))
    indices = np.linspace(0, len(raw) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 2.1 * n_show), sharex=True)
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for row_index, data_index in enumerate(indices):
        axes[row_index, 0].plot(CFG.time_grid, raw[data_index], color="#777777", lw=0.8)
        axes[row_index, 1].plot(CFG.time_grid, normalized[data_index], color="#3b82f6", lw=0.8)
        axes[row_index, 2].plot(CFG.time_grid, smoothed[data_index], color="#d62728", lw=1.0)
        axes[row_index, 0].set_ylabel(f"S{data_index:03d}", fontsize=7)
    for axis, title in zip(axes[0], ["Akima / 10-min", "MaxAbs normalized", "Savitzky-Golay (71, 2)"]):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Time (h)")
    fig.suptitle("Paper-aligned preprocessing audit (representative curves)", y=1.002)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_preprocessing_overview.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for curve in smoothed:
        ax.plot(CFG.time_grid, curve, color="#4c78a8", alpha=0.16, lw=0.6)
    ax.set(xlabel="Time (h)", ylabel="Normalized PCE / power", title=f"All {len(smoothed)} selected curves after preprocessing")
    ax.set_xlim(0, CFG.window_hours)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_all_preprocessed_curves.png", dpi=300)
    plt.close(fig)


def plot_som(som: MiniSom, data: np.ndarray, labels: np.ndarray, summary: pd.DataFrame, mapping: dict[int, str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for cluster, ax in enumerate(axes.flat):
        members = data[labels == cluster]
        for curve in members:
            ax.plot(CFG.time_grid, curve, color="#8c8c8c", alpha=0.18, lw=0.6)
        if len(members):
            ax.plot(CFG.time_grid, members.mean(axis=0), color="black", lw=2.2)
        ax.set_title(f"Neuron {cluster // 2},{cluster % 2}: {mapping[cluster].replace('_', ' ')} (n={len(members)})")
        ax.set_xlim(0, CFG.window_hours)
        ax.set_ylim(min(-0.1, float(np.min(data)) - 0.03), max(1.05, float(np.max(data)) + 0.03))
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE / power")
    fig.suptitle("2 x 2 SOM, sigma=0.5, learning rate=0.1, 50,000 iterations")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_som_clusters.png", dpi=300)
    plt.close(fig)

    ordered = summary.sort_values("paper_shape_label")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(ordered["paper_shape_label"].str.replace("_", " "), ordered["n_curves"], color="#4c78a8")
    for bar, value in zip(bars, ordered["n_curves"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(int(value)), ha="center")
    ax.set_ylabel("Number of curves")
    ax.set_title("SOM cluster distribution")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_som_cluster_counts.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(som.distance_map().T, cmap="viridis", origin="lower")
    for x in range(CFG.som_x):
        for y in range(CFG.som_y):
            ax.text(x, y, f"{som.distance_map()[x, y]:.3f}", ha="center", va="center", color="white")
    ax.set_xticks(range(CFG.som_x))
    ax.set_yticks(range(CFG.som_y))
    ax.set_xlabel("SOM x")
    ax.set_ylabel("SOM y")
    ax.set_title("SOM unified distance matrix")
    fig.colorbar(image, ax=ax, label="Normalized neighbor distance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_som_umatrix.png", dpi=300)
    plt.close(fig)


def plot_pce_statistics(pce: pd.DataFrame, groups: pd.DataFrame, regression: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    values = [pce.loc[pce["pce_group"] == group, "relative_pce_loss_200h_percent"] for group in range(1, 6)]
    ax.boxplot(values, tick_labels=[f"G{group}" for group in range(1, 6)], showmeans=True)
    ax.set_xlabel("Equal-count maximum-PCE group")
    ax.set_ylabel("Relative PCE loss at 200 h (%)")
    ax.set_title("Absolute-PCE subset: 200 h relative loss")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_pce_loss_by_group.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = groups["max_pce_mean"].to_numpy()
    y = groups["relative_loss_mean"].to_numpy()
    # Draw the actual Q25-Q75 intervals directly. In this heterogeneous small
    # subset a mean can lie outside its IQR, so mean-centered errorbar lengths
    # would become negative and would also misstate the statistics.
    for row in groups.itertuples(index=False):
        ax.vlines(row.max_pce_mean, row.relative_loss_q25, row.relative_loss_q75, color="#1f77b4")
        ax.hlines(row.relative_loss_mean, row.max_pce_q25, row.max_pce_q75, color="#1f77b4")
    ax.scatter(x, y, color="#1f77b4", zorder=3)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = regression["intercept"] + regression["slope"] * line_x
    ax.plot(line_x, line_y, "--", color="#d62728", label=f"slope={regression['slope']:.3f}, R²={regression['r_squared']:.3f}")
    ax.set_xlabel("Mean maximum PCE in group (%)")
    ax.set_ylabel("Mean relative PCE loss at 200 h (%)")
    ax.set_title("Group-mean regression (exploratory subset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_pce_group_mean_regression.png", dpi=300)
    plt.close(fig)

    pivot = (
        pce.groupby(["pce_group", "paper_shape_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(1, 6), fill_value=0)
    )
    fractions = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(fractions))
    for column in fractions.columns:
        ax.bar([f"G{i}" for i in fractions.index], fractions[column], bottom=bottom, label=column.replace("_", " "))
        bottom += fractions[column].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction within PCE group")
    ax.set_xlabel("Equal-count maximum-PCE group")
    ax.set_title("SOM shape frequency vs. maximum-PCE group")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_som_shape_fraction_by_pce_group.png", dpi=300)
    plt.close(fig)


def plot_dtw(data: np.ndarray, labels: np.ndarray | None, centers: np.ndarray | None) -> None:
    if labels is None or centers is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for cluster, ax in enumerate(axes.flat):
        members = data[labels == cluster]
        for curve in members:
            ax.plot(CFG.time_grid, curve, color="#8c8c8c", alpha=0.16, lw=0.6)
        ax.plot(CFG.time_grid, centers[cluster], color="black", lw=2.2)
        ax.set_title(f"DTW k-means cluster {cluster} (n={len(members)})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized PCE / power")
        ax.set_xlim(0, CFG.window_hours)
    fig.suptitle("TimeSeriesKMeans(n_clusters=4, metric='dtw')")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "10_dtw_kmeans_clusters.png", dpi=300)
    plt.close(fig)


def write_method_evidence() -> None:
    text = """# 原文参数证据与本次 200h 映射

| 环节 | 原论文/原 notebook | 本次实现 |
|---|---|---|
| 截止时间 | 150h；短于 150h 排除，长曲线只取前 150h | 唯一目标改动：200h；短于 200h 排除，长曲线只取前 200h |
| 最大值位置 | 最大 PCE 在截止时间之后的曲线排除 | 全曲线全局最大值位于 200h 后则排除，并保留审计字段 |
| 时间采样 | 10min | 10min，1200 点（10min 至 200h） |
| 插值 | Akima | Akima |
| 归一化 | MaxAbsScaler，按单条曲线最大绝对值 | 相同数学运算 |
| 平滑 | Savitzky-Golay，window=71，polyorder=2 | 完全相同 |
| SOM | MiniSom 2.2.9；2×2；sigma=0.5；learning_rate=0.1 | 完全相同 |
| 初始化/训练 | random_weights_init；50,000 次；默认顺序；未设置随机种子 | 完全相同 |
| 对照 | TimeSeriesKMeans(n_clusters=4, metric='dtw') | 相同显式参数与默认参数调用 |
| PCE 稳定性 | top-3 最大 PCE 均值；截止点 PCE；5 个等数量组 | 截止点改成 200h，其余相同 |

证据位置：`work.md` 的 Data analysis 和 SOM 段落；`Supplementary.md` 的 Supplementary Fig. 2、6、7、8；两个原 notebook 的预处理、SOM 与 DTW k-means 代码单元。

## 数据集导致、且无法伪装成“完全一致”的限制

本地输入是从论文图中数字化的稀疏 `x/y` 曲线，不是作者原始每 2min MPPT 表。因此 10min 重采样与 Akima 合并为“在 10min 网格上做 Akima 重建”，然后才归一化和平滑。曲线级 N₂/air、1 sun、温度、封装、pixel filter、材料堆栈元数据不存在，不能验证或筛选。统计 PCE 分组只在纵轴明确标为绝对百分比且未标为 normalized/relative 的子集上进行，并标为探索性结果。

循环轴、单位不明、非 PCE/效率/功率纵轴、少于 200h、全局最大值在 200h 后，以及明显不合理的超过 100,000h 数字化时间跨度均逐条排除；不按数值范围猜测单位。
"""
    (REFERENCES_DIR / "METHOD_EVIDENCE_CN.md").write_text(text, encoding="utf-8")


def write_report(
    audit: pd.DataFrame,
    selected: pd.DataFrame,
    som_summary: pd.DataFrame,
    som: MiniSom,
    pce: pd.DataFrame,
    regression: dict[str, Any],
    dtw_metrics: dict[str, Any],
) -> None:
    excluded = len(audit) - len(selected)
    cluster_lines = "\n".join(
        f"- {row.paper_shape_label}: {int(row.n_curves)} 条（SOM neuron {int(row.neuron_x)},{int(row.neuron_y)}）"
        for row in som_summary.sort_values("paper_shape_label").itertuples()
    )
    reason_table = (
        audit.assign(reason=audit["exclusion_reasons"].replace("", "selected"))
        .groupby("reason")
        .size()
        .sort_values(ascending=False)
    )
    reason_lines = "\n".join(f"- `{reason}`: {count}" for reason, count in reason_table.items())
    dtw_status = dtw_metrics.get("status", "unknown")
    ari = dtw_metrics.get("adjusted_rand_index_vs_som")
    ari_text = f"，与 SOM 的 ARI={ari:.4f}" if ari is not None else ""
    report = f"""# 200h PvkSOM 严格参数复现结果

## 结论摘要

对 218 条数字化曲线完成了全量审计，最终有 {len(selected)} 条满足可确认时间/PCE 轴、跨度不少于 200h、全局最大 PCE 不在 200h 之后以及基础数值质量要求。提取后矩阵为 `{len(selected)} × {CFG.n_points}`，时间分辨率 10min，截止 200h。

SOM 使用论文原参数：MiniSom 2.2.9、2×2、sigma=0.5、learning rate=0.1、50,000 iterations、random_weights_init、顺序训练、random seed=None。该次训练的量化误差为 {som.quantization_error(np.load(DATA_DIR / 'analysis_matrices.npz')['smoothed']):.6f}，拓扑误差为 {som.topographic_error(np.load(DATA_DIR / 'analysis_matrices.npz')['smoothed']):.6f}。

{cluster_lines}

DTW k-means 对照状态：`{dtw_status}`{ari_text}。

## 数据筛选

- 原始曲线：{len(audit)}
- 纳入：{len(selected)}
- 排除：{excluded}

审计组合原因（同一条曲线可同时命中多项，因此此处是组合计数）：

{reason_lines}

所有曲线和排除原因见 `data_0_200h/all_218_curves_audit.csv`。前 200h 的原值、归一化值和平滑值分别保存在同目录三个矩阵 CSV 以及 `analysis_matrices.npz`。

## 绝对 PCE 子集

只有 {len(pce)} 条曲线的纵轴能明确确认是绝对 PCE/效率百分比。严格按原 notebook 的 top-3 最大值均值、200h 末点、5 个等数量 PCE 组计算后，组均值回归斜率为 {regression['slope']:.6f}，R²={regression['r_squared']:.6f}。由于输入缺少 N₂、1 sun、温度等条件元数据，这一部分只能视作探索性结果，不能与原论文 2,245 个同质器件的统计结论等同。

## 重要边界

输入并非原论文的原始 2min MPPT 数据，而是 218 条文献图数字化曲线，且包含不同研究条件。能保持一致的是公开方法与数值参数；无法从输入核验的实验条件没有被假定为满足。具体参数证据和不可避免的适配见 `source_references/METHOD_EVIDENCE_CN.md`。

## 复现

```bash
cd {ROOT}
MPLCONFIGDIR=/tmp/mplconfig_som200h_paper_exact conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```

原 notebook 没有设置随机种子。本次也没有新增 seed，以保持参数一致；因此重新训练可能得到旋转/置换或局部差异。当前运行的初始权重、最终权重和 pickle 模型均已保存，可用于精确追溯本次结果。
"""
    (ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")


def write_manifest() -> None:
    manifest_path = ROOT / "checksums.sha256"
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name == ".DS_Store":
            continue
        if "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs() -> dict[str, Any]:
    audit = pd.read_csv(DATA_DIR / "all_218_curves_audit.csv")
    selected = pd.read_csv(DATA_DIR / "selected_curve_metadata.csv")
    matrices = np.load(DATA_DIR / "analysis_matrices.npz")
    assignments = pd.read_csv(RESULTS_DIR / "som_cluster_assignments.csv")
    som_metrics = json.loads((RESULTS_DIR / "som_metrics.json").read_text(encoding="utf-8"))
    dtw_metrics = json.loads((RESULTS_DIR / "dtw_kmeans_metrics.json").read_text(encoding="utf-8"))
    with (RESULTS_DIR / "som_model_minisom_2_2_9.pkl").open("rb") as handle:
        stored_som = pickle.load(handle)
    replayed_labels = np.array(
        [
            np.ravel_multi_index(stored_som.winner(row), (CFG.som_x, CFG.som_y))
            for row in matrices["smoothed"]
        ],
        dtype=int,
    )
    expected_som_parameters = {
        "window_hours": 200,
        "interval_minutes": 10,
        "savgol_window": 71,
        "savgol_order": 2,
        "som_x": 2,
        "som_y": 2,
        "som_sigma": 0.5,
        "som_learning_rate": 0.1,
        "som_iterations": 50_000,
        "som_random_seed": None,
    }
    expected_dtw_parameters = {
        "n_clusters": 4,
        "metric": "dtw",
        "max_iter": 50,
        "tol": 1e-6,
        "n_init": 1,
        "max_iter_barycenter": 100,
        "metric_params": None,
        "n_jobs": None,
        "dtw_inertia": False,
        "verbose": 0,
        "random_state": None,
        "init": "k-means++",
    }
    checks = {
        "input_curve_count_is_218": len(audit) == CFG.expected_input_curves,
        "selected_curve_count_is_114": len(selected) == CFG.expected_selected_curves,
        "raw_shape_is_114x1200": matrices["raw"].shape == (CFG.expected_selected_curves, CFG.n_points),
        "normalized_shape_matches": matrices["normalized"].shape == matrices["raw"].shape,
        "smoothed_shape_matches": matrices["smoothed"].shape == matrices["raw"].shape,
        "all_matrices_finite": all(np.isfinite(matrices[key]).all() for key in ("raw", "normalized", "smoothed")),
        "maxabs_normalization_recomputes_exactly": bool(
            np.allclose(
                matrices["normalized"],
                matrices["raw"] / np.max(np.abs(matrices["raw"]), axis=1, keepdims=True),
            )
        ),
        "savgol_71_2_recomputes_exactly": bool(
            np.allclose(
                matrices["smoothed"],
                savgol_filter(matrices["normalized"], 71, 2, axis=1),
            )
        ),
        "time_grid_ends_at_200h": bool(np.isclose(matrices["time_hours"][-1], CFG.window_hours)),
        "time_grid_spacing_is_10min": bool(np.allclose(np.diff(matrices["time_hours"]), 1 / 6)),
        "all_selected_duration_ge_200h": bool((selected["duration_hours"] >= CFG.window_hours).all()),
        "no_selected_global_max_after_200h": bool((selected["time_of_global_max_hours"] <= CFG.window_hours).all()),
        "assignment_count_matches": len(assignments) == len(selected),
        "assignment_curve_ids_match_selected_order": assignments["curve_id"].tolist() == selected["curve_id"].tolist(),
        "saved_som_replays_all_assignments": bool(
            np.array_equal(replayed_labels, assignments["som_neuron_id"].to_numpy(int))
        ),
        "saved_som_weights_match_npy": bool(
            np.array_equal(stored_som.get_weights(), np.load(RESULTS_DIR / "som_final_weights.npy"))
        ),
        "som_parameters_match_paper_except_200h": all(
            som_metrics["parameters"].get(key) == value for key, value in expected_som_parameters.items()
        ),
        "dtw_parameters_match_tslearn_0_5_2_defaults": dtw_metrics.get("parameters") == expected_dtw_parameters,
        "all_four_som_neurons_populated": assignments["som_neuron_id"].nunique() == 4,
        "vendored_minisom_2_2_9_used": str(VENDOR.resolve()) in str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    report = {"passed": all(checks.values()), "checks": checks}
    (ROOT / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    ensure_directories()
    logger = configure_logging()
    logger.info("Starting paper-aligned 200 h analysis")
    write_environment()
    copy_reference_files()
    write_method_evidence()
    (ROOT / "run_config.json").write_text(
        json.dumps(asdict(CFG) | {"n_points": CFG.n_points}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit, curves = audit_curves(logger)
    if len(audit) != CFG.expected_input_curves:
        raise RuntimeError(f"Expected {CFG.expected_input_curves} input curves, found {len(audit)}")
    if int(audit["selected"].sum()) != CFG.expected_selected_curves:
        raise RuntimeError(
            f"Expected {CFG.expected_selected_curves} selected curves, found {int(audit['selected'].sum())}"
        )
    selected, raw, normalized, smoothed = extract_and_preprocess(audit, curves, logger)
    save_data_outputs(audit, selected, raw, normalized, smoothed)

    plot_selection(audit)
    plot_preprocessing(raw, normalized, smoothed, selected)
    som, som_labels = train_som(smoothed, logger)
    assignments, som_summary, semantic_mapping = save_som_outputs(
        som, smoothed, selected, som_labels
    )
    plot_som(som, smoothed, som_labels, som_summary, semantic_mapping)

    pce, pce_groups, regression = pce_statistics(raw, selected, som_labels, semantic_mapping)
    plot_pce_statistics(pce, pce_groups, regression)

    dtw_labels, dtw_centers, dtw_metrics = run_dtw_kmeans(smoothed, logger)
    match_dtw_to_som(som_labels, dtw_labels, dtw_metrics)
    if dtw_labels is not None:
        dtw_assignments = assignments[["curve_id", "som_neuron_id", "paper_shape_label"]].copy()
        dtw_assignments["dtw_kmeans_cluster"] = dtw_labels
        dtw_assignments.to_csv(RESULTS_DIR / "dtw_kmeans_assignments.csv", index=False)
    plot_dtw(smoothed, dtw_labels, dtw_centers)
    dtw_metrics = json.loads((RESULTS_DIR / "dtw_kmeans_metrics.json").read_text(encoding="utf-8"))

    write_report(audit, selected, som_summary, som, pce, regression, dtw_metrics)
    verification = verify_outputs()
    write_manifest()
    if not verification["passed"]:
        raise RuntimeError("Output verification failed; see verification_report.json")
    logger.info("Analysis complete. All verification checks passed.")


if __name__ == "__main__":
    main()
