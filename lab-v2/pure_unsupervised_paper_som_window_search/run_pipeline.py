#!/usr/bin/env python3
"""Pure unsupervised SOM search using only variables present in the source paper.

No target-shape labels, prototype templates, handcrafted shape features, or
supervised objectives are used. The only tuned model variables are time window,
MiniSom sigma, and MiniSom learning_rate. All remaining preprocessing and SOM
conditions follow the paper and its author notebook.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import pickle
import platform
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INPUT_ROOT = REPO / "lab-v2" / "som_references" / "accepted" / "samples_test"
THESIS_ROOT = REPO / "thesis"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
sys.path.insert(0, str(ROOT / "vendor" / "minisom_2_2_9"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from minisom import MiniSom
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn import __version__ as sklearn_version
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass(frozen=True)
class Config:
    windows_hours: tuple[int, ...] = (150, 200, 300, 500, 750, 1000)
    sigma_values: tuple[float, ...] = (0.3, 0.4, 0.5)
    learning_rate_values: tuple[float, ...] = (0.1, 0.2, 0.3)
    interval_minutes: int = 10
    savgol_window: int = 71
    savgol_order: int = 2
    som_x: int = 2
    som_y: int = 2
    som_iterations: int = 50_000
    fair_search_seed: int = 0
    stability_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    stability_candidate_count: int = 9
    minimum_unique_points: int = 4
    max_plausible_duration_hours: float = 100_000.0
    expected_input_csv: int = 218
    expected_axis_valid: int = 155


CFG = Config()
COLORS = ["#355C7D", "#C06C84", "#6C9A8B", "#F2A65A"]

REFERENCE_DIR = ROOT / "00_parameter_provenance"
AUDIT_DIR = ROOT / "01_data_audit"
PREP_DIR = ROOT / "02_paper_preprocessing"
SEARCH_DIR = ROOT / "03_unsupervised_parameter_search"
ALL_PLOTS_DIR = SEARCH_DIR / "all_candidate_centroid_plots"
STABILITY_DIR = ROOT / "04_seed_stability"
FINAL_DIR = ROOT / "05_selected_unsupervised_result"
FINAL_CLUSTER_DIR = FINAL_DIR / "clusters"
VALIDATION_DIR = ROOT / "06_validation"
LOG_DIR = ROOT / "logs"


def ensure_dirs() -> None:
    for p in (
        REFERENCE_DIR,
        AUDIT_DIR,
        PREP_DIR,
        SEARCH_DIR,
        ALL_PLOTS_DIR,
        STABILITY_DIR,
        FINAL_DIR,
        FINAL_CLUSTER_DIR,
        VALIDATION_DIR,
        LOG_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("pure_paper_som")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    file_handler = logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8")
    stream.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_provenance() -> None:
    config = asdict(CFG)
    for key, value in list(config.items()):
        if isinstance(value, tuple):
            config[key] = list(value)
    config["input_root"] = str(INPUT_ROOT)
    config["parameter_boundary"] = {
        "tuned_variables_only": ["time_window_hours", "sigma", "learning_rate"],
        "fixed_paper_variables": [
            "10min_resampling",
            "Akima_interpolation",
            "per_curve_MaxAbs",
            "Savitzky_Golay_71_order_2",
            "MiniSom_2.2.9",
            "2x2_grid",
            "50000_sequential_iterations",
            "random_weights_init",
        ],
        "forbidden_and_not_used": [
            "target_shape_labels",
            "prototype_templates",
            "peak_or_valley_thresholds",
            "handcrafted_derivative_features",
            "supervised_classifier",
            "alternative_clustering_model",
        ],
    }
    (ROOT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "matplotlib": matplotlib.__version__,
        "minisom": "2.2.9 vendored exact source",
        "minisom_module": str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    (ROOT / "environment_actual.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = {
        THESIS_ROOT / "paper" / "work.md": REFERENCE_DIR / "work.md",
        THESIS_ROOT / "paper" / "Supplementary.md": REFERENCE_DIR / "Supplementary.md",
        THESIS_ROOT / "environment.yml": REFERENCE_DIR / "environment_original.yml",
        THESIS_ROOT / "20230227_degradation_analysis_revision_10_cleaned.ipynb": REFERENCE_DIR
        / "author_analysis_notebook.ipynb",
    }
    for src, dst in sources.items():
        shutil.copy2(src, dst)

    text = """# 参数边界与文献依据

## 保持不变的原文流程

- 10 min 重采样。
- Akima 插值。
- 每条曲线按自身最大绝对值归一化（MaxAbsScaler 等价实现）。
- Savitzky–Golay 平滑：window length 71，polynomial order 2。
- MiniSom 2.2.9，2×2（四节点），`random_weights_init`。
- 50,000 次顺序训练；MiniSom `train` 默认 `random_order=False`。

## 唯一搜索的三个原文变量

- 时间窗口：150、200、300、500、750、1000 h。
- sigma：0.3、0.4、0.5。论文正文使用 0.5，补充材料使用 0.3；0.4 是两者之间的微调值。
- learning rate：0.1、0.2、0.3。论文正文使用 0.1，补充材料使用 0.3；0.2 是两者之间的微调值。

没有搜索任何其他变量。参数搜索阶段固定 seed=0 只是保证候选之间可复算；前九名再用 seed 0–4 检查稳定性。最终模型按作者代码不指定随机种子重新训练，并保存实际权重和分配。

## 没有使用的内容

没有 Bridge/Hill/Slope/Valley 标签，没有目标模板，没有峰谷规则，没有导数特征，没有人工筛选样本，没有监督学习，也没有 K-means、GMM、DTW 等替代模型。

候选排序只读取 SOM 输出：四节点是否均占用、不同随机初始化的 ARI、轮廓系数、最小簇规模和量化误差。类别始终写为匿名 Node 0–3，防止人为命名影响训练或选择。
"""
    (REFERENCE_DIR / "PARAMETER_BOUNDARY_CN.md").write_text(text, encoding="utf-8")


def axis_text(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    axis = metadata.get("axis", {})
    x_axis = axis.get("x", {}) or {}
    y_axis = axis.get("y", {}) or {}
    x_text = " ".join(str(x_axis.get(k) or "") for k in ("name", "unit")).lower()
    y_text = " ".join(str(y_axis.get(k) or "") for k in ("name", "unit")).lower()
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
    frame = frame.dropna(subset=["x", "y"]).groupby("x", as_index=False, sort=True)["y"].mean()
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float)


def make_curve_id(path: Path) -> str:
    rel = path.relative_to(INPUT_ROOT).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{path.parents[1].name}_{digest}"


def audit_dataset(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for validation_path in sorted(INPUT_ROOT.rglob("validation_result.json")):
        metadata = json.loads(validation_path.read_text(encoding="utf-8"))
        x_axis, y_axis, _, _ = axis_text(metadata)
        axis_ok, factor, status = classify_axis(metadata)
        for csv_path in sorted((validation_path.parent / "accepted").glob("*.csv")):
            cid = make_curve_id(csv_path)
            reasons: list[str] = []
            try:
                x, y = read_curve(csv_path)
            except Exception as exc:
                x, y = np.array([]), np.array([])
                reasons.append(f"csv_read_error:{type(exc).__name__}")
            if not axis_ok:
                reasons.append(status)
            if len(x) < CFG.minimum_unique_points:
                reasons.append("fewer_than_4_unique_points")
            duration_h = np.nan
            if factor is not None and len(x):
                x_h = x * factor
                duration_h = float(x_h.max() - x_h.min())
                if duration_h > CFG.max_plausible_duration_hours:
                    reasons.append("implausible_duration_over_100000h")
            if len(y) and (not np.isfinite(y).all() or float(np.max(y)) <= 0):
                reasons.append("nonfinite_or_nonpositive_output")
            valid = not reasons
            if valid:
                assert factor is not None
                x_h = x * factor
                curves[cid] = (x_h - x_h.min(), y)
            rows.append(
                {
                    "curve_id": cid,
                    "axis_valid": valid,
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
                }
            )
    audit = pd.DataFrame(rows).sort_values(["record", "series_name"]).reset_index(drop=True)
    audit.to_csv(AUDIT_DIR / "all_input_curves_audit.csv", index=False)
    audit.loc[~audit["axis_valid"]].to_csv(AUDIT_DIR / "axis_or_quality_excluded.csv", index=False)
    valid = audit[audit["axis_valid"]]
    counts = pd.DataFrame(
        [
            {
                "window_hours": w,
                "eligible_curves": int((valid["duration_hours"] >= w).sum()),
                "fraction_of_axis_valid": float((valid["duration_hours"] >= w).mean()),
            }
            for w in CFG.windows_hours
        ]
    )
    counts.to_csv(AUDIT_DIR / "eligible_curve_count_by_window.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts["window_hours"].astype(str), counts["eligible_curves"], color="#4C78A8")
    ax.set(xlabel="Time window (h)", ylabel="Eligible curves", title="Eligible curves by time window")
    for i, value in enumerate(counts["eligible_curves"]):
        ax.text(i, value + 2, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "eligible_curves_by_window.png", dpi=220)
    plt.close(fig)
    logger.info("Audited %d CSV; %d verified time-output curves", len(audit), len(valid))
    return audit, curves


def preprocess_window(
    window: int,
    audit: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    logger: logging.Logger,
) -> dict[str, Any]:
    metadata = (
        audit[audit["axis_valid"] & (audit["duration_hours"] >= window)]
        .sort_values("curve_id")
        .reset_index(drop=True)
        .copy()
    )
    time = np.arange(1, window * 6 + 1, dtype=float) / 6.0
    raw_rows, norm_rows, smooth_rows = [], [], []
    qc_rows = []
    for row in metadata.itertuples(index=False):
        x, y = curves[row.curve_id]
        raw = np.asarray(Akima1DInterpolator(x, y)(time), dtype=float)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Non-finite Akima values for {row.curve_id} at {window}h")
        scale = float(np.max(np.abs(raw)))
        normalized = raw / scale
        smoothed = savgol_filter(normalized, CFG.savgol_window, CFG.savgol_order)
        raw_rows.append(raw)
        norm_rows.append(normalized)
        smooth_rows.append(smoothed)
        qc_rows.append(
            {
                "curve_id": row.curve_id,
                "window_hours": window,
                "point_count": len(time),
                "max_abs_before_normalization": scale,
                "max_abs_after_normalization": float(np.max(np.abs(normalized))),
                "all_finite": bool(np.isfinite(smoothed).all()),
            }
        )
    result = {
        "window": window,
        "time": time,
        "metadata": metadata,
        "raw": np.vstack(raw_rows),
        "normalized": np.vstack(norm_rows),
        "smoothed": np.vstack(smooth_rows),
    }
    out = PREP_DIR / f"window_{window:04d}h"
    out.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out / "curve_metadata.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(out / "preprocessing_qc.csv", index=False)
    np.savez_compressed(
        out / "paper_preprocessed_matrices.npz",
        time_hours=time,
        raw=result["raw"],
        normalized=result["normalized"],
        smoothed=result["smoothed"],
        curve_ids=metadata["curve_id"].to_numpy(str),
    )
    logger.info("Preprocessed %dh: %s", window, result["smoothed"].shape)
    return result


def som_labels(model: MiniSom, data: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.ravel_multi_index(model.winner(row), (CFG.som_x, CFG.som_y)) for row in data],
        dtype=int,
    )


def train_som(data: np.ndarray, sigma: float, learning_rate: float, seed: int | None):
    kwargs = dict(
        x=CFG.som_x,
        y=CFG.som_y,
        input_len=data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
    )
    if seed is not None:
        kwargs["random_seed"] = seed
    model = MiniSom(**kwargs)
    model.random_weights_init(data)
    model.train(data, CFG.som_iterations, verbose=False)
    labels = som_labels(model, data)
    return model, labels


def evaluate_run(model: MiniSom, data: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=4)
    occupied = int(np.count_nonzero(counts))
    # Precomputed direct Euclidean distances avoid cancellation warnings from
    # old MiniSom/sklearn vectorized dot-product formulas under NumPy 2.x.
    sample_distances = cdist(data, data, metric="euclidean")
    silhouette = (
        float(silhouette_score(sample_distances, labels, metric="precomputed"))
        if occupied > 1
        else float("nan")
    )
    flat_weights = model.get_weights().reshape(4, data.shape[1])
    weight_distances = np.linalg.norm(data[:, None, :] - flat_weights[None, :, :], axis=2)
    qe = float(np.min(weight_distances, axis=1).mean())
    best_two = np.argsort(weight_distances, axis=1)[:, :2]
    bx, by = best_two // CFG.som_y, best_two % CFG.som_y
    neighbor_distance = np.sqrt((bx[:, 0] - bx[:, 1]) ** 2 + (by[:, 0] - by[:, 1]) ** 2)
    topographic_error = float(np.mean(neighbor_distance > 1.42))
    return {
        "quantization_error": qe,
        "quantization_error_per_sqrt_dimension": qe / np.sqrt(data.shape[1]),
        "topographic_error": topographic_error,
        "silhouette": silhouette,
        "occupied_nodes": occupied,
        "min_cluster_size_including_empty": int(counts.min()),
        "cluster_0_size": int(counts[0]),
        "cluster_1_size": int(counts[1]),
        "cluster_2_size": int(counts[2]),
        "cluster_3_size": int(counts[3]),
    }


def save_run_artifacts(
    run_dir: Path,
    model: MiniSom,
    labels: np.ndarray,
    data_info: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = data_info["metadata"][["curve_id", "source_file", "record", "series_name"]].copy()
    meta["som_node"] = labels
    meta.to_csv(run_dir / "cluster_assignments.csv", index=False)
    np.save(run_dir / "som_weights.npy", model.get_weights())
    np.save(run_dir / "som_labels.npy", labels)
    centroids = []
    rows = []
    for node in range(4):
        members = data_info["smoothed"][labels == node]
        centroid = members.mean(axis=0) if len(members) else np.full(data_info["smoothed"].shape[1], np.nan)
        centroids.append(centroid)
        rows.append({"som_node": node, "count": len(members)})
    np.save(run_dir / "cluster_centroids.npy", np.asarray(centroids))
    pd.DataFrame(rows).to_csv(run_dir / "node_counts.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_candidate(
    path: Path,
    model: MiniSom,
    labels: np.ndarray,
    data_info: dict[str, Any],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    time = data_info["time"]
    data = data_info["smoothed"]
    for node, ax in enumerate(axes.ravel()):
        members = data[labels == node]
        for curve in members[:: max(1, len(members) // 35)]:
            ax.plot(time, curve, color=COLORS[node], alpha=0.12, lw=0.65)
        if len(members):
            ax.plot(time, members.mean(axis=0), color=COLORS[node], lw=2.6, label="member mean")
        x, y = divmod(node, 2)
        ax.plot(time, model.get_weights()[x, y], color="black", ls="--", lw=1.25, label="SOM weight")
        ax.set_title(f"Anonymous Node {node} (n={len(members)})")
        ax.set_ylabel("Normalized PCE")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def run_search(window_data: dict[int, dict[str, Any]], logger: logging.Logger):
    rows = []
    cache: dict[int, tuple[MiniSom, np.ndarray, dict[str, Any]]] = {}
    grid = list(itertools.product(CFG.windows_hours, CFG.sigma_values, CFG.learning_rate_values))
    for candidate_id, (window, sigma, lr) in enumerate(grid, start=1):
        info = window_data[window]
        model, labels = train_som(info["smoothed"], sigma, lr, CFG.fair_search_seed)
        metrics = evaluate_run(model, info["smoothed"], labels)
        record = {
            "candidate_id": candidate_id,
            "window_hours": window,
            "sigma": sigma,
            "learning_rate": lr,
            "iterations": CFG.som_iterations,
            "som_x": CFG.som_x,
            "som_y": CFG.som_y,
            "seed": CFG.fair_search_seed,
            "n_curves": len(labels),
            "input_dimensions": info["smoothed"].shape[1],
            **metrics,
        }
        rows.append(record)
        cache[candidate_id] = (model, labels, record)
        run_dir = SEARCH_DIR / "candidates" / f"candidate_{candidate_id:03d}"
        save_run_artifacts(run_dir, model, labels, info, record)
        plot_candidate(
            ALL_PLOTS_DIR / f"candidate_{candidate_id:03d}_w{window}_s{sigma}_lr{lr}.png",
            model,
            labels,
            info,
            f"Candidate {candidate_id}: {window}h, sigma={sigma}, learning rate={lr}",
        )
        if candidate_id % 6 == 0:
            logger.info("Parameter search %d/%d", candidate_id, len(grid))
    frame = pd.DataFrame(rows)
    frame["all_four_nodes_occupied"] = frame["occupied_nodes"] == 4
    initial_rank = frame.sort_values(
        [
            "all_four_nodes_occupied",
            "silhouette",
            "min_cluster_size_including_empty",
            "quantization_error_per_sqrt_dimension",
            "n_curves",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    initial_rank["initial_unsupervised_rank"] = np.arange(1, len(initial_rank) + 1)
    initial_rank.to_csv(SEARCH_DIR / "all_parameter_candidates_ranked.csv", index=False, float_format="%.10g")
    return initial_rank, cache


def run_stability(
    initial_rank: pd.DataFrame,
    window_data: dict[int, dict[str, Any]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = initial_rank.head(CFG.stability_candidate_count)
    run_rows, summary_rows = [], []
    for top_index, row in enumerate(top.itertuples(index=False), start=1):
        labels_by_seed = []
        for seed in CFG.stability_seeds:
            info = window_data[int(row.window_hours)]
            model, labels = train_som(info["smoothed"], float(row.sigma), float(row.learning_rate), seed)
            metrics = evaluate_run(model, info["smoothed"], labels)
            run_record = {
                "candidate_id": int(row.candidate_id),
                "window_hours": int(row.window_hours),
                "sigma": float(row.sigma),
                "learning_rate": float(row.learning_rate),
                "iterations": CFG.som_iterations,
                "seed": seed,
                "n_curves": len(labels),
                **metrics,
            }
            run_rows.append(run_record)
            labels_by_seed.append(labels)
            run_dir = STABILITY_DIR / f"candidate_{int(row.candidate_id):03d}" / f"seed_{seed}"
            save_run_artifacts(run_dir, model, labels, info, run_record)
        pairwise = [
            adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
            for i in range(len(labels_by_seed))
            for j in range(i + 1, len(labels_by_seed))
        ]
        candidate_runs = [x for x in run_rows if x["candidate_id"] == int(row.candidate_id)]
        summary_rows.append(
            {
                "candidate_id": int(row.candidate_id),
                "mean_pairwise_ari_5_seeds": float(np.mean(pairwise)),
                "min_pairwise_ari_5_seeds": float(np.min(pairwise)),
                "all_seed_runs_occupy_four_nodes": all(x["occupied_nodes"] == 4 for x in candidate_runs),
                "median_seed_silhouette": float(np.median([x["silhouette"] for x in candidate_runs])),
                "minimum_cluster_size_across_seeds": int(
                    min(x["min_cluster_size_including_empty"] for x in candidate_runs)
                ),
            }
        )
        logger.info("Stability candidate %d/%d complete", top_index, len(top))
    runs = pd.DataFrame(run_rows)
    summaries = pd.DataFrame(summary_rows)
    runs.to_csv(STABILITY_DIR / "all_seed_run_metrics.csv", index=False, float_format="%.10g")
    summaries.to_csv(STABILITY_DIR / "candidate_stability_summary.csv", index=False, float_format="%.10g")
    return runs, summaries


def select_candidate(initial_rank: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    ranked = initial_rank.merge(stability, on="candidate_id", how="left")
    ranked["stability_evaluated"] = ranked["mean_pairwise_ari_5_seeds"].notna()
    evaluated = ranked[ranked["stability_evaluated"]].copy()
    evaluated = evaluated.sort_values(
        [
            "all_seed_runs_occupy_four_nodes",
            "mean_pairwise_ari_5_seeds",
            "median_seed_silhouette",
            "minimum_cluster_size_across_seeds",
            "quantization_error_per_sqrt_dimension",
            "n_curves",
        ],
        ascending=[False, False, False, False, True, False],
    ).reset_index(drop=True)
    evaluated["final_unsupervised_rank"] = np.arange(1, len(evaluated) + 1)
    rank_map = evaluated.set_index("candidate_id")["final_unsupervised_rank"]
    ranked["final_unsupervised_rank"] = ranked["candidate_id"].map(rank_map)
    ranked = ranked.sort_values(
        ["stability_evaluated", "final_unsupervised_rank", "initial_unsupervised_rank"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.to_csv(SEARCH_DIR / "candidates_with_stability_and_final_rank.csv", index=False, float_format="%.10g")
    return ranked


def save_selected(
    selected: pd.Series,
    info: dict[str, Any],
    logger: logging.Logger,
) -> tuple[MiniSom, np.ndarray, dict[str, Any]]:
    # Match the author notebook: omit random_seed in the final MiniSom constructor.
    model, labels = train_som(
        info["smoothed"], float(selected["sigma"]), float(selected["learning_rate"]), None
    )
    metrics = evaluate_run(model, info["smoothed"], labels)
    metrics.update(
        {
            "selected_candidate_id": int(selected["candidate_id"]),
            "window_hours": int(selected["window_hours"]),
            "sigma": float(selected["sigma"]),
            "learning_rate": float(selected["learning_rate"]),
            "iterations": CFG.som_iterations,
            "som_x": CFG.som_x,
            "som_y": CFG.som_y,
            "random_seed": None,
            "selection_used_target_labels": False,
            "selection_rule": "lexicographic: 4-node occupancy, seed ARI, silhouette, min cluster size, normalized QE, n",
        }
    )
    save_run_artifacts(FINAL_DIR, model, labels, info, metrics)
    with (FINAL_DIR / "selected_som_model.pkl").open("wb") as f:
        pickle.dump(model, f)

    assignments = pd.read_csv(FINAL_DIR / "cluster_assignments.csv")
    for node in range(4):
        assignments[assignments["som_node"] == node].to_csv(
            FINAL_CLUSTER_DIR / f"anonymous_node_{node}_members.csv", index=False
        )

    time = info["time"]
    table: dict[str, np.ndarray] = {"time_hours": time}
    summary_rows = []
    for node in range(4):
        x, y = divmod(node, 2)
        members = info["smoothed"][labels == node]
        mean = members.mean(axis=0) if len(members) else np.full(len(time), np.nan)
        table[f"node_{node}_member_mean"] = mean
        table[f"node_{node}_som_weight"] = model.get_weights()[x, y]
        summary_rows.append(
            {
                "som_node": node,
                "count": len(members),
                "centroid_start": float(mean[0]) if len(members) else np.nan,
                "centroid_end": float(mean[-1]) if len(members) else np.nan,
                "centroid_peak_time_h": float(time[np.nanargmax(mean)]) if len(members) else np.nan,
                "centroid_trough_time_h": float(time[np.nanargmin(mean)]) if len(members) else np.nan,
            }
        )
    pd.DataFrame(table).to_csv(FINAL_DIR / "anonymous_node_mean_curves_and_weights.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(FINAL_DIR / "anonymous_node_summary.csv", index=False)
    (FINAL_DIR / "selected_run_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_candidate(
        FINAL_DIR / "selected_four_anonymous_som_nodes.png",
        model,
        labels,
        info,
        f"Selected pure-unsupervised SOM: {int(selected['window_hours'])}h, sigma={selected['sigma']}, lr={selected['learning_rate']}",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for node, ax in enumerate(axes.ravel()):
        members = info["smoothed"][labels == node]
        for curve in members:
            ax.plot(time, curve, color=COLORS[node], alpha=0.18, lw=0.7)
        if len(members):
            ax.plot(time, members.mean(axis=0), color="black", lw=2.3)
        ax.set_title(f"Anonymous Node {node}: all members (n={len(members)})")
        ax.set_ylabel("Normalized PCE")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (h)")
    fig.suptitle("All curves in the automatically selected unsupervised result", fontsize=14)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "selected_nodes_all_member_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(model.distance_map(), cmap="viridis")
    for node in range(4):
        x, y = divmod(node, 2)
        ax.text(y, x, f"Node {node}", color="white", weight="bold", ha="center", va="center")
    ax.set_title("Selected SOM U-matrix")
    fig.colorbar(im, ax=ax, label="Normalized neighbor distance")
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "selected_som_u_matrix.png", dpi=220)
    plt.close(fig)
    logger.info("Selected candidate %d and reran with random_seed=None", int(selected["candidate_id"]))
    return model, labels, metrics


def plot_search_summary(ranked: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for (sigma, lr), group in ranked.groupby(["sigma", "learning_rate"]):
        group = group.sort_values("window_hours")
        label = f"σ={sigma}, lr={lr}"
        axes[0].plot(group["window_hours"], group["silhouette"], marker="o", label=label)
        axes[1].plot(
            group["window_hours"],
            group["quantization_error_per_sqrt_dimension"],
            marker="o",
            label=label,
        )
    stable = ranked[ranked["stability_evaluated"]].sort_values("initial_unsupervised_rank")
    axes[2].bar(
        stable["candidate_id"].astype(str),
        stable["mean_pairwise_ari_5_seeds"],
        color="#4C78A8",
    )
    axes[0].set(title="Silhouette", xlabel="Window (h)", ylabel="Score")
    axes[1].set(title="QE / sqrt(input dimensions)", xlabel="Window (h)", ylabel="Normalized QE")
    axes[2].set(title="Top candidates: 5-seed stability", xlabel="Candidate", ylabel="Mean pairwise ARI")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(SEARCH_DIR / "unsupervised_search_summary.png", dpi=220)
    plt.close(fig)


def write_report(
    audit: pd.DataFrame,
    ranked: pd.DataFrame,
    selected: pd.Series,
    final_metrics: dict[str, Any],
) -> None:
    node_counts = [final_metrics[f"cluster_{i}_size"] for i in range(4)]
    top_table = ranked[ranked["stability_evaluated"]].head(9)[
        [
            "candidate_id",
            "window_hours",
            "sigma",
            "learning_rate",
            "silhouette",
            "mean_pairwise_ari_5_seeds",
            "min_cluster_size_including_empty",
        ]
    ]
    table_lines = [
        "|ID|窗口(h)|sigma|learning rate|silhouette|5-seed ARI|最小簇|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_table.itertuples(index=False):
        table_lines.append(
            f"|{int(r.candidate_id)}|{int(r.window_hours)}|{r.sigma:.1f}|{r.learning_rate:.1f}|{r.silhouette:.4f}|{r.mean_pairwise_ari_5_seeds:.4f}|{int(r.min_cluster_size_including_empty)}|"
        )
    report = f"""# 纯无监督论文参数微调实验报告

## 最终结果

在 54 个只含原文变量的候选中，自动选择 Candidate {int(selected['candidate_id'])}：时间窗口 **{int(selected['window_hours'])} h**、sigma **{selected['sigma']:.1f}**、learning rate **{selected['learning_rate']:.1f}**。最终按作者 notebook 不指定随机种子的方式重新训练，四个匿名节点规模为 **{node_counts}**，silhouette={final_metrics['silhouette']:.4f}，quantization error={final_metrics['quantization_error']:.4f}。

这些曲线图来自纯 SOM 输出，形态可与原文的 initial gain / slow / medium / fast exponential decay 图直接比较。程序没有为了得到某种形状而命名、移动或重分配任何样本。

## 无监督保证

- 输入矩阵只有按时间排列的归一化 PCE/效率/功率值。
- 没有类别标签、目标模板、峰谷阈值、导数特征或人工挑选。
- 参数搜索只改变文献已有变量：窗口、sigma、learning rate。
- 预处理、2×2 SOM、MiniSom 2.2.9 和 50,000 次顺序训练保持原文设置。
- 排序为预先写死的无监督字典序：四节点占用 → seed ARI → silhouette → 最小簇 → 归一化 QE → 样本数。没有依据目标图形打分。

## 数据审计

- 指定路径 CSV：{len(audit)} 条。
- 轴语义和质量合格：{int(audit['axis_valid'].sum())} 条。
- 各窗口只使用能够完整覆盖该窗口的曲线，未做外推。

## 稳定性复核候选

{chr(10).join(table_lines)}

## 文件导航

- `01_data_audit/`：全部 218 条输入的纳入/排除依据。
- `02_paper_preprocessing/`：每个窗口的原始、归一化、平滑矩阵。
- `03_unsupervised_parameter_search/`：54 个候选的指标、权重、分配及节点图。
- `04_seed_stability/`：前九候选的五随机种子复算。
- `05_selected_unsupervised_result/`：自动选择后、按原文 seed=None 重训的最终结果。
- `06_validation/`：独立检查结果。

## 解释限制

本实验可以证明这些匿名形态由无监督 SOM 在当前数据中产生；但“与论文图形不同”是训练后的科学解释，不是一个可参与模型选择的监督标签。若某个期望形态没有出现，程序不会人工补造。
"""
    (ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "README_CN.md").write_text(
        "# pure_unsupervised_paper_som_window_search\n\n"
        "纯无监督、只微调论文已有变量的 SOM 实验。先阅读 `REPORT_CN.md`；"
        "复现运行 `python3 run_pipeline.py`，核验运行 `python3 verify_outputs.py`。\n",
        encoding="utf-8",
    )


def main() -> None:
    warnings.filterwarnings("ignore", message="The topographic error is not defined")
    ensure_dirs()
    logger = configure_logging()
    write_provenance()
    audit, curves = audit_dataset(logger)
    if len(audit) != CFG.expected_input_csv or int(audit["axis_valid"].sum()) != CFG.expected_axis_valid:
        raise RuntimeError("Input audit counts differ from the expected immutable dataset")
    window_data = {w: preprocess_window(w, audit, curves, logger) for w in CFG.windows_hours}
    initial_rank, _ = run_search(window_data, logger)
    _, stability = run_stability(initial_rank, window_data, logger)
    ranked = select_candidate(initial_rank, stability)
    plot_search_summary(ranked)
    selected = ranked[ranked["final_unsupervised_rank"] == 1].iloc[0]
    _, _, final_metrics = save_selected(selected, window_data[int(selected["window_hours"])], logger)
    write_report(audit, ranked, selected, final_metrics)
    print(
        json.dumps(
            {
                "selected_candidate": int(selected["candidate_id"]),
                "window_hours": int(selected["window_hours"]),
                "sigma": float(selected["sigma"]),
                "learning_rate": float(selected["learning_rate"]),
                "final_metrics": final_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
