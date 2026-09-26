#!/usr/bin/env python3
"""Strict literature-bounded SOM cluster-count search on samples_test.

The training matrix contains only normalized PCE/efficiency values over time.
IFO labels, target templates, derivatives, peak/valley features, and supervised
objectives are deliberately absent from preprocessing, training, and K choice.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INPUT_ROOT = REPO / "lab-v2" / "som_references" / "accepted" / "samples_test"
THESIS_ROOT = REPO / "thesis"
VENDOR_MINISOM = (
    REPO
    / "lab-v2"
    / "pure_unsupervised_paper_som_window_search"
    / "vendor"
    / "minisom_2_2_9"
)
sys.path.insert(0, str(VENDOR_MINISOM))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from minisom import MiniSom
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Config:
    window_hours: int = 150
    interval_minutes: int = 10
    savgol_window: int = 71
    savgol_order: int = 2
    iterations: int = 50_000
    comparison_seed: int = 0
    n_values: tuple[int, ...] = tuple(range(2, 11))
    parameter_pairs: tuple[tuple[float, float], ...] = (
        (0.5, 0.1),
        (0.3, 0.1),
        (0.5, 0.3),
    )
    minimum_unique_points: int = 4


CFG = Config()
LAYOUTS: dict[int, tuple[int, int]] = {
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (1, 5),
    6: (2, 3),
    7: (1, 7),
    8: (2, 4),
    9: (3, 3),
    10: (2, 5),
    16: (4, 4),
}
COLORS = plt.get_cmap("tab20")

METHOD_DIR = ROOT / "00_method_boundary"
AUDIT_DIR = ROOT / "01_data_audit"
PREP_DIR = ROOT / "02_preprocessed_150h"
SEARCH_DIR = ROOT / "03_k_search_n2_10"
N16_DIR = ROOT / "04_n16_exploratory_not_selected"
VALIDATION_DIR = ROOT / "05_validation"


def ensure_dirs() -> None:
    for path in (METHOD_DIR, AUDIT_DIR, PREP_DIR, SEARCH_DIR, N16_DIR, VALIDATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def axis_decision(metadata: dict[str, Any]) -> tuple[bool, float | None, str]:
    axis = metadata.get("axis") or {}
    x_axis = axis.get("x") or {}
    y_axis = axis.get("y") or {}
    x_name = clean_text(x_axis.get("name"))
    x_unit = clean_text(x_axis.get("unit"))
    y_name = clean_text(y_axis.get("name"))
    x_text = f"{x_name} {x_unit}"

    if "cycle" in x_text:
        return False, None, "x_axis_is_cycles"
    if not ("time" in x_text or "duration" in x_text):
        return False, None, "x_axis_not_explicit_time"
    if not ("pce" in y_name or "efficien" in y_name):
        return False, None, "y_axis_not_explicit_pce_or_efficiency"
    if re.search(r"\bdays?\b|\(d\)|^d$", x_unit):
        return True, 24.0, "days_to_hours"
    if re.search(r"\bhours?\b|\bhrs?\b|\(h\)|^h$", x_unit) or "hour" in x_name:
        return True, 1.0, "hours"
    return False, None, "time_unit_not_explicit_hour_or_day"


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    frame = pd.read_csv(path, usecols=["x", "y"])
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["x", "y"])
    duplicate_rows = int(frame.duplicated("x", keep=False).sum())
    # The source workflow explicitly resamples with mean aggregation. Averaging
    # repeated digitized x positions is therefore a documented preprocessing act.
    frame = frame.groupby("x", as_index=False, sort=True)["y"].mean()
    return frame["x"].to_numpy(float), frame["y"].to_numpy(float), duplicate_rows


def curve_id(path: Path) -> str:
    rel = path.relative_to(INPUT_ROOT).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{path.parents[1].name}_{digest}"


def audit_input() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    usable: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for validation_path in sorted(INPUT_ROOT.rglob("validation_result.json")):
        metadata = json.loads(validation_path.read_text(encoding="utf-8"))
        axis = metadata.get("axis") or {}
        x_axis = axis.get("x") or {}
        y_axis = axis.get("y") or {}
        axis_ok, factor, axis_status = axis_decision(metadata)
        record = validation_path.parent.name

        for csv_path in sorted((validation_path.parent / "accepted").glob("*.csv")):
            cid = curve_id(csv_path)
            reasons: list[str] = []
            try:
                x, y, duplicate_rows = read_curve(csv_path)
            except Exception as exc:
                x, y, duplicate_rows = np.array([]), np.array([]), 0
                reasons.append(f"csv_read_error:{type(exc).__name__}")

            if not axis_ok:
                reasons.append(axis_status)
            if record == "图片140":
                reasons.append("record_140_implausible_hour_scale")
            if record == "图片35":
                reasons.append("record_35_no_observed_early_0_150h_segment")
            if len(x) < CFG.minimum_unique_points:
                reasons.append("fewer_than_4_unique_points")
            if len(y) and not (np.isfinite(y).all() and float(np.max(y)) > 0):
                reasons.append("nonfinite_or_nonpositive_series")
            if len(y) and float(np.ptp(y)) <= 1e-12:
                reasons.append("constant_series")

            duration_h = np.nan
            x_min_h = np.nan
            x_max_h = np.nan
            if factor is not None and len(x):
                x_h = x * factor
                x_min_h = float(x_h.min())
                x_max_h = float(x_h.max())
                duration_h = float(x_h.max() - x_h.min())
                if duration_h < CFG.window_hours:
                    reasons.append("duration_shorter_than_150h")

            included = len(reasons) == 0
            if included:
                assert factor is not None
                x_h = x * factor
                x_relative = x_h - x_h.min()
                usable[cid] = (x_relative, y)

            rows.append(
                {
                    "curve_id": cid,
                    "included_strict_150h": included,
                    "exclusion_reasons": ";".join(reasons),
                    "source_file": csv_path.relative_to(REPO).as_posix(),
                    "record": record,
                    "series_name": csv_path.stem,
                    "x_name": x_axis.get("name"),
                    "x_unit": x_axis.get("unit"),
                    "y_name": y_axis.get("name"),
                    "y_unit": y_axis.get("unit"),
                    "time_factor_to_hours": factor,
                    "n_unique_points_after_mean": len(x),
                    "duplicate_x_rows_aggregated_by_mean": duplicate_rows,
                    "x_min_hours_before_rezero": x_min_h,
                    "x_max_hours_before_rezero": x_max_h,
                    "duration_hours": duration_h,
                }
            )

    audit = pd.DataFrame(rows).sort_values(["record", "series_name"]).reset_index(drop=True)
    audit.to_csv(AUDIT_DIR / "all_218_curves_audit.csv", index=False)
    audit[audit["included_strict_150h"]].to_csv(AUDIT_DIR / "included_strict_150h.csv", index=False)
    audit[~audit["included_strict_150h"]].to_csv(AUDIT_DIR / "excluded_with_reasons.csv", index=False)
    return audit, usable


def preprocess(
    audit: pd.DataFrame, curves: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    metadata = (
        audit[audit["included_strict_150h"]]
        .sort_values("curve_id")
        .reset_index(drop=True)
        .copy()
    )
    # Matches the released author array: 150*6 values after dropping the first
    # resampled row, i.e. 1/6 h through 150 h, not 901 values including zero.
    time_h = np.arange(1, CFG.window_hours * 6 + 1, dtype=float) / 6.0
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    smoothed_rows: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []

    for row in metadata.itertuples(index=False):
        x, y = curves[row.curve_id]
        # SciPy 1.7 (the available runtime) has no ``extrapolate`` keyword.
        # The audited relative domain already spans the complete 0-150 h grid;
        # the subsequent finite-value assertion prevents silent extrapolation.
        interpolator = Akima1DInterpolator(x, y)
        raw = np.asarray(interpolator(time_h), dtype=float)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Akima produced non-finite values for {row.curve_id}")
        scale = float(np.max(np.abs(raw)))
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"Invalid MaxAbs scale for {row.curve_id}")
        normalized = raw / scale
        smoothed = savgol_filter(normalized, CFG.savgol_window, CFG.savgol_order)
        raw_rows.append(raw)
        normalized_rows.append(normalized)
        smoothed_rows.append(smoothed)
        qc_rows.append(
            {
                "curve_id": row.curve_id,
                "n_time_points": len(time_h),
                "time_start_h": time_h[0],
                "time_end_h": time_h[-1],
                "max_abs_before_normalization": scale,
                "max_abs_after_normalization": float(np.max(np.abs(normalized))),
                "all_finite_after_smoothing": bool(np.isfinite(smoothed).all()),
            }
        )

    result = {
        "metadata": metadata,
        "time_h": time_h,
        "raw": np.vstack(raw_rows),
        "normalized": np.vstack(normalized_rows),
        "smoothed": np.vstack(smoothed_rows),
    }
    metadata.to_csv(PREP_DIR / "curve_metadata.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(PREP_DIR / "preprocessing_qc.csv", index=False)
    np.savez_compressed(
        PREP_DIR / "strict_paper_preprocessed_150h.npz",
        time_hours=time_h,
        raw=result["raw"],
        normalized=result["normalized"],
        smoothed=result["smoothed"],
        curve_ids=metadata["curve_id"].to_numpy(str),
    )
    return result


def flat_labels(model: MiniSom, data: np.ndarray, layout: tuple[int, int]) -> np.ndarray:
    winners = [model.winner(row) for row in data]
    return np.asarray([np.ravel_multi_index(w, layout) for w in winners], dtype=int)


def train_run(
    data: np.ndarray,
    n: int,
    sigma: float,
    learning_rate: float,
    seed: int | None,
) -> tuple[MiniSom, np.ndarray]:
    x_size, y_size = LAYOUTS[n]
    kwargs: dict[str, Any] = {
        "x": x_size,
        "y": y_size,
        "input_len": data.shape[1],
        "sigma": sigma,
        "learning_rate": learning_rate,
    }
    if seed is not None:
        kwargs["random_seed"] = seed
    model = MiniSom(**kwargs)
    model.random_weights_init(data)
    model.train(data, CFG.iterations, verbose=False)
    return model, flat_labels(model, data, (x_size, y_size))


def direct_quantization_error(model: MiniSom, data: np.ndarray) -> float:
    weights = model.get_weights().reshape(-1, data.shape[1])
    distances = np.linalg.norm(data[:, None, :] - weights[None, :, :], axis=2)
    return float(np.min(distances, axis=1).mean())


def member_centroids(data: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    rows = []
    for node in range(n):
        members = data[labels == node]
        rows.append(members.mean(axis=0) if len(members) else np.full(data.shape[1], np.nan))
    return np.vstack(rows)


def save_run(
    run_dir: Path,
    model: MiniSom,
    labels: np.ndarray,
    prepared: dict[str, Any],
    title: str,
    config_record: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    data = prepared["smoothed"]
    time_h = prepared["time_h"]
    metadata = prepared["metadata"]
    n = int(config_record["n_clusters"])
    layout = tuple(config_record["layout"])
    weights = model.get_weights().reshape(n, data.shape[1])
    centroids = member_centroids(data, labels, n)
    counts = np.bincount(labels, minlength=n)
    qe = direct_quantization_error(model, data)
    occupied = int(np.count_nonzero(counts))

    metrics = {
        **config_record,
        "n_input_curves": int(len(data)),
        "n_dimensions": int(data.shape[1]),
        "occupied_nodes": occupied,
        "quantization_error": qe,
        "quantization_error_per_sqrt_dimension": qe / float(np.sqrt(data.shape[1])),
        "cluster_sizes": counts.tolist(),
        "minimum_occupied_cluster_size": int(counts[counts > 0].min()) if occupied else 0,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(run_dir / "som_weights.npy", model.get_weights())
    np.save(run_dir / "som_labels.npy", labels)
    np.save(run_dir / "member_mean_centroids.npy", centroids)
    with (run_dir / "som_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    assignments = metadata[
        ["curve_id", "source_file", "record", "series_name", "x_name", "x_unit", "y_name", "y_unit"]
    ].copy()
    assignments["anonymous_som_node"] = labels
    assignments["node_x"] = labels // layout[1]
    assignments["node_y"] = labels % layout[1]
    assignments.to_csv(run_dir / "cluster_assignments.csv", index=False)

    curve_frame: dict[str, Any] = {"time_hours": time_h}
    for node in range(n):
        curve_frame[f"node_{node}_member_mean"] = centroids[node]
        curve_frame[f"node_{node}_som_weight"] = weights[node]
    pd.DataFrame(curve_frame).to_csv(run_dir / "anonymous_node_curves.csv", index=False)
    pd.DataFrame(
        {
            "anonymous_som_node": np.arange(n),
            "count": counts,
            "node_x": np.arange(n) // layout[1],
            "node_y": np.arange(n) % layout[1],
        }
    ).to_csv(run_dir / "node_counts.csv", index=False)

    rows, cols = layout
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(8.0, 3.2 * cols), max(3.2, 2.8 * rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for node, ax in enumerate(axes.ravel()):
        members = data[labels == node]
        color = COLORS(node % 20)
        step = max(1, len(members) // 45)
        for curve in members[::step]:
            ax.plot(time_h, curve, color=color, alpha=0.13, lw=0.65)
        if len(members):
            ax.plot(time_h, centroids[node], color=color, lw=2.4, label="member mean")
        ax.plot(time_h, weights[node], color="black", lw=1.15, ls="--", label="SOM weight")
        ax.set_title(f"Anonymous node {node} (n={len(members)})", fontsize=9)
        ax.grid(alpha=0.2)
        if node % cols == 0:
            ax.set_ylabel("Normalized PCE")
        if node // cols == rows - 1:
            ax.set_xlabel("Time (h)")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "anonymous_nodes_all_members_and_centers.png", dpi=210)
    plt.close(fig)
    return metrics


def write_provenance(audit: pd.DataFrame, prepared: dict[str, Any]) -> None:
    config = asdict(CFG)
    for key, value in list(config.items()):
        if isinstance(value, tuple):
            config[key] = [list(v) if isinstance(v, tuple) else v for v in value]
    config.update(
        {
            "layouts": {str(k): list(v) for k, v in LAYOUTS.items()},
            "input_root": str(INPUT_ROOT),
            "strict_input_count": int(len(prepared["metadata"])),
            "model_input_variables": ["normalized PCE/efficiency values on the 10-minute time grid"],
            "explicitly_not_used": [
                "IFO target labels",
                "shape templates",
                "derivatives or peak/valley features",
                "supervised classifier",
                "extra material/device variables",
                "window length search",
                "seed search",
            ],
            "n16_status": "paper-reported exploratory resolution; excluded from K elbow and final K selection",
        }
    )
    (ROOT / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "minisom": "2.2.9 vendored exact source",
        "minisom_module": str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    (ROOT / "environment_actual.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reasons = (
        audit.loc[~audit["included_strict_150h"], "exclusion_reasons"]
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict()
    )
    boundary = f"""# 文献参数边界与类别数判据

## 固定流程

- 仅使用明确的 PCE/efficiency-time 曲线；严格窗口为前 150 h。
- 10 min 重采样，Akima 插值，逐曲线 MaxAbs 归一化，Savitzky-Golay `(71, 2)`。
- MiniSom 2.2.9，`random_weights_init`，50,000 次顺序训练。
- 实际矩阵每条 900 点，对应 `1/6 h ... 150 h`；这与作者发布的 `(2245, 900)` 数组一致。

## 唯一允许搜索的论文参数

- 类别数 `n=2...10`。
- 三组 `(sigma, learning_rate)`：`(0.5,0.1)`、`(0.3,0.1)`、`(0.5,0.3)`。
- `seed=0` 只用于让候选公平复算，不作为搜索变量。

## 类别数规则

论文规则不是固定 K=4，也不是一个额外的自动指标：先看 quantization-error elbow，
再检查较小 K 是否欠分、较大 K 的中心是否开始重叠或不可区分，选择仍能捕获主形态的最小 K。
IFO-Bridge/Hill/Slope/Valley 不参与训练、排序或 K 选择，只能在结果冻结后事后解释。

`n=16` 仅因为 Supplementary Fig. 1 明确展示而作为探索对照；它不进入 `n=2...10`
肘部，也不能被冒充为论文最优类别数。

## 数据审计

- 总 CSV：{len(audit)}。
- 严格纳入：{len(prepared['metadata'])}。
- 排除原因计数（同一曲线可有多项）：`{json.dumps(reasons, ensure_ascii=False)}`。

## 复现歧义

论文没有完整列出每个 n 的 SOM 网格形状。这里采用与补图一致且尽量接近方形的因子布局：
`{json.dumps({k: v for k, v in LAYOUTS.items() if k <= 10})}`。布局会影响拓扑，必须与结果一起报告。

## 文献证据

- `thesis/paper/work.md`：Data analysis、Degradation curve shape clustering。
- `thesis/paper/Supplementary.md`：Supplementary Figs. 6、7、15-17 与 SOM Quantisation Error。
"""
    (METHOD_DIR / "METHOD_BOUNDARY_CN.md").write_text(boundary, encoding="utf-8")


def run_all(prepared: dict[str, Any]) -> pd.DataFrame:
    data = prepared["smoothed"]
    rows: list[dict[str, Any]] = []
    for pair_index, (sigma, learning_rate) in enumerate(CFG.parameter_pairs, start=1):
        pair_dir = SEARCH_DIR / f"pair_{pair_index}_sigma_{sigma:g}_lr_{learning_rate:g}"
        for n in CFG.n_values:
            layout = LAYOUTS[n]
            model, labels = train_run(data, n, sigma, learning_rate, CFG.comparison_seed)
            record = {
                "parameter_pair_id": pair_index,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "n_clusters": n,
                "layout": list(layout),
                "iterations": CFG.iterations,
                "comparison_seed": CFG.comparison_seed,
                "selection_role": "paper_K_search_n2_10",
            }
            metrics = save_run(
                pair_dir / f"n_{n:02d}_{layout[0]}x{layout[1]}",
                model,
                labels,
                prepared,
                f"n={n}, layout={layout[0]}x{layout[1]}, sigma={sigma:g}, lr={learning_rate:g}",
                record,
            )
            rows.append(metrics)
            print(
                f"pair {pair_index}/3 n={n}: QE={metrics['quantization_error']:.6f}, "
                f"occupied={metrics['occupied_nodes']}, sizes={metrics['cluster_sizes']}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame["layout"] = frame["layout"].apply(lambda value: "x".join(map(str, value)))
    frame["cluster_sizes"] = frame["cluster_sizes"].apply(json.dumps)
    frame.to_csv(SEARCH_DIR / "all_27_run_metrics.csv", index=False, float_format="%.10g")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for ax, (pair_index, group) in zip(axes, frame.groupby("parameter_pair_id", sort=True)):
        group = group.sort_values("n_clusters")
        ax.plot(group["n_clusters"], group["quantization_error"], "o-", color="#2878B5")
        ax.set_xticks(CFG.n_values)
        ax.set_xlabel("Number of SOM nodes (n)")
        ax.set_ylabel("Quantization error")
        row = group.iloc[0]
        ax.set_title(f"sigma={row.sigma:g}, learning rate={row.learning_rate:g}")
        ax.grid(alpha=0.25)
    fig.suptitle("Paper-rule evidence: inspect elbow first, then centroid overlap")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(SEARCH_DIR / "quantization_error_elbows_three_paper_pairs.png", dpi=240)
    plt.close(fig)
    return frame


def run_n16(prepared: dict[str, Any]) -> None:
    sigma, learning_rate = CFG.parameter_pairs[0]
    n = 16
    model, labels = train_run(
        prepared["smoothed"], n, sigma, learning_rate, CFG.comparison_seed
    )
    record = {
        "parameter_pair_id": 1,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "n_clusters": n,
        "layout": list(LAYOUTS[n]),
        "iterations": CFG.iterations,
        "comparison_seed": CFG.comparison_seed,
        "selection_role": "exploratory_only_not_part_of_K_selection",
    }
    save_run(
        N16_DIR,
        model,
        labels,
        prepared,
        "Literature-reported n=16 exploratory map (NOT an optimum-K result)",
        record,
    )


def write_initial_report(audit: pd.DataFrame, metrics: pd.DataFrame) -> None:
    baseline = metrics[metrics["parameter_pair_id"] == 1].sort_values("n_clusters")
    table_lines = [
        "|n|layout|quantization error|occupied nodes|cluster sizes|",
        "|---:|:---:|---:|---:|:---|",
    ]
    for row in baseline.itertuples(index=False):
        table_lines.append(
            f"|{int(row.n_clusters)}|{row.layout}|{row.quantization_error:.6f}|"
            f"{int(row.occupied_nodes)}|`{row.cluster_sizes}`|"
        )
    table = "\n".join(table_lines)
    report = f"""# 严格文献参数 SOM 类别数实验

## 当前状态

已完成数据审计、固定论文预处理、`n=2...10 × 3` 组论文参数的 27 次 SOM，及一个
不参与最优 K 选择的 `n=16` 文献探索对照。

总输入 {len(audit)} 条，严格纳入 {int(audit['included_strict_150h'].sum())} 条。模型输入只有
900 个按时间排列的归一化 PCE/efficiency 值；没有 IFO 标签、目标模板、额外变量或形态特征。

## 基线参数的量化误差证据

{table}

类别数尚未在程序内自动宣布。原因是原论文规则本身要求：先目视肘部，再比较候选 K 的
中心曲线是否欠分或开始重叠。人工复核决定将单独写入 `K_SELECTION_DECISION_CN.md`，并且
不能使用四个 IFO 目标反向选择 K。

## 目录

- `01_data_audit/`：218 条曲线逐条纳入/排除原因。
- `02_preprocessed_150h/`：严格输入、900点原值/归一化/平滑矩阵与QC。
- `03_k_search_n2_10/`：27 次训练的权重、分配、中心、成员图和QE肘图。
- `04_n16_exploratory_not_selected/`：只作为 Supplementary Fig.1 对照。
"""
    (ROOT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "README_CN.md").write_text(
        "# literature_k_som_search\n\n先阅读 `REPORT_CN.md` 与 `00_method_boundary/METHOD_BOUNDARY_CN.md`。\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    audit, curves = audit_input()
    prepared = preprocess(audit, curves)
    write_provenance(audit, prepared)
    print(f"strict input: {len(prepared['metadata'])} / {len(audit)} curves", flush=True)
    metrics = run_all(prepared)
    run_n16(prepared)
    write_initial_report(audit, metrics)
    print("Completed strict literature-bounded K search.", flush=True)


if __name__ == "__main__":
    main()
