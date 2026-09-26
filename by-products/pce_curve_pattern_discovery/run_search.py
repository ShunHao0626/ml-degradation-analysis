#!/usr/bin/env python3
"""Literature-bounded SOM search for digitized PCE-time curves.

The model sees only normalized PCE/efficiency values on the time grid.  Target
IFO labels, target templates, derivatives, material metadata, and engineered
shape features are deliberately absent from preprocessing, training, and K
selection.
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
REPO = ROOT.parent
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
    k_values: tuple[int, ...] = tuple(range(2, 11))
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

SCOPE_DIR = ROOT / "00_scope"
AUDIT_DIR = ROOT / "01_data_audit"
PREP_DIR = ROOT / "02_preprocessed_150h"
SEARCH_DIR = ROOT / "03_som_search"
SELECTION_DIR = ROOT / "04_k_selection"
VALIDATION_DIR = ROOT / "08_validation"


def ensure_dirs() -> None:
    for path in (
        SCOPE_DIR,
        AUDIT_DIR,
        PREP_DIR,
        SEARCH_DIR,
        SELECTION_DIR,
        VALIDATION_DIR,
        ROOT / "logs",
    ):
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
            # These two record-level exclusions are axis/coverage QC only and
            # were fixed before any SOM run or IFO inspection.
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

            x_min_h = x_max_h = duration_h = np.nan
            if factor is not None and len(x):
                x_h = x * factor
                x_min_h = float(x_h.min())
                x_max_h = float(x_h.max())
                duration_h = float(x_h.max() - x_h.min())
                if duration_h < CFG.window_hours:
                    reasons.append("duration_shorter_than_150h")

            included = not reasons
            if included:
                assert factor is not None
                x_h = x * factor
                usable[cid] = (x_h - x_h.min(), y)

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
    audit.to_csv(AUDIT_DIR / "all_input_curves.csv", index=False)
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
    time_h = np.arange(1, CFG.window_hours * 6 + 1, dtype=float) / 6.0
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    smoothed_rows: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []

    for row in metadata.itertuples(index=False):
        x, y = curves[row.curve_id]
        raw = np.asarray(Akima1DInterpolator(x, y)(time_h), dtype=float)
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

    prepared = {
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
        raw=prepared["raw"],
        normalized=prepared["normalized"],
        smoothed=prepared["smoothed"],
        curve_ids=metadata["curve_id"].to_numpy(str),
    )
    return prepared


def flat_labels(model: MiniSom, data: np.ndarray, layout: tuple[int, int]) -> np.ndarray:
    winners = [model.winner(row) for row in data]
    return np.asarray([np.ravel_multi_index(w, layout) for w in winners], dtype=int)


def train(data: np.ndarray, k: int, sigma: float, learning_rate: float) -> tuple[MiniSom, np.ndarray]:
    rows, cols = LAYOUTS[k]
    model = MiniSom(
        rows,
        cols,
        data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=CFG.comparison_seed,
    )
    model.random_weights_init(data)
    model.train(data, CFG.iterations, random_order=False, verbose=False)
    return model, flat_labels(model, data, (rows, cols))


def member_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack(
        [data[labels == node].mean(axis=0) if np.any(labels == node) else np.full(data.shape[1], np.nan) for node in range(k)]
    )


def direct_qe(model: MiniSom, data: np.ndarray) -> float:
    weights = model.get_weights().reshape(-1, data.shape[1])
    distances = np.linalg.norm(data[:, None, :] - weights[None, :, :], axis=2)
    return float(np.min(distances, axis=1).mean())


def minimum_centroid_rmse(centroids: np.ndarray) -> float:
    valid = centroids[np.isfinite(centroids).all(axis=1)]
    if len(valid) < 2:
        return float("nan")
    values = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            values.append(float(np.sqrt(np.mean((valid[i] - valid[j]) ** 2))))
    return min(values)


def plot_run(
    path: Path,
    time_h: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    layout: tuple[int, int],
    title: str,
) -> None:
    rows, cols = layout
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(max(8.0, cols * 3.0), max(3.2, rows * 2.8)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for node, ax in enumerate(axes.ravel()):
        members = data[labels == node]
        step = max(1, len(members) // 40)
        for curve in members[::step]:
            ax.plot(time_h, curve, color="#7b8da0", alpha=0.14, lw=0.6)
        if len(members):
            ax.plot(time_h, centroids[node], color="#c51b7d", lw=2.2, label="member mean")
        ax.plot(time_h, weights[node], color="black", ls="--", lw=1.0, label="SOM weight")
        ax.set_title(f"Anonymous node {node} (n={len(members)})", fontsize=9)
        ax.grid(alpha=0.2)
        if node % cols == 0:
            ax.set_ylabel("Normalized PCE")
        if node // cols == rows - 1:
            ax.set_xlabel("Time (h)")
    handles, labels_text = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_text, loc="upper right", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def run_search(prepared: dict[str, Any]) -> pd.DataFrame:
    data = prepared["smoothed"]
    rows: list[dict[str, Any]] = []
    for pair_id, (sigma, learning_rate) in enumerate(CFG.parameter_pairs, start=1):
        pair_dir = SEARCH_DIR / f"pair_{pair_id}_sigma_{sigma:g}_lr_{learning_rate:g}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        for k in CFG.k_values:
            layout = LAYOUTS[k]
            model, labels = train(data, k, sigma, learning_rate)
            weights = model.get_weights().reshape(k, data.shape[1])
            centroids = member_centroids(data, labels, k)
            counts = np.bincount(labels, minlength=k)
            record = {
                "pair_id": pair_id,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "k": k,
                "layout": f"{layout[0]}x{layout[1]}",
                "quantization_error": direct_qe(model, data),
                "qe_per_sqrt_dimension": direct_qe(model, data) / np.sqrt(data.shape[1]),
                "occupied_nodes": int(np.count_nonzero(counts)),
                "minimum_occupied_node_size": int(counts[counts > 0].min()),
                "minimum_member_centroid_rmse": minimum_centroid_rmse(centroids),
                "cluster_sizes": json.dumps(counts.tolist()),
            }
            rows.append(record)
            run_dir = pair_dir / f"k_{k:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                run_dir / "run_arrays.npz",
                labels=labels,
                weights=weights,
                member_centroids=centroids,
                counts=counts,
            )
            with (run_dir / "som_model.pkl").open("wb") as handle:
                pickle.dump(model, handle)
            (run_dir / "metrics.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            plot_run(
                run_dir / "anonymous_nodes.png",
                prepared["time_h"],
                data,
                labels,
                centroids,
                weights,
                layout,
                f"K={k}, sigma={sigma:g}, learning rate={learning_rate:g}",
            )

    metrics = pd.DataFrame(rows).sort_values(["pair_id", "k"]).reset_index(drop=True)
    metrics["qe_drop_from_previous_percent"] = metrics.groupby("pair_id")["quantization_error"].pct_change().mul(-100)
    metrics["qe_drop_to_next_percent"] = metrics.groupby("pair_id")["quantization_error"].shift(-1).sub(metrics["quantization_error"]).div(metrics["quantization_error"]).mul(-100)
    metrics.to_csv(SEARCH_DIR / "all_27_run_metrics.csv", index=False)
    metrics.to_csv(SELECTION_DIR / "k_selection_evidence.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for pair_id, group in metrics.groupby("pair_id"):
        label = f"pair {pair_id}: sigma={group.sigma.iloc[0]:g}, lr={group.learning_rate.iloc[0]:g}"
        axes[0].plot(group["k"], group["quantization_error"], marker="o", label=label)
        axes[1].plot(group["k"], group["minimum_member_centroid_rmse"], marker="o", label=label)
    axes[0].set(xlabel="K", ylabel="Quantization error", title="Paper-rule elbow evidence")
    axes[1].set(xlabel="K", ylabel="Minimum centroid RMSE", title="Indistinguishability diagnostic")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_xticks(list(CFG.k_values))
    fig.tight_layout()
    fig.savefig(SELECTION_DIR / "qe_elbow_and_overlap_diagnostic.png", dpi=220)
    plt.close(fig)
    return metrics


def run_n16_exploratory(prepared: dict[str, Any]) -> None:
    data = prepared["smoothed"]
    model, labels = train(data, 16, 0.5, 0.1)
    weights = model.get_weights().reshape(16, data.shape[1])
    centroids = member_centroids(data, labels, 16)
    counts = np.bincount(labels, minlength=16)
    out = SEARCH_DIR / "n16_paper_exploratory_not_for_k_selection"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "run_arrays.npz", labels=labels, weights=weights, member_centroids=centroids, counts=counts)
    pd.DataFrame({"anonymous_node": np.arange(16), "count": counts}).to_csv(out / "node_counts.csv", index=False)
    plot_run(
        out / "anonymous_nodes.png",
        prepared["time_h"],
        data,
        labels,
        centroids,
        weights,
        LAYOUTS[16],
        "K=16 paper exploratory resolution; excluded from final K selection",
    )


def write_scope(audit: pd.DataFrame, prepared: dict[str, Any]) -> None:
    config = asdict(CFG)
    config["k_values"] = list(CFG.k_values)
    config["parameter_pairs"] = [list(v) for v in CFG.parameter_pairs]
    config["layouts"] = {str(k): list(v) for k, v in LAYOUTS.items()}
    config["input_root"] = str(INPUT_ROOT)
    config["included_curve_count"] = int(len(prepared["metadata"]))
    config["model_input"] = "only normalized PCE/efficiency values over time"
    config["forbidden_and_unused"] = [
        "IFO target labels or templates",
        "derivatives, extrema, or engineered shape variables",
        "material/device/testing-condition variables",
        "supervised labels or classifiers",
        "algorithm replacement",
        "random-seed search",
    ]
    (ROOT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    reason_counts = (
        audit.loc[~audit["included_strict_150h"], "exclusion_reasons"]
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict()
    )
    scope = f"""# 方法边界与参数白名单

## 固定流程

- 输入根目录：`{INPUT_ROOT}`。
- 仅纳入横轴明确为 hour/day 时间且纵轴明确为 PCE/efficiency 的曲线。
- 前 150 h，10 min 网格，Akima 插值，逐曲线 MaxAbs 归一化。
- Savitzky–Golay `(window=71, order=2)`。
- MiniSom 2.2.9，随机权重初始化，50,000 次顺序训练。

## 唯一扫描的论文参数

- `K=2...10`。
- `(sigma, learning_rate)` 只取论文展示的 `(0.5,0.1)`、`(0.3,0.1)`、`(0.5,0.3)`。
- `seed=0` 只保证 27 个候选公平复算，不参与搜索。

## 类别数规则

按论文方法先看 quantization-error elbow；再检查低 K 是否欠分、高 K 是否产生重叠、
不可区分或极小节点，选择能够覆盖主要形态的最小 K。目标 IFO 名称在 K 冻结前不可使用。
`K=16` 只复现 Supplementary Fig. 1 的探索分辨率，不进入 K=2...10 的最终选择。

## 数据审计结果

- 发现 CSV：{len(audit)}。
- 严格纳入：{len(prepared['metadata'])}。
- 排除原因计数（同一曲线可有多项）：`{json.dumps(reason_counts, ensure_ascii=False)}`。

## 明确没有做

没有修改 SOM 算法，没有增加导数、峰谷、材料或实验条件变量，没有目标模板，
没有按 Bridge/Hill/Slope/Valley 选择参数，也没有强制 K=4。
"""
    (SCOPE_DIR / "METHOD_BOUNDARY_CN.md").write_text(scope, encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "minisom": "2.2.9 vendored source",
        "minisom_source": str(Path(sys.modules["minisom"].__file__).resolve()),
    }
    (ROOT / "environment_actual.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(audit: pd.DataFrame, prepared: dict[str, Any], metrics: pd.DataFrame) -> None:
    checks = {
        "input_csv_count_is_218": len(audit) == 218,
        "included_count_is_nonzero": len(prepared["metadata"]) > 0,
        "matrix_has_900_time_values": prepared["smoothed"].shape[1] == 900,
        "all_model_values_finite": bool(np.isfinite(prepared["smoothed"]).all()),
        "all_27_literature_parameter_runs_present": len(metrics) == 27,
        "k_values_exactly_2_through_10": sorted(metrics["k"].unique().tolist()) == list(range(2, 11)),
        "only_three_paper_parameter_pairs": len(metrics[["sigma", "learning_rate"]].drop_duplicates()) == 3,
        "all_nodes_occupied": bool((metrics["occupied_nodes"] == metrics["k"]).all()),
    }
    report = {"passed": all(checks.values()), "checks": checks}
    (VALIDATION_DIR / "search_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError(f"Verification failed: {report}")


def main() -> None:
    ensure_dirs()
    audit, curves = audit_input()
    prepared = preprocess(audit, curves)
    write_scope(audit, prepared)
    metrics = run_search(prepared)
    run_n16_exploratory(prepared)
    validate(audit, prepared, metrics)
    print(
        json.dumps(
            {
                "input_csv": len(audit),
                "included_150h": len(prepared["metadata"]),
                "matrix_shape": list(prepared["smoothed"].shape),
                "parameter_runs": len(metrics),
                "status": "search_complete_k_not_yet_selected",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
