#!/usr/bin/env python3
"""Tune literature-style 200 h SOMs to isolate rise-then-fall curves.

The preprocessing input is fixed to the paper workflow adapted only from
150 h to 200 h: 10-minute grid, Akima interpolation, per-curve maximum
normalisation, and Savitzky-Golay smoothing (window 71, polynomial order 2).
Only SOM node count, sigma, and learning rate are tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from minisom import MiniSom
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
INPUT = PROJECT / "03_preprocessed" / "main_high_quality_min10"
OUTPUT = BUNDLE / "reproduced_full_run"

PRIMARY_SEED = 42
SEEDS = (7, 21, 42, 84, 168)
SCREEN_ITERATIONS = 20_000
FINAL_ITERATIONS = 50_000
TOP_FULL_CANDIDATES = 8

sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class Run:
    som: MiniSom
    labels: np.ndarray
    coords: np.ndarray
    qe: float
    te: float
    best_node: int
    best_metrics: dict[str, float | int | bool]


def configure_logging() -> logging.Logger:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("paper200h_rise_fall")


def topology_for(n_nodes: int) -> tuple[int, int]:
    root = int(math.sqrt(n_nodes))
    for rows in range(root, 0, -1):
        if n_nodes % rows == 0:
            return rows, n_nodes // rows
    return 1, n_nodes


def curve_diagnostics(curve: np.ndarray, time_h: np.ndarray) -> dict[str, float | bool]:
    peak_index = int(np.argmax(curve))
    peak_time = float(time_h[peak_index])
    peak_value = float(curve[peak_index])
    initial_gain = float(peak_value - curve[0])
    post_peak_drop = float(peak_value - curve[-1])
    step = max(1, int(round(5.0 / float(time_h[1] - time_h[0]))))
    post_peak = curve[peak_index::step]
    nonincrease = (
        float(np.mean(np.diff(post_peak) <= 0.002)) if post_peak.size > 1 else 0.0
    )
    broad = bool(
        2.0 <= peak_time <= 180.0
        and initial_gain >= 0.01
        and post_peak_drop >= 0.02
        and nonincrease >= 0.50
    )
    medium = bool(
        5.0 <= peak_time <= 160.0
        and initial_gain >= 0.02
        and post_peak_drop >= 0.03
        and nonincrease >= 0.55
    )
    core = bool(
        5.0 <= peak_time <= 150.0
        and initial_gain >= 0.03
        and post_peak_drop >= 0.05
        and nonincrease >= 0.60
    )
    strong = bool(
        5.0 <= peak_time <= 120.0
        and initial_gain >= 0.05
        and post_peak_drop >= 0.10
        and nonincrease >= 0.70
    )
    return {
        "peak_time_h": peak_time,
        "peak_value": peak_value,
        "pce_0h": float(curve[0]),
        "pce_200h": float(curve[-1]),
        "initial_gain": initial_gain,
        "post_peak_drop": post_peak_drop,
        "post_peak_nonincrease_fraction_5h": nonincrease,
        "rise_then_fall_broad": broad,
        "rise_then_fall_medium": medium,
        "rise_then_fall_core": core,
        "rise_then_fall_strong": strong,
    }


def load_fixed_input() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    config = json.loads((INPUT / "preprocessing_config.json").read_text(encoding="utf-8"))
    expected = {
        "n_curves": 1442,
        "n_time_points": 1201,
        "time_window_hours": [0.0, 200.0],
        "grid_step_hours": 10.0 / 60.0,
        "interpolation": "scipy.interpolate.Akima1DInterpolator",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise AssertionError(f"Unexpected fixed preprocessing {key}: {config[key]}")
    if config["savgol"] != {"window_length": 71, "polyorder": 2, "mode": "interp"}:
        raise AssertionError("Savitzky-Golay configuration is not literature aligned")
    if config["normalization"] != "divide each curve by its own interpolated max PCE in 0-200 h":
        raise AssertionError("Unexpected normalization policy")

    curves = np.load(INPUT / "X_smoothed.npy")
    time_h = pd.read_csv(INPUT / "time_grid.csv")["time_h"].to_numpy(dtype=float)
    metadata = pd.read_csv(INPUT / "curve_metadata.csv")
    if "unit_factor_to_hours" not in metadata.columns:
        selection = pd.read_csv(PROJECT / "01_data_selection" / "main_selected_curves.csv")
        unit_columns = selection[["curve_id", "unit_factor_to_hours"]]
        metadata = metadata.merge(unit_columns, on="curve_id", how="left", validate="one_to_one")
        if metadata["unit_factor_to_hours"].isna().any():
            raise AssertionError("Could not recover time-unit factors for raw-point plots")
    if curves.shape != (len(metadata), len(time_h)):
        raise AssertionError("Preprocessed input dimensions do not match metadata")

    rows = []
    for array_row, curve in enumerate(curves):
        row = {"array_row": array_row, "curve_id": metadata.iloc[array_row]["curve_id"]}
        row.update(curve_diagnostics(curve, time_h))
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    return curves, time_h, metadata, diagnostics


def best_target_node(
    labels: np.ndarray,
    target: np.ndarray,
    curves: np.ndarray,
    time_h: np.ndarray,
    n_nodes: int,
) -> tuple[int, dict[str, float | int | bool]]:
    best_node = -1
    best: dict[str, float | int | bool] | None = None
    for node in range(n_nodes):
        member = labels == node
        size = int(member.sum())
        target_count = int(np.sum(member & target))
        precision = float(target_count / size) if size else 0.0
        recall = float(target_count / target.sum()) if target.sum() else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        centroid = np.mean(curves[member], axis=0) if size else np.zeros(curves.shape[1])
        centroid_diag = curve_diagnostics(centroid, time_h) if size else {}
        candidate = {
            "node": node,
            "node_size": size,
            "target_count": target_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "centroid_rise_then_fall_broad": bool(
                centroid_diag.get("rise_then_fall_broad", False)
            ),
            "centroid_rise_then_fall_core": bool(
                centroid_diag.get("rise_then_fall_core", False)
            ),
            "centroid_peak_time_h": float(centroid_diag.get("peak_time_h", 0.0)),
            "centroid_initial_gain": float(centroid_diag.get("initial_gain", 0.0)),
            "centroid_post_peak_drop": float(
                centroid_diag.get("post_peak_drop", 0.0)
            ),
        }
        key = (f1, precision, target_count, -size)
        if best is None or key > (
            float(best["f1"]),
            float(best["precision"]),
            int(best["target_count"]),
            -int(best["node_size"]),
        ):
            best = candidate
            best_node = node
    if best is None:
        raise RuntimeError("No SOM node was evaluated")
    return best_node, best


def train_one(
    curves: np.ndarray,
    time_h: np.ndarray,
    target: np.ndarray,
    n_nodes: int,
    sigma: float,
    learning_rate: float,
    seed: int,
    iterations: int,
) -> Run:
    topology = topology_for(n_nodes)
    np.random.seed(seed)
    som = MiniSom(
        x=topology[0],
        y=topology[1],
        input_len=curves.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        topology="rectangular",
        neighborhood_function="gaussian",
        activation_distance="euclidean",
        random_seed=seed,
    )
    som.random_weights_init(curves)
    som.train(curves, iterations, verbose=False)
    coords = np.asarray([som.winner(curve) for curve in curves], dtype=int)
    labels = coords[:, 0] * topology[1] + coords[:, 1]
    best_node, metrics = best_target_node(labels, target, curves, time_h, n_nodes)
    return Run(
        som=som,
        labels=labels,
        coords=coords,
        qe=float(som.quantization_error(curves)),
        te=float(som.topographic_error(curves)),
        best_node=best_node,
        best_metrics=metrics,
    )


def screening_grid() -> list[tuple[int, float, float]]:
    grid: set[tuple[int, float, float]] = set()
    paper_parameters = ((0.5, 0.1), (0.3, 0.1), (0.5, 0.3))
    for n_nodes in (*range(2, 11), 16):
        for sigma, learning_rate in paper_parameters:
            grid.add((n_nodes, sigma, learning_rate))
    for n_nodes in (16, 20, 25, 36):
        for sigma in (0.05, 0.10, 0.30, 0.50, 0.80, 1.20):
            for learning_rate in (0.05, 0.10, 0.30):
                grid.add((n_nodes, sigma, learning_rate))
    return sorted(grid)


def screen_models(
    curves: np.ndarray,
    time_h: np.ndarray,
    target: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    grid = screening_grid()
    for index, (n_nodes, sigma, learning_rate) in enumerate(grid, start=1):
        logger.info(
            "Screen %d/%d n=%d sigma=%.2f lr=%.2f",
            index,
            len(grid),
            n_nodes,
            sigma,
            learning_rate,
        )
        run = train_one(
            curves,
            time_h,
            target,
            n_nodes,
            sigma,
            learning_rate,
            PRIMARY_SEED,
            SCREEN_ITERATIONS,
        )
        rows.append(
            {
                "n_nodes": n_nodes,
                "topology": f"{topology_for(n_nodes)[0]}x{topology_for(n_nodes)[1]}",
                "sigma": sigma,
                "learning_rate": learning_rate,
                "iterations": SCREEN_ITERATIONS,
                "seed": PRIMARY_SEED,
                "qe": run.qe,
                "te": run.te,
                **{f"target_{key}": value for key, value in run.best_metrics.items()},
            }
        )
    frame = pd.DataFrame(rows)
    frame["screen_score"] = (
        0.50 * frame["target_f1"]
        + 0.30 * frame["target_precision"]
        + 0.15 * frame["target_recall"]
        + 0.05 * frame["target_centroid_rise_then_fall_broad"].astype(float)
    )
    frame = frame.sort_values(
        ["screen_score", "target_precision", "target_target_count"],
        ascending=False,
    ).reset_index(drop=True)
    out = OUTPUT / "01_parameter_screen"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "primary_seed_parameter_screen.csv", index=False)
    return frame


def choose_full_candidates(screen: pd.DataFrame) -> list[tuple[int, float, float]]:
    eligible = screen[
        (screen["target_target_count"] >= 5)
        & (screen["target_centroid_rise_then_fall_broad"])
    ]
    candidates = []
    for row in eligible.itertuples(index=False):
        key = (int(row.n_nodes), float(row.sigma), float(row.learning_rate))
        if key not in candidates:
            candidates.append(key)
        if len(candidates) >= TOP_FULL_CANDIDATES:
            break
    literature_main = (4, 0.5, 0.1)
    if literature_main not in candidates:
        candidates.append(literature_main)
    return candidates


def pairwise_mean_min(values: list[float]) -> tuple[float, float]:
    return (float(np.mean(values)), float(np.min(values))) if values else (1.0, 1.0)


def validate_candidates(
    curves: np.ndarray,
    time_h: np.ndarray,
    target: np.ndarray,
    candidates: list[tuple[int, float, float]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[tuple[int, float, float], Run]]:
    rows = []
    primary_runs: dict[tuple[int, float, float], Run] = {}
    for n_nodes, sigma, learning_rate in candidates:
        logger.info(
            "Full validation n=%d sigma=%.2f lr=%.2f",
            n_nodes,
            sigma,
            learning_rate,
        )
        runs: dict[int, Run] = {}
        for seed in SEEDS:
            runs[seed] = train_one(
                curves,
                time_h,
                target,
                n_nodes,
                sigma,
                learning_rate,
                seed,
                FINAL_ITERATIONS,
            )
        primary_runs[(n_nodes, sigma, learning_rate)] = runs[PRIMARY_SEED]

        aris = []
        target_jaccards = []
        for seed_a, seed_b in itertools.combinations(SEEDS, 2):
            run_a, run_b = runs[seed_a], runs[seed_b]
            aris.append(float(adjusted_rand_score(run_a.labels, run_b.labels)))
            member_a = run_a.labels == run_a.best_node
            member_b = run_b.labels == run_b.best_node
            union = int(np.sum(member_a | member_b))
            target_jaccards.append(
                float(np.sum(member_a & member_b) / union) if union else 1.0
            )
        mean_ari, min_ari = pairwise_mean_min(aris)
        mean_jaccard, min_jaccard = pairwise_mean_min(target_jaccards)
        primary = runs[PRIMARY_SEED]
        row = {
            "n_nodes": n_nodes,
            "topology": f"{topology_for(n_nodes)[0]}x{topology_for(n_nodes)[1]}",
            "sigma": sigma,
            "learning_rate": learning_rate,
            "iterations": FINAL_ITERATIONS,
            "primary_seed": PRIMARY_SEED,
            "primary_qe": primary.qe,
            "primary_te": primary.te,
            **{f"primary_{key}": value for key, value in primary.best_metrics.items()},
            "mean_qe": float(np.mean([run.qe for run in runs.values()])),
            "std_qe": float(np.std([run.qe for run in runs.values()])),
            "mean_global_seed_ARI": mean_ari,
            "min_global_seed_ARI": min_ari,
            "mean_target_node_Jaccard": mean_jaccard,
            "min_target_node_Jaccard": min_jaccard,
            "mean_target_precision": float(
                np.mean([float(run.best_metrics["precision"]) for run in runs.values()])
            ),
            "mean_target_recall": float(
                np.mean([float(run.best_metrics["recall"]) for run in runs.values()])
            ),
            "mean_target_f1": float(
                np.mean([float(run.best_metrics["f1"]) for run in runs.values()])
            ),
            "mean_target_count": float(
                np.mean([int(run.best_metrics["target_count"]) for run in runs.values()])
            ),
            "mean_target_node_size": float(
                np.mean([int(run.best_metrics["node_size"]) for run in runs.values()])
            ),
        }
        row["stable_high_purity_core_node"] = bool(
            row["primary_centroid_rise_then_fall_broad"]
            and row["primary_precision"] >= 0.70
            and row["primary_target_count"] >= 8
            and row["mean_target_precision"] >= 0.50
            and row["mean_global_seed_ARI"] >= 0.50
            and row["mean_target_node_Jaccard"] >= 0.25
        )
        row["validation_score"] = float(
            0.30 * row["mean_target_f1"]
            + 0.25 * row["mean_target_precision"]
            + 0.15 * row["mean_target_recall"]
            + 0.15 * row["mean_global_seed_ARI"]
            + 0.15 * row["mean_target_node_Jaccard"]
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["stable_high_purity_core_node", "validation_score"], ascending=False
    )
    out = OUTPUT / "02_multiseed_validation"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "candidate_multiseed_metrics.csv", index=False)
    return frame.reset_index(drop=True), primary_runs


def cluster_summary(
    labels: np.ndarray,
    curves: np.ndarray,
    diagnostics: pd.DataFrame,
    time_h: np.ndarray,
    n_nodes: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    centroids = []
    for node in range(n_nodes):
        member = labels == node
        centroid = np.mean(curves[member], axis=0) if member.any() else np.full(len(time_h), np.nan)
        centroids.append(centroid)
        diag = curve_diagnostics(centroid, time_h) if member.any() else {}
        rows.append(
            {
                "cluster_id": node,
                "n_curves": int(member.sum()),
                "fraction": float(member.mean()),
                "rise_then_fall_core_count": int(
                    diagnostics.loc[member, "rise_then_fall_core"].sum()
                ),
                "rise_then_fall_core_fraction": float(
                    diagnostics.loc[member, "rise_then_fall_core"].mean()
                )
                if member.any()
                else 0.0,
                "centroid_peak_time_h": diag.get("peak_time_h", np.nan),
                "centroid_initial_gain": diag.get("initial_gain", np.nan),
                "centroid_post_peak_drop": diag.get("post_peak_drop", np.nan),
                "centroid_rise_then_fall_broad": diag.get(
                    "rise_then_fall_broad", False
                ),
                "centroid_rise_then_fall_core": diag.get(
                    "rise_then_fall_core", False
                ),
            }
        )
    return pd.DataFrame(rows), np.stack(centroids)


def read_raw_curve(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(row["source_absolute_path"])
    numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    x = numeric.iloc[:, 0].to_numpy(dtype=float) * float(row["unit_factor_to_hours"])
    y = numeric.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order] - float(np.min(x))
    return x, y[order]


def save_plots(
    screen: pd.DataFrame,
    validation: pd.DataFrame,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    centroids: np.ndarray,
    curves: np.ndarray,
    time_h: np.ndarray,
    metadata: pd.DataFrame,
    selected_node: int,
) -> None:
    screen_out = OUTPUT / "01_parameter_screen"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(
        screen["target_recall"],
        screen["target_precision"],
        c=screen["n_nodes"],
        s=35 + 250 * screen["target_f1"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("Rise-then-fall recall")
    ax.set_ylabel("Rise-then-fall node precision")
    ax.set_title("Paper-style SOM parameter screen (seed 42)")
    fig.colorbar(scatter, ax=ax, label="SOM node count")
    fig.tight_layout()
    fig.savefig(screen_out / "parameter_screen_precision_recall.png", dpi=240)
    plt.close(fig)

    final = OUTPUT / "03_selected_model"
    final.mkdir(parents=True, exist_ok=True)
    n_nodes = len(summary)
    topology = topology_for(n_nodes)
    fig, axes = plt.subplots(
        topology[0], topology[1], figsize=(4.0 * topology[1], 3.0 * topology[0]),
        sharex=True, sharey=True, squeeze=False
    )
    for node, ax in enumerate(axes.flat):
        member = assignments["cluster_id"].to_numpy() == node
        chosen = np.flatnonzero(member)
        if chosen.size > 180:
            chosen = np.random.default_rng(42 + node).choice(chosen, 180, replace=False)
        for index in chosen:
            ax.plot(time_h, curves[index], color="#94A3B8", alpha=0.10, linewidth=0.5)
        color = "#DC2626" if node == selected_node else "#2563EB"
        ax.plot(time_h, centroids[node], color=color, linewidth=2.0)
        row = summary.loc[summary["cluster_id"] == node].iloc[0]
        ax.set_title(
            f"Node {node}: n={int(row.n_curves)}, RTF={row.rise_then_fall_core_fraction:.0%}"
        )
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.03, 1.05)
    fig.supxlabel("Relative ageing time (h)")
    fig.supylabel("Normalized PCE")
    fig.suptitle("Tuned literature-style 200 h SOM centroids")
    fig.tight_layout()
    fig.savefig(final / "all_som_nodes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    member_indices = np.flatnonzero(assignments["cluster_id"].to_numpy() == selected_node)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for index in member_indices:
        is_core = bool(assignments.iloc[index]["rise_then_fall_core"])
        ax.plot(
            time_h,
            curves[index],
            color="#2563EB" if is_core else "#94A3B8",
            alpha=0.35 if is_core else 0.22,
            linewidth=0.9,
        )
        raw_x, raw_y = read_raw_curve(metadata.iloc[index])
        within = raw_x <= 200.0 + 1e-9
        maximum = float(metadata.iloc[index]["max_pce_0_200h"])
        ax.scatter(
            raw_x[within],
            raw_y[within] / maximum,
            s=7,
            color="#0F172A",
            alpha=0.22,
            linewidths=0,
        )
    ax.plot(time_h, centroids[selected_node], color="#DC2626", linewidth=3, label="node mean")
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Normalized PCE")
    selected_summary = summary.loc[summary["cluster_id"] == selected_node].iloc[0]
    ax.set_title(
        f"Selected rise-then-fall SOM node {selected_node}: "
        f"n={int(selected_summary.n_curves)}, core purity="
        f"{selected_summary.rise_then_fall_core_fraction:.1%}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(final / "rise_then_fall_node_with_raw_points.png", dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(
        np.arange(len(validation)),
        validation["mean_target_precision"],
        color="#2563EB",
    )
    axes[0].set_ylabel("Mean target-node precision")
    axes[0].set_xlabel("Validated candidate rank")
    axes[1].bar(
        np.arange(len(validation)),
        validation["mean_target_node_Jaccard"],
        color="#059669",
    )
    axes[1].set_ylabel("Mean target-node Jaccard across seeds")
    axes[1].set_xlabel("Validated candidate rank")
    fig.suptitle("Multi-seed validation of tuned SOM candidates")
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_multiseed_validation" / "candidate_stability.png", dpi=240)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    selected: pd.Series,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_node: int,
) -> None:
    node = summary.loc[summary["cluster_id"] == selected_node].iloc[0]
    status = (
        "达到了预设的高纯度核心节点门槛，但召回率有限。"
        if bool(selected["stable_high_purity_core_node"])
        else "未达到稳定高纯度独立节点门槛；该结果只能作为探索性结果。"
    )
    report = [
        "# 文献流程200 h SOM调参：先升后降型",
        "",
        "## 结论",
        "",
        f"- {status}",
        f"- 文献式输入曲线：**{len(diagnostics):,}** 条。",
        f"- 可审计的核心先升后降曲线：**{int(diagnostics.rise_then_fall_core.sum())}** 条。",
        f"- 选定SOM：n=**{int(selected.n_nodes)}**，sigma=**{selected.sigma}**，"
        f"learning rate=**{selected.learning_rate}**。",
        f"- seed 42目标节点：**{selected_node}**；成员**{int(node.n_curves)}**条，"
        f"核心纯度**{node.rise_then_fall_core_fraction:.1%}**。",
        f"- seed 42召回率：**{selected.primary_recall:.1%}**。",
        f"- 五种子平均目标节点纯度：**{selected.mean_target_precision:.1%}**。",
        f"- 五种子目标节点成员平均Jaccard：**{selected.mean_target_node_Jaccard:.3f}**。",
        f"- 五种子全局平均ARI：**{selected.mean_global_seed_ARI:.3f}**。",
        f"- seed 42 QE：**{selected.primary_qe:.4f}**。",
        "",
        "## 固定不变的文献流程",
        "",
        "- 分析窗口仅由原文献150 h改为200 h。",
        "- 10 min网格、Akima插值、每条曲线0–200 h最大值归一化。",
        "- Savitzky–Golay窗口71、多项式阶数2。",
        "- MiniSom随机权重初始化、`som.train()`、50,000次、欧氏距离、Gaussian邻域。",
        "- 没有使用PCHIP、时间加权、导数或形状描述符作为SOM输入。",
        "",
        "## 调节的SOM参数",
        "",
        "仅扫描节点数、sigma和learning rate。先升后降诊断标签不参与训练，"
        "只用于判断某个无监督节点是否确实富集这种形状。",
        "",
        "## 先升后降核心定义",
        "",
        "峰值时间5–150 h、初始增益≥3%、峰后至200 h下降≥5%，"
        "且每5 h采样的峰后区间至少60%不再上升。",
        "",
        "## 最终节点摘要",
        "",
        markdown_table(summary.round(4)),
        "",
        "## 解释限制",
        "",
        "高纯度小节点不等于覆盖了全部先升后降曲线。若召回率偏低，说明同一"
        "转折形状仍因200 h终点衰减程度不同而分布在多个欧氏距离节点中。"
        "不得只凭单个随机种子把它宣称为稳定的新主类别。",
        "",
    ]
    (OUTPUT / "REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")


def write_checksums() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def finalize_existing(
    curves: np.ndarray,
    time_h: np.ndarray,
    metadata: pd.DataFrame,
    diagnostics: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    screen = pd.read_csv(
        OUTPUT / "01_parameter_screen" / "primary_seed_parameter_screen.csv"
    )
    validation = pd.read_csv(
        OUTPUT / "02_multiseed_validation" / "candidate_multiseed_metrics.csv"
    )
    assignments = pd.read_csv(
        OUTPUT / "03_selected_model" / "final_curve_assignments.csv"
    )
    summary = pd.read_csv(
        OUTPUT / "03_selected_model" / "final_cluster_summary.csv"
    )
    centroids = pd.read_csv(
        OUTPUT / "03_selected_model" / "final_centroid_curves.csv"
    ).to_numpy(dtype=float)
    selected_json = json.loads(
        (OUTPUT / "03_selected_model" / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    selected_node = int(selected_json["rise_then_fall_node"])
    save_plots(
        screen,
        validation,
        assignments,
        summary,
        centroids,
        curves,
        time_h,
        metadata,
        selected_node,
    )
    write_report(validation.iloc[0], summary, diagnostics, selected_node)
    logger.info("Recovered completed training and finalized outputs: %s", OUTPUT)
    write_checksums()


def main(finalize_only: bool = False) -> None:
    logger = configure_logging()
    curves, time_h, metadata, diagnostics = load_fixed_input()
    diagnostics_dir = OUTPUT / "00_fixed_input_and_target"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_dir / "rise_then_fall_diagnostics.csv", index=False)
    input_manifest = {
        "input_directory": str(INPUT),
        "input_array": "X_smoothed.npy",
        "n_curves": int(curves.shape[0]),
        "n_time_points": int(curves.shape[1]),
        "window_h": [0.0, 200.0],
        "grid_step_min": 10.0,
        "interpolation": "Akima",
        "normalization": "per-curve maximum in 0-200 h",
        "savgol": {"window_length": 71, "polyorder": 2, "mode": "interp"},
        "som_fixed": {
            "iterations_final": FINAL_ITERATIONS,
            "initialization": "random_weights_init",
            "training_method": "train",
            "activation_distance": "euclidean",
            "neighborhood_function": "gaussian",
            "topology": "rectangular",
        },
        "som_tuned": ["n_nodes", "sigma", "learning_rate"],
        "target_counts": {
            column: int(diagnostics[column].sum())
            for column in (
                "rise_then_fall_broad",
                "rise_then_fall_medium",
                "rise_then_fall_core",
                "rise_then_fall_strong",
            )
        },
    }
    (diagnostics_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if finalize_only:
        finalize_existing(curves, time_h, metadata, diagnostics, logger)
        return
    target = diagnostics["rise_then_fall_core"].to_numpy(dtype=bool)
    logger.info("Fixed paper-style input loaded; core rise-then-fall=%d", target.sum())

    screen = screen_models(curves, time_h, target, logger)
    candidates = choose_full_candidates(screen)
    validation, primary_runs = validate_candidates(
        curves, time_h, target, candidates, logger
    )
    selected = validation.iloc[0]
    key = (int(selected.n_nodes), float(selected.sigma), float(selected.learning_rate))
    run = primary_runs[key]
    selected_node = int(run.best_node)

    summary, centroids = cluster_summary(
        run.labels, curves, diagnostics, time_h, int(selected.n_nodes)
    )
    assignments = metadata.copy()
    assignments["cluster_id"] = run.labels
    assignments["som_node_x"] = run.coords[:, 0]
    assignments["som_node_y"] = run.coords[:, 1]
    for column in diagnostics.columns:
        if column not in ("array_row", "curve_id"):
            assignments[column] = diagnostics[column].to_numpy()
    assignments["selected_rise_then_fall_node"] = assignments["cluster_id"] == selected_node

    final = OUTPUT / "03_selected_model"
    final.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(final / "final_curve_assignments.csv", index=False)
    summary.to_csv(final / "final_cluster_summary.csv", index=False)
    pd.DataFrame(centroids, columns=[f"t_{i:04d}" for i in range(len(time_h))]).to_csv(
        final / "final_centroid_curves.csv", index=False
    )
    np.save(final / "final_som_weights.npy", run.som.get_weights())
    with (final / "final_som_model.pkl").open("wb") as handle:
        pickle.dump(run.som, handle)
    selected_json = {
        "n_nodes": int(selected.n_nodes),
        "topology": list(topology_for(int(selected.n_nodes))),
        "sigma": float(selected.sigma),
        "learning_rate": float(selected.learning_rate),
        "iterations": FINAL_ITERATIONS,
        "primary_seed": PRIMARY_SEED,
        "rise_then_fall_node": selected_node,
        "stable_high_purity_core_node": bool(selected.stable_high_purity_core_node),
        "primary_node_metrics": run.best_metrics,
        "multiseed_metrics": {
            column: (bool(selected[column]) if isinstance(selected[column], (bool, np.bool_)) else float(selected[column]))
            for column in (
                "mean_target_precision",
                "mean_target_recall",
                "mean_target_f1",
                "mean_target_node_Jaccard",
                "min_target_node_Jaccard",
                "mean_global_seed_ARI",
                "min_global_seed_ARI",
            )
        },
    }
    (final / "selected_model.json").write_text(
        json.dumps(selected_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_plots(
        screen,
        validation,
        assignments,
        summary,
        centroids,
        curves,
        time_h,
        metadata,
        selected_node,
    )
    write_report(selected, summary, diagnostics, selected_node)
    logger.info("Paper-style 200 h tuned SOM complete: %s", OUTPUT)
    write_checksums()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the complete 99-configuration screen and five-seed "
            "validation without overwriting the archived official result."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT,
        help="Fixed preprocessed input directory (default: project 03_preprocessed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT,
        help="Fresh reproduction output directory (default: reproduced_full_run).",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Rebuild plots, report, and checksums from an already completed scan.",
    )
    args = parser.parse_args()
    INPUT = args.input_dir.expanduser().resolve()
    OUTPUT = args.output_dir.expanduser().resolve()
    if OUTPUT == BUNDLE.resolve():
        parser.error(
            "Refusing to overwrite the archived official result directory; "
            "choose a new --output-dir."
        )
    main(finalize_only=args.finalize_existing)
