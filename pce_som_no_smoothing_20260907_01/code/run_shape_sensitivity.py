#!/usr/bin/env python3
"""Literature-inspired shape-sensitive SOM and post-hoc IFO candidate audit."""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplcache"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import run_analysis as core

OUT = PROJECT / "08_shape_sensitive"


def train_search(zdata: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, labels = [], {}
    total = len(cfg["som_settings"]) * len(cfg["candidate_k"]) * len(cfg["screen_seeds"])
    done = 0
    for setting in cfg["som_settings"]:
        for k in cfg["candidate_k"]:
            for seed in cfg["screen_seeds"]:
                _, lab, metric = core.train_one(zdata, k, setting, seed, cfg["screen_iterations"])
                rows.append(metric); labels[(setting["name"], k, seed)] = lab
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"shape search: {done}/{total}", flush=True)
    runs = pd.DataFrame(rows)
    summary = core.aggregate_search(runs, labels)
    runs.to_csv(OUT / "all_runs.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    core.plot_qe(summary, OUT / "qe_elbows.png", "Shape-sensitive z-normalized SOM QE")
    return runs, summary


def fit_final(zdata: np.ndarray, raw: np.ndarray, meta: pd.DataFrame, grid: np.ndarray, cfg: dict, chosen: dict, k: int):
    setting = next(s for s in cfg["som_settings"] if s["name"] == chosen["setting"])
    models, label_sets, metrics = [], [], []
    for seed in cfg["final_seeds"]:
        som, lab, metric = core.train_one(zdata, k, setting, seed, cfg["final_iterations"])
        models.append(som); label_sets.append(lab); metrics.append(metric)
    ari = np.eye(len(models))
    for i, j in itertools.combinations(range(len(models)), 2):
        ari[i, j] = ari[j, i] = adjusted_rand_score(label_sets[i], label_sets[j])
    medoid = int(np.argmax(ari.mean(axis=1)))
    labels = label_sets[medoid]
    # Curves are plotted in the paper's MaxAbs space; medians are aggregate
    # representatives, not smoothing applied to any individual curve.
    centres = np.vstack([np.median(raw[labels == c], axis=0) for c in range(k)])
    assignments = meta.copy(); assignments["shape_cluster"] = labels
    assignments.to_csv(OUT / "assignments.csv", index=False)
    np.save(OUT / "labels.npy", labels); np.save(OUT / "raw_space_cluster_medians.npy", centres)
    pd.DataFrame(metrics).to_csv(OUT / "final_seed_metrics.csv", index=False)
    pd.DataFrame(ari, index=cfg["final_seeds"], columns=cfg["final_seeds"]).to_csv(OUT / "final_seed_ari_matrix.csv")
    core.dump_json(OUT / "final_model.json", {
        "representation": "per-curve z-normalized unsmoothed trajectory",
        "selected_k": k, "setting": setting, "selected_seed": cfg["final_seeds"][medoid],
        "mean_pairwise_seed_ari": core.pairwise_ari(label_sets),
        "raw_space_silhouette": float(silhouette_score(raw, labels, sample_size=min(1000, len(raw)), random_state=0)),
        "smoothing_enabled": False, "labels_used": False,
    })
    descriptors = core.posthoc_descriptors(centres, raw, labels, grid, OUT / "cluster_shape_descriptors.csv")
    plot_clusters(raw, labels, centres, grid)
    return labels, centres, descriptors


def plot_clusters(raw: np.ndarray, labels: np.ndarray, centres: np.ndarray, grid: np.ndarray) -> None:
    k = len(centres); cols = min(3, k); rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), sharex=True, sharey=True, squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        subset = raw[labels == c]
        for curve in subset:
            ax.plot(grid, curve, color="#3b82a0", lw=.4, alpha=min(.18, max(.015, 8 / len(subset))))
        ax.plot(grid, centres[c], color="black", lw=2.2)
        ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Shape cluster {c} (n={len(subset)})")
        ax.grid(alpha=.15)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("MaxAbs-normalized PCE")
    fig.suptitle("Shape-sensitive SOM result shown in unsmoothed PCE space")
    fig.tight_layout(); fig.savefig(OUT / "clusters.png", dpi=220); plt.close(fig)


def candidate_scores(curve: np.ndarray, grid: np.ndarray) -> dict:
    peak_i, trough_i = int(np.argmax(curve)), int(np.argmin(curve))
    peak, trough, start, end = curve[peak_i], curve[trough_i], curve[0], curve[-1]
    rise = max(0.0, peak - start); decay = max(0.0, peak - end)
    prominence = min(rise, decay)
    baseline = max(start, end)
    threshold = baseline + .5 * max(0.0, peak - baseline)
    width = float(np.mean(curve >= threshold)) if peak > baseline else 0.0
    peak_before_200 = 1.0 if 0 < grid[peak_i] <= 200 else 0.15
    post = curve[trough_i:]
    post_peak_i = trough_i + int(np.argmax(post))
    drop_to_trough = max(0.0, start - trough)
    recovery = max(0.0, curve[post_peak_i] - trough)
    late_decay = max(0.0, curve[post_peak_i] - end)
    valley_early = 1.0 if 0 < grid[trough_i] <= 200 and post_peak_i > trough_i else 0.1
    monotonic_share = float(np.mean(np.diff(curve) <= 0))
    return {
        "Bridge": prominence * width * peak_before_200,
        "Hill": prominence * (1.0 - width) * peak_before_200,
        "Slope": max(0.0, start - end) * monotonic_share,
        "Valley": min(drop_to_trough, recovery) * (1.0 + late_decay) * valley_early,
        "peak_time_h": float(grid[peak_i]), "trough_time_h": float(grid[trough_i]),
        "peak_width_fraction": width, "rise": rise, "post_peak_decay": decay,
        "post_trough_recovery": recovery,
    }


def posthoc_candidate_gallery(raw: np.ndarray, meta: pd.DataFrame, grid: np.ndarray) -> None:
    rows = []
    for i, curve in enumerate(raw):
        scores = candidate_scores(curve, grid)
        base = {"row_index": i, "curve_id": meta.iloc[i]["curve_id"], "source_file": meta.iloc[i]["source_file"], "legend": meta.iloc[i]["legend"]}
        rows.append(base | scores)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "posthoc_individual_ifo_scores.csv", index=False)
    types = ["Bridge", "Hill", "Slope", "Valley"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    selected_rows = []
    for kind, ax in zip(types, axes.flat):
        top = scores.sort_values(kind, ascending=False).head(8)
        for rank, (_, item) in enumerate(top.iterrows(), 1):
            idx = int(item["row_index"])
            x, y = core.read_curve(Path(item["source_file"]))
            rec = meta.iloc[idx]; elapsed = (x - x[0]) * float(rec["time_factor_to_hours"])
            mask = elapsed <= grid[-1] + 1e-9; scale = np.max(np.abs(y[mask]))
            ax.plot(elapsed[mask], y[mask] / scale, marker=".", ms=2.5, lw=.9, alpha=.72, label=item["curve_id"] if rank <= 4 else None)
            selected_rows.append({"ifo_candidate": kind, "rank": rank, "score": item[kind], "curve_id": item["curve_id"], "source_file": item["source_file"], "legend": item["legend"]})
        ax.axvline(200, color="grey", ls="--", lw=.8); ax.set_title(f"Post-hoc {kind}-like candidates")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("Raw-point PCE / in-window MaxAbs")
    fig.suptitle("Morphology-ranked individual candidates (post-hoc only; not cluster labels; smoothing OFF)")
    fig.tight_layout(); fig.savefig(OUT / "posthoc_individual_ifo_candidate_gallery.png", dpi=220); plt.close(fig)
    pd.DataFrame(selected_rows).to_csv(OUT / "posthoc_individual_ifo_candidate_provenance.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = core.load_config()
    raw = np.load(PROJECT / "02_preprocessed/primary_500h_curves.npy")
    grid = np.load(PROJECT / "02_preprocessed/primary_500h_time_hours.npy")
    meta = pd.read_csv(PROJECT / "02_preprocessed/primary_500h_included_metadata.csv")
    zdata = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-12)
    runs, summary = train_search(zdata, cfg)
    chosen, k, decisions = core.choose_final(summary)
    decisions.to_csv(OUT / "setting_and_k_decisions.csv", index=False)
    labels, centres, descriptors = fit_final(zdata, raw, meta, grid, cfg, chosen, k)
    posthoc_candidate_gallery(raw, meta, grid)
    baseline = json.loads((PROJECT / "05_final_model/final_model.json").read_text())
    shape = json.loads((OUT / "final_model.json").read_text())
    comparison = pd.DataFrame([
        {"method": "paper_MaxAbs_SOM", "selected_k": baseline["k"], "representation": "MaxAbs unsmoothed trajectory", "mean_seed_ari": baseline["mean_pairwise_seed_ari"], "raw_space_silhouette": float(pd.read_csv(PROJECT / "05_final_model/final_seed_metrics.csv")["silhouette"].median()), "status": "primary_selected_by_paper_protocol"},
        {"method": "shape_z_SOM", "selected_k": shape["selected_k"], "representation": shape["representation"], "mean_seed_ari": shape["mean_pairwise_seed_ari"], "raw_space_silhouette": shape["raw_space_silhouette"], "status": "sensitivity_only"},
    ])
    comparison.to_csv(OUT / "method_comparison_common_raw_space.csv", index=False)
    counts = descriptors["posthoc_ifo_name"].value_counts().to_dict()
    text = f"""# 形状敏感无监督对照\n\n逐曲线 z-normalization 受 k-Shape 的形状归一化思想启发，但仍使用论文的 SOM、QE elbow 和中心重叠规则；没有 cross-correlation、时间扭曲、平滑或标签。\n\n该对照自动选择 K={k}，参数 `{chosen['setting']}`。冻结后中心的 IFO-like 描述计数为 `{counts}`。它分离出 Bridge-like 与 Valley-like 中心，但没有独立 Hill-like 中心。\n\n此对照不替代主结果：共同的原始 MaxAbs 空间 silhouette 为 {shape['raw_space_silhouette']:.4f}，主论文流程为 {comparison.iloc[0]['raw_space_silhouette']:.4f}。此外 z-normalization 刻意消除了幅度信息。因此主结论仍采用论文 MaxAbs 流程的 K={baseline['k']}。\n\n`posthoc_individual_ifo_candidate_gallery.png` 只是在最终模型冻结后按峰谷形态连续评分列出个体候选，用于回答数据中是否能找到四类外观；它不是聚类标签，不能作为‘无监督发现四个簇’的证据。\n"""
    (OUT / "shape_sensitive_report_cn.md").write_text(text)
    print(f"shape sensitivity complete: K={k}, centres={counts}")


if __name__ == "__main__":
    main()
