#!/usr/bin/env python3
"""Quality-filtered MaxAbs sensitivity experiment.

This script never uses IFO shape labels.  It subsets curves only by axis metadata,
observed duration, and basic numerical validity, then repeats the paper-style SOM
QE-elbow analysis with smoothing disabled.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/curve_discovery_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import itertools
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import adjusted_rand_score
from minisom import MiniSom
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "supplementary_filtered_maxabs_20260907"
FIG = OUT / "figures"
EXP = OUT / "experiments"
DATA = OUT / "data"

sys.path.insert(0, str(HERE))
from run_experiments import elbow, piecewise_elbow, grid  # noqa: E402

KS = list(range(1, 17))
SEEDS = [11, 29, 47, 71, 101]
SIGMA = 0.5
LR = 0.3
ITERATIONS = 50000


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(v):
    return None if pd.isna(v) else float(v)


def load_inputs():
    inv = pd.read_csv(ROOT / "data/curve_inventory.csv")
    curves = json.loads((ROOT / "data/raw_curves.json").read_text(encoding="utf-8"))
    rep = np.load(ROOT / "data/representation_maxabs.npz")
    X = rep["X"]
    assert len(inv) == len(curves) == len(X)
    assert inv.curve_id.tolist() == [c["id"] for c in curves]
    return inv, curves, X


def build_masks(inv):
    time_ok = inv.axis_status.eq("time_verified")
    plausible_duration = inv.duration_h.lt(100000).fillna(False)
    nonconstant = ~inv.constant_y.astype(bool)
    nonnegative = inv.y_min.ge(0)
    quality = time_ok & plausible_duration & nonconstant & nonnegative
    return {
        "Q156_quality": quality,
        "H200_n126": quality & inv.duration_h.ge(200),
        "H500_n105_recommended": quality & inv.duration_h.ge(500),
        "H750_n88_strict": quality & inv.duration_h.ge(750),
        "P15_n153_sensitivity": quality & inv.n_points.ge(15),
    }


def exclusion_reason(row):
    reasons = []
    if row.axis_status != "time_verified":
        reasons.append("x axis is not verified elapsed hours")
    if pd.notna(row.duration_h) and row.duration_h >= 100000:
        reasons.append("implausible duration >=100000 h")
    if bool(row.constant_y):
        reasons.append("constant y")
    if row.y_min < 0:
        reasons.append("negative y values")
    return "; ".join(reasons)


def summarize_model(X, labels, weights):
    k = len(weights)
    sizes = np.bincount(labels, minlength=k)
    occupied = np.flatnonzero(sizes)
    if len(occupied) > 1:
        # Explicit distance implementation avoids platform-dependent BLAS paths.
        distances = cdist(X, X)
        sample_sil = []
        for i, cluster in enumerate(labels):
            same = labels == cluster
            if same.sum() == 1:
                sample_sil.append(0.0)
                continue
            a = float(distances[i, same].sum() / (same.sum() - 1))
            b = min(float(distances[i, labels == other].mean()) for other in occupied if other != cluster)
            sample_sil.append((b - a) / max(a, b, 1e-15))
        sil = float(np.mean(sample_sil))
        centers = np.array([X[labels == j].mean(axis=0) for j in occupied])
        between = squareform(pdist(centers))
        between[between == 0] = np.nan
        min_between = float(np.nanmin(between))
        within = []
        for j, ctr in zip(occupied, centers):
            within.append(np.sqrt(np.mean(np.sum((X[labels == j] - ctr) ** 2, axis=1))))
        pooled_within = float(np.mean(within))
        separation_ratio = min_between / max(pooled_within, 1e-15)
    else:
        sil = None
        min_between = None
        pooled_within = None
        separation_ratio = None
    return {
        "cluster_sizes": sizes.tolist(),
        "silhouette": sil,
        "minimum_centroid_distance": min_between,
        "mean_within_cluster_rms": pooled_within,
        "separation_ratio": separation_ratio,
    }


def train_qe_only(X, k, sigma, lr, seed, iterations):
    """Train SOM and compute only selection statistics; silhouette is final-only."""
    a, b = grid(k)
    som = MiniSom(a, b, X.shape[1], sigma=sigma, learning_rate=lr, random_seed=seed)
    som.random_weights_init(X)
    som.train(X, iterations, random_order=True)
    weights = som.get_weights().reshape(k, -1)
    dist = cdist(X, weights)
    labels = dist.argmin(axis=1)
    return {
        "k": k, "sigma": sigma, "lr": lr, "seed": seed,
        "iterations": iterations, "grid": f"{a}x{b}",
        "qe": float(dist.min(axis=1).mean()),
        "occupied": int(len(np.unique(labels))),
    }, labels, weights


def run_subset(name, idx, Xall):
    X = Xall[idx]
    rows, labels_by = [], {}
    weights_by = {}
    for k, seed in itertools.product(KS, SEEDS):
        r, labels, weights = train_qe_only(X, k, SIGMA, LR, seed, iterations=ITERATIONS)
        r["subset"] = name
        rows.append(r)
        labels_by[k, seed] = labels
        weights_by[k, seed] = weights
    df = pd.DataFrame(rows)
    q = df.groupby("k").qe.median()
    selected_k, chord_gap = elbow(q.index.to_numpy(), q.to_numpy())
    selected_seed = int(df[df.k == selected_k].sort_values("qe").iloc[0].seed)
    labels = labels_by[selected_k, selected_seed]
    weights = weights_by[selected_k, selected_seed]
    stability = [
        adjusted_rand_score(labels_by[selected_k, a], labels_by[selected_k, b])
        for a, b in itertools.combinations(SEEDS, 2)
    ]
    range_elbows = {
        str(end): elbow(q.index[q.index <= end].to_numpy(), q[q.index <= end].to_numpy())[0]
        for end in [8, 10, 12, 16]
    }
    seed_elbows = {
        str(seed): elbow(KS, df[df.seed == seed].sort_values("k").qe.to_numpy())[0]
        for seed in SEEDS
    }
    model_stats = summarize_model(X, labels, weights)
    summary = {
        "subset": name,
        "n_curves": int(len(idx)),
        "selected_k": int(selected_k),
        "selected_seed": selected_seed,
        "sigma": SIGMA,
        "learning_rate": LR,
        "iterations": ITERATIONS,
        "median_qe": float(q.loc[selected_k]),
        "qe_over_k1": float(q.loc[selected_k] / q.loc[1]),
        "seed_ari_mean": float(np.mean(stability)),
        "seed_ari_min": float(np.min(stability)),
        "elbow_by_range": range_elbows,
        "elbow_by_seed": seed_elbows,
        "piecewise_elbow": int(piecewise_elbow(KS, q.to_numpy())),
        "chord_gap": chord_gap,
        **model_stats,
    }
    # Preserve best-QE labels for every k so k=3/4/5 overlap can be reviewed.
    save = {"indices": idx, "selected_labels": labels, "selected_weights": weights}
    for k in KS:
        seed = int(df[df.k == k].sort_values("qe").iloc[0].seed)
        save[f"labels_k{k}"] = labels_by[k, seed]
        save[f"weights_k{k}"] = weights_by[k, seed]
        save[f"seed_k{k}"] = np.array(seed)
    np.savez_compressed(EXP / f"model_{name}.npz", **save)
    return df, summary


def maxabs_curve(c):
    y = np.asarray(c["y"], float)
    return y / max(float(np.max(np.abs(y))), 1e-15)


def sampled_profiles(indices, labels, curves, grid):
    profiles = np.array([
        np.interp(grid, np.asarray(curves[i]["u"], float), maxabs_curve(curves[i]))
        for i in indices
    ])
    out = []
    for cluster in sorted(np.unique(labels)):
        member = profiles[labels == cluster]
        median = np.median(member, axis=0)
        q25, q75 = np.quantile(member, [0.25, 0.75], axis=0)
        d = np.sqrt(np.mean((member - median) ** 2, axis=1))
        local = np.flatnonzero(labels == cluster)[np.argmin(d)]
        out.append((cluster, member, median, q25, q75, indices[local]))
    return out


def plot_selected(summaries, masks, curves):
    grid = np.linspace(0, 1, 401)
    names = ["Q156_quality", "H200_n126", "H500_n105_recommended", "H750_n88_strict"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, name in zip(axes.flat, names):
        model = np.load(EXP / f"model_{name}.npz")
        idx = model["indices"]
        labels = model["selected_labels"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
        for (cluster, member, median, q25, q75, medoid), color in zip(
            sampled_profiles(idx, labels, curves, grid), colors
        ):
            ax.fill_between(grid, q25, q75, color=color, alpha=0.14)
            ax.plot(grid, median, color=color, lw=2.3,
                    label=f"C{cluster + 1}: n={len(member)}, medoid={curves[medoid]['id']}")
        s = summaries[name]
        ax.set_title(f"{name} | selected k={s['selected_k']} | silhouette={s['silhouette']:.3f}")
        ax.set_ylabel("y / max(|y|), no smoothing")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("relative observed progress u")
    fig.suptitle("Filtered MaxAbs SOM: selected cluster median and interquartile band", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG / "01_filtered_selected_clusters.png", dpi=180)
    plt.close(fig)


def plot_qe(sweeps, summaries):
    names = ["Q156_quality", "H200_n126", "H500_n105_recommended", "H750_n88_strict", "P15_n153_sensitivity"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        d = sweeps[sweeps.subset == name].groupby("k").qe.median()
        y = d / d.loc[1]
        ax.plot(d.index, y, marker="o", ms=3, label=f"{name} (k*={summaries[name]['selected_k']})")
        k = summaries[name]["selected_k"]
        ax.scatter([k], [y.loc[k]], s=65)
    ax.set_xticks(KS)
    ax.set_xlabel("number of SOM units k")
    ax.set_ylabel("median QE / median QE at k=1")
    ax.set_title("QE curves after label-free quality filtering")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_filtered_qe_elbows.png", dpi=180)
    plt.close(fig)


def plot_adjacent_k(curves):
    name = "H500_n105_recommended"
    model = np.load(EXP / f"model_{name}.npz")
    idx = model["indices"]
    grid = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, [2, 3, 4, 5, 6, 7]):
        labels = model[f"labels_k{k}"]
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for (cluster, member, median, q25, q75, medoid), color in zip(
            sampled_profiles(idx, labels, curves, grid), colors
        ):
            ax.plot(grid, median, color=color, lw=2, label=f"n={len(member)}; {curves[medoid]['id']}")
        ax.set_title(f"k={k}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("relative observed progress u")
    for ax in axes[:, 0]:
        ax.set_ylabel("MaxAbs y")
    fig.suptitle("H500 adjacent-k shape-overlap review (median profiles; no labels used)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG / "03_H500_adjacent_k_overlap_review.png", dpi=180)
    plt.close(fig)


def export_assignments(inv, masks, summaries):
    out = inv.copy()
    for name, mask in masks.items():
        out[f"included_{name}"] = mask.to_numpy()
        out[f"cluster_{name}"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        model = np.load(EXP / f"model_{name}.npz")
        idx, labels = model["indices"], model["selected_labels"]
        out.loc[idx, f"cluster_{name}"] = labels + 1
    out["quality_exclusion_reason"] = inv.apply(exclusion_reason, axis=1)
    out.to_csv(DATA / "filtered_assignments.csv", index=False)


def export_h500_curves(inv, curves):
    """Make the recommended filtered set directly reusable without source lookup."""
    selected_dir = DATA / "H500_selected_curves"
    selected_dir.mkdir(parents=True, exist_ok=True)
    model = np.load(EXP / "model_H500_n105_recommended.npz")
    cluster_by_index = dict(zip(model["indices"].tolist(), (model["selected_labels"] + 1).tolist()))
    manifest = []
    for i in model["indices"]:
        c = curves[int(i)]
        row = inv.iloc[int(i)]
        x = np.asarray(c["x"], float)
        y = np.asarray(c["y"], float)
        u = np.asarray(c["u"], float)
        elapsed = (x - x[0]) * float(row.hours_factor)
        out = pd.DataFrame({"original_x": x, "original_y": y,
                            "relative_progress": u, "elapsed_h": elapsed,
                            "y_maxabs": maxabs_curve(c)})
        out["H500_cluster"] = cluster_by_index[int(i)]
        path = selected_dir / f"{c['id']}.csv"
        out.to_csv(path, index=False)
        manifest.append({"curve_id": c["id"], "figure": row.figure,
                         "duration_h": row.duration_h, "n_points": row.n_points,
                         "cluster": cluster_by_index[int(i)],
                         "export_path": str(path.relative_to(OUT)),
                         "source_path": row.source_path})
    pd.DataFrame(manifest).to_csv(DATA / "H500_manifest.csv", index=False)


def comparison_ari(masks):
    names = list(masks)
    rows = []
    loaded = {n: np.load(EXP / f"model_{n}.npz") for n in names}
    for a, b in itertools.combinations(names, 2):
        ia, ib = loaded[a]["indices"], loaded[b]["indices"]
        common = np.intersect1d(ia, ib)
        posa = {v: p for p, v in enumerate(ia)}
        posb = {v: p for p, v in enumerate(ib)}
        la = loaded[a]["selected_labels"][[posa[v] for v in common]]
        lb = loaded[b]["selected_labels"][[posb[v] for v in common]]
        l4a = loaded[a]["labels_k4"][[posa[v] for v in common]]
        l4b = loaded[b]["labels_k4"][[posb[v] for v in common]]
        rows.append({"subset_a": a, "subset_b": b, "n_common": len(common),
                     "ari_selected_k": adjusted_rand_score(la, lb),
                     "ari_fixed_k4": adjusted_rand_score(l4a, l4b)})
    return pd.DataFrame(rows)


def baseline_comparison(masks):
    base = np.load(ROOT / "experiments/models_maxabs.npz")
    rows = []
    for name in masks:
        model = np.load(EXP / f"model_{name}.npz")
        idx = model["indices"]
        rows.append({
            "subset": name,
            "n": len(idx),
            "ari_subset_selected_vs_full_k5": adjusted_rand_score(model["selected_labels"], base["labels"][idx]),
            "ari_subset_k4_vs_full_k4": adjusted_rand_score(model["labels_k4"], base["labels_k4"][idx]),
            "full_k5_labels_present": json.dumps(np.unique(base["labels"][idx]).tolist()),
        })
    return pd.DataFrame(rows)


def write_readme(inv, masks, summaries, ari):
    q = summaries["Q156_quality"]
    h2 = summaries["H200_n126"]
    h5 = summaries["H500_n105_recommended"]
    h7 = summaries["H750_n88_strict"]
    p15 = summaries["P15_n153_sensitivity"]
    lines = [
        "# 数据质量筛选补充实验（MaxAbs、无平滑）", "",
        "本目录是原 218 条曲线实验的独立补充，不覆盖原结果。筛选和训练全程未使用 IFO-Bridge/Hill/Slope/Valley 标签。", "",
        "## 最重要的结论", "",
        f"- 质量筛选组保留 {q['n_curves']}/218 条（{q['n_curves']/218:.1%}），QE chord elbow 选择 k={q['selected_k']}。",
        f"- 至少 200 h 组保留 {h2['n_curves']} 条，选择 k={h2['selected_k']}。",
        f"- 推荐补充组 H500 保留 {h5['n_curves']} 条，选择 k={h5['selected_k']}；它同时提供至少 200 h 之前和至少 300 h 之后的观测。",
        f"- 严格 H750 组保留 {h7['n_curves']} 条，选择 k={h7['selected_k']}。",
        f"- 删除仅 3 条少于 15 点的曲线后，P15 组选择 k={p15['selected_k']}；该结果用于判断稀疏曲线是否主导结论。", "",
        "**建议把 H500 的 k=3 作为补充实验主结论。** 它的不同 k 上限、5 个随机种子和分段线性 elbow 全部给出 3；H750 也给出 3，且两组共同 88 条曲线的聚类 ARI 为 0.978。相邻 k 图显示 k=4 首先增加的是 C063 单样本极端衰减簇，k=5 以后主要继续拆分衰减幅度。按论文的重叠检查，不把它们解释为稳定的新总体类型。", "",
        "Q156 的 k=4 可作为广覆盖探索结果，但稳定性弱于 H500：分段 elbow 为 3，逐 seed 为 4/4/6/4/5，且其中一簇只有 6 条。H200 的全范围 chord 给出 5，但 k 上限≤12和分段 elbow均为3，并出现 4 条小簇；P15 的 k=6 同样随 k 范围和 seed 改变。因此这两组不能作为“确定有 5 类/6 类”的证据。", "",
        "筛选后的 MaxAbs 结果没有把 Bridge/Hill/Slope/Valley 稳定恢复为四个独立总体簇；长时长组主要分成三种衰减强度。该现象本身是补充实验结果，说明原数据中的少数峰谷形态、短曲线及非小时曲线不足以支撑稳定的四总体结构。", "",
        "最终类别数以 QE 曲线为主，并同时报告不同 k 上限、逐随机种子、分段线性 elbow、seed ARI、silhouette 和相邻 k 的簇中位曲线。若这些诊断不一致，应表述为类别数不稳定，而不能为了得到四类而指定 k=4。", "",
        "## 筛选规则", "",
        "Q156 同时满足：x 轴经元数据确认可换算为小时；观测时长 <100000 h；y 非常数；y 不含负值。随后只用时长建立 H200、H500 和 H750 嵌套组。负 x 起点、重复 x、点数少不会自动删除；负 x 常来自图片数字化偏差，重复 x 在分段线性 L2 中是零时长跳变。", "",
        "| 组 | 条数 | 图源数 | 用途 |", "|---|---:|---:|---|",
    ]
    for name, mask in masks.items():
        d = inv[mask]
        purpose = {
            "Q156_quality": "广覆盖质量组",
            "H200_n126": "确认跨过 200 h",
            "H500_n105_recommended": "推荐补充实验",
            "H750_n88_strict": "长时长严格敏感性",
            "P15_n153_sensitivity": "稀疏点敏感性",
        }[name]
        lines.append(f"| {name} | {len(d)} | {d.figure.nunique()} | {purpose} |")
    lines += ["", "## 坐标、归一化与距离", "",
              "每条曲线的 x 映射为 `u=(x-x_min)/(x_max-x_min)`，因此完整观测区间均为 0–1；真实小时数仍保存在名单中。每条曲线的 y 使用 `y/max(abs(y))`。没有 smoothing、插值平滑、截断或外推。原始点之间仅作分段线性连接，SOM 输入保持其精确 L2 距离。", "",
              "H200/H500/H750 的时长只决定曲线能否进入该组，不进入距离函数。因此本实验回答的是：在曲线至少被观察这么久的前提下，完整生命周期相对形态能分成几类。它不能直接比较绝对 200 h 发生点的位置。", "",
              "## SOM 参数", "",
              f"固定 MaxAbs 全数据参数：sigma={SIGMA}，learning rate={LR}。k=1–16；随机种子={SEEDS}；每次 {ITERATIONS} 次更新。固定参数用于隔离筛选本身的影响。每个 k 取 5 个种子的 QE 中位数，归一化后取端点弦线最大距离作为主 elbow。", "",
              "## 结果表", "",
              "| 组 | k* | QE/k1 | silhouette | seed ARI mean | seed elbows | range elbows (8/10/12/16) | piecewise | sizes |", "|---|---:|---:|---:|---:|---|---|---:|---|" ]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['selected_k']} | {s['qe_over_k1']:.3f} | {s['silhouette']:.3f} | {s['seed_ari_mean']:.3f} | {list(s['elbow_by_seed'].values())} | {list(s['elbow_by_range'].values())} | {s['piecewise_elbow']} | {s['cluster_sizes']} |")
    lines += ["", "## 如何看图", "",
              "1. `figures/01_filtered_selected_clusters.png`：各筛选层级最终选中 k 的簇中位曲线与四分位带，是主要结果图。",
              "2. `figures/02_filtered_qe_elbows.png`：类别数判定依据。",
              "3. `figures/03_H500_adjacent_k_overlap_review.png`：推荐 H500 组在 k=2–7 的相邻类别形状；用于检查新增单元是否只是在拆分相似曲线。", "",
              "`data/filtered_assignments.csv` 给出每条曲线在各组是否入选、排除原因及最终簇。`data/H500_selected_curves/` 是可直接使用的 105 条推荐补充曲线，`data/H500_manifest.csv` 是其清单。`experiments/qe_sweeps.csv` 保存每次训练。`experiments/subset_comparison_ari.csv` 同时给出各组在各自 k* 与固定 k=4 下的 ARI；`experiments/baseline_comparison.csv` 对比原 218 条 MaxAbs 模型。", "",
              "## 复现", "", "```bash", "cd /Users/shunhao/Desktop/ML/curve_discovery_unsupervised_20260906_01", "/usr/bin/python3 code/run_filtered_supplement.py", "```", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.time()
    for d in [OUT, FIG, EXP, DATA]:
        d.mkdir(parents=True, exist_ok=True)
    inv, curves, X = load_inputs()
    masks = build_masks(inv)
    expected = {"Q156_quality": 156, "H200_n126": 126, "H500_n105_recommended": 105,
                "H750_n88_strict": 88, "P15_n153_sensitivity": 153}
    actual = {name: int(mask.sum()) for name, mask in masks.items()}
    assert actual == expected, (actual, expected)
    config = {
        "source_result": str(ROOT), "source_dataset": str(ROOT.parent / "lab-v2/som_references/accepted/samples_test"),
        "normalization": "per-curve maxabs", "x_transform": "per-curve relative observed progress",
        "smoothing": False, "target_shape_labels_used": False, "ks": KS, "seeds": SEEDS,
        "sigma": SIGMA, "learning_rate": LR, "iterations": ITERATIONS,
        "selection": "median QE normalized chord elbow; overlap diagnostics are reported, never optimized against IFO labels",
        "subsets": actual,
    }
    dump(EXP / "config.json", config)
    sweeps = []
    summaries = {}
    for name, mask in masks.items():
        idx = np.flatnonzero(mask.to_numpy())
        df, summary = run_subset(name, idx, X)
        sweeps.append(df)
        summaries[name] = summary
        print("FINISHED", name, "n", len(idx), "k", summary["selected_k"],
              "seed ARI", round(summary["seed_ari_mean"], 3), flush=True)
    sweeps = pd.concat(sweeps, ignore_index=True)
    sweeps.to_csv(EXP / "qe_sweeps.csv", index=False)
    dump(EXP / "selection.json", summaries)
    flat = []
    for name, s in summaries.items():
        flat.append({k: v for k, v in s.items() if k not in {"chord_gap", "elbow_by_range", "elbow_by_seed", "cluster_sizes"}} |
                    {"cluster_sizes": json.dumps(s["cluster_sizes"]),
                     "elbow_by_range": json.dumps(s["elbow_by_range"]),
                     "elbow_by_seed": json.dumps(s["elbow_by_seed"])})
    pd.DataFrame(flat).to_csv(EXP / "subset_summary.csv", index=False)
    export_assignments(inv, masks, summaries)
    export_h500_curves(inv, curves)
    ari = comparison_ari(masks)
    ari.to_csv(EXP / "subset_comparison_ari.csv", index=False)
    baseline_comparison(masks).to_csv(EXP / "baseline_comparison.csv", index=False)
    plot_selected(summaries, masks, curves)
    plot_qe(sweeps, summaries)
    plot_adjacent_k(curves)
    write_readme(inv, masks, summaries, ari)
    dump(EXP / "run_complete.json", {"complete": True, "elapsed_seconds": time.time() - start})
    print("COMPLETE", OUT, "seconds", round(time.time() - start, 1), flush=True)


def finalize_only():
    """Rebuild summaries/plots from completed model files without retraining."""
    inv, curves, X = load_inputs()
    masks = build_masks(inv)
    sweeps = pd.read_csv(EXP / "qe_sweeps.csv")
    summaries = json.loads((EXP / "selection.json").read_text(encoding="utf-8"))
    for name in masks:
        model = np.load(EXP / f"model_{name}.npz")
        idx = model["indices"]
        summaries[name].update(summarize_model(X[idx], model["selected_labels"], model["selected_weights"]))
    dump(EXP / "selection.json", summaries)
    flat = []
    for name, s in summaries.items():
        flat.append({k: v for k, v in s.items() if k not in {"chord_gap", "elbow_by_range", "elbow_by_seed", "cluster_sizes"}} |
                    {"cluster_sizes": json.dumps(s["cluster_sizes"]),
                     "elbow_by_range": json.dumps(s["elbow_by_range"]),
                     "elbow_by_seed": json.dumps(s["elbow_by_seed"])})
    pd.DataFrame(flat).to_csv(EXP / "subset_summary.csv", index=False)
    export_assignments(inv, masks, summaries)
    export_h500_curves(inv, curves)
    ari = comparison_ari(masks)
    ari.to_csv(EXP / "subset_comparison_ari.csv", index=False)
    baseline_comparison(masks).to_csv(EXP / "baseline_comparison.csv", index=False)
    plot_selected(summaries, masks, curves)
    plot_qe(sweeps, summaries)
    plot_adjacent_k(curves)
    write_readme(inv, masks, summaries, ari)
    dump(EXP / "run_complete.json", {"complete": True, "finalized_from_saved_models": True})
    print("FINALIZE COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        if "--finalize-only" in sys.argv:
            finalize_only()
        else:
            main()
