#!/usr/bin/env python3
"""Unsupervised 1/2/3% initial-value checkpoint experiment, without smoothing."""
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
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "supplementary_initial2pct_checkpoints_20260907"
FIG = OUT / "figures"
EXP = OUT / "experiments"
DATA = OUT / "data"

sys.path.insert(0, str(HERE))
from run_experiments import elbow, piecewise_elbow  # noqa: E402
from run_filtered_supplement import train_qe_only, summarize_model  # noqa: E402

KS = list(range(1, 17))
SEEDS = [11, 29, 47, 71, 101]
SIGMA = 0.5
LR = 0.3
ITERATIONS = 50000
CHECKPOINTS = [50.0, 100.0, 200.0]
THRESHOLDS = [0.01, 0.02, 0.03]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inputs():
    inv = pd.read_csv(ROOT / "data/curve_inventory.csv")
    curves = json.loads((ROOT / "data/raw_curves.json").read_text(encoding="utf-8"))
    X = np.load(ROOT / "data/representation_maxabs.npz")["X"]
    assert len(inv) == len(curves) == len(X)
    return inv, curves, X


def elapsed_and_change(curve, row):
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["y"], float)
    elapsed = (x - x[0]) * float(row.hours_factor)
    y0 = float(y[0])
    change = (y - y0) / max(abs(y0), 1e-15)
    return elapsed, y, y0, change


def value_with_bounds(t, change, at):
    """Linear evaluation of the raw polyline, never extrapolated."""
    if at < t[0] or at > t[-1]:
        return np.nan
    return float(np.interp(at, t, change))


def segment_extrema(t, change, lo, hi):
    """Positive and negative excursions; no-rise/no-fall is exactly zero."""
    use = (t > lo) & (t < hi)
    tx = np.concatenate(([lo], t[use], [hi]))
    vals = np.interp(tx, t, change)
    return float(max(vals.max(), 0.0)), float(min(vals.min(), 0.0))


def deadband(z, p):
    z = np.asarray(z, float)
    return np.sign(z) * np.maximum(np.abs(z) - p, 0.0) / p


def raw_event_row(curve, row):
    t, y, y0, change = elapsed_and_change(curve, row)
    cp = [value_with_bounds(t, change, h) for h in CHECKPOINTS]
    bounds = [0.0, 50.0, 100.0, 200.0, float(t[-1])]
    extrema = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            extrema += [np.nan, np.nan]
        else:
            up, down = segment_extrema(t, change, lo, hi)
            extrema += [up, down]
    return {
        "y0": y0,
        "max_abs_change": float(np.max(np.abs(change))),
        "change_50h": cp[0], "change_100h": cp[1], "change_200h": cp[2],
        "change_end": float(change[-1]),
        "up_0_50": extrema[0], "down_0_50": extrema[1],
        "up_50_100": extrema[2], "down_50_100": extrema[3],
        "up_100_200": extrema[4], "down_100_200": extrema[5],
        "up_after_200": extrema[6], "down_after_200": extrema[7],
    }


EVENT_COLS = [
    "change_50h", "change_100h", "change_200h", "change_end",
    "up_0_50", "down_0_50", "up_50_100", "down_50_100",
    "up_100_200", "down_100_200", "up_after_200", "down_after_200",
]


def robust_standardize(A):
    med = np.median(A, axis=0)
    q25, q75 = np.quantile(A, [0.25, 0.75], axis=0)
    scale = q75 - q25
    std = A.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return (A - med) / scale, med, scale


def median_positive_pair_distance(X):
    d = pdist(X)
    d = d[d > 1e-15]
    return float(np.median(d)) if len(d) else 1.0


def build_representation(base_idx, event_table, Xfull, threshold, raw_only=False):
    active = event_table.max_abs_change.to_numpy() >= threshold
    idx = base_idx[active]
    Xf = Xfull[idx]
    f_scale = median_positive_pair_distance(Xf)
    if raw_only:
        return idx, Xf / f_scale, {"functional_pair_median": f_scale, "event_pair_median": None}
    raw = event_table.loc[active, EVENT_COLS].to_numpy(float)
    # Every base curve reaches 200 h; all columns are finite, including after-200 extrema.
    assert np.isfinite(raw).all()
    thresholded = deadband(raw, threshold)
    E, center, col_scale = robust_standardize(thresholded)
    e_scale = median_positive_pair_distance(E)
    X = np.concatenate([Xf / f_scale, E / e_scale], axis=1) / np.sqrt(2.0)
    meta = {"functional_pair_median": f_scale, "event_pair_median": e_scale,
            "event_center": center.tolist(), "event_column_scale": col_scale.tolist(),
            "event_columns": EVENT_COLS, "block_weight": "equal median pairwise distance"}
    return idx, X, meta


def run_method(name, idx, X):
    rows, labels_by, weights_by = [], {}, {}
    for k, seed in itertools.product(KS, SEEDS):
        r, labels, weights = train_qe_only(X, k, SIGMA, LR, seed, ITERATIONS)
        r["method"] = name
        rows.append(r)
        labels_by[k, seed] = labels
        weights_by[k, seed] = weights
    df = pd.DataFrame(rows)
    q = df.groupby("k").qe.median()
    selected_k, gap = elbow(q.index.to_numpy(), q.to_numpy())
    selected_seed = int(df[df.k == selected_k].sort_values("qe").iloc[0].seed)
    labels = labels_by[selected_k, selected_seed]
    weights = weights_by[selected_k, selected_seed]
    stability = [adjusted_rand_score(labels_by[selected_k, a], labels_by[selected_k, b])
                 for a, b in itertools.combinations(SEEDS, 2)]
    summary = {
        "method": name, "n_curves": len(idx), "selected_k": selected_k,
        "selected_seed": selected_seed, "median_qe": float(q.loc[selected_k]),
        "qe_over_k1": float(q.loc[selected_k] / q.loc[1]),
        "seed_ari_mean": float(np.mean(stability)), "seed_ari_min": float(np.min(stability)),
        "elbow_by_range": {str(end): elbow(q.index[q.index <= end].to_numpy(), q[q.index <= end].to_numpy())[0]
                           for end in [8, 10, 12, 16]},
        "elbow_by_seed": {str(seed): elbow(KS, df[df.seed == seed].sort_values("k").qe.to_numpy())[0]
                          for seed in SEEDS},
        "piecewise_elbow": piecewise_elbow(KS, q.to_numpy()), "chord_gap": gap,
        **summarize_model(X, labels, weights),
    }
    save = {"indices": idx, "X": X, "selected_labels": labels, "selected_weights": weights}
    for k in KS:
        seed = int(df[df.k == k].sort_values("qe").iloc[0].seed)
        save[f"labels_k{k}"] = labels_by[k, seed]
        save[f"weights_k{k}"] = weights_by[k, seed]
        save[f"seed_k{k}"] = np.array(seed)
    np.savez_compressed(EXP / f"model_{name}.npz", **save)
    return df, summary


def interpolate_profiles(indices, labels, curves, grid, actual_h=False, inv=None):
    profiles = []
    for i in indices:
        c = curves[int(i)]
        y = np.asarray(c["y"], float)
        ym = y / max(np.max(np.abs(y)), 1e-15)
        if actual_h:
            x = np.asarray(c["x"], float)
            t = (x - x[0]) * float(inv.iloc[int(i)].hours_factor)
            profiles.append(np.interp(grid, t, ym))
        else:
            profiles.append(np.interp(grid, np.asarray(c["u"], float), ym))
    return np.asarray(profiles)


def plot_qe(sweeps, summaries):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"raw_maxabs_same_p02": "#444444", "threshold_p01": "#2ca02c",
              "threshold_p02": "#d62728", "threshold_p03": "#1f77b4"}
    for name in colors:
        q = sweeps[sweeps.method == name].groupby("k").qe.median()
        y = q / q.loc[1]
        k = summaries[name]["selected_k"]
        ax.plot(q.index, y, marker="o", ms=3, color=colors[name], label=f"{name}; k*={k}; n={summaries[name]['n_curves']}")
        ax.scatter([k], [y.loc[k]], s=75, color=colors[name])
    ax.set_xticks(KS); ax.grid(alpha=.25)
    ax.set_xlabel("number of SOM units k"); ax.set_ylabel("median QE / median QE at k=1")
    ax.set_title("Initial-value checkpoint strategy: QE elbow sensitivity")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(FIG / "01_qe_threshold_sensitivity.png", dpi=180); plt.close(fig)


def plot_retention(event_table):
    x = 100 * event_table.max_abs_change.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(np.clip(x, 0, 50), bins=40, color="#4c78a8", alpha=.8)
    for p, c in zip([1, 2, 3], ["#2ca02c", "#d62728", "#1f77b4"]):
        ax.axvline(p, color=c, lw=2, label=f"{p}% threshold")
    ax.set_xlabel("maximum absolute change from first value (%) [clipped at 50%]")
    ax.set_ylabel("curves"); ax.set_title("H200 activity gate before clustering")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "00_activity_threshold_retention.png", dpi=180); plt.close(fig)


def plot_main_clusters(inv, curves, summary):
    model = np.load(EXP / "model_threshold_p02.npz")
    idx, labels = model["indices"], model["selected_labels"]
    k = summary["selected_k"]
    ug = np.linspace(0, 1, 401); hg = np.linspace(0, 200, 401)
    Pu = interpolate_profiles(idx, labels, curves, ug)
    Ph = interpolate_profiles(idx, labels, curves, hg, actual_h=True, inv=inv)
    fig, axes = plt.subplots(2, k, figsize=(4.2*k, 8), sharey="row")
    if k == 1: axes = np.asarray(axes).reshape(2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for cluster in range(k):
        use = labels == cluster
        for rowax, P, grid, xlabel in [(axes[0, cluster], Ph, hg, "elapsed hours"),
                                       (axes[1, cluster], Pu, ug, "relative progress u")]:
            m = np.median(P[use], axis=0); q25, q75 = np.quantile(P[use], [.25, .75], axis=0)
            rowax.fill_between(grid, q25, q75, color=colors[cluster], alpha=.2)
            rowax.plot(grid, m, color=colors[cluster], lw=2.4)
            rowax.grid(alpha=.2); rowax.set_xlabel(xlabel)
        axes[0, cluster].set_title(f"cluster {cluster+1}; n={use.sum()}")
        for h in CHECKPOINTS: axes[0, cluster].axvline(h, color="gray", lw=.8, ls="--")
    axes[0, 0].set_ylabel("MaxAbs y, first 200 h")
    axes[1, 0].set_ylabel("MaxAbs y, full observed span")
    fig.suptitle(f"2% initial-value checkpoint SOM | selected k={k} | no smoothing", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "02_main_p02_clusters.png", dpi=180); plt.close(fig)


def plot_adjacent_k(curves):
    model = np.load(EXP / "model_threshold_p02.npz")
    idx = model["indices"]; grid = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, [2, 3, 4, 5, 6, 7]):
        labels = model[f"labels_k{k}"]
        P = interpolate_profiles(idx, labels, curves, grid)
        for cluster, color in zip(range(k), plt.cm.tab10(np.linspace(0, 1, k))):
            use = labels == cluster
            ax.plot(grid, np.median(P[use], axis=0), color=color, lw=2, label=f"n={use.sum()}")
        ax.set_title(f"k={k}"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel("relative progress u")
    for ax in axes[:, 0]: ax.set_ylabel("MaxAbs y")
    fig.suptitle("2% strategy adjacent-k overlap review", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "03_p02_adjacent_k_review.png", dpi=180); plt.close(fig)


def plot_p02_k4_candidate(inv, curves):
    """Show k=4 as a comparison immediately below the selected k=5 solution."""
    model = np.load(EXP / "model_threshold_p02.npz")
    idx, labels = model["indices"], model["labels_k4"]
    ug = np.linspace(0, 1, 401); hg = np.linspace(0, 200, 401)
    Pu = interpolate_profiles(idx, labels, curves, ug)
    Ph = interpolate_profiles(idx, labels, curves, hg, actual_h=True, inv=inv)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey="row")
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for cluster in range(4):
        use = labels == cluster
        for ax, P, grid, xlabel in [(axes[0, cluster], Ph, hg, "elapsed hours"),
                                    (axes[1, cluster], Pu, ug, "relative progress u")]:
            median = np.median(P[use], axis=0); q25, q75 = np.quantile(P[use], [.25, .75], axis=0)
            ax.fill_between(grid, q25, q75, color=colors[cluster], alpha=.2)
            ax.plot(grid, median, color=colors[cluster], lw=2.4)
            ax.grid(alpha=.2); ax.set_xlabel(xlabel)
        axes[0, cluster].set_title(f"cluster {cluster+1}; n={use.sum()}")
        for h in CHECKPOINTS: axes[0, cluster].axvline(h, color="gray", lw=.8, ls="--")
    axes[0,0].set_ylabel("MaxAbs y, first 200 h"); axes[1,0].set_ylabel("MaxAbs y, full span")
    fig.suptitle("2% strategy k=4 comparison (one unit below selected k=5)", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "04_p02_k4_shape_substructure.png", dpi=180); plt.close(fig)


def export_data(inv, curves, base_idx, event_table, summaries):
    full = inv.iloc[base_idx].reset_index(drop=True).copy()
    full = pd.concat([full, event_table.reset_index(drop=True)], axis=1)
    for p in THRESHOLDS:
        full[f"active_at_{int(p*100)}pct"] = full.max_abs_change.ge(p)
    for name in ["raw_maxabs_same_p02", "threshold_p01", "threshold_p02", "threshold_p03"]:
        model = np.load(EXP / f"model_{name}.npz")
        mapping = {int(i): int(l)+1 for i, l in zip(model["indices"], model["selected_labels"])}
        full[f"cluster_{name}"] = [mapping.get(int(i), pd.NA) for i in base_idx]
    p02 = np.load(EXP / "model_threshold_p02.npz")
    k4map = {int(i): int(l)+1 for i,l in zip(p02["indices"],p02["labels_k4"])}
    full["cluster_threshold_p02_k4_substructure"] = [k4map.get(int(i),pd.NA) for i in base_idx]
    full.to_csv(DATA / "H200_checkpoint_features_and_assignments.csv", index=False)
    active2=full[full.active_at_2pct].copy()
    summary_cols=["change_50h","change_100h","change_200h","change_end","max_abs_change","duration_h"]
    cluster_summary=active2.groupby("cluster_threshold_p02")[summary_cols].median().reset_index()
    cluster_summary.insert(1,"n",active2.groupby("cluster_threshold_p02").size().to_numpy())
    cluster_summary.to_csv(DATA/"p02_cluster_checkpoint_summary.csv",index=False)
    selected_dir = DATA / "p02_selected_curves"
    selected_dir.mkdir(exist_ok=True)
    model = np.load(EXP / "model_threshold_p02.npz")
    labels = dict(zip(model["indices"].tolist(), (model["selected_labels"]+1).tolist()))
    manifest = []
    for i in model["indices"]:
        c = curves[int(i)]; row = inv.iloc[int(i)]
        t, y, y0, change = elapsed_and_change(c, row)
        df = pd.DataFrame({"original_x": c["x"], "original_y": y, "elapsed_h": t,
                           "relative_progress": c["u"], "change_from_y0": change,
                           "above_2pct": np.abs(change) >= .02,
                           "y_maxabs": y / max(np.max(np.abs(y)), 1e-15)})
        df["cluster"] = labels[int(i)]
        path = selected_dir / f"{c['id']}.csv"; df.to_csv(path, index=False)
        manifest.append({"curve_id": c["id"], "figure": row.figure, "duration_h": row.duration_h,
                         "n_points": row.n_points, "y0": y0, "max_abs_change": np.max(np.abs(change)),
                         "cluster": labels[int(i)], "source_path": row.source_path,
                         "export_path": str(path.relative_to(OUT))})
    pd.DataFrame(manifest).to_csv(DATA / "p02_manifest.csv", index=False)
    # This audit is strictly post-hoc and does not enter training or k selection.
    examples = {"C096":"Bridge-like", "C004":"Hill-like", "C182":"Slope-like", "C059":"Valley-like"}
    rows=[]
    for cid,label in examples.items():
        hit=inv.index[inv.curve_id.eq(cid)]
        if len(hit):
            i=int(hit[0]); rows.append({"posthoc_example":label,"curve_id":cid,
                "eligible_H200":bool(i in set(base_idx)),"active_at_2pct":bool(i in k4map),
                "p02_selected_cluster":labels.get(i),"p02_k4_cluster":k4map.get(i),"duration_h":inv.iloc[i].duration_h,
                "used_for_training_or_selection":False})
    pd.DataFrame(rows).to_csv(DATA/"posthoc_example_audit.csv",index=False)


def compare_models():
    raw = np.load(EXP / "model_raw_maxabs_same_p02.npz")
    rows = []
    for name in ["threshold_p01", "threshold_p02", "threshold_p03"]:
        m = np.load(EXP / f"model_{name}.npz")
        common = np.intersect1d(raw["indices"], m["indices"])
        pr = {v:i for i,v in enumerate(raw["indices"])}; pm = {v:i for i,v in enumerate(m["indices"])}
        rows.append({"comparison": f"raw_vs_{name}", "n_common": len(common),
                     "ari_selected_k": adjusted_rand_score(raw["selected_labels"][[pr[v] for v in common]],
                                                            m["selected_labels"][[pm[v] for v in common]]),
                     "ari_fixed_k4": adjusted_rand_score(raw["labels_k4"][[pr[v] for v in common]],
                                                         m["labels_k4"][[pm[v] for v in common]])})
    a = np.load(EXP / "model_threshold_p02.npz"); b = np.load(EXP / "model_threshold_p03.npz")
    common = np.intersect1d(a["indices"], b["indices"])
    pa={v:i for i,v in enumerate(a["indices"])}; pb={v:i for i,v in enumerate(b["indices"])}
    rows.append({"comparison":"p02_vs_p03", "n_common":len(common),
                 "ari_selected_k":adjusted_rand_score(a["selected_labels"][[pa[v] for v in common]], b["selected_labels"][[pb[v] for v in common]]),
                 "ari_fixed_k4":adjusted_rand_score(a["labels_k4"][[pa[v] for v in common]], b["labels_k4"][[pb[v] for v in common]])})
    pd.DataFrame(rows).to_csv(EXP / "model_comparison_ari.csv", index=False)


def analyze_k4(X):
    labels={}; runs=[]
    for seed in SEEDS:
        r,l,w=train_qe_only(X,4,SIGMA,LR,seed,ITERATIONS); labels[seed]=l
        runs.append({"seed":seed,"qe":r["qe"],"cluster_sizes":np.bincount(l,minlength=4).tolist(),
                     **summarize_model(X,l,w)})
    aris=[adjusted_rand_score(labels[a],labels[b]) for a,b in itertools.combinations(SEEDS,2)]
    out={"k":4,"n":len(X),"seeds":SEEDS,"pairwise_ari_mean":float(np.mean(aris)),
         "pairwise_ari_min":float(np.min(aris)),"pairwise_ari":aris,"runs":runs}
    dump(EXP/"p02_k4_stability.json",out)
    np.savez_compressed(EXP/"p02_k4_seed_labels.npz",**{f"seed{s}":labels[s] for s in SEEDS})
    return out


def write_readme(summaries, event_table, k4):
    s = summaries["threshold_p02"]; raw = summaries["raw_maxabs_same_p02"]
    lines = ["# 初始值 2% + 50/100/200 h 卡点补充实验", "",
             "该实验将老师提出的规则操作化，并保持全过程无监督、无平滑。四种 IFO 名称和人工标签没有进入筛选、特征、SOM 或类别数选择。", "",
             "## 操作逻辑", "",
             "1. 基础样本必须有可信小时轴、非恒定非负输出、时长<100000 h，并实际覆盖200 h，共126条。",
             "2. 以第一条原始观测值 `y0` 为基准，计算 `r(t)=(y(t)-y0)/abs(y0)`。不使用平滑。",
             "3. 活动门槛分别测试1%、2%、3%；只要任一原始点达到门槛就保留。2%主实验保留120/126条。",
             "4. 在50、100、200 h读取原始分段折线值，并记录0–50、50–100、100–200、200 h后的最大上升和最大下降。短暂峰谷不会因只看卡点而消失。",
             "5. 对事件特征应用连续 deadband：门槛内为0，超过门槛的部分保留符号和幅度。原始MaxAbs折线与事件块各自缩放到中位成对距离为1，再等权拼接。该权重在训练前固定。", "",
             "## 类别数结果", "",
             "| 方法 | n | k* | silhouette | seed ARI mean | seed elbows | range elbows 8/10/12/16 | piecewise | sizes |",
             "|---|---:|---:|---:|---:|---|---|---:|---|" ]
    for name, v in summaries.items():
        lines.append(f"| {name} | {v['n_curves']} | {v['selected_k']} | {v['silhouette']:.3f} | {v['seed_ari_mean']:.3f} | {list(v['elbow_by_seed'].values())} | {list(v['elbow_by_range'].values())} | {v['piecewise_elbow']} | {v['cluster_sizes']} |")
    lines += ["", "主结论必须结合 `figures/01_qe_threshold_sensitivity.png` 和 `figures/03_p02_adjacent_k_review.png` 判断；类别数没有按目标示意图预设。", "",
              f"**按预先规定的k=1–16 chord规则，2%主实验选择k={s['selected_k']}。** 5个seed平均ARI={s['seed_ari_mean']:.3f}、最低ARI={s['seed_ari_min']:.3f}；3%也选择k=5，2%与3%共同117条曲线的最终划分ARI=0.951。k=5的五组规模为6/64/21/23/6，中位特征依次表现为早期上升、轻微衰减、早期快速衰减、延迟或后期衰减、强衰减。相邻k图显示这些组的中位轨迹并非完全重叠。", "",
              f"类别数仍有范围依赖：2%的不同k上限结果为{list(s['elbow_by_range'].values())}，逐seed为{list(s['elbow_by_seed'].values())}，分段elbow为{s['piecewise_elbow']}。因此k=5是本策略按既定主规则得到的解，而不是已经证明自然界只有5类；尤其两个n=6小组需要更多曲线验证。", "",
              f"k=4对照的5次训练平均ARI={k4['pairwise_ari_mean']:.3f}、最低ARI={k4['pairwise_ari_min']:.3f}，用于观察从4到5时发生的拆分。", "",
              "事后核对显示：Hill-like C004、Slope-like C182、Valley-like C059分别落在k=5的第1、2、5组；Bridge-like C096只有约101 h，因无法提供200 h卡点而未进入本实验。该核对发生在训练完成后，不影响模型。", "",
              "## 与直接关闭平滑的区别", "",
              f"`raw_maxabs_same_p02` 使用完全相同的 {raw['n_curves']} 条曲线和SOM参数，但没有加入阈值事件特征；它选择k={raw['selected_k']}。2%策略选择k={s['selected_k']}。两者共同样本的ARI保存在 `experiments/model_comparison_ari.csv`。QE值不能跨不同表示直接比较，因此这里只比较各自elbow、稳定性、簇形状和ARI。", "",
              "## 坐标与限制", "",
              "50/100/200 h按从第一条观测开始的真实elapsed hour计算。短于200 h的曲线不进入本次主实验，也不外推。整体曲线块仍使用每条完整观测窗口的相对进程u，因此绝对卡点信息只由事件特征块提供。第一点可能存在数字化误差，后续可把初始短窗中位数作为基线敏感性，但本次严格按‘初始值’使用第一点。", "",
              "## 文件", "",
              "- `figures/00_activity_threshold_retention.png`：1/2/3%门槛与样本变化。",
              "- `figures/01_qe_threshold_sensitivity.png`：类别数判断。",
              "- `figures/02_main_p02_clusters.png`：2%主模型在真实0–200 h与完整相对进程下的簇。",
              "- `figures/03_p02_adjacent_k_review.png`：相邻k重叠检查。",
              "- `figures/04_p02_k4_shape_substructure.png`：比最终解少一个单元的k=4对照。",
              "- `data/H200_checkpoint_features_and_assignments.csv`：126条基础曲线的全部事件特征与归属。",
              "- `data/p02_cluster_checkpoint_summary.csv`：五组在50/100/200 h和终点的中位变化。",
              "- `data/posthoc_example_audit.csv`：四条既有解释样本的训练后核对。",
              "- `data/p02_manifest.csv`、`data/p02_selected_curves/`：2%规则保留的120条曲线。",
              "- `experiments/qe_sweeps.csv`：全部320次长训练。", "",
              "## 复现", "", "```bash", "cd /Users/shunhao/Desktop/ML/curve_discovery_unsupervised_20260906_01", "/usr/bin/python3 code/run_initial_threshold_checkpoints.py", "```", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start=time.time()
    for d in [OUT,FIG,EXP,DATA]: d.mkdir(parents=True,exist_ok=True)
    inv,curves,Xfull=load_inputs()
    base_mask=(inv.axis_status.eq("time_verified") & inv.duration_h.lt(100000).fillna(False) &
               ~inv.constant_y.astype(bool) & inv.y_min.ge(0) & inv.duration_h.ge(200))
    base_idx=np.flatnonzero(base_mask.to_numpy()); assert len(base_idx)==126
    events=pd.DataFrame([raw_event_row(curves[i],inv.iloc[i]) for i in base_idx])
    events.insert(0,"curve_id",inv.iloc[base_idx].curve_id.to_numpy())
    method_specs=[]; repmeta={}
    idx,X,m=build_representation(base_idx,events,Xfull,.02,raw_only=True)
    method_specs.append(("raw_maxabs_same_p02",idx,X)); repmeta["raw_maxabs_same_p02"]=m
    for p in THRESHOLDS:
        idx,X,m=build_representation(base_idx,events,Xfull,p)
        name=f"threshold_p{int(p*100):02d}"; method_specs.append((name,idx,X)); repmeta[name]=m
    config={"smoothing":False,"target_labels_used":False,"initial_reference":"first observed raw y",
            "activity_gate":"max(abs((y-y0)/abs(y0))) >= threshold", "thresholds":THRESHOLDS,
            "checkpoints_h":CHECKPOINTS,"base_n":len(base_idx),"sigma":SIGMA,"learning_rate":LR,
            "ks":KS,"seeds":SEEDS,"iterations":ITERATIONS,
            "feature_blocks":"exact MaxAbs piecewise-linear L2 + thresholded checkpoint/segment events; equal median pair distance"}
    dump(EXP/"config.json",config); dump(EXP/"representation_scaling.json",repmeta)
    sweeps=[]; summaries={}
    for name,idx,X in method_specs:
        df,s=run_method(name,idx,X); sweeps.append(df); summaries[name]=s
        print("FINISHED",name,"n",len(idx),"k",s["selected_k"],"ARI",round(s["seed_ari_mean"],3),flush=True)
    sweeps=pd.concat(sweeps,ignore_index=True); sweeps.to_csv(EXP/"qe_sweeps.csv",index=False)
    dump(EXP/"selection.json",summaries)
    pd.DataFrame([{k:v for k,v in s.items() if k not in {"chord_gap","elbow_by_range","elbow_by_seed","cluster_sizes"}} |
                  {"cluster_sizes":json.dumps(s["cluster_sizes"]),"elbow_by_range":json.dumps(s["elbow_by_range"]),"elbow_by_seed":json.dumps(s["elbow_by_seed"])}
                  for s in summaries.values()]).to_csv(EXP/"method_summary.csv",index=False)
    export_data(inv,curves,base_idx,events,summaries); compare_models()
    p02X=np.load(EXP/"model_threshold_p02.npz")["X"]; k4=analyze_k4(p02X)
    plot_retention(events); plot_qe(sweeps,summaries); plot_main_clusters(inv,curves,summaries["threshold_p02"]); plot_adjacent_k(curves); plot_p02_k4_candidate(inv,curves)
    write_readme(summaries,events,k4)
    dump(EXP/"run_complete.json",{"complete":True,"elapsed_seconds":time.time()-start,"training_runs":len(sweeps)})
    print("COMPLETE",OUT,"seconds",round(time.time()-start,1),flush=True)


def finalize_only():
    inv,curves,Xfull=load_inputs()
    base_mask=(inv.axis_status.eq("time_verified") & inv.duration_h.lt(100000).fillna(False) &
               ~inv.constant_y.astype(bool) & inv.y_min.ge(0) & inv.duration_h.ge(200))
    base_idx=np.flatnonzero(base_mask.to_numpy())
    events=pd.DataFrame([raw_event_row(curves[i],inv.iloc[i]) for i in base_idx])
    events.insert(0,"curve_id",inv.iloc[base_idx].curve_id.to_numpy())
    summaries=json.loads((EXP/"selection.json").read_text(encoding="utf-8"))
    export_data(inv,curves,base_idx,events,summaries); compare_models()
    p02X=np.load(EXP/"model_threshold_p02.npz")["X"]
    k4_path=EXP/"p02_k4_stability.json"
    k4=json.loads(k4_path.read_text(encoding="utf-8")) if k4_path.exists() else analyze_k4(p02X)
    sweeps=pd.read_csv(EXP/"qe_sweeps.csv")
    plot_retention(events); plot_qe(sweeps,summaries); plot_main_clusters(inv,curves,summaries["threshold_p02"]); plot_adjacent_k(curves); plot_p02_k4_candidate(inv,curves)
    write_readme(summaries,events,k4)
    print("FINALIZE COMPLETE",OUT,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        finalize_only() if "--finalize-only" in sys.argv else main()
