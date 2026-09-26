#!/usr/bin/env python3
"""Expanded label-blind SOM search for unsmoothed PCE trajectories."""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import common


def cfg() -> dict:
    return json.loads((PROJECT / "config.json").read_text())


def ensure_dirs() -> None:
    for name in (
        "01_data_audit", "02_variants", "03_kmeans_screen", "04_som_search",
        "05_frozen_selection", "06_final_model/clusters", "07_posthoc_ifo",
        "08_stability", "09_figures", "reports", "logs",
    ):
        (PROJECT / name).mkdir(parents=True, exist_ok=True)


def variant_id(window: int, cohort: str, representation: str) -> str:
    return f"w{window:04d}_{cohort}_{representation}"


def prepare(config: dict) -> pd.DataFrame:
    curves, inventory = common.load_curves(Path(config["input_root"]))
    inventory["smoothing_applied"] = False
    inventory.to_csv(PROJECT / "01_data_audit/input_inventory.csv", index=False)
    accepted = inventory[inventory["accepted"]]
    rows = []
    for window in config["windows_hours"]:
        for cohort, rule in config["cohorts"].items():
            for representation in config["representations"]:
                vid = variant_id(window, cohort, representation)
                data, display, meta, grid = common.build_variant(
                    curves, window, cohort, rule, representation, config["grid_points"]
                )
                np.savez_compressed(PROJECT / "02_variants" / f"{vid}.npz", data=data, display=display, grid=grid)
                meta.to_csv(PROJECT / "02_variants" / f"{vid}_metadata.csv", index=False)
                rows.append({
                    "variant_id": vid, "window_hours": window, "cohort": cohort,
                    "representation": representation, "n": len(data), "dimensions": data.shape[1] if len(data) else 0,
                    "accepted_input_total": len(accepted), "retention_fraction": len(data) / max(1, len(accepted)),
                    "primary_window_eligible": window >= config["primary_minimum_window_hours"],
                    "smoothing_applied": False, "labels_used": False,
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(PROJECT / "02_variants/variant_inventory.csv", index=False)
    common.dump_json(PROJECT / "01_data_audit/audit_summary.json", {
        "accepted_csv_scanned": int(len(inventory)), "true_pce_curves_accepted": int(len(accepted)),
        "excluded": int((~inventory["accepted"]).sum()),
        "exclusion_counts": inventory.loc[~inventory["accepted"], "exclusion_reason"].value_counts().to_dict(),
        "windows": config["windows_hours"], "cohorts": config["cohorts"],
        "smoothing_enabled": False, "source_tree_modified": False,
    })
    return summary


def load_variant(vid: str):
    z = np.load(PROJECT / "02_variants" / f"{vid}.npz")
    meta = pd.read_csv(PROJECT / "02_variants" / f"{vid}_metadata.csv")
    return z["data"], z["display"], z["grid"], meta


def kmeans_screen(config: dict, variants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows, decision_rows = [], []
    total = len(variants)
    for pos, v in variants.iterrows():
        vid = v["variant_id"]
        data, _, _, _ = load_variant(vid)
        labels_by_k: dict[int, list[np.ndarray]] = {}
        for k in config["candidate_k_kmeans"]:
            labels_by_k[k] = []
            for seed in config["kmeans_seeds"]:
                labels, centres, inertia = common.safe_kmeans(data, k, seed, n_init=3, max_iter=100)
                labels_by_k[k].append(labels)
                counts = np.bincount(labels, minlength=k)
                all_rows.append({
                    "variant_id": vid, "k": k, "seed": seed, "inertia": inertia,
                    "silhouette": common.safe_silhouette(data, labels),
                    "min_cluster_fraction": float(counts.min() / len(data)),
                })
        local = pd.DataFrame([r for r in all_rows if r["variant_id"] == vid])
        agg = local.groupby("k", as_index=False).agg(
            inertia_median=("inertia", "median"), silhouette_median=("silhouette", "median"),
            min_cluster_fraction_median=("min_cluster_fraction", "median"),
        )
        agg["seed_ari"] = [common.pairwise_ari(labels_by_k[int(k)]) for k in agg["k"]]
        elbow, strength = common.geometric_elbow(agg["k"].to_numpy(), agg["inertia_median"].to_numpy())
        base_labels = labels_by_k[elbow][0]
        subsample_labels = []
        for repeat in range(config["kmeans_subsample_repeats"]):
            rng = np.random.default_rng(9000 + repeat)
            idx = rng.choice(len(data), size=max(elbow * 5, int(0.8 * len(data))), replace=False)
            _, centres, _ = common.safe_kmeans(data[idx], elbow, 9000 + repeat, n_init=5, max_iter=100)
            subsample_labels.append(common.squared_to_centres(data, centres).argmin(axis=1))
        erow = agg[agg["k"] == elbow].iloc[0]
        decision_rows.append({
            **v.to_dict(), "elbow_k": elbow, "elbow_strength": strength,
            "inertia_at_elbow": erow["inertia_median"], "silhouette_at_elbow": erow["silhouette_median"],
            "seed_ari_at_elbow": erow["seed_ari"],
            "subsample_ari_to_full": float(np.mean([adjusted_rand_score(base_labels, x) for x in subsample_labels])),
            "min_cluster_fraction_at_elbow": erow["min_cluster_fraction_median"],
        })
        print(f"KMeans {pos + 1}/{total}: {vid} K={elbow}", flush=True)
    runs = pd.DataFrame(all_rows)
    decisions = pd.DataFrame(decision_rows)
    eligible = decisions["primary_window_eligible"] & (decisions["n"] >= 50)
    decisions["passes_support"] = eligible & (decisions["min_cluster_fraction_at_elbow"] >= config["minimum_cluster_fraction"])
    rank_source = decisions.loc[eligible].copy()
    for col, ascending in (
        ("silhouette_at_elbow", False), ("subsample_ari_to_full", False), ("seed_ari_at_elbow", False),
        ("elbow_strength", False), ("min_cluster_fraction_at_elbow", False), ("n", False),
    ):
        decisions.loc[eligible, "rank_" + col] = rank_source[col].rank(ascending=ascending, method="average")
    rank_cols = [c for c in decisions.columns if c.startswith("rank_")]
    decisions["kmeans_rank_sum"] = decisions[rank_cols].sum(axis=1, min_count=len(rank_cols))
    decisions["selected_for_som"] = False
    selected = []
    # One best primary candidate per pre-registered representation.
    for representation in config["representations"]:
        pool = decisions[(decisions["representation"] == representation) & decisions["passes_support"]]
        if len(pool):
            selected.append(pool.sort_values(["kmeans_rank_sum", "variant_id"]).index[0])
    decisions.loc[selected[:config["top_dataset_variants_for_som"]], "selected_for_som"] = True
    runs.to_csv(PROJECT / "03_kmeans_screen/all_runs.csv", index=False)
    decisions.sort_values(["selected_for_som", "kmeans_rank_sum"], ascending=[False, True]).to_csv(
        PROJECT / "03_kmeans_screen/variant_decisions.csv", index=False
    )
    common.dump_json(PROJECT / "03_kmeans_screen/frozen_som_inputs.json", {
        "selected_variant_ids": decisions.loc[decisions["selected_for_som"], "variant_id"].tolist(),
        "selection_used_ifo_names": False,
        "selection_rule": "best eligible unsupervised KMeans screen candidate per pre-registered representation",
    })
    return runs, decisions


def aggregate_som(runs: pd.DataFrame, labels_store: dict) -> pd.DataFrame:
    rows = []
    groups = ["variant_id", "setting_id", "k"]
    for keys, g in runs.groupby(groups, sort=False):
        vid, sid, k = keys
        labels = [labels_store[(vid, sid, int(k), int(seed))] for seed in g["seed"]]
        row = {"variant_id": vid, "setting_id": sid, "k": int(k), "seed_ari": common.pairwise_ari(labels)}
        for col in (
            "quantization_error", "qe_per_sqrt_dimension", "topographic_error", "silhouette",
            "occupied_nodes", "empty_nodes", "min_cluster_fraction", "minimum_center_rmse",
            "median_within_rmse", "center_separation_ratio",
        ):
            row[col + "_median"] = float(g[col].median())
        for col in ("sigma", "learning_rate", "random_order", "window_hours", "cohort", "representation", "n", "dimensions", "retention_fraction"):
            row[col] = g.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows)


def som_search(config: dict, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = decisions[decisions["selected_for_som"]].copy()
    settings = [(s, lr, ro) for s in config["som_sigmas"] for lr in config["som_learning_rates"] for ro in config["som_random_order"]]
    rows, labels_store = [], {}
    total = len(selected) * len(settings) * len(config["candidate_k_som"]) * len(config["screen_seeds"])
    done = 0
    for _, v in selected.iterrows():
        data, _, _, _ = load_variant(v["variant_id"])
        for sigma, lr, ro in settings:
            sid = f"sigma{sigma:g}_lr{lr:g}_{'random' if ro else 'sequential'}"
            for k in config["candidate_k_som"]:
                for seed in config["screen_seeds"]:
                    _, _, labels, metric = common.train_som(
                        data, k, sigma, lr, ro, seed, config["som_screen_iterations"]
                    )
                    metric.update({
                        "variant_id": v["variant_id"], "setting_id": sid, "k": k, "seed": seed,
                        "sigma": sigma, "learning_rate": lr, "random_order": ro,
                        "window_hours": v["window_hours"], "cohort": v["cohort"],
                        "representation": v["representation"], "n": v["n"], "dimensions": v["dimensions"],
                        "retention_fraction": v["retention_fraction"],
                    })
                    rows.append(metric); labels_store[(v["variant_id"], sid, k, seed)] = labels
                    done += 1
                    if done % 100 == 0 or done == total:
                        print(f"SOM search {done}/{total}", flush=True)
    runs = pd.DataFrame(rows)
    # Persist the expensive raw grid before any downstream aggregation.
    runs.to_csv(PROJECT / "04_som_search/all_runs.csv", index=False)
    summary = aggregate_som(runs, labels_store)
    summary.to_csv(PROJECT / "04_som_search/summary_by_k.csv", index=False)
    decision_rows = []
    for (vid, sid), group in summary.groupby(["variant_id", "setting_id"], sort=False):
        group = group.sort_values("k")
        elbow, strength = common.geometric_elbow(group["k"].to_numpy(), group["quantization_error_median"].to_numpy())
        r = group[group["k"] == elbow].iloc[0]
        decision_rows.append({
            "variant_id": vid, "setting_id": sid, "elbow_k": elbow, "qe_elbow_strength": strength,
            **{c: r[c] for c in (
                "sigma", "learning_rate", "random_order", "window_hours", "cohort", "representation", "n", "dimensions",
                "retention_fraction", "quantization_error_median", "qe_per_sqrt_dimension_median",
                "topographic_error_median", "silhouette_median", "seed_ari", "occupied_nodes_median",
                "empty_nodes_median", "min_cluster_fraction_median", "center_separation_ratio_median",
            )},
        })
    som_decisions = pd.DataFrame(decision_rows)
    som_decisions["passes_support"] = (
        (som_decisions["empty_nodes_median"] == 0)
        & (som_decisions["min_cluster_fraction_median"] >= config["minimum_cluster_fraction"])
    )
    valid = som_decisions[som_decisions["passes_support"]]
    for col, ascending in (
        ("silhouette_median", False), ("seed_ari", False), ("topographic_error_median", True),
        ("center_separation_ratio_median", False), ("min_cluster_fraction_median", False),
        ("retention_fraction", False), ("qe_elbow_strength", False),
    ):
        som_decisions.loc[valid.index, "rank_" + col] = valid[col].rank(ascending=ascending, method="average")
    rank_cols = [c for c in som_decisions if c.startswith("rank_")]
    som_decisions["som_rank_sum"] = som_decisions[rank_cols].sum(axis=1, min_count=len(rank_cols))
    som_decisions.sort_values(["passes_support", "som_rank_sum"], ascending=[False, True]).to_csv(
        PROJECT / "04_som_search/setting_decisions.csv", index=False
    )
    return runs, summary, som_decisions


def freeze_selection(config: dict, som_decisions: pd.DataFrame) -> dict:
    chosen = som_decisions[som_decisions["passes_support"]].sort_values(["som_rank_sum", "variant_id", "setting_id"]).iloc[0].to_dict()
    frozen = {
        "variant_id": chosen["variant_id"], "window_hours": int(chosen["window_hours"]),
        "cohort": chosen["cohort"], "representation": chosen["representation"],
        "selected_k": int(chosen["elbow_k"]), "sigma": float(chosen["sigma"]),
        "learning_rate": float(chosen["learning_rate"]), "random_order": bool(chosen["random_order"]),
        "screen_rank_sum": float(chosen["som_rank_sum"]),
        "selection_used_ifo_names": False, "selection_used_target_k4": False,
        "smoothing_enabled": False,
        "selection_rule": "QE elbow within each SOM setting; label-blind rank aggregation across eligible settings",
    }
    common.dump_json(PROJECT / "05_frozen_selection/frozen_selection.json", frozen)
    return frozen


def final_fit(config: dict, frozen: dict):
    data, display, grid, meta = load_variant(frozen["variant_id"])
    models, weights_list, label_sets, metrics = [], [], [], []
    for seed in config["final_seeds"]:
        model, weights, labels, metric = common.train_som(
            data, frozen["selected_k"], frozen["sigma"], frozen["learning_rate"],
            frozen["random_order"], seed, config["som_final_iterations"]
        )
        models.append(model); weights_list.append(weights); label_sets.append(labels)
        metrics.append({"seed": seed, **metric})
        print(f"final seed {seed}: QE={metric['quantization_error']:.5f}", flush=True)
    ari = np.eye(len(label_sets))
    for i, j in itertools.combinations(range(len(label_sets)), 2):
        ari[i, j] = ari[j, i] = adjusted_rand_score(label_sets[i], label_sets[j])
    medoid_model = int(np.argmax(ari.mean(axis=1)))
    weights = weights_list[medoid_model]; labels = label_sets[medoid_model]
    medoids = common.medoid_indices(data, labels, weights)
    assignment = meta.copy()
    assignment["cluster"] = labels
    distances = np.linalg.norm(data[:, None, :] - weights[None, :, :], axis=2)
    ordered = np.sort(distances, axis=1)
    assignment["bmu_distance"] = distances[np.arange(len(data)), labels]
    assignment["assignment_margin"] = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered[:, 1], 1e-12)
    assignment.to_csv(PROJECT / "06_final_model/assignments.csv", index=False)
    np.save(PROJECT / "06_final_model/weights.npy", weights)
    np.save(PROJECT / "06_final_model/labels.npy", labels)
    pd.DataFrame(metrics).to_csv(PROJECT / "06_final_model/final_seed_metrics.csv", index=False)
    pd.DataFrame(ari, index=config["final_seeds"], columns=config["final_seeds"]).to_csv(
        PROJECT / "06_final_model/final_seed_ari_matrix.csv"
    )
    common.dump_json(PROJECT / "06_final_model/final_model.json", {
        **frozen, "selected_seed": config["final_seeds"][medoid_model],
        "mean_pairwise_seed_ari": common.pairwise_ari(label_sets),
        "cluster_sizes": np.bincount(labels, minlength=frozen["selected_k"]).astype(int).tolist(),
        "n": len(data), "dimensions": data.shape[1], "final_iterations": config["som_final_iterations"],
    })
    return data, display, grid, meta, labels, weights, medoids


def bootstrap_stability(config: dict, frozen: dict, data: np.ndarray, reference_labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in config["bootstrap_seeds"]:
        rng = np.random.default_rng(seed)
        boot = rng.choice(len(data), size=len(data), replace=True)
        _, weights, _, metrics = common.train_som(
            data[boot], frozen["selected_k"], frozen["sigma"], frozen["learning_rate"],
            frozen["random_order"], seed, config["som_bootstrap_iterations"]
        )
        predicted = np.argmin(np.linalg.norm(data[:, None, :] - weights[None, :, :], axis=2), axis=1)
        rows.append({
            "seed": seed, "ari_to_frozen": adjusted_rand_score(reference_labels, predicted),
            "occupied_nodes": len(np.unique(predicted)), "min_cluster_fraction": np.bincount(predicted, minlength=frozen["selected_k"]).min() / len(data),
            "bootstrap_qe": metrics["quantization_error"],
        })
        print(f"bootstrap seed {seed}: ARI={rows[-1]['ari_to_frozen']:.4f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / "08_stability/bootstrap_stability.csv", index=False)
    common.dump_json(PROJECT / "08_stability/stability_summary.json", {
        "bootstrap_repeats": len(result), "ari_mean": float(result["ari_to_frozen"].mean()),
        "ari_median": float(result["ari_to_frozen"].median()), "ari_p05": float(result["ari_to_frozen"].quantile(.05)),
        "ari_p95": float(result["ari_to_frozen"].quantile(.95)),
        "all_bootstraps_occupied_all_nodes": bool((result["occupied_nodes"] == frozen["selected_k"]).all()),
    })
    return result


def plot_outputs(config: dict, variants: pd.DataFrame, km_decisions: pd.DataFrame, som_summary: pd.DataFrame,
                 frozen: dict, display: np.ndarray, grid: np.ndarray, meta: pd.DataFrame,
                 labels: np.ndarray, medoids: list[int], descriptors: pd.DataFrame, stability: pd.DataFrame) -> None:
    # Coverage / quality cohorts.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    q = variants[variants["representation"] == config["representations"][0]]
    for cohort, g in q.groupby("cohort"):
        ax.plot(g["window_hours"], g["n"], marker="o", label=cohort)
    ax.axvline(config["primary_minimum_window_hours"], color="grey", ls="--", lw=1)
    ax.set(xlabel="Time window (h)", ylabel="Included curves", title="Objective curve retention by time window and QC cohort")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(PROJECT / "09_figures/01_window_cohort_coverage.png", dpi=220); plt.close(fig)

    # KMeans representation screen.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
    for rep, ax in zip(config["representations"], axes.flat):
        g = km_decisions[(km_decisions["representation"] == rep) & km_decisions["primary_window_eligible"]]
        sizes = 25 + 100 * g["subsample_ari_to_full"].clip(0, 1)
        ax.scatter(g["n"], g["silhouette_at_elbow"], s=sizes, alpha=.7)
        chosen = g[g["selected_for_som"]]
        if len(chosen):
            r = chosen.iloc[0]; ax.scatter([r["n"]], [r["silhouette_at_elbow"]], marker="*", s=220, color="black")
            ax.annotate(f"{int(r['window_hours'])}h/{r['cohort']}/K={int(r['elbow_k'])}", (r["n"], r["silhouette_at_elbow"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(rep); ax.set_xlabel("n"); ax.set_ylabel("Silhouette at elbow"); ax.grid(alpha=.15)
    fig.suptitle("Label-blind representation screen (marker size = subsample stability)")
    fig.tight_layout(); fig.savefig(PROJECT / "09_figures/02_kmeans_representation_screen.png", dpi=220); plt.close(fig)

    # QE curves for final representation, colored by sigma and split by learning rate/order only for readability.
    chosen_summary = som_summary[som_summary["variant_id"] == frozen["variant_id"]]
    best_settings = chosen_summary.groupby("setting_id")["silhouette_median"].max().nlargest(12).index
    fig, ax = plt.subplots(figsize=(10, 6))
    for sid in best_settings:
        g = chosen_summary[chosen_summary["setting_id"] == sid].sort_values("k")
        ax.plot(g["k"], g["quantization_error_median"], marker="o", lw=1, alpha=.75, label=sid)
    ax.axvline(frozen["selected_k"], color="black", ls="--", lw=1, label="final frozen K")
    ax.set(xlabel="SOM nodes (K)", ylabel="Median quantization error", title="QE curves for top parameter settings of the frozen representation")
    ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(PROJECT / "09_figures/03_som_qe_elbows.png", dpi=220); plt.close(fig)

    # Final clusters: all aligned curves are piecewise linear only; medoid is an observed curve.
    k = frozen["selected_k"]; cols = min(3, k); rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), sharex=True, sharey=True, squeeze=False)
    desc_map = descriptors.set_index("cluster").to_dict("index")
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        subset = display[labels == c]
        for curve in subset:
            ax.plot(grid, curve, lw=.38, alpha=min(.15, max(.012, 8 / max(1, len(subset)))), color="#357aa1")
        ax.plot(grid, display[medoids[c]], color="black", lw=2.0, label="actual medoid curve")
        if grid[-1] >= 200:
            ax.axvline(200, color="grey", ls="--", lw=.8)
        shape = desc_map.get(c, {}).get("posthoc_ifo_name", "")
        ax.set_title(f"Cluster {c}: n={len(subset)}; {shape}"); ax.grid(alpha=.15)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("PCE / in-window MaxAbs")
    fig.suptitle("Frozen unsupervised SOM — all members and actual medoid (smoothing OFF)")
    fig.tight_layout(); fig.savefig(PROJECT / "06_final_model/clusters/all_clusters.png", dpi=240); plt.close(fig)

    # Raw digitized points for nearest representatives, not model grid lines.
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), sharex=True, sharey=True, squeeze=False)
    provenance = []
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        idx = np.flatnonzero(labels == c)
        distance = np.linalg.norm(display[idx] - display[medoids[c]], axis=1)
        chosen_idx = idx[np.argsort(distance)[:min(8, len(idx))]]
        for rank, i in enumerate(chosen_idx, 1):
            x, y = common.read_xy(Path(meta.iloc[i]["source_file"]))
            t = (x - x[0]) * float(meta.iloc[i]["time_factor_to_hours"])
            mask = t <= grid[-1] + 1e-9
            scale = np.max(np.abs(y[mask]))
            ax.plot(t[mask], y[mask] / scale, marker=".", ms=2.8, lw=.65, alpha=.75)
            provenance.append({"cluster": c, "rank": rank, "curve_id": meta.iloc[i]["curve_id"], "source_file": meta.iloc[i]["source_file"], "legend": meta.iloc[i]["legend"]})
        if grid[-1] >= 200: ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Cluster {c}: nearest raw-point members"); ax.grid(alpha=.15)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("Raw-point PCE / in-window MaxAbs")
    fig.suptitle("Original digitized points for frozen clusters (no smoothing)")
    fig.tight_layout(); fig.savefig(PROJECT / "07_posthoc_ifo/raw_point_representatives.png", dpi=240); plt.close(fig)
    pd.DataFrame(provenance).to_csv(PROJECT / "07_posthoc_ifo/representative_provenance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(stability["ari_to_frozen"], bins=np.linspace(-.05, 1.0, 22), color="#357aa1", edgecolor="white")
    ax.axvline(stability["ari_to_frozen"].median(), color="black", ls="--", label=f"median={stability['ari_to_frozen'].median():.3f}")
    ax.set(xlabel="ARI to frozen partition", ylabel="Bootstrap runs", title="Sample-bootstrap SOM stability")
    ax.legend(); ax.grid(axis="y", alpha=.15); fig.tight_layout(); fig.savefig(PROJECT / "09_figures/04_bootstrap_stability.png", dpi=220); plt.close(fig)


def write_report(config: dict, variants: pd.DataFrame, km_decisions: pd.DataFrame, som_decisions: pd.DataFrame,
                 frozen: dict, final: dict, descriptors: pd.DataFrame, stability: pd.DataFrame) -> None:
    counts = descriptors["posthoc_ifo_name"].value_counts().to_dict()
    exact_four = frozen["selected_k"] == 4 and set(descriptors["posthoc_ifo_name"]) == {"Bridge-like", "Hill-like", "Slope-like", "Valley-like"}
    top = som_decisions.sort_values(["passes_support", "som_rank_sum"], ascending=[False, True]).head(12)
    cols = ["variant_id", "setting_id", "elbow_k", "silhouette_median", "seed_ari", "topographic_error_median", "min_cluster_fraction_median", "som_rank_sum"]
    desc_cols = ["cluster", "n", "posthoc_ifo_name", "peak_time_h", "trough_time_h", "rise", "post_peak_drop", "post_trough_recovery", "early_slope", "late_slope"]
    def md(frame: pd.DataFrame) -> str:
        f = frame.copy()
        for c in f:
            f[c] = f[c].map(lambda x: f"{x:.5g}" if isinstance(x, (float, np.floating)) else str(x))
        return "| " + " | ".join(f.columns) + " |\n| " + " | ".join("---" for _ in f.columns) + " |\n" + "\n".join("| " + " | ".join(row.astype(str)) + " |" for _, row in f.iterrows())
    conclusion = (
        "冻结模型自然产生了四个节点，且后验形态恰好一一对应 Bridge/Hill/Slope/Valley。"
        if exact_four else
        "冻结模型没有同时形成四个一一对应 Bridge/Hill/Slope/Valley 的稳定节点；因此不能宣称已无监督发现目标四类。"
    )
    text = f"""# 最终报告：无监督 PCE 曲线细节搜索

## 直接结论

本轮最终选择 **K={frozen['selected_k']}**，输入为 `{frozen['variant_id']}`，即 {frozen['window_hours']} h、`{frozen['cohort']}` 质量子集、`{frozen['representation']}` 表征。SOM 参数为 sigma={frozen['sigma']}、learning rate={frozen['learning_rate']}、random_order={frozen['random_order']}。

{conclusion}

冻结后形态计数为 `{counts}`。这些名称没有进入样本筛选、参数搜索或 K 判断。

最终 10 个随机种子的平均两两 ARI 为 {final['mean_pairwise_seed_ari']:.4f}；20 次样本 bootstrap 相对冻结分区的 ARI 中位数为 {stability['ari_to_frozen'].median():.4f}，5%–95% 区间为 {stability['ari_to_frozen'].quantile(.05):.4f}–{stability['ari_to_frozen'].quantile(.95):.4f}。

## 数据与无平滑保证

- 扫描 accepted CSV：{json.loads((PROJECT/'01_data_audit/audit_summary.json').read_text())['accepted_csv_scanned']} 条；识别为真实 PCE 且时间有效：{json.loads((PROJECT/'01_data_audit/audit_summary.json').read_text())['true_pce_curves_accepted']} 条。
- 搜索窗口：{config['windows_hours']} h；coverage/detail/dense 使用预注册的点数、首末覆盖和最大时间空档规则。
- 最终纳入 {final['n']} 条曲线。任何曲线都没有因为不像目标四类而删除。
- 所有曲线局部起伏均保留。固定输入网格仅是相邻真实点之间的分段线性插值；没有滤波、样条、Savitzky–Golay、LOWESS 或移动平均。
- 最终代表图 `raw_point_representatives.png` 回到原始数字化散点，避免密集网格造成“细节增加”的错觉。

## 搜索规模

- 数据表示组合：{len(variants)} 个 window × cohort × representation 变体。
- K-means 快速筛选运行：{len(pd.read_csv(PROJECT/'03_kmeans_screen/all_runs.csv'))} 次；每种表征独立保留一个候选进入 SOM。
- SOM 参数运行：{len(pd.read_csv(PROJECT/'04_som_search/all_runs.csv'))} 次，覆盖 sigma={config['som_sigmas']}、learning rate={config['som_learning_rates']}、顺序/随机训练、K={config['candidate_k_som']} 和多个种子。
- 类别数没有固定为4。每个 SOM 设置先按 QE 几何 elbow 选 K，再用完全无标签的质量指标进行秩聚合。

## 排名前十二的 SOM 决策

{md(top[cols])}

## 冻结后的形态描述

{md(descriptors[desc_cols])}

## 科学边界

- 参数搜索很宽并不意味着可以把某个恰好出现四种外观的次优运行升级为主结论；主结论必须服从预注册的无标签选择规则。
- 文献图像数字化曲线的采样密度、实验条件和上游处理并不一致，簇不能直接解释为单一退化机理。
- 一阶/二阶有限差分保留局部变化，但也会放大数字化噪声；因此这些表征和 level-only 表征被共同比较，而不是预先指定为正确答案。
- 150/200 h 结果保留为窗口敏感性，但主结论至少需要 300 h 以观察较长期行为。

## 文件入口

- `05_frozen_selection/frozen_selection.json`：在 IFO 命名前冻结的选择。
- `06_final_model/clusters/all_clusters.png`：所有成员与真实 medoid。
- `07_posthoc_ifo/raw_point_representatives.png`：原始数字化点。
- `07_posthoc_ifo/cluster_shape_descriptors.csv`：冻结后形态解释。
- `08_stability/bootstrap_stability.csv`：20 次样本 bootstrap。
- `03_kmeans_screen/variant_decisions.csv` 与 `04_som_search/setting_decisions.csv`：完整选择证据。
"""
    (PROJECT / "reports/final_report_cn.md").write_text(text)
    common.dump_json(PROJECT / "FINAL_RESULT.json", {
        "selected_k": frozen["selected_k"], "selected_variant": frozen["variant_id"],
        "posthoc_ifo_counts": counts, "exact_four_ifo_like_clusters": bool(exact_four),
        "mean_seed_ari": final["mean_pairwise_seed_ari"], "bootstrap_ari_median": float(stability["ari_to_frozen"].median()),
        "smoothing_enabled": False, "labels_used_before_freeze": False,
    })


def main() -> None:
    started = time.time(); ensure_dirs(); config = cfg()
    common.dump_json(PROJECT / "run_manifest.json", {
        "status": "running", "started_unix": started, "python": sys.version,
        "smoothing_enabled": False, "labels_used_for_training_or_selection": False,
    })
    variant_path = PROJECT / "02_variants/variant_inventory.csv"
    km_path = PROJECT / "03_kmeans_screen/variant_decisions.csv"
    variants = pd.read_csv(variant_path) if variant_path.exists() else prepare(config)
    if km_path.exists() and (PROJECT / "03_kmeans_screen/frozen_som_inputs.json").exists():
        km_decisions = pd.read_csv(km_path)
        km_decisions["selected_for_som"] = km_decisions["selected_for_som"].astype(bool)
        print("reusing completed KMeans screen", flush=True)
    else:
        _, km_decisions = kmeans_screen(config, variants)
    _, som_summary, som_decisions = som_search(config, km_decisions)
    frozen = freeze_selection(config, som_decisions)
    data, display, grid, meta, labels, weights, medoids = final_fit(config, frozen)
    # Post-hoc names are first computed only after frozen_selection.json exists.
    descriptors = common.describe_ifo(display, labels, medoids, grid)
    descriptors["curve_id"] = [meta.iloc[i]["curve_id"] for i in medoids]
    descriptors["source_file"] = [meta.iloc[i]["source_file"] for i in medoids]
    descriptors.to_csv(PROJECT / "07_posthoc_ifo/cluster_shape_descriptors.csv", index=False)
    stability = bootstrap_stability(config, frozen, data, labels)
    final = json.loads((PROJECT / "06_final_model/final_model.json").read_text())
    plot_outputs(config, variants, km_decisions, som_summary, frozen, display, grid, meta, labels, medoids, descriptors, stability)
    write_report(config, variants, km_decisions, som_decisions, frozen, final, descriptors, stability)
    common.dump_json(PROJECT / "run_manifest.json", {
        "status": "complete", "started_unix": started, "elapsed_seconds": time.time() - started,
        "selected_k": frozen["selected_k"], "selected_variant": frozen["variant_id"],
        "smoothing_enabled": False, "labels_used_for_training_or_selection": False,
        "source_tree_modified": False, "python": sys.version,
    })
    print(f"COMPLETE K={frozen['selected_k']} variant={frozen['variant_id']}", flush=True)


if __name__ == "__main__":
    main()
