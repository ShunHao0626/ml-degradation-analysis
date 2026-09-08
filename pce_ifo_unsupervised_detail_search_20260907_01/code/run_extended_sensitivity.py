#!/usr/bin/env python3
"""All-window/all-cohort SOM sensitivity, performed after the primary freeze."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplcache"))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import common
import run_pipeline as pipeline

OUT = PROJECT / "11_extended_window_cohort_sensitivity"
SETTINGS = [
    {"id": "paper_sigma03_lr01_seq", "sigma": 0.3, "learning_rate": 0.1, "random_order": False},
    {"id": "paper_sigma05_lr01_seq", "sigma": 0.5, "learning_rate": 0.1, "random_order": False},
    {"id": "paper_sigma05_lr03_seq", "sigma": 0.5, "learning_rate": 0.3, "random_order": False},
    {"id": "detail_sigma12_lr05_seq", "sigma": 1.2, "learning_rate": 0.5, "random_order": False},
    {"id": "random_sigma05_lr01", "sigma": 0.5, "learning_rate": 0.1, "random_order": True},
    {"id": "random_sigma12_lr03", "sigma": 1.2, "learning_rate": 0.3, "random_order": True},
]
TARGET = {"Bridge-like", "Hill-like", "Slope-like", "Valley-like"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((PROJECT / "config.json").read_text())
    if not (PROJECT / "05_frozen_selection/frozen_selection.json").exists():
        raise SystemExit("Primary result must be frozen first")
    variants = pd.read_csv(PROJECT / "02_variants/variant_inventory.csv")
    rows, labels_store = [], {}
    total = len(variants) * len(SETTINGS) * len(config["candidate_k_som"]) * len(config["screen_seeds"])
    done = 0
    for _, v in variants.iterrows():
        data, _, _, _ = pipeline.load_variant(v["variant_id"])
        for setting in SETTINGS:
            for k in config["candidate_k_som"]:
                for seed in config["screen_seeds"]:
                    _, _, labels, metric = common.train_som(
                        data, k, setting["sigma"], setting["learning_rate"], setting["random_order"],
                        seed, config["som_screen_iterations"]
                    )
                    metric.update({
                        "variant_id": v["variant_id"], "setting_id": setting["id"], "k": k, "seed": seed,
                        "sigma": setting["sigma"], "learning_rate": setting["learning_rate"],
                        "random_order": setting["random_order"], "window_hours": v["window_hours"],
                        "cohort": v["cohort"], "representation": v["representation"], "n": v["n"],
                        "dimensions": v["dimensions"], "retention_fraction": v["retention_fraction"],
                    })
                    rows.append(metric); labels_store[(v["variant_id"], setting["id"], k, seed)] = labels
                    done += 1
                    if done % 300 == 0 or done == total:
                        print(f"extended SOM {done}/{total}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "all_runs.csv", index=False)
    summary = pipeline.aggregate_som(runs, labels_store)
    summary.to_csv(OUT / "summary_by_k.csv", index=False)
    decisions = []
    for (vid, sid), group in summary.groupby(["variant_id", "setting_id"], sort=False):
        group = group.sort_values("k")
        elbow, strength = common.geometric_elbow(group["k"].to_numpy(), group["quantization_error_median"].to_numpy())
        r = group[group["k"] == elbow].iloc[0]
        decisions.append({
            "variant_id": vid, "setting_id": sid, "elbow_k": elbow, "qe_elbow_strength": strength,
            **{c: r[c] for c in (
                "sigma", "learning_rate", "random_order", "window_hours", "cohort", "representation", "n",
                "dimensions", "retention_fraction", "quantization_error_median", "topographic_error_median",
                "silhouette_median", "seed_ari", "empty_nodes_median", "min_cluster_fraction_median",
                "center_separation_ratio_median",
            )},
        })
    decisions = pd.DataFrame(decisions)
    decisions["primary_window_eligible"] = decisions["window_hours"] >= config["primary_minimum_window_hours"]
    decisions["passes_support"] = (
        decisions["primary_window_eligible"] & (decisions["empty_nodes_median"] == 0)
        & (decisions["min_cluster_fraction_median"] >= config["minimum_cluster_fraction"])
    )
    valid = decisions[decisions["passes_support"]]
    for col, ascending in (
        ("silhouette_median", False), ("seed_ari", False), ("topographic_error_median", True),
        ("center_separation_ratio_median", False), ("min_cluster_fraction_median", False),
        ("retention_fraction", False), ("qe_elbow_strength", False),
    ):
        decisions.loc[valid.index, "rank_" + col] = valid[col].rank(ascending=ascending, method="average")
    rank_cols = [c for c in decisions if c.startswith("rank_")]
    decisions["rank_sum"] = decisions[rank_cols].sum(axis=1, min_count=len(rank_cols))
    decisions = decisions.sort_values(["passes_support", "rank_sum"], ascending=[False, True])
    decisions.to_csv(OUT / "label_blind_decisions.csv", index=False)
    best = decisions[decisions["passes_support"]].iloc[0].to_dict()
    common.dump_json(OUT / "extended_label_blind_selection.json", {
        "selection_completed_before_ifo_naming": True,
        "best": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in best.items() if not str(k).startswith("rank_")},
        "settings": SETTINGS, "smoothing_enabled": False, "labels_used": False,
    })

    # Only now perform morphology audit of every already-selected elbow model.
    audit_rows = []
    for pos, d in decisions.iterrows():
        data, display, grid, meta = pipeline.load_variant(d["variant_id"])
        label_sets, weights_sets = [], []
        for seed in config["screen_seeds"]:
            _, weights, labels, _ = common.train_som(
                data, int(d["elbow_k"]), float(d["sigma"]), float(d["learning_rate"]), bool(d["random_order"]),
                int(seed), config["som_screen_iterations"]
            )
            label_sets.append(labels); weights_sets.append(weights)
        ari = np.eye(len(label_sets))
        for i, j in itertools.combinations(range(len(label_sets)), 2):
            ari[i, j] = ari[j, i] = adjusted_rand_score(label_sets[i], label_sets[j])
        m = int(np.argmax(ari.mean(axis=1)))
        medoids = common.medoid_indices(data, label_sets[m], weights_sets[m])
        desc = common.describe_ifo(display, label_sets[m], medoids, grid)
        names = set(desc["posthoc_ifo_name"])
        audit_rows.append({
            "variant_id": d["variant_id"], "setting_id": d["setting_id"], "elbow_k": int(d["elbow_k"]),
            "window_hours": d["window_hours"], "cohort": d["cohort"], "representation": d["representation"],
            "n": d["n"], "seed_ari": d["seed_ari"], "silhouette_median": d["silhouette_median"],
            "label_blind_rank_sum": d["rank_sum"], "target_type_coverage": len(names & TARGET),
            "exact_four": int(d["elbow_k"]) == 4 and names == TARGET,
            "names": ";".join(sorted(names)),
        })
        if len(audit_rows) % 72 == 0:
            print(f"extended posthoc {len(audit_rows)}/{len(decisions)}", flush=True)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["exact_four", "target_type_coverage", "seed_ari", "silhouette_median", "label_blind_rank_sum"],
        ascending=[False, False, False, False, True],
    )
    audit.to_csv(OUT / "posthoc_morphology_audit.csv", index=False)
    common.dump_json(OUT / "extended_summary.json", {
        "variants": len(variants), "settings": len(SETTINGS), "som_runs": len(runs),
        "elbow_models_audited": len(audit), "exact_four_count": int(audit["exact_four"].sum()),
        "maximum_target_type_coverage": int(audit["target_type_coverage"].max()),
        "best_posthoc_candidate": audit.iloc[0].to_dict(),
        "important_boundary": "posthoc morphology audit does not replace label-blind selection",
    })
    (OUT / "README_CN.md").write_text(
        "# 全窗口/质量子集 SOM 敏感性\n\n"
        f"在主模型冻结后，对72个数据变体分别运行6个代表性 SOM 设置、K=2–8和3个种子，共 {len(runs)} 次。"
        "分类与 K 判断不使用 IFO 名称。\n\n"
        f"冻结后审计 {len(audit)} 个 elbow 模型；同时覆盖四种 IFO-like 形态的模型数为 {int(audit['exact_four'].sum())}，"
        f"最大覆盖 {int(audit['target_type_coverage'].max())}/4。详细结果见 `posthoc_morphology_audit.csv`。\n"
    )
    print(f"EXTENDED COMPLETE exact_four={int(audit['exact_four'].sum())} max_coverage={int(audit['target_type_coverage'].max())}")


if __name__ == "__main__":
    main()
