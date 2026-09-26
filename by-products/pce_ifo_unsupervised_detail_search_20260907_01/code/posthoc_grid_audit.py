#!/usr/bin/env python3
"""Post-freeze audit: do any already-searched unsupervised models show all four IFO forms?"""

from __future__ import annotations

import itertools
import json
import math
import os
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
import run_pipeline as pipeline

OUT = PROJECT / "10_posthoc_grid_audit"
TARGET = {"Bridge-like", "Hill-like", "Slope-like", "Valley-like"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((PROJECT / "config.json").read_text())
    frozen_path = PROJECT / "05_frozen_selection/frozen_selection.json"
    if not frozen_path.exists():
        raise SystemExit("Primary model must be frozen before this audit")
    decisions = pd.read_csv(PROJECT / "04_som_search/setting_decisions.csv")
    rows = []
    best_payload = None
    total = len(decisions)
    for pos, d in decisions.iterrows():
        data, display, grid, meta = pipeline.load_variant(d["variant_id"])
        labels_list, weights_list = [], []
        for seed in config["screen_seeds"]:
            _, weights, labels, _ = common.train_som(
                data, int(d["elbow_k"]), float(d["sigma"]), float(d["learning_rate"]),
                bool(d["random_order"]), int(seed), config["som_screen_iterations"]
            )
            labels_list.append(labels); weights_list.append(weights)
        ari = np.eye(len(labels_list))
        for i, j in itertools.combinations(range(len(labels_list)), 2):
            ari[i, j] = ari[j, i] = adjusted_rand_score(labels_list[i], labels_list[j])
        medoid_seed = int(np.argmax(ari.mean(axis=1)))
        labels, weights = labels_list[medoid_seed], weights_list[medoid_seed]
        medoids = common.medoid_indices(data, labels, weights)
        desc = common.describe_ifo(display, labels, medoids, grid)
        names = set(desc["posthoc_ifo_name"])
        coverage = len(names & TARGET)
        exact = int(d["elbow_k"]) == 4 and names == TARGET
        row = {
            "variant_id": d["variant_id"], "setting_id": d["setting_id"], "elbow_k": int(d["elbow_k"]),
            "sigma": d["sigma"], "learning_rate": d["learning_rate"], "random_order": d["random_order"],
            "n": d["n"], "silhouette_median": d["silhouette_median"], "seed_ari_from_search": d["seed_ari"],
            "posthoc_target_type_coverage": coverage, "posthoc_exact_four": exact,
            "posthoc_names": ";".join(sorted(names)),
            "bridge_count": int((desc["posthoc_ifo_name"] == "Bridge-like").sum()),
            "hill_count": int((desc["posthoc_ifo_name"] == "Hill-like").sum()),
            "slope_count": int((desc["posthoc_ifo_name"] == "Slope-like").sum()),
            "valley_count": int((desc["posthoc_ifo_name"] == "Valley-like").sum()),
            "other_count": int((desc["posthoc_ifo_name"] == "Other/mixed").sum()),
            "primary_som_rank_sum": d["som_rank_sum"],
        }
        rows.append(row)
        key = (int(exact), coverage, float(d["seed_ari"]), float(d["silhouette_median"]), -float(d["som_rank_sum"]))
        if best_payload is None or key > best_payload[0]:
            best_payload = (key, row, data, display, grid, meta, labels, weights, medoids, desc)
        if (pos + 1) % 32 == 0 or pos + 1 == total:
            print(f"posthoc audit {pos + 1}/{total}", flush=True)
    result = pd.DataFrame(rows).sort_values(
        ["posthoc_exact_four", "posthoc_target_type_coverage", "seed_ari_from_search", "silhouette_median", "primary_som_rank_sum"],
        ascending=[False, False, False, False, True],
    )
    result.to_csv(OUT / "all_setting_posthoc_audit.csv", index=False)
    assert best_payload is not None
    _, best, data, display, grid, meta, labels, weights, medoids, desc = best_payload
    desc["curve_id"] = [meta.iloc[i]["curve_id"] for i in medoids]
    desc["source_file"] = [meta.iloc[i]["source_file"] for i in medoids]
    desc.to_csv(OUT / "best_morphology_candidate_descriptors.csv", index=False)
    assignment = meta.copy(); assignment["cluster"] = labels
    assignment.to_csv(OUT / "best_morphology_candidate_assignments.csv", index=False)
    common.dump_json(OUT / "audit_summary.json", {
        "audit_started_only_after_primary_freeze": True,
        "audited_settings": len(result), "exact_four_setting_count": int(result["posthoc_exact_four"].sum()),
        "maximum_target_type_coverage": int(result["posthoc_target_type_coverage"].max()),
        "best_candidate": best,
        "classification_algorithm_used_labels": False,
        "important_boundary": "IFO names selected this exploratory display post hoc; it does not replace the label-blind primary model",
    })
    plot_candidate(best, display, grid, labels, medoids, desc)
    report = f"""# 冻结后全 SOM 网格形态审计

该审计只在主模型与 K 已写入 `05_frozen_selection/frozen_selection.json` 后运行。每个被审计的分类本身仍是无标签 SOM，但这里按 IFO 名称寻找候选属于**事后目标导向检查**，不能替代全过程标签盲的主选择。

- 审计设置数：{len(result)}。
- 同时覆盖 Bridge/Hill/Slope/Valley 的 K=4 设置数：{int(result['posthoc_exact_four'].sum())}。
- 任一设置最多覆盖目标形态数：{int(result['posthoc_target_type_coverage'].max())}/4。
- 最佳形态覆盖候选：`{best['variant_id']}` / `{best['setting_id']}` / K={best['elbow_k']}。
- 该候选的搜索期 seed ARI={best['seed_ari_from_search']:.4f}，silhouette={best['silhouette_median']:.4f}，事后名称为 `{best['posthoc_names']}`。

如果 exact-four 数为0，则本轮广泛搜索仍没有找到可被诚实描述为四个目标簇的配置。如果大于0，候选图只能证明“存在某些无监督运行产生四种外观”，还需结合其稳定性与主选择排名判断是否可信。
"""
    (OUT / "README_CN.md").write_text(report)


def plot_candidate(best: dict, display: np.ndarray, grid: np.ndarray, labels: np.ndarray, medoids: list[int], desc: pd.DataFrame) -> None:
    k = int(best["elbow_k"]); cols = min(3, k); rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), sharex=True, sharey=True, squeeze=False)
    names = desc.set_index("cluster")["posthoc_ifo_name"].to_dict()
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.axis("off"); continue
        subset = display[labels == c]
        for curve in subset:
            ax.plot(grid, curve, color="#a7bfd0", lw=.35, alpha=min(.16, max(.015, 8 / max(1, len(subset)))))
        ax.plot(grid, display[medoids[c]], color="black", lw=2)
        if grid[-1] >= 200: ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Cluster {c}: n={len(subset)}; {names.get(c, '')}"); ax.grid(alpha=.15)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("PCE / in-window MaxAbs")
    fig.suptitle("Best post-freeze morphology-coverage candidate (not the primary model; smoothing OFF)")
    fig.tight_layout(); fig.savefig(OUT / "best_morphology_candidate.png", dpi=240); plt.close(fig)


if __name__ == "__main__":
    main()
