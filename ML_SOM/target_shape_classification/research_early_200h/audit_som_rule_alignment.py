#!/usr/bin/env python3
"""Measure how the frozen SOM nodes align with independent stage-rule candidates.

The rule labels are pseudo labels, not human references. Node-to-class mappings
are fitted on training DOI groups only and scored as agreement on held-out DOI
groups. No SOM weights or four-class deliveries are changed.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, normalized_mutual_info_score)

from early_data import CLASSES, HERE, digest, rows, write_rows


OUT = HERE / "evaluation" / "som_rule_alignment_20260925"
SPLITS = ("train", "validation", "test")
NODES = 100


def scores(truth, predicted):
    return dict(count=len(truth), agreement_with_rule=float(np.mean(np.array(truth) == np.array(predicted))),
                macro_f1_against_rule=float(f1_score(truth, predicted, labels=CLASSES,
                                                    average="macro", zero_division=0)),
                balanced_accuracy_against_rule=float(balanced_accuracy_score(truth, predicted)),
                confusion_against_rule=confusion_matrix(truth, predicted, labels=CLASSES).tolist())


def render_map(path, train_counts, majority, class_balanced):
    colors = ("#5373a5", "#d28759", "#779469", "#bc72a8")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    occupancy = np.array([sum(train_counts[(node, cls)] for cls in CLASSES)
                          for node in range(NODES)]).reshape(10, 10)
    majority_grid = np.array([CLASSES.index(majority[node]) for node in range(NODES)]).reshape(10, 10)
    balanced_grid = np.array([CLASSES.index(class_balanced[node]) for node in range(NODES)]).reshape(10, 10)
    entropy = []
    for node in range(NODES):
        counts = np.array([train_counts[(node, cls)] for cls in CLASSES], dtype=float)
        prob = counts / counts.sum() if counts.sum() else np.zeros(len(CLASSES))
        nonzero = prob[prob > 0]
        entropy.append(float(-np.sum(nonzero * np.log2(nonzero)) / 2))
    entropy_grid = np.array(entropy).reshape(10, 10)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    specs = ((occupancy, "Training curves per SOM node", "viridis", None),
             (majority_grid, "Training rule-label majority", cmap, norm),
             (balanced_grid, "Training class-balanced rule mapping", cmap, norm),
             (entropy_grid, "Training rule-label entropy / 2 bits", "magma", None))
    for ax, (values, title, palette, color_norm) in zip(axes.flat, specs):
        image = ax.imshow(values, cmap=palette, norm=color_norm, interpolation="nearest",
                          vmin=None if color_norm else 0,
                          vmax=None if color_norm or title.startswith("Training curves") else 1)
        ax.set(title=title, xlabel="SOM column", ylabel="SOM row")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.grid(color="white", linewidth=.4, alpha=.6)
        if color_norm is None:
            fig.colorbar(image, ax=ax, shrink=.8)
    handles = [Patch(facecolor=colors[i], label=cls.title()) for i, cls in enumerate(CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.025))
    fig.suptitle("Frozen E2 SOM vs 0–200 h automatic stage-rule candidates (training DOI groups)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    old = {r["curve_id"]: r for r in rows(HERE / "delivery" / "class_index.csv")}
    stage = rows(HERE / "diagram_stage_20260925" / "stage_features.csv")
    selected = json.loads((HERE / "runs" / "summary.json").read_text(encoding="utf-8"))
    if selected["selected_run"] != "E2_seed43" or len(stage) != 2205 or len(old) != 2205:
        raise ValueError("frozen SOM or stage cohort changed")
    if digest(HERE / "runs" / "E2_seed43.npz") != selected["selected_model_sha256"]:
        raise ValueError("frozen SOM model hash changed")
    records = []
    for row in stage:
        previous = old[row["curve_id"]]
        if (row["source_group"], row["split"], row["source_sha256"]) != (
                previous["source_group"], previous["split"], previous["source_sha256"]):
            raise ValueError("source identity or DOI split changed: " + row["curve_id"])
        node = int(previous["som_bmu"])
        if not 0 <= node < NODES or row["candidate_class"] not in CLASSES:
            raise ValueError("missing SOM node or stage-rule candidate: " + row["curve_id"])
        records.append(dict(curve_id=row["curve_id"], source_group=row["source_group"],
                            split=row["split"], som_bmu=node,
                            rule_candidate_class=row["candidate_class"],
                            evidence_status=row["evidence_status"],
                            som_posthoc_class=previous["som_posthoc_class"]))
    groups_by_split = {split: {r["source_group"] for r in records if r["split"] == split}
                       for split in SPLITS}
    if any(groups_by_split[a] & groups_by_split[b] for a in SPLITS for b in SPLITS if a < b):
        raise ValueError("DOI source groups overlap across splits")
    train = [r for r in records if r["split"] == "train"]
    counts = Counter((r["som_bmu"], r["rule_candidate_class"]) for r in train)
    class_sizes = Counter(r["rule_candidate_class"] for r in train)
    if set(class_sizes) != set(CLASSES):
        raise ValueError("all four candidate classes must appear in training")
    # Both mappings are frozen from training DOI groups; lexical order settles ties.
    majority = {node: max(CLASSES, key=lambda cls: (counts[(node, cls)], -CLASSES.index(cls)))
                for node in range(NODES)}
    class_balanced = {node: max(CLASSES,
                                key=lambda cls: (counts[(node, cls)] / class_sizes[cls],
                                                 -CLASSES.index(cls)))
                      for node in range(NODES)}
    rows_out = []
    for record in records:
        rows_out.append(dict(record, train_node_majority=majority[record["som_bmu"]],
                             train_node_class_balanced=class_balanced[record["som_bmu"]]))
    nodes_out = []
    for node in range(NODES):
        for split in SPLITS:
            members = [r for r in records if r["som_bmu"] == node and r["split"] == split]
            class_counts = Counter(r["rule_candidate_class"] for r in members)
            nodes_out.append(dict(som_bmu=node, split=split, count=len(members),
                                  bridge=class_counts["bridge"], hill=class_counts["hill"],
                                  slope=class_counts["slope"], valley=class_counts["valley"],
                                  train_node_majority=majority[node],
                                  train_node_class_balanced=class_balanced[node]))
    report = dict(protocol="frozen_som_vs_diagram_stage_rule_v1",
                  som_model_sha256=selected["selected_model_sha256"],
                  stage_features_sha256=digest(HERE / "diagram_stage_20260925" /
                                               "stage_features.csv"),
                  cohort=len(records), train_source_groups=len({r["source_group"] for r in train}),
                  class_order=CLASSES, train_class_counts=dict(class_sizes),
                  mapping_rules=dict(
                      majority="Argmax training count per BMU; class order breaks ties.",
                      class_balanced="Argmax training node count divided by training class count; "
                                     "class order breaks ties. No validation tuning."),
                  splits={})
    for split in SPLITS:
        subset = [r for r in rows_out if r["split"] == split]
        truth = [r["rule_candidate_class"] for r in subset]
        bmu = [r["som_bmu"] for r in subset]
        split_report = dict(count=len(subset), source_groups=len({r["source_group"] for r in subset}),
                            class_counts=dict(Counter(truth)),
                            adjusted_mutual_information=float(adjusted_mutual_info_score(truth, bmu)),
                            normalized_mutual_information=float(normalized_mutual_info_score(truth, bmu)),
                            global_slope=scores(truth, ["slope"] * len(subset)),
                            train_node_majority=scores(truth, [r["train_node_majority"] for r in subset]),
                            train_node_class_balanced=scores(
                                truth, [r["train_node_class_balanced"] for r in subset]))
        high = [r for r in subset if r["evidence_status"] == "stage_candidate"]
        if high:
            split_report["stage_candidate_subset"] = dict(
                count=len(high), class_counts=dict(Counter(r["rule_candidate_class"] for r in high)),
                train_node_majority=scores([r["rule_candidate_class"] for r in high],
                                           [r["train_node_majority"] for r in high]),
                train_node_class_balanced=scores([r["rule_candidate_class"] for r in high],
                                                 [r["train_node_class_balanced"] for r in high]))
        report["splits"][split] = split_report
    report["interpretation"] = ("All four-class quantities are agreement with an automatic 0–200 h "
                                "stage rule, not accuracy against independent human references. "
                                "The frozen SOM is not refitted or selected with these classes.")
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "per_curve.csv", rows_out, list(rows_out[0]))
    write_rows(OUT / "node_class_counts.csv", nodes_out, list(nodes_out[0]))
    render_map(OUT / "node_rule_alignment.png", counts, majority, class_balanced)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cohort": report["cohort"],
                      "validation": {key: report["splits"]["validation"][key]
                                     for key in ("adjusted_mutual_information", "global_slope",
                                                 "train_node_majority", "train_node_class_balanced")},
                      "test": {key: report["splits"]["test"][key]
                               for key in ("adjusted_mutual_information", "global_slope",
                                           "train_node_majority", "train_node_class_balanced")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
