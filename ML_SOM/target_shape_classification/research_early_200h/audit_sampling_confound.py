#!/usr/bin/env python3
"""Descriptive check whether SOM nodes track sampling support."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from early_data import HERE, rows, write_rows


def eta_squared(groups, values):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mean = values.mean()
    denominator = np.sum((values - mean) ** 2)
    if denominator == 0:
        return 0.
    return float(sum(np.sum(groups == node) * (values[groups == node].mean() - mean) ** 2
                     for node in np.unique(groups)) / denominator)


def main():
    records = [r for r in rows(HERE / "delivery/class_index.csv") if r["split"] == "train"]
    groups = np.array([int(r["som_bmu"]) for r in records])
    metrics = {key: np.array([float(r[key]) for r in records]) for key in
               ("observation_end_used_h", "window_points", "max_gap_h", "model_score")}
    nodes = []
    for node in sorted(set(groups)):
        members = [r for r in records if int(r["som_bmu"]) == node]
        nodes.append(dict(node=node, train_count=len(members),
                          median_end_h=float(np.median([float(r["observation_end_used_h"]) for r in members])),
                          median_points=float(np.median([float(r["window_points"]) for r in members])),
                          median_max_gap_h=float(np.median([float(r["max_gap_h"]) for r in members]))))
    write_rows(HERE / "evaluation/sampling_by_node.csv", nodes, list(nodes[0]))
    result = dict(train_count=len(records), occupied_nodes=len(nodes),
                  eta_squared_by_node={key: eta_squared(groups, metrics[key]) for key in
                                       ("observation_end_used_h", "window_points", "max_gap_h")},
                  spearman_quantization_vs_points=float(spearmanr(metrics["model_score"],
                                                                   metrics["window_points"]).statistic),
                  note="Descriptive association only; node identity can reflect both shape and sampling support.")
    (HERE / "evaluation/sampling_confound.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
