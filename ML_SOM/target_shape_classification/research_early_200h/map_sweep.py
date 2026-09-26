#!/usr/bin/env python3
"""Label-free E2 map-size sensitivity on the pre-locked DOI split."""
from __future__ import annotations

import json
import statistics

import numpy as np

from early_data import HERE, rows, write_rows
from masked_som import MaskedSOM


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    frozen = {r["canonical_id"]: r["split"] for r in rows(HERE / "manifests/frozen_inputs.csv")}
    stored = np.load(HERE / "manifests/strict_features.npz")
    data, mask = stored["data"], stored["mask"]
    ids = [str(x) for x in stored["curve_ids"]]
    sizes = [int(x) for x in stored["group_sizes"]]
    train = np.array([i for i, fid in enumerate(ids) if frozen[fid] == "train"])
    val = np.array([i for i, fid in enumerate(ids) if frozen[fid] == "validation"])
    weights = [cfg["group_weights"][key] for key in ("trajectory", "velocity", "detail")]
    records = []
    prior = json.loads((HERE / "runs/summary.json").read_text())
    baseline_size = f"{cfg['map_rows']}x{cfg['map_columns']}"
    for run in prior["runs"]:
        if run["variant"] == "E2" and run["seed"] in (41, 42, 43):
            metrics = run["metrics"]["validation"]
            records.append(dict(map_size=baseline_size, seed=run["seed"], **metrics))
    for side in (6, 8, 10):
        if f"{side}x{side}" == baseline_size:
            continue
        for seed in (41, 42, 43):
            model = MaskedSOM(side, side, sizes, weights, seed)
            model.fit(data[train], mask[train], cfg["epochs"])
            metrics = model.metrics(data[val], mask[val])
            records.append(dict(map_size=f"{side}x{side}", seed=seed, **metrics))
            path = HERE / "runs" / f"E2_{side}x{side}_seed{seed}.npz"
            np.savez_compressed(path, prototypes=model.prototypes, supported=model.supported,
                                coordinates=model.coordinates, feature_sizes=np.array(sizes),
                                weights=np.array(weights), train_ids=np.array(ids)[train], seed=seed)
            print(side, seed, json.dumps(metrics), flush=True)
    write_rows(HERE / "evaluation/map_size_sweep.csv", records, list(records[0]))
    grouped = {}
    for size in ("6x6", "8x8", "10x10"):
        subset = [r for r in records if r["map_size"] == size]
        grouped[size] = dict(qe_mean=statistics.mean(r["quantization_error"] for r in subset),
                             te_mean=statistics.mean(r["topology_error"] for r in subset),
                             occupied_mean=statistics.mean(r["occupied_nodes"] for r in subset))
    best_q = min(x["qe_mean"] for x in grouped.values())
    eligible = {size: x for size, x in grouped.items() if x["qe_mean"] <= 1.05 * best_q}
    selected = min(eligible, key=lambda size: (eligible[size]["te_mean"], int(size.split('x')[0])))
    report = dict(grouped=grouped, suggested_size=selected,
                  selection_note="Mean validation QE within 5% of best, then lowest mean topology error; descriptive sensitivity, no labels used.")
    (HERE / "evaluation/map_size_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
