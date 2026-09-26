#!/usr/bin/env python3
"""Label-free 25/50/100 epoch sensitivity for the selected map size."""
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
    prior = json.loads((HERE / "runs/summary.json").read_text())
    records = []
    for run in prior["runs"]:
        if run["variant"] == "E2" and run["seed"] in (41, 42, 43):
            records.append(dict(epochs=50, seed=run["seed"], **run["metrics"]["validation"]))
    for epochs in (25, 100):
        for seed in (41, 42, 43):
            model = MaskedSOM(cfg["map_rows"], cfg["map_columns"], sizes, weights, seed)
            model.fit(data[train], mask[train], epochs)
            metrics = model.metrics(data[val], mask[val])
            records.append(dict(epochs=epochs, seed=seed, **metrics))
            path = HERE / "runs" / f"E2_{cfg['map_rows']}x{cfg['map_columns']}_{epochs}epochs_seed{seed}.npz"
            np.savez_compressed(path, prototypes=model.prototypes, supported=model.supported,
                                coordinates=model.coordinates, feature_sizes=np.array(sizes),
                                weights=np.array(weights), train_ids=np.array(ids)[train], seed=seed)
            print(epochs, seed, json.dumps(metrics), flush=True)
    write_rows(HERE / "evaluation/epoch_sweep.csv", records, list(records[0]))
    means = {str(n): dict(qe_mean=statistics.mean(r["quantization_error"] for r in records if r["epochs"] == n),
                          te_mean=statistics.mean(r["topology_error"] for r in records if r["epochs"] == n))
             for n in (25, 50, 100)}
    best_q = min(r["qe_mean"] for r in means.values())
    eligible = {n: r for n, r in means.items() if r["qe_mean"] <= 1.05 * best_q}
    suggested = min(eligible, key=lambda n: (eligible[n]["te_mean"], int(n)))
    report = dict(means=means, suggested_epochs=int(suggested),
                  selection_note="Within 5% of lowest mean validation QE, then lowest mean topology error; 3 seeds, no labels.")
    (HERE / "evaluation/epoch_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
