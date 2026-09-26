#!/usr/bin/env python3
"""Check if cumulative digitized variation makes the SOM track point density."""
from __future__ import annotations

import json

import numpy as np

from audit_sampling_confound import eta_squared
from early_data import HERE, rows, write_rows
from masked_som import MaskedSOM


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    saved = np.load(HERE / "manifests/strict_features.npz")
    data, full_mask = saved["data"], saved["mask"]
    ids = [str(x) for x in saved["curve_ids"]]
    sizes = [int(x) for x in saved["group_sizes"]]
    start = sizes[0] + sizes[1]
    metadata = {r["curve_id"]: r for r in rows(HERE / "delivery/class_index.csv")}
    train = np.array([i for i, fid in enumerate(ids) if metadata[fid]["split"] == "train"])
    val = np.array([i for i, fid in enumerate(ids) if metadata[fid]["split"] == "validation"])
    point_counts = np.array([float(metadata[ids[i]]["window_points"]) for i in train])
    end_times = np.array([float(metadata[ids[i]]["observation_end_used_h"]) for i in train])
    weights = [cfg["group_weights"][key] for key in ("trajectory", "velocity", "detail")]
    records = []
    for variant in ("full", "extrema_only"):
        mask = full_mask.copy()
        if variant == "extrema_only":
            # Each 10 h bin retains its raw min/max. Cumulative positive and
            # negative digital line variation remains in the audit, not SOM input.
            for k in range(20):
                mask[:, start + 4 * k + 2:start + 4 * k + 4] = False
        for seed in (41, 42, 43):
            if variant == "full":
                old = np.load(HERE / "runs" / f"E2_seed{seed}.npz")
                model = MaskedSOM(cfg["map_rows"], cfg["map_columns"], sizes, weights, seed)
                model.prototypes, model.supported = old["prototypes"], old["supported"]
            else:
                model = MaskedSOM(cfg["map_rows"], cfg["map_columns"], sizes, weights, seed)
                model.fit(data[train], mask[train], cfg["epochs"])
                np.savez_compressed(HERE / "runs" / f"E2_extrema_only_seed{seed}.npz",
                                    prototypes=model.prototypes, supported=model.supported,
                                    coordinates=model.coordinates, feature_sizes=np.array(sizes),
                                    weights=np.array(weights), train_ids=np.array(ids)[train], seed=seed)
            bmu = model.assign(data[train], mask[train])[0]
            metrics = model.metrics(data[val], mask[val])
            records.append(dict(variant=variant, seed=seed,
                                eta2_node_vs_point_count=eta_squared(bmu, point_counts),
                                eta2_node_vs_window_end=eta_squared(bmu, end_times), **metrics))
            print(json.dumps(records[-1]), flush=True)
    write_rows(HERE / "evaluation/detail_ablation.csv", records, list(records[0]))
    (HERE / "evaluation/detail_ablation.json").write_text(json.dumps(dict(
        note="Distances differ between variants; compare sampling association and within-variant seed stability, not raw QE across variants.",
        records=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
