#!/usr/bin/env python3
"""DOI-group bootstrap stability for the frozen anonymous E2 map."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from early_data import HERE, rows, write_rows
from masked_som import MaskedSOM


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    summary = json.loads((HERE / "runs/summary.json").read_text())
    split = {r["source_group"]: r["split"] for r in rows(HERE / "manifests/source_split.csv")}
    source = {r["file_id"]: r["source_group"] for r in rows(cfg["legacy_results_root"] + "/canonical_curves.csv")}
    stored = np.load(HERE / "manifests/strict_features.npz")
    data, mask = stored["data"], stored["mask"]
    ids = [str(x) for x in stored["curve_ids"]]
    sizes = [int(x) for x in stored["group_sizes"]]
    weights = [cfg["group_weights"][key] for key in ("trajectory", "velocity", "detail")]
    by_group = {}
    validation = []
    for i, fid in enumerate(ids):
        group = source[fid]
        if split[group] == "train":
            by_group.setdefault(group, []).append(i)
        elif split[group] == "validation":
            validation.append(i)
    validation = np.array(validation)
    saved = np.load(HERE / "runs" / f"{summary['selected_run']}.npz")
    baseline = MaskedSOM(cfg["map_rows"], cfg["map_columns"], sizes, weights, int(saved["seed"]))
    baseline.prototypes, baseline.supported = saved["prototypes"], saved["supported"]
    reference_bmu = baseline.assign(data[validation], mask[validation])[0]
    groups = sorted(by_group)
    records = []
    for replication in range(3):
        rng = np.random.default_rng(cfg["split_seed"] + 100 + replication)
        draw = rng.choice(groups, size=len(groups), replace=True)
        sampled = np.concatenate([by_group[group] for group in draw])
        model = MaskedSOM(cfg["map_rows"], cfg["map_columns"], sizes, weights, 47 + replication)
        model.fit(data[sampled], mask[sampled], cfg["epochs"])
        new_bmu = model.assign(data[validation], mask[validation])[0]
        metrics = model.metrics(data[validation], mask[validation])
        records.append(dict(replication=replication + 1, sampled_DOI_groups=len(draw),
                            distinct_DOI_groups=len(set(draw)), validation_bmu_ARI_vs_frozen=float(
                                adjusted_rand_score(reference_bmu, new_bmu)), **metrics))
        print(json.dumps(records[-1]), flush=True)
    write_rows(HERE / "evaluation/source_resampling.csv", records, list(records[0]))
    (HERE / "evaluation/source_resampling.json").write_text(json.dumps(dict(
        method="Train DOI groups sampled with replacement; compare validation BMU partitions using permutation-invariant ARI.",
        selected_run=summary["selected_run"], records=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
