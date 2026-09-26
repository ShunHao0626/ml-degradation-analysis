"""Unlabeled stability check under small independent PCE perturbations."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from run import CONFIG, OUT, all_pairs, distances, embed_and_cluster, exact_neighbor_graph, features, observations, read_rows


def best_jaccard(base: np.ndarray, changed: np.ndarray) -> list[dict]:
    result: list[dict] = []
    for cluster in sorted(set(base)):
        members = base == cluster
        options = [(np.count_nonzero(members & (changed == c)) /
                    np.count_nonzero(members | (changed == c)), int(c),
                    int(np.count_nonzero(members & (changed == c))),
                    int(np.count_nonzero(changed == c))) for c in set(changed)]
        overlap, best_cluster, intersection, changed_count = max(options)
        result.append(dict(original_cluster=int(cluster), original_count=int(np.count_nonzero(members)),
                           best_changed_cluster=best_cluster, changed_count=changed_count,
                           shared_count=intersection, best_jaccard=round(float(overlap), 6)))
    return result


def main() -> None:
    index = read_rows(OUT / "input_index.csv")
    base = read_rows(OUT / "anonymous_assignments.csv")
    assert [r["curve_id"] for r in index] == [r["curve_id"] for r in base]
    pairs = all_pairs(len(index))
    rng = np.random.default_rng(20260926)
    grid = CONFIG["rank_grid_points"]
    report = {}
    for noise_sd in (0.003, 0.005):
        arrays = [[], [], []]
        for record in index:
            values = observations(record)
            changed = np.maximum(values + rng.normal(0, noise_sd, len(values)), 0)
            changed /= max(float(np.max(changed)), 1e-12)
            channels = features(changed, grid)
            for target, channel in zip(arrays, channels):
                target.append(channel)
        level, short, long = [np.ascontiguousarray(a, dtype=np.float64) for a in arrays]
        d = distances(level, short, long, pairs, save=False)
        edges, edge_d = exact_neighbor_graph(pairs, d, len(index), save=False)
        clusters = embed_and_cluster(edges, edge_d, len(index), save=False)
        item = {}
        for k, changed in clusters.items():
            original = np.array([int(r[f"anonymous_k{k}"]) for r in base])
            rows = best_jaccard(original, changed)
            with (OUT / f"noise_{str(noise_sd).replace('.', 'p')}_k{k}_overlap.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            overlaps = [r["best_jaccard"] for r in rows]
            item[str(k)] = dict(ari=round(float(adjusted_rand_score(original, changed)), 4),
                                median_best_jaccard=round(float(np.median(overlaps)), 4),
                                clusters_best_jaccard_at_least_0_5=sum(v >= 0.5 for v in overlaps),
                                cluster_count=len(overlaps))
        report[str(noise_sd)] = item
        print(f"Perturbation SD {noise_sd}: {item}", flush=True)
    (OUT / "unlabeled_noise_sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
