#!/usr/bin/env python3
"""Exploratory, label-free grouping of the locked E2 SOM prototypes.

No rule, human, or final four-class label is read when forming groups or
selecting k. Semantic naming and independent validation remain separate.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from early_data import HERE, digest, rows, write_rows
from masked_som import MaskedSOM


OUT = HERE / "evaluation/unsupervised_groups_20260925"
KS = range(2, 13)
SEEDS = range(41, 51)


def load_model(path: Path) -> tuple[MaskedSOM, np.ndarray]:
    saved = np.load(path)
    model = MaskedSOM(10, 10, list(saved["feature_sizes"]),
                      list(saved["weights"]), int(saved["seed"]))
    model.prototypes = saved["prototypes"]
    model.supported = saved["supported"]
    return model, saved["train_ids"]


def prototype_distances(model: MaskedSOM) -> np.ndarray:
    """The frozen SOM's weighted block RMS distance, with common support."""
    squared = np.zeros((100, 100))
    denominator = 0.0
    for weight, sl in zip(model.weights, model.slices):
        support = model.supported[sl]
        if not support.any():
            continue
        block = model.prototypes[:, sl][:, support]
        squared += weight * np.mean((block[:, None, :] - block[None, :, :]) ** 2, axis=2)
        denominator += weight
    distance = np.sqrt(squared / denominator)
    np.fill_diagonal(distance, 0)
    return distance


def labels_for_k(distance: np.ndarray, k: int) -> np.ndarray:
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                    linkage="average").fit_predict(distance)
    return labels.astype(int)


def numbered_groups(labels: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Name anonymous groups G1, G2, ... by descending member count."""
    sizes = np.bincount(labels, weights=occupancy, minlength=labels.max() + 1)
    order = sorted(range(len(sizes)), key=lambda group: (-sizes[group], group))
    lookup = {old: new + 1 for new, old in enumerate(order)}
    return np.array([lookup[value] for value in labels], dtype=int)


def strict_points(path: Path) -> np.ndarray:
    by_hour = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                hour, pce = float(row["hour"]), float(row["normalized_pce"])
            except (ValueError, KeyError):
                continue
            if 0 <= hour <= 200 and np.isfinite(pce):
                by_hour[hour].append(pce)
    result = np.asarray([(hour, np.median(by_hour[hour])) for hour in sorted(by_hour)])
    if result.ndim != 2 or len(result) < 2:
        raise ValueError(f"Missing strict numeric points: {path}")
    return result


def representative(group: int, memberships: np.ndarray, bmu: np.ndarray,
                   train: np.ndarray, data: np.ndarray, mask: np.ndarray,
                   model: MaskedSOM, distances: np.ndarray, occupancy: np.ndarray,
                   index: dict[str, dict], ids: list[str]) -> str:
    nodes = np.flatnonzero(memberships == group)
    center = nodes[np.argmin((distances[np.ix_(nodes, nodes)] * occupancy[nodes][None, :]).sum(axis=1))]
    candidates = []
    for i in np.flatnonzero((memberships[bmu] == group) & train):
        points = strict_points(HERE / "delivery" / index[ids[i]]["curve_csv"])
        if len(points) >= 5 and points[0, 0] <= 40 and points[-1, 0] >= 160 \
                and np.max(np.diff(points[:, 0])) <= 40:
            candidates.append((model.distance_one(data[i], mask[i])[center], ids[i]))
    if not candidates:
        for i in np.flatnonzero((memberships[bmu] == group) & train):
            candidates.append((model.distance_one(data[i], mask[i])[center], ids[i]))
    return min(candidates)[1]


def plot_map(path: Path, groups: np.ndarray, occupancy: np.ndarray, k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    axes[0].imshow(groups.reshape(10, 10), cmap="tab10", vmin=1, vmax=10,
                   interpolation="nearest")
    axes[1].imshow(occupancy.reshape(10, 10), cmap="Blues", interpolation="nearest")
    for ax, title in zip(axes, (f"Anonymous prototype groups: k={k}",
                                "All-curve SOM occupancy")):
        ax.set(title=title, xlabel="SOM column", ylabel="SOM row")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
    for node, group in enumerate(groups):
        axes[0].text(node % 10, node // 10, str(group), ha="center", va="center",
                     fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_selection(path: Path, rows_in: list[dict]) -> None:
    k = [int(row["k"]) for row in rows_in]
    node = [float(row["node_silhouette"]) for row in rows_in]
    weighted = [float(row["occupied_curve_weighted_silhouette"]) for row in rows_in]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(k, node, "o-", label="Each prototype weighted equally")
    ax.plot(k, weighted, "s-", label="Weighted by curve occupancy")
    ax.axvline(4, color="#bd772c", linestyle="--", linewidth=1,
               label="Predefined four-shape hypothesis")
    ax.set(xlabel="Number of anonymous groups k", ylabel="Prototype silhouette",
           title="Label-free SOM prototype grouping: k=2–12", xticks=k)
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_curves(path: Path, groups: np.ndarray, bmu: np.ndarray, ids: list[str],
                index: dict[str, dict], representative_ids: dict[int, str], k: int) -> None:
    columns = 2
    rows_count = k // 2
    fig, axes = plt.subplots(rows_count, columns, figsize=(11, 4.8 * rows_count), dpi=150,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    curve_groups = groups[bmu]
    for group, ax in enumerate(axes, 1):
        members = np.flatnonzero(curve_groups == group)
        last_hours = []
        for i in members:
            points = strict_points(HERE / "delivery" / index[ids[i]]["curve_csv"])
            last_hours.append(points[-1, 0])
            breaks = np.flatnonzero(np.diff(points[:, 0]) > 40) + 1
            for segment in np.split(points, breaks):
                if len(segment) >= 2:
                    ax.plot(segment[:, 0], segment[:, 1], color="#526371",
                            alpha=(.27 if len(members) < 20 else
                                   .045 if len(members) < 500 else .012), linewidth=.7)
        fid = representative_ids[group]
        full_window_count = int(np.sum(np.asarray(last_hours) >= 160))
        if full_window_count / len(members) >= .2:
            points = strict_points(HERE / "delivery" / index[fid]["curve_csv"])
            breaks = np.flatnonzero(np.diff(points[:, 0]) > 40) + 1
            for segment in np.split(points, breaks):
                if len(segment) >= 2:
                    ax.plot(segment[:, 0], segment[:, 1], color="#c6262e", linewidth=2.2)
        ax.set(title=f"G{group}: {len(members)} measured curves", xlabel="Hour (h)",
               xlim=(0, 200), ylim=(-.05, 1.20))
        ax.grid(alpha=.15)
        note = (f"Red: actual train example {fid}" if full_window_count / len(members) >= .2
                else f"Only {full_window_count}/{len(members)} extend to 160 h; no example highlighted")
        ax.text(.02, .04, note, transform=ax.transAxes,
                fontsize=8, bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none"})
    axes[0].set_ylabel("Normalized PCE")
    fig.suptitle(f"Frozen SOM prototype groups (k={k}); 0–200 h only; no semantic labels",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((HERE / "runs/summary.json").read_text())
    assert summary["selected_run"] == "E2_seed43"
    model_path = HERE / "runs/E2_seed43.npz"
    assert digest(model_path) == summary["selected_model_sha256"]
    model, _ = load_model(model_path)
    stored = np.load(HERE / "manifests/strict_features.npz")
    ids = [str(x) for x in stored["curve_ids"]]
    assert len(ids) == 2205 and len(set(ids)) == len(ids)
    index = {row["curve_id"]: row for row in rows(HERE / "delivery/class_index.csv")}
    assert set(ids).issubset(index)
    data, mask = stored["data"], stored["mask"]
    bmu = model.assign(data, mask)[0]
    assert all(int(index[fid]["som_bmu"]) == int(bmu[i]) for i, fid in enumerate(ids))
    occupancy = np.bincount(bmu, minlength=100)
    train = np.array([index[fid]["split"] == "train" for fid in ids])
    val = np.array([index[fid]["split"] == "validation" for fid in ids])
    assert np.sum(val) == 420 and np.all(occupancy > 0)
    distance = prototype_distances(model)
    all_labels = {k: labels_for_k(distance, k) for k in KS}
    score_rows = []
    for k, labels in all_labels.items():
        silhouette = silhouette_samples(distance, labels, metric="precomputed")
        counts = np.bincount(labels, weights=occupancy)
        score_rows.append(dict(k=k, node_silhouette=f"{silhouette.mean():.6f}",
                               occupied_curve_weighted_silhouette=f"{np.average(silhouette, weights=occupancy):.6f}",
                               min_group_curves=int(counts.min()), max_group_curves=int(counts.max()),
                               group_curve_counts="|".join(str(int(x)) for x in sorted(counts, reverse=True))))
    write_rows(OUT / "k_selection.csv", score_rows, list(score_rows[0]))
    plot_selection(OUT / "k_selection.png", score_rows)
    preferred_k = max(score_rows, key=lambda row: float(row["node_silhouette"]))["k"]
    assert preferred_k == 2

    seed_rows = []
    for seed in SEEDS:
        other, _ = load_model(HERE / "runs" / f"E2_seed{seed}.npz")
        other_bmu = other.assign(data[val], mask[val])[0]
        other_distances = prototype_distances(other)
        record = {"seed": seed}
        for k in (2, 4):
            other_labels = labels_for_k(other_distances, k)
            reference = all_labels[k][bmu[val]]
            record[f"k{k}_node_silhouette"] = f"{silhouette_samples(other_distances, other_labels, metric='precomputed').mean():.6f}"
            record[f"k{k}_validation_ARI_vs_frozen"] = f"{adjusted_rand_score(reference, other_labels[other_bmu]):.6f}"
        seed_rows.append(record)
    write_rows(OUT / "seed_stability.csv", seed_rows, list(seed_rows[0]))

    # Refit the SOM on three DOI-group bootstrap samples. Semantic classes are
    # never read; score only the resulting anonymous validation partitions.
    cfg = summary["config"]
    by_source = defaultdict(list)
    for i, fid in enumerate(ids):
        if train[i]:
            by_source[index[fid]["source_group"]].append(i)
    sources = sorted(by_source)
    bootstrap_rows = []
    for replication in range(3):
        rng = np.random.default_rng(cfg["split_seed"] + 100 + replication)
        drawn = rng.choice(sources, size=len(sources), replace=True)
        sampled = np.concatenate([by_source[source] for source in drawn])
        resampled = MaskedSOM(10, 10, list(stored["group_sizes"]), list(model.weights),
                              47 + replication)
        resampled.fit(data[sampled], mask[sampled], cfg["epochs"])
        np.savez_compressed(OUT / f"source_bootstrap_model_rep{replication + 1}.npz",
                            prototypes=resampled.prototypes, supported=resampled.supported,
                            feature_sizes=stored["group_sizes"], weights=model.weights,
                            seed=resampled.seed)
        resampled_bmu = resampled.assign(data[val], mask[val])[0]
        resampled_distance = prototype_distances(resampled)
        record = {"replication": replication + 1, "drawn_DOI_groups": len(drawn),
                  "distinct_DOI_groups": len(set(drawn))}
        for k in (2, 4):
            observed = labels_for_k(resampled_distance, k)[resampled_bmu]
            reference = all_labels[k][bmu[val]]
            record[f"k{k}_validation_ARI_vs_frozen"] = f"{adjusted_rand_score(reference, observed):.6f}"
        bootstrap_rows.append(record)
        print(f"DOI bootstrap {replication + 1}/3: {record}", flush=True)
    write_rows(OUT / "source_bootstrap_stability.csv", bootstrap_rows, list(bootstrap_rows[0]))

    numbered = {k: numbered_groups(all_labels[k], occupancy) for k in (2, 4)}
    assignment_rows = []
    for i, fid in enumerate(ids):
        row = index[fid]
        assignment_rows.append(dict(curve_id=fid, source_group=row["source_group"], split=row["split"],
                                    som_bmu=int(bmu[i]), anonymous_k2=f"G{numbered[2][bmu[i]]}",
                                    anonymous_k4=f"G{numbered[4][bmu[i]]}",
                                    curve_csv=row["curve_csv"], source_csv=row["source_csv"]))
    write_rows(OUT / "group_assignments.csv", assignment_rows, list(assignment_rows[0]))
    sampling_rows = []
    for k in (2, 4):
        for group in range(1, k + 1):
            members = [row for row in assignment_rows if row[f"anonymous_k{k}"] == f"G{group}"]
            source = [index[row["curve_id"]] for row in members]
            points = np.array([float(row["window_points"]) for row in source])
            end = np.array([float(row["observation_end_used_h"]) for row in source])
            gaps = np.array([float(row["max_gap_h"]) for row in source])
            sampling_rows.append(dict(k=k, group=f"G{group}", curves=len(members),
                                      DOI_groups=len({row["source_group"] for row in members}),
                                      median_window_points=f"{np.median(points):.3f}",
                                      median_last_strict_hour=f"{np.median(end):.3f}",
                                      median_max_gap_h=f"{np.median(gaps):.3f}",
                                      count_last_strict_hour_at_least_160=int(np.sum(end >= 160)),
                                      count_common_support_to_160_h=int(np.sum((end >= 160) & (points >= 5) & (gaps <= 40))),
                                      count_at_least_5_points_and_gap_at_most_40=int(np.sum((points >= 5) & (gaps <= 40)))))
    write_rows(OUT / "sampling_diagnostics.csv", sampling_rows, list(sampling_rows[0]))
    node_rows = [dict(som_bmu=i, row=i // 10, column=i % 10,
                      all_curve_count=int(occupancy[i]), anonymous_k2=f"G{numbered[2][i]}",
                      anonymous_k4=f"G{numbered[4][i]}") for i in range(100)]
    write_rows(OUT / "node_groups.csv", node_rows, list(node_rows[0]))
    examples = {}
    for k in (2, 4):
        selected = {g: representative(g, numbered[k], bmu, train, data, mask, model, distance,
                                      occupancy, index, ids) for g in range(1, k + 1)}
        examples[k] = {f"G{g}": fid for g, fid in selected.items()}
        plot_map(OUT / f"som_groups_k{k}.png", numbered[k], occupancy, k)
        plot_curves(OUT / f"measured_curves_k{k}.png", numbered[k], bmu, ids, index, selected, k)

    result = dict(status="exploratory_unlabeled_structure_only", model="E2_seed43",
                  model_sha256=digest(model_path), source_feature_sha256=digest(HERE / "manifests/strict_features.npz"),
                  selection="Maximum unweighted prototype silhouette among k=2..12; average linkage on SOM-weighted prototype RMS distances.",
                  selected_k=int(preferred_k), four_groups_for_hypothesis_only=True,
                  curves=2205, prototype_nodes=100,
                  k2_group_counts=dict(Counter(row["anonymous_k2"] for row in assignment_rows)),
                  k4_group_counts=dict(Counter(row["anonymous_k4"] for row in assignment_rows)),
                  representative_real_training_curves=examples,
                  seed_stability="Validation curve partition ARI versus E2_seed43, across ten preset E2 seeds; same data, not DOI bootstrap.",
                  source_bootstrap_stability="Three train DOI-group bootstrap SOM refits; compare anonymous k=2 and k=4 validation partitions against frozen E2_seed43 by ARI.",
                  human_semantic_validation="Not available; no Bridge/Hill/Slope/Valley labels inferred from these groups.")
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
