"""Fit four anonymous groups after unlabeled input quality curation.

The feature/model pipeline is frozen from the previous exploration; its
historical selection was target-informed. This round's input threshold was
chosen using only unlabeled quality and perturbation metrics. This fitting
code reads no candidate classes or user annotations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INDEX = (HERE / CONFIG["input_index"]).resolve()
SHAPES = (HERE / CONFIG["input_shapes"]).resolve()
POINTS_ROOT = (HERE / CONFIG["input_points_root"]).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_observations(row: dict[str, str]) -> np.ndarray:
    path = (POINTS_ROOT / row["normalized_csv"]).resolve()
    if not path.is_relative_to(POINTS_ROOT):
        raise ValueError("Point path escapes source directory")
    by_hour: dict[float, list[float]] = defaultdict(list)
    for point in read_rows(path):
        by_hour[float(point["hour"])].append(float(point["normalized_pce"]))
    # Hour orders original observations; elapsed time is absent from clustering.
    return np.array([np.median(by_hour[h]) for h in sorted(by_hour)], dtype=float)


def encode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amplitudes = raw.max(axis=1) - raw.min(axis=1)
    denominator = np.maximum(amplitudes, CONFIG["amplitude_floor"])
    normalized = (raw - raw[:, :1]) / denominator[:, None]
    smoothed = gaussian_filter1d(normalized, CONFIG["gaussian_rank_sigma"], axis=1)
    rank = np.linspace(-1, 1, raw.shape[1])
    coefficients = np.polynomial.chebyshev.chebfit(rank, smoothed.T,
                                                    CONFIG["chebyshev_degree"]).T[:, 1:]
    scaler = RobustScaler(quantile_range=tuple(CONFIG["robust_scaler_quantiles"]))
    embedding = scaler.fit_transform(coefficients)
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite shape coefficients")
    return embedding, normalized, amplitudes, coefficients


def fit(embedding: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = NearestNeighbors(n_neighbors=CONFIG["density_neighbors"], metric="euclidean")
    neighbors.fit(embedding)
    radius = neighbors.kneighbors(return_distance=True)[0][:, -1]
    weights = np.clip(radius / np.median(radius), *CONFIG["density_weight_clip"])
    model = KMeans(n_clusters=CONFIG["cluster_count"], n_init=CONFIG["kmeans_restarts"],
                   random_state=CONFIG["random_seed"] if seed is None else seed)
    # This local NumPy/BLAS build reports spurious floating-point warnings for
    # finite matrix products in sklearn's k-means++ initializer. Validate all
    # inputs and outputs explicitly rather than relying on these warnings.
    if not np.isfinite(embedding).all() or not np.isfinite(weights).all():
        raise ValueError("Nonfinite data before KMeans")
    with np.errstate(all="ignore"):
        labels = model.fit_predict(embedding, sample_weight=weights)
    if not np.isfinite(model.cluster_centers_).all() or not np.isfinite(model.inertia_):
        raise ValueError("Nonfinite KMeans result")
    return labels.astype(np.int32), weights, model.cluster_centers_


def representatives(rows: list[dict], embedding: np.ndarray, labels: np.ndarray,
                    centers: np.ndarray) -> list[dict]:
    output = []
    for cluster in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(embedding[members] - centers[cluster], axis=1)
        selected = members[np.argsort(distances)[:5]]
        output.append(dict(anonymous_cluster=int(cluster), member_count=len(members),
                           sparse_4_to_7=int(sum(int(rows[i]["distinct_observations"]) < 8 for i in members)),
                           source_groups=len({rows[i]["source_group"] for i in members}),
                           representative_ids=";".join(rows[i]["curve_id"] for i in selected)))
    return output


def plot(raw: np.ndarray, rows: list[dict], labels: np.ndarray, reps: list[dict]) -> None:
    position = {r["curve_id"]: i for i, r in enumerate(rows)}
    rank = np.linspace(0, 1, raw.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, rep in zip(axes.flat, reps):
        cluster = rep["anonymous_cluster"]
        members = np.flatnonzero(labels == cluster)
        for i in members:
            ax.plot(rank, raw[i], color="#6e8792", alpha=min(0.2, 12 / len(members)), lw=0.55)
        for cid in rep["representative_ids"].split(";"):
            ax.plot(rank, raw[position[cid]], lw=1.7, label=cid)
        ax.set_title(f"Anonymous {cluster}: {len(members)} curves")
        ax.set_ylim(0, 1.07)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("Relative observation rank")
    for ax in axes[:, 0]: ax.set_ylabel("Normalized PCE")
    fig.suptitle("Four anonymous shape groups; original PCE scale", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "anonymous_four_groups.png", dpi=180)
    plt.close(fig)
    # Reproduce the reference's curve style: transparent member trajectories
    # with a dark center line. The clustering and anonymous IDs are unchanged.
    display_order = (0, 1, 2, 3)
    colors = {0: "#719b82", 1: "#416987", 2: "#ad945e", 3: "#907192"}
    # Derive each display window from its anonymous median. Values and
    # assignments are unchanged; out-of-range member segments are clipped.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6), sharex=True)
    for ax, cluster in zip(axes.flat, display_order):
        members = np.flatnonzero(labels == cluster)
        ax.set_facecolor("#eaf1fb")
        ax.set_axisbelow(True)
        member_values = raw[members]
        segments = [np.column_stack((rank, raw[i])) for i in members]
        alpha = .018 if len(members) > 500 else .12 if len(members) > 100 else .19
        ax.add_collection(LineCollection(segments, colors=colors[cluster],
                                         linewidths=.7, alpha=alpha, zorder=1))
        median = gaussian_filter1d(np.median(member_values, axis=0),
                                   sigma=1.2, mode="nearest")
        median_span = np.ptp(median)
        pad = max(.02, .5 * median_span)
        lo, hi = max(-.05, float(median.min()-pad)), min(1.05, float(median.max()+pad))
        ax.plot(rank, median, color="#171b20", linewidth=2.0, zorder=3)
        ax.set_title(f"Anonymous cluster {cluster} · {len(members)} curves",
                     fontsize=12, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        ax.grid(color="white", linewidth=1.3)
        ax.tick_params(length=0, colors="#3c4b58")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel("Normalized PCE")
    for ax in axes[-1]:
        ax.set_xlabel("Relative observation rank (0–1)")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(OUT / "anonymous_four_groups_zoom.png", dpi=180)
    fig.savefig(OUT / "anonymous_four_groups_zoom.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_rows = read_rows(INDEX)
    with np.load(SHAPES) as arrays:
        raw_all = arrays["raw_rank"]
        ids = arrays["curve_id"].tolist()
    assert ids == [r["curve_id"] for r in all_rows]
    originals = [original_observations(row) for row in all_rows]
    variation = np.array([np.abs(np.diff(y)).sum() for y in originals])
    raw_range = np.array([np.ptp(y) for y in originals])
    roughness = variation / np.maximum(raw_range, CONFIG["roughness_range_floor"])
    eligible = [i for i, r in enumerate(all_rows)
                if int(r["distinct_observations"]) >= CONFIG["minimum_distinct_observations"]]
    positions = [i for i in eligible
                 if roughness[i] <= CONFIG["maximum_total_variation_to_range"]]
    retained = set(positions)
    excluded = [dict(curve_id=all_rows[i]["curve_id"],
                     distinct_observations=all_rows[i]["distinct_observations"],
                     total_variation_to_range=f"{roughness[i]:.10g}",
                     reason=("fewer_than_8_distinct_observations"
                             if int(all_rows[i]["distinct_observations"]) < 8
                             else "excessive_oscillation"))
                for i in range(len(all_rows)) if i not in retained]
    write_rows(OUT / "excluded_input_curves.csv", excluded,
               ["curve_id", "distinct_observations", "total_variation_to_range", "reason"])
    rows = [all_rows[i] for i in positions]
    raw = raw_all[positions]
    embedding, normalized, amplitudes, coefficients = encode(raw)
    labels, weights, centers = fit(embedding)
    assignments = [dict(curve_id=row["curve_id"], anonymous_cluster=int(labels[i]),
                        distinct_observations=row["distinct_observations"],
                        source_group=row["source_group"], normalized_csv=row["normalized_csv"],
                        observed_range=f"{amplitudes[i]:.10g}",
                        total_variation_to_range=f"{roughness[positions[i]]:.10g}",
                        density_weight=f"{weights[i]:.10g}") for i, row in enumerate(rows)]
    write_rows(OUT / "anonymous_assignments.csv", assignments, list(assignments[0]))
    reps = representatives(rows, embedding, labels, centers)
    write_rows(OUT / "representatives.csv", reps, list(reps[0]))
    np.savez_compressed(OUT / "model.npz", curve_id=np.array(ids)[positions], raw_rank=raw,
                        embedding=embedding, normalized_shape=normalized,
                        coefficients=coefficients, centers=centers, weights=weights,
                        amplitudes=amplitudes, original_positions=np.array(positions))
    plot(raw, rows, labels, reps)
    summary = dict(protocol=CONFIG["protocol"], input_index_sha256=sha(INDEX),
                   input_shapes_sha256=sha(SHAPES), config_sha256=sha(HERE / "config.json"),
                   code_sha256=sha(HERE / "run.py"), included=len(rows),
                   excluded_sparse=sum(r["reason"] == "fewer_than_8_distinct_observations"
                                       for r in excluded),
                   excluded_rough=sum(r["reason"] == "excessive_oscillation"
                                      for r in excluded),
                   excluded_from_input=len(all_rows)-len(rows),
                   cluster_sizes=dict(Counter(map(int, labels))),
                   sparse_4_to_7=sum(int(r["distinct_observations"]) < 8 for r in rows),
                   class_names_used_in_fit=False,
                   current_input_selection_used_old_candidates=False,
                   previous_feature_model_selection_was_target_informed=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"included": len(rows), "excluded": summary["excluded_from_input"],
                      "cluster_sizes": summary["cluster_sizes"]}, indent=2))


if __name__ == "__main__":
    main()
