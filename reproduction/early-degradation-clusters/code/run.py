"""Fit four shape clusters to the fixed set of 1,842 curves."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


HERE = Path(__file__).resolve().parent.parent
CONFIG = json.loads((HERE / "code" / "config.json").read_text(encoding="utf-8"))
OUT = (HERE / CONFIG["output_dir"]).resolve()
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
        distances = np.linalg.norm(embedding[members] - centers[cluster - 1], axis=1)
        selected = members[np.argsort(distances)[:5]]
        output.append(dict(cluster=int(cluster), member_count=len(members),
                           sparse_4_to_7=int(sum(int(rows[i]["distinct_observations"]) < 8 for i in members)),
                           source_groups=len({rows[i]["source_group"] for i in members}),
                           representative_ids=";".join(rows[i]["curve_id"] for i in selected)))
    return output


def plot(raw: np.ndarray, rows: list[dict], labels: np.ndarray, reps: list[dict]) -> None:
    position = {r["curve_id"]: i for i, r in enumerate(rows)}
    rank = np.linspace(0, 1, raw.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, rep in zip(axes.flat, reps):
        cluster = rep["cluster"]
        members = np.flatnonzero(labels == cluster)
        for i in members:
            ax.plot(rank, raw[i], color="#6e8792", alpha=min(0.2, 12 / len(members)), lw=0.55)
        for cid in rep["representative_ids"].split(";"):
            ax.plot(rank, raw[position[cid]], lw=1.7, label=cid)
        ax.set_title(f"Cluster {cluster}: {len(members)} curves")
        ax.set_ylim(0, 1.07)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("Relative observation rank")
    for ax in axes[:, 0]: ax.set_ylabel("Normalized PCE")
    fig.suptitle("Four shape clusters; original PCE scale", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "cluster_four_groups.png", dpi=180)
    plt.close(fig)


def export_cluster_sources(assignments: list[dict]) -> None:
    """Copy source DOI folders and write an exact curve membership list per cluster."""
    mode = CONFIG["cluster_export_mode"]
    if mode not in {"full_doi", "matching_csv"}:
        raise ValueError(f"Unknown cluster export mode: {mode}")
    source_root = (HERE / "data" / "raw_csv").resolve()
    source_rows = read_rows(HERE / "data" / "curve_index.csv")
    source_by_id = {row["curve_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Duplicate curve IDs in raw source index")
    members: dict[int, list[dict]] = {cluster: [] for cluster in range(1, CONFIG["cluster_count"] + 1)}
    fields = ["curve_id", "cluster", "source_group", "source_csv", "copied_csv", "csv_sha256"]
    with tempfile.TemporaryDirectory(prefix=".cluster_export_", dir=OUT) as temporary:
        staging = Path(temporary)
        for assignment in assignments:
            curve_id = assignment["curve_id"]
            source_row = source_by_id[curve_id]
            source_csv = (HERE / "data" / source_row["raw_csv"]).resolve()
            if not source_csv.is_relative_to(source_root) or not source_csv.is_file():
                raise ValueError(f"Invalid raw CSV path for {curve_id}")
            cluster = int(assignment["cluster"])
            destination_root = staging / f"cluster_{cluster}"
            copied_csv = source_csv.relative_to(source_root)
            if mode == "full_doi":
                source_doi = source_csv.parent
                destination_doi = destination_root / copied_csv.parent
                if not destination_doi.exists():
                    shutil.copytree(source_doi, destination_doi)
            else:
                destination_csv = destination_root / copied_csv
                destination_csv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_csv, destination_csv)
            members[cluster].append(dict(curve_id=curve_id, cluster=cluster,
                                         source_group=source_row["source_group"],
                                         source_csv=source_row["source_csv"],
                                         copied_csv=copied_csv.as_posix(),
                                         csv_sha256=source_row["csv_sha256"]))
        for cluster, rows in members.items():
            folder = staging / f"cluster_{cluster}"
            folder.mkdir(exist_ok=True)
            write_rows(folder / "members.csv", rows, fields)
            note = ("这里完整复制了对应的 DOI 文件夹。同一 DOI 可能包含不同簇的曲线；"
                    "请以 members.csv 判断哪些 CSV 真正属于本簇。"
                    if mode == "full_doi" else
                    "这里只复制本簇的原始 CSV，并保留 DOI 文件夹结构。")
            (folder / "README.md").write_text(
                f"# Cluster {cluster}\n\n本簇有 {len(rows)} 条曲线。{note}\n", encoding="utf-8")
        for cluster in members:
            destination = OUT / f"cluster_{cluster}"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(staging / f"cluster_{cluster}"), destination)


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
    rows = [all_rows[i] for i in positions]
    raw = raw_all[positions]
    embedding, normalized, amplitudes, coefficients = encode(raw)
    labels, weights, centers = fit(embedding)
    cluster_ids = labels + 1  # Public cluster IDs are 1–4; NumPy centers remain zero-indexed.
    assignments = [dict(curve_id=row["curve_id"], cluster=int(cluster_ids[i]),
                        distinct_observations=row["distinct_observations"],
                        source_group=row["source_group"], normalized_csv=row["normalized_csv"],
                        observed_range=f"{amplitudes[i]:.10g}",
                        total_variation_to_range=f"{roughness[positions[i]]:.10g}",
                        density_weight=f"{weights[i]:.10g}") for i, row in enumerate(rows)]
    write_rows(OUT / "cluster_assignments.csv", assignments, list(assignments[0]))
    reps = representatives(rows, embedding, cluster_ids, centers)
    write_rows(OUT / "representatives.csv", reps, list(reps[0]))
    np.savez_compressed(OUT / "model.npz", curve_id=np.array(ids)[positions], cluster=cluster_ids,
                        center_cluster=np.arange(1, CONFIG["cluster_count"] + 1), raw_rank=raw,
                        embedding=embedding, normalized_shape=normalized,
                        coefficients=coefficients, centers=centers, weights=weights,
                        amplitudes=amplitudes, original_positions=np.array(positions))
    plot(raw, rows, cluster_ids, reps)
    summary = dict(protocol=CONFIG["protocol"], input_index_sha256=sha(INDEX),
                   input_shapes_sha256=sha(SHAPES), config_sha256=sha(HERE / "code" / "config.json"),
                   code_sha256=sha(HERE / "code" / "run.py"), included=len(rows),
                   excluded_sparse=sum(r["reason"] == "fewer_than_8_distinct_observations"
                                       for r in excluded),
                   excluded_rough=sum(r["reason"] == "excessive_oscillation"
                                      for r in excluded),
                   excluded_from_input=len(all_rows)-len(rows),
                   cluster_sizes=dict(sorted(Counter(map(int, cluster_ids)).items())),
                   sparse_4_to_7=sum(int(r["distinct_observations"]) < 8 for r in rows),
                   class_names_used_in_fit=False,
                   current_input_selection_used_old_candidates=True,
                   previous_feature_model_selection_was_target_informed=True)
    summary["cluster_export_mode"] = CONFIG["cluster_export_mode"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from render_si_zoom import render_zoom
    render_zoom(OUT)
    export_cluster_sources(assignments)
    from build_dashboard import build_dashboard
    build_dashboard(OUT)
    print(json.dumps({"included": len(rows), "excluded": summary["excluded_from_input"],
                      "cluster_sizes": summary["cluster_sizes"]}, indent=2))


if __name__ == "__main__":
    main()
