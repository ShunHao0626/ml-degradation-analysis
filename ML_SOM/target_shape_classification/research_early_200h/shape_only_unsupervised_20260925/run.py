"""Anonymous shape-only DTW clustering of observed 0–200 h PCE sequences.

Absolute hour values, prior shape names, class rules, and old assignments are
never loaded into the model. Hour is used only to order and deduplicate raw
observations before conversion to relative observation rank.
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUT_INDEX = (HERE / CONFIG["input_index"]).resolve()
POINTS_ROOT = (HERE / CONFIG["input_points_root"]).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(record: dict[str, str]) -> np.ndarray:
    path = (POINTS_ROOT / record["normalized_csv"]).resolve()
    if not path.is_relative_to(POINTS_ROOT):
        raise ValueError("Point path escapes source directory")
    rows = read_rows(path)
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row["hour"])].append(float(row["normalized_pce"]))
    # The hour value orders measured points; its magnitude never enters a
    # feature, pair distance, neighbor search, or graph weight.
    return np.array([np.median(grouped[h]) for h in sorted(grouped)], dtype=np.float64)


def features(values: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative_rank = np.linspace(0, len(values) - 1, grid)
    y = np.interp(relative_rank, np.arange(len(values)), values)
    smooth = np.convolve(np.pad(y, (1, 1), mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    indices = np.arange(grid)
    short_radius = CONFIG["short_difference_radius"]
    long_radius = CONFIG["long_difference_radius"]
    short = smooth[np.minimum(indices + short_radius, grid - 1)] - smooth[np.maximum(indices - short_radius, 0)]
    long = smooth[np.minimum(indices + long_radius, grid - 1)] - smooth[np.maximum(indices - long_radius, 0)]
    return (y - y[0],
            np.tanh(short / CONFIG["short_difference_soft_scale"]),
            np.tanh(long / CONFIG["long_difference_soft_scale"]))


def build_inputs() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(INPUT_INDEX)
    index = []
    level, short, long, raw = [], [], [], []
    grid = CONFIG["rank_grid_points"]
    for row in rows:
        if not row["normalized_csv"]:
            continue
        values = observations(row)
        if len(values) < CONFIG["minimum_distinct_observations"]:
            continue
        a, b, c = features(values, grid)
        index.append(dict(curve_id=row["curve_id"], source_group=row["source_group"],
                          source_csv=row["source_csv"], normalized_csv=row["normalized_csv"],
                          distinct_observations=len(values),
                          evidence_tier="more_observations" if len(values) >= 8 else "sparse_4_to_7"))
        level.append(a)
        short.append(b)
        long.append(c)
        raw.append(np.interp(np.linspace(0, len(values)-1, grid), np.arange(len(values)), values))
    write_rows(OUT / "input_index.csv", index,
               ["curve_id", "source_group", "source_csv", "normalized_csv",
                "distinct_observations", "evidence_tier"])
    arrays = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in (level, short, long, raw))
    np.savez_compressed(OUT / "shape_inputs.npz", curve_id=np.array([r["curve_id"] for r in index]),
                        level=arrays[0], short_change=arrays[1], long_change=arrays[2], raw_rank=arrays[3])
    return index, *arrays


def all_pairs(n: int) -> np.ndarray:
    a, b = np.triu_indices(n, k=1)
    return np.ascontiguousarray(np.column_stack((a, b)), dtype=np.int32)


def distances(level: np.ndarray, short: np.ndarray, long: np.ndarray,
              pairs: np.ndarray, save: bool = True) -> np.ndarray:
    libpath = HERE / "shape_dtw.dylib"
    subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(HERE / "shape_dtw.cpp"),
                    "-o", str(libpath)], check=True)
    library = ctypes.CDLL(str(libpath))
    double = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    int32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    library.shape_pair_dtw.argtypes = [double, double, double, ctypes.c_int, int32,
                                       ctypes.c_int, ctypes.c_int, double, double]
    output = np.empty(len(pairs), dtype=np.float64)
    weights = np.ascontiguousarray(CONFIG["dtw_channel_weights"], dtype=np.float64)
    library.shape_pair_dtw(level, short, long, level.shape[1], pairs, len(pairs),
                           int(level.shape[1] * CONFIG["warping_fraction_of_rank_grid"]),
                           weights, output)
    if not np.all(np.isfinite(output)):
        raise ValueError("Nonfinite DTW distance")
    if save:
        np.savez_compressed(OUT / "all_pair_dtw_distances.npz", pairs=pairs, distance=output)
    return output


def exact_neighbor_graph(all_pairs_: np.ndarray, all_distances: np.ndarray,
                         n: int, save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    dense = np.full((n, n), np.inf, dtype=np.float64)
    dense[all_pairs_[:, 0], all_pairs_[:, 1]] = all_distances
    dense[all_pairs_[:, 1], all_pairs_[:, 0]] = all_distances
    neighbors = np.argpartition(dense, kth=CONFIG["exact_dtw_neighbors"] - 1,
                                axis=1)[:, :CONFIG["exact_dtw_neighbors"]]
    edges = {(min(i, int(j)), max(i, int(j))) for i, js in enumerate(neighbors)
             for j in js if i != j}
    # Exact nearest neighbors can form isolated islands. Connect each island
    # by its closest DTW edge, still without looking at any semantic labels.
    while True:
        pair_array = np.array(sorted(edges), dtype=np.int32)
        structure = coo_matrix((np.ones(len(pair_array) * 2),
                                (np.r_[pair_array[:, 0], pair_array[:, 1]],
                                 np.r_[pair_array[:, 1], pair_array[:, 0]])), shape=(n, n)).tocsr()
        components, component_ids = connected_components(structure, directed=False)
        if components == 1:
            break
        best = (math.inf, -1, -1)
        for c in range(components):
            inside = np.flatnonzero(component_ids == c)
            outside = np.flatnonzero(component_ids != c)
            block = dense[np.ix_(inside, outside)]
            a, b = np.unravel_index(np.argmin(block), block.shape)
            if block[a, b] < best[0]:
                best = (float(block[a, b]), int(inside[a]), int(outside[b]))
        edges.add((min(best[1:]), max(best[1:])))
    pair_array = np.array(sorted(edges), dtype=np.int32)
    edge_distances = dense[pair_array[:, 0], pair_array[:, 1]]
    if save:
        np.savez_compressed(OUT / "shape_dtw_edges.npz", pairs=pair_array,
                            distance=edge_distances)
    return pair_array, edge_distances


def embed_and_cluster(pairs: np.ndarray, distances_: np.ndarray, n: int,
                      save: bool = True) -> dict[int, np.ndarray]:
    neighbor_distances: dict[int, list[float]] = defaultdict(list)
    for (i, j), d in zip(pairs, distances_):
        neighbor_distances[int(i)].append(float(d))
        neighbor_distances[int(j)].append(float(d))
    localscale = np.array([np.median(sorted(neighbor_distances[i])[:10]) for i in range(n)])
    floor = max(float(np.quantile(localscale, 0.05)), 0.001)
    localscale = np.maximum(localscale, floor)
    weight = np.exp(-np.minimum(distances_ ** 2 / (localscale[pairs[:, 0]] * localscale[pairs[:, 1]]), 20))
    graph = coo_matrix((np.r_[weight, weight],
                        (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                       shape=(n, n)).tocsr()
    components, component_ids = connected_components(graph, directed=False)
    if components != 1:
        raise ValueError(f"Candidate graph has {components} components; inspect before clustering")
    lmatrix = laplacian(graph, normed=True)
    dim = max(CONFIG["cluster_counts"]) // 2
    eigenvalues, eigenvectors = eigsh(lmatrix, k=dim + 1, which="SM", tol=1e-3,
                                       maxiter=20000, v0=np.random.default_rng(CONFIG["random_seed"]).normal(size=n))
    order = np.argsort(eigenvalues)
    embedding = eigenvectors[:, order[1:]]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    hierarchy = linkage(embedding, method="ward", optimal_ordering=False)
    clusters = {k: fcluster(hierarchy, t=k, criterion="maxclust").astype(np.int32) - 1
                for k in CONFIG["cluster_counts"]}
    if save:
        np.savez_compressed(OUT / "graph_embedding.npz", embedding=embedding,
                            eigenvalues=eigenvalues[order], local_scale=localscale,
                            graph_weight=weight, linkage=hierarchy)
    return clusters


def representatives(index: list[dict], labels: np.ndarray, pairs: np.ndarray,
                    distance: np.ndarray) -> list[dict]:
    adjacent: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), d in zip(pairs, distance):
        adjacent[int(a)].append((int(b), float(d)))
        adjacent[int(b)].append((int(a), float(d)))
    reps = []
    for cluster in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster)
        member_set = set(members)
        ranks = []
        for i in members:
            local = [d for j, d in adjacent[int(i)] if j in member_set]
            ranks.append((-len(local), np.mean(local) if local else math.inf, index[i]["curve_id"], int(i)))
        selected = [r[3] for r in sorted(ranks)[:5]]
        reps.append(dict(cluster=int(cluster), member_count=len(members),
                         source_groups=len({index[i]["source_group"] for i in members}),
                         sparse_4_to_7=sum(index[i]["evidence_tier"] == "sparse_4_to_7" for i in members),
                         representative_ids=";".join(index[i]["curve_id"] for i in selected)))
    return reps


def plot_clusters(index: list[dict], raw: np.ndarray, labels: np.ndarray, reps: list[dict]) -> None:
    id_to_position = {r["curve_id"]: i for i, r in enumerate(index)}
    x = np.linspace(0, 1, raw.shape[1])
    for page in range((len(reps) + 15) // 16):
        fig, axes = plt.subplots(4, 4, figsize=(17, 15), sharex=True, sharey=True)
        for ax, row in zip(axes.flat, reps[page * 16:(page + 1) * 16]):
            cluster = row["cluster"]
            members = np.flatnonzero(labels == cluster)
            for i in members:
                ax.plot(x, raw[i], color="#557385", lw=0.6, alpha=0.12)
            for cid in row["representative_ids"].split(";"):
                ax.plot(x, raw[id_to_position[cid]], lw=1.5, alpha=0.85, label=cid)
            ax.set_title(f"Anonymous {cluster} · n={len(members)} · sparse={row['sparse_4_to_7']}", fontsize=10)
            ax.set_ylim(0, 1.07)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=5.5, loc="lower left", ncol=2)
        for ax in axes[-1]: ax.set_xlabel("Relative observation rank")
        for ax in axes[:, 0]: ax.set_ylabel("Normalized PCE")
        fig.suptitle(f"Shape-only anonymous DTW · k={len(reps)} · page {page+1}", fontsize=16)
        fig.tight_layout()
        fig.savefig(OUT / f"anonymous_k{len(reps)}_page{page+1}.png", dpi=170)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    start = time.monotonic()
    index, level, short, long, raw = build_inputs()
    print(f"Prepared {len(index)} curves", flush=True)
    all_pairs_ = all_pairs(len(index))
    print(f"Exact pairwise DTW: {len(all_pairs_)} pairs", flush=True)
    all_distances = distances(level, short, long, all_pairs_)
    print(f"All DTW distances done in {time.monotonic()-start:.1f}s", flush=True)
    pairs, dist = exact_neighbor_graph(all_pairs_, all_distances, len(index))
    print(f"Exact neighbor graph: {len(pairs)} edges", flush=True)
    clusters = embed_and_cluster(pairs, dist, len(index))
    rows = []
    for i, item in enumerate(index):
        rows.append(dict(**item, **{f"anonymous_k{k}": int(labels[i]) for k, labels in clusters.items()}))
    write_rows(OUT / "anonymous_assignments.csv", rows,
               list(index[0]) + [f"anonymous_k{k}" for k in CONFIG["cluster_counts"]])
    for k, labels in clusters.items():
        reps = representatives(index, labels, pairs, dist)
        write_rows(OUT / f"representatives_k{k}.csv", reps,
                   ["cluster", "member_count", "source_groups", "sparse_4_to_7", "representative_ids"])
        plot_clusters(index, raw, labels, reps)
    summary = dict(protocol=CONFIG["protocol"], input_index_sha256=digest(INPUT_INDEX),
                   config_sha256=digest(HERE / "config.json"), code_sha256=digest(HERE / "run.py"),
                   cpp_sha256=digest(HERE / "shape_dtw.cpp"), eligible=len(index),
                   sparse_4_to_7=sum(r["evidence_tier"] == "sparse_4_to_7" for r in index),
                   exact_dtw_pairs=len(all_pairs_), neighbor_graph_edges=len(pairs),
                   elapsed_seconds=round(time.monotonic()-start, 1),
                   cluster_sizes={str(k): dict(Counter(map(int, v))) for k, v in clusters.items()},
                   absolute_hours_used_in_distance=False, shape_names_used_in_fit=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"eligible": len(index), "seconds": summary["elapsed_seconds"],
                      "cluster_counts": CONFIG["cluster_counts"]}), flush=True)


if __name__ == "__main__":
    main()
