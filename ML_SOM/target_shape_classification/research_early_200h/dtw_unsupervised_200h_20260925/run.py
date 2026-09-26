"""Strict 0-200 h, source-only, anonymous DTW clustering experiment.

No existing four-class assignment, review label, or class template enters any
input feature, candidate-neighbor search, DTW distance, or cluster selection.
The old rule output is read only after all anonymous assignments are saved.
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUTPUT = HERE / "results"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GRID = np.arange(0, CONFIG["window_h"] + 0.001, CONFIG["grid_step_h"], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_path(relative: str) -> Path:
    root = Path(CONFIG["source_root"]).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path escapes root")
    return path


def source_points(row: dict) -> list[tuple[float, float, int]]:
    path = source_path(row["source_csv"])
    if sha256(path) != row["source_sha256"]:
        raise ValueError("source hash mismatch")
    factor = float(row["time_factor"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("hour conversion unresolved")
    if row["folder"] == "final_data":
        if factor != 1:
            raise ValueError("final_data hour conversion must equal one")
        columns = ("time_h", "normalized_pce")
    else:
        columns = ("x", "y")
    points = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError("missing coordinate columns")
        for source_row, item in enumerate(reader, 2):
            try:
                hour = float(item[columns[0]]) * factor
                value = float(item[columns[1]])
            except (ValueError, TypeError):
                continue
            if math.isfinite(hour) and math.isfinite(value) and 0 <= hour <= CONFIG["window_h"]:
                points.append((hour, value, source_row))
    return sorted(points)


def grid_from_observations(hours: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level = np.zeros(len(GRID), dtype=float)
    valid = np.zeros(len(GRID), dtype=bool)
    # No extrapolation. Gaps above the declared limit provide no continuous support.
    for i in range(len(hours) - 1):
        gap = hours[i + 1] - hours[i]
        if gap <= 0 or gap > CONFIG["maximum_interpolation_gap_h"]:
            continue
        select = (GRID >= hours[i]) & (GRID <= hours[i + 1])
        valid[select] = True
        level[select] = np.interp(GRID[select], hours[i:i + 2], values[i:i + 2])
    # A measured point near a grid cell is represented only if its own cell was
    # reached by supported interpolation; the source point is always saved below.
    derivative = np.zeros(len(GRID), dtype=float)
    radius = CONFIG["derivative_window_h"]
    indices = np.flatnonzero(valid)
    for j in indices:
        local = indices[np.abs(GRID[indices] - GRID[j]) <= radius]
        # Do not bridge across a missing interval inside the regression window.
        if len(local) < 3 or np.any(np.diff(local) != 1):
            continue
        t = GRID[local] - GRID[local].mean()
        derivative[j] = float(np.dot(t, level[local] - level[local].mean()) / np.dot(t, t))
    derivative *= CONFIG["derivative_scale_h"]
    return level, derivative, valid


def prepare() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    canonical = read_rows(Path(CONFIG["canonical_manifest"]))
    original = read_rows(Path(CONFIG["all_file_manifest"]))
    if len(canonical) != 2246 or len(original) != 2250:
        raise ValueError("source manifest counts changed; inspect before continuing")
    for row in original:
        if sha256(source_path(row["source_csv"])) != row["source_sha256"]:
            raise ValueError("source hash mismatch: " + row["file_id"])
    write_rows(OUTPUT / "all_file_index.csv", [dict(file_id=r["file_id"],
                 canonical_id=r["canonical_id"], source_csv=r["source_csv"],
                 source_sha256=r["source_sha256"]) for r in original],
               ["file_id", "canonical_id", "source_csv", "source_sha256"])
    index = []
    level_arrays, derivative_arrays, masks = [], [], []
    for row in canonical:
        curve_id = row["file_id"]
        record = dict(curve_id=curve_id, source_csv=row["source_csv"],
                      source_sha256=row["source_sha256"], source_group=row["source_group"],
                      y_kind=row["y_kind"], time_factor=row["time_factor"],
                      original_points_0_200=0, unique_hours_0_200=0,
                      first_hour="", last_hour="", normalization_divisor="",
                      grid_support_points=0, model_status="", normalized_csv="")
        if row["y_kind"] != "pce":
            record["model_status"] = "non_pce_or_unresolved_axis"
            index.append(record)
            continue
        try:
            points = source_points(row)
        except (OSError, ValueError) as exc:
            record["model_status"] = "source_error:" + str(exc)
            index.append(record)
            continue
        record["original_points_0_200"] = len(points)
        if not points:
            record["model_status"] = "no_0_200h_observation"
            index.append(record)
            continue
        divisor = max(p[1] for p in points)
        if divisor <= 0:
            record["model_status"] = "nonpositive_window_maximum"
            index.append(record)
            continue
        by_hour: dict[float, list[float]] = defaultdict(list)
        for hour, value, _ in points:
            by_hour[hour].append(value / divisor)
        hours = np.array(sorted(by_hour), dtype=float)
        values = np.array([np.median(by_hour[h]) for h in hours], dtype=float)
        record.update(unique_hours_0_200=len(hours), first_hour=f"{hours[0]:.12g}",
                      last_hour=f"{hours[-1]:.12g}",
                      normalization_divisor=f"{divisor:.12g}",
                      normalized_csv=f"points/{curve_id}.csv")
        write_rows(OUTPUT / record["normalized_csv"],
                   [dict(hour=f"{h:.12g}", source_y=f"{y:.12g}",
                         normalized_pce=f"{y / divisor:.12g}", source_row=source_row)
                    for h, y, source_row in points],
                   ["hour", "source_y", "normalized_pce", "source_row"])
        level, derivative, valid = grid_from_observations(hours, values)
        record["grid_support_points"] = int(valid.sum())
        if len(hours) < CONFIG["minimum_original_unique_hours"]:
            record["model_status"] = "too_few_distinct_original_hours"
        elif hours[-1] - hours[0] < CONFIG["minimum_observed_span_h"]:
            record["model_status"] = "observed_span_too_short"
        elif valid.sum() < CONFIG["minimum_supported_grid_points"]:
            record["model_status"] = "too_little_supported_grid"
        else:
            record["model_status"] = "model_eligible"
            level_arrays.append(level)
            derivative_arrays.append(derivative)
            masks.append(valid)
        index.append(record)
    write_rows(OUTPUT / "curve_index.csv", index, list(index[0]))
    eligible = [r for r in index if r["model_status"] == "model_eligible"]
    arrays = (np.stack(level_arrays), np.stack(derivative_arrays), np.stack(masks))
    np.savez_compressed(OUTPUT / "model_inputs.npz", grid_hour=GRID,
                        curve_id=np.array([r["curve_id"] for r in eligible]),
                        level=arrays[0], derivative=arrays[1], mask=arrays[2])
    return index, *arrays


def candidate_pairs(level: np.ndarray, derivative: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Three anonymous views reduce the chance that a rare change-of-direction
    # neighbor is lost by an amplitude-only screening step.
    views = []
    for data in (level, derivative):
        filled = data.copy()
        for j in range(data.shape[1]):
            values = data[mask[:, j], j]
            filled[~mask[:, j], j] = np.median(values) if len(values) else 0
        views.append(filled)
    joined = np.concatenate((views[0] / np.sqrt(2), views[1] / np.sqrt(2)), axis=1)
    views = [views[0], views[1], joined]
    pairs: set[tuple[int, int]] = set()
    n_neighbors = min(CONFIG["candidate_neighbors_per_view"] + 1, len(level))
    for view in views:
        feature = np.concatenate((view, 0.10 * mask.astype(float)), axis=1)
        neighbor = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        neighbor.fit(feature)
        ids = neighbor.kneighbors(feature, return_distance=False)
        for a, row in enumerate(ids):
            for b in row:
                if a != b:
                    pairs.add((min(a, int(b)), max(a, int(b))))
    out = np.array(sorted(pairs), dtype=np.int32)
    np.savez_compressed(OUTPUT / "candidate_pairs.npz", pairs=out)
    return out


def dtw_edges(level: np.ndarray, derivative: np.ndarray, mask: np.ndarray,
              pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    libpath = HERE / ("masked_dtw.dylib" if platform.system() == "Darwin" else "masked_dtw.so")
    subprocess.run(["c++", "-std=c++17", "-O3", "-shared", "-fPIC",
                    str(HERE / "masked_dtw.cpp"), "-o", str(libpath)], check=True)
    lib = ctypes.CDLL(str(libpath))
    func = lib.masked_pair_dtw
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                     ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    a = np.ascontiguousarray(level, dtype=np.float64)
    b = np.ascontiguousarray(derivative, dtype=np.float64)
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    p = np.ascontiguousarray(pairs, dtype=np.int32)
    distances = np.empty((len(pairs), 3), dtype=np.float64)
    func(a.ctypes.data, b.ctypes.data, m.ctypes.data, len(GRID), p.ctypes.data,
         len(p), round(CONFIG["dtw_warp_limit_h"] / CONFIG["grid_step_h"]),
         CONFIG["minimum_pair_shared_grid_points"],
         round(CONFIG["minimum_pair_shared_span_h"] / CONFIG["grid_step_h"]),
         distances.ctypes.data)
    keep = np.isfinite(distances).all(axis=1)
    pairs, distances = pairs[keep], distances[keep]
    np.savez_compressed(OUTPUT / "dtw_distances.npz", pairs=pairs,
                        level_dtw=distances[:, 0], derivative_dtw=distances[:, 1],
                        fused_dtw=distances[:, 2])
    return pairs, distances


def weighted_graph(n: int, pairs: np.ndarray, distances: np.ndarray) -> coo_matrix:
    # Local scale is estimated from anonymous DTW neighbors; no class frequency
    # or target-label agreement is used in graph construction.
    neighbors: list[list[float]] = [[] for _ in range(n)]
    for (a, b), d in zip(pairs, distances):
        neighbors[a].append(float(d))
        neighbors[b].append(float(d))
    positive = distances[distances > 1e-12]
    floor = max(float(np.percentile(positive, 5)) * 0.25, 1e-5) if len(positive) else 1e-5
    scale = np.array([max(np.median(sorted(v)[:10]), floor) if v else floor
                      for v in neighbors])
    # An extremely weak positive edge makes spectral eigenvectors numerically
    # singular when a small component is attached only by that edge.
    weight = np.exp(-np.clip(distances ** 2 / (scale[pairs[:, 0]] * scale[pairs[:, 1]]), 0, 8))
    row = np.concatenate((pairs[:, 0], pairs[:, 1]))
    col = np.concatenate((pairs[:, 1], pairs[:, 0]))
    data = np.concatenate((weight, weight))
    return coo_matrix((data, (row, col)), shape=(n, n)).tocsr()


def cluster_graphs(index: list[dict], pairs: np.ndarray, distances: np.ndarray) -> tuple[list[dict], dict]:
    eligible = [r for r in index if r["model_status"] == "model_eligible"]
    n = len(eligible)
    fused = weighted_graph(n, pairs, distances[:, 2])
    component_count, component = connected_components(fused, directed=False)
    sizes = np.bincount(component)
    largest = int(np.argmax(sizes))
    active = np.flatnonzero(component == largest)
    if len(active) < 50:
        raise ValueError("largest comparable-overlap component unexpectedly small")
    assignment = {r["curve_id"]: dict(curve_id=r["curve_id"],
                  model_status=r["model_status"], graph_component="",
                  cluster_fused_k4="", cluster_fused_k8="", cluster_fused_k12="",
                  cluster_level_k8="", cluster_derivative_k8="") for r in index}
    for i, r in enumerate(eligible):
        assignment[r["curve_id"]]["graph_component"] = int(component[i])
        if component[i] != largest:
            assignment[r["curve_id"]]["model_status"] = "outside_largest_overlap_component"
    graph_views = {"fused": fused,
                   "level": weighted_graph(n, pairs, distances[:, 0]),
                   "derivative": weighted_graph(n, pairs, distances[:, 1])}
    for view, graph in graph_views.items():
        for k in (CONFIG["diagnostic_cluster_counts"] if view == "fused" else [8]):
            model = SpectralClustering(n_clusters=k, affinity="precomputed",
                                       assign_labels="cluster_qr",
                                       random_state=CONFIG["random_seed"])
            labels = model.fit_predict(graph[active][:, active])
            for i, label in zip(active, labels):
                assignment[eligible[i]["curve_id"]][f"cluster_{view}_k{k}"] = int(label)
    records = [assignment[r["curve_id"]] for r in index]
    write_rows(OUTPUT / "anonymous_assignments.csv", records, list(records[0]))
    summary = dict(eligible=n, comparable_edges=len(pairs),
                   graph_components=int(component_count), largest_component=len(active),
                   other_component_curves=int(n - len(active)),
                   primary_cluster_count=CONFIG["primary_cluster_count"],
                   primary_cluster_sizes=dict(Counter(r["cluster_fused_k8"] for r in records
                                                      if r["cluster_fused_k8"] != "")))
    return records, summary


def show_clusters(index: list[dict], assignments: list[dict], pairs: np.ndarray,
                  distances: np.ndarray) -> list[dict]:
    eligible = [r for r in index if r["model_status"] == "model_eligible"]
    id_to_position = {r["curve_id"]: i for i, r in enumerate(eligible)}
    id_to_index = {r["curve_id"]: r for r in index}
    members: dict[int, list[str]] = defaultdict(list)
    for r in assignments:
        if r["cluster_fused_k8"] != "":
            members[int(r["cluster_fused_k8"])].append(r["curve_id"])
    edges_by_node: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), d in zip(pairs, distances[:, 2]):
        edges_by_node[int(a)].append((int(b), float(d)))
        edges_by_node[int(b)].append((int(a), float(d)))
    representative_rows = []
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True, sharey=True)
    for ax, (cluster, ids) in zip(axes.flat, sorted(members.items())):
        positions = {id_to_position[cid] for cid in ids}
        ranks = []
        for cid in ids:
            pos = id_to_position[cid]
            local = [d for q, d in edges_by_node[pos] if q in positions]
            # Dense, central members become *observed* exemplars; no interpolated
            # or synthetic prototype is displayed as a measured curve.
            ranks.append((-(len(local)), np.mean(local) if local else float("inf"), cid))
        exemplar_ids = [item[2] for item in sorted(ranks)[:5]]
        representative_rows.append(dict(cluster=cluster, member_count=len(ids),
                                        source_groups=len({id_to_index[cid]["source_group"] for cid in ids}),
                                        representative_ids=";".join(exemplar_ids)))
        def draw(cid: str, color: str, alpha: float, width: float, label: str | None = None) -> None:
            with (OUTPUT / id_to_index[cid]["normalized_csv"]).open(newline="") as f:
                points = list(csv.DictReader(f))
            hour = np.array([float(p["hour"]) for p in points])
            pce = np.array([float(p["normalized_pce"]) for p in points])
            cuts = np.flatnonzero(np.diff(hour) > CONFIG["maximum_interpolation_gap_h"]) + 1
            for j, segment in enumerate(np.split(np.arange(len(hour)), cuts)):
                ax.plot(hour[segment], pce[segment], color=color,
                        alpha=alpha, linewidth=width, label=label if j == 0 else None)
        for cid in ids:
            draw(cid, "#4b6b83", 0.035, 0.7)
        for cid in exemplar_ids:
            draw(cid, None, 0.83, 1.5, cid)
        ax.set_title(f"Anonymous cluster {cluster}: {len(ids)} curves")
        ax.set_xlim(0, CONFIG["window_h"])
        ax.set_ylim(0, 1.07)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Hour (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized PCE")
    fig.suptitle("Anonymous fused-DTW clusters; original 0–200 h observations", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "anonymous_cluster_overview.png", dpi=160)
    plt.close(fig)
    write_rows(OUTPUT / "cluster_representatives.csv", representative_rows,
               ["cluster", "member_count", "source_groups", "representative_ids"])
    return representative_rows


def event_review_queue(index: list[dict], assignments: list[dict]) -> None:
    """Post-clustering shape-event retrieval; does not affect any cluster."""
    cluster = {r["curve_id"]: r["cluster_fused_k8"] for r in assignments}
    records = []
    for r in index:
        if not r["normalized_csv"] or r["model_status"] != "model_eligible":
            continue
        with (OUTPUT / r["normalized_csv"]).open(newline="") as f:
            points = list(csv.DictReader(f))
        by_hour: dict[float, list[float]] = defaultdict(list)
        for p in points:
            by_hour[float(p["hour"])].append(float(p["normalized_pce"]))
        hours = np.array(sorted(by_hour))
        values = np.array([np.median(by_hour[h]) for h in hours])
        # The score requires both sides of a potential turn; an isolated one-step
        # excursion remains a review candidate, never an accepted four-class label.
        pre_min = np.minimum.accumulate(values)
        pre_max = np.maximum.accumulate(values)
        post_min = np.minimum.accumulate(values[::-1])[::-1]
        post_max = np.maximum.accumulate(values[::-1])[::-1]
        hill_score = np.minimum(values - pre_min, values - post_min)
        valley_score = np.minimum(pre_max - values, post_max - values)
        hill_score[0] = hill_score[-1] = 0
        valley_score[0] = valley_score[-1] = 0
        records.append(dict(curve_id=r["curve_id"], anonymous_cluster=cluster[r["curve_id"]],
                            source_group=r["source_group"], unique_hours=len(hours),
                            maximum_gap_h=f"{np.max(np.diff(hours)):.6g}",
                            rise_then_fall_score=f"{max(hill_score):.6g}",
                            fall_then_rise_score=f"{max(valley_score):.6g}",
                            rise_turn_h=f"{hours[np.argmax(hill_score)]:.6g}",
                            fall_turn_h=f"{hours[np.argmax(valley_score)]:.6g}",
                            normalized_csv=r["normalized_csv"]))
    write_rows(OUTPUT / "event_review_scores.csv", records, list(records[0]))
    queue = []
    for kind, score in (("rise_then_fall", "rise_then_fall_score"),
                        ("fall_then_rise", "fall_then_rise_score")):
        ordered = sorted(records, key=lambda r: -float(r[score]))
        seen_groups = set()
        for r in ordered:
            if r["source_group"] in seen_groups:
                continue
            queue.append(dict(review_event=kind, rank=len(seen_groups) + 1, **r))
            seen_groups.add(r["source_group"])
            if len(seen_groups) == 100:
                break
    write_rows(OUTPUT / "event_review_queue.csv", queue, list(queue[0]))


def posthoc_rule_audit(assignments: list[dict]) -> dict:
    # Deliberately called after assignments and figures have been saved.
    old = PROJECT / "target_shape_classification/research_early_200h/final_integrated_20260925/class_index.csv"
    old_by_id = {r["curve_id"]: r for r in read_rows(old)}
    comparison = []
    for r in assignments:
        prior = old_by_id.get(r["curve_id"], {})
        comparison.append(dict(curve_id=r["curve_id"],
                 anonymous_cluster_k8=r["cluster_fused_k8"],
                 anonymous_cluster_k4=r["cluster_fused_k4"],
                 prior_rule_candidate=prior.get("integrated_class", ""),
                 prior_evidence_status=prior.get("evidence_status", ""),
                 prior_classification_origin=prior.get("classification_origin", "")))
    write_rows(OUTPUT / "posthoc_rule_comparison.csv", comparison, list(comparison[0]))
    for k, filename in ((8, "posthoc_contingency.csv"),
                        (4, "posthoc_k4_contingency.csv")):
        field = f"anonymous_cluster_k{k}"
        contingency = []
        for cluster in sorted({r[field] for r in comparison if r[field] != ""}, key=int):
            subset = [r for r in comparison if r[field] == cluster]
            count = Counter(r["prior_rule_candidate"] or "unassigned" for r in subset)
            contingency.append(dict(anonymous_cluster=cluster, total=len(subset),
                   bridge=count["bridge"], hill=count["hill"], slope=count["slope"],
                   valley=count["valley"], other=count["unassigned"]))
        write_rows(OUTPUT / filename, contingency,
                   ["anonymous_cluster", "total", "bridge", "hill", "slope", "valley", "other"])
    subset = [r for r in comparison if r["anonymous_cluster_k8"] != "" and
              r["prior_rule_candidate"] in {"bridge", "hill", "slope", "valley"}]
    return dict(prior_rule_comparison_n=len(subset),
                prior_rule_ari_k8=float(adjusted_rand_score(
                    [r["prior_rule_candidate"] for r in subset],
                    [r["anonymous_cluster_k8"] for r in subset])) if subset else None,
                prior_rule_ari_k4=float(adjusted_rand_score(
                    [r["prior_rule_candidate"] for r in subset],
                    [r["anonymous_cluster_k4"] for r in subset])) if subset else None,
                warning="Prior labels are automatic rule candidates, not independent truth; this is not accuracy.")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Preparing original 0-200 h curves...", flush=True)
    index, level, derivative, mask = prepare()
    print("Eligible curves:", len(level), flush=True)
    pairs = candidate_pairs(level, derivative, mask)
    print("Anonymous candidate pairs:", len(pairs), flush=True)
    pairs, distances = dtw_edges(level, derivative, mask, pairs)
    print("Comparable DTW pairs:", len(pairs), flush=True)
    assignments, graph_info = cluster_graphs(index, pairs, distances)
    representatives = show_clusters(index, assignments, pairs, distances)
    event_review_queue(index, assignments)
    print("Anonymous assignments saved; now computing posthoc comparison.", flush=True)
    audit = posthoc_rule_audit(assignments)
    summary = dict(protocol=CONFIG["protocol"], config_sha256=sha256(HERE / "config.json"),
                   code_sha256=sha256(HERE / "run.py"),
                   cpp_sha256=sha256(HERE / "masked_dtw.cpp"),
                   canonical_manifest_sha256=sha256(Path(CONFIG["canonical_manifest"])),
                   all_file_manifest_sha256=sha256(Path(CONFIG["all_file_manifest"])),
                   canonical_count=len(index), original_file_count=len(read_rows(Path(CONFIG["all_file_manifest"]))),
                   statuses=dict(Counter(r["model_status"] for r in index)),
                   graph=graph_info, representatives=representatives, posthoc=audit)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"statuses": summary["statuses"], "graph": graph_info,
                      "posthoc": audit}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
