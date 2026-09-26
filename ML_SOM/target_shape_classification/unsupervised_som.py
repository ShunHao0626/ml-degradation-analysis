#!/usr/bin/env python3
"""SOM-first discovery of four PCE trajectory motifs.

The SOM sees no reviewed class or template. Source reviews are used for
post-training diagnostics and source-group-held-out sensitivity. Prototype
naming is a separate, auditable shape interpretation after unsupervised training.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

import run
import som_evidence


CLASSES = run.CLASSES
N_GRID = 96
N_PHASE = 16


def input_curves(root: Path, canonical: list[dict]):
    """Load PCE curves without smoothing or deleting original observations."""
    values, raw_points, metadata = [], [], []
    for row in canonical:
        if row["y_kind"] != "pce" or float(row["normalization_max"] or 0) <= 0:
            continue
        path = root / row["analysis_version"]
        if not path.exists() and row["analysis_version"] != row["source_csv"]:
            path = Path(__file__).resolve().parent / "results" / row["analysis_version"]
        try:
            t, y, _, _ = run.read_curve(path, row["folder"])
            if len(t) < 2 or t[-1] <= t[0] or np.max(y) <= 0:
                continue
            normalized = y / np.max(y)
            view = run.shape_view({"t": t, "y": normalized}, N_GRID)
            if view is None:
                continue
            values.append(view[0])
            raw_points.append(((t - t[0]) / (t[-1] - t[0]), normalized))
            metadata.append(row)
        except (OSError, ValueError, KeyError):
            continue
    return np.asarray(values), raw_points, metadata


def features(values: np.ndarray, raw_points, derivative_weight: float = .7,
             variation_weight: float = 0, drawup_weight: float = 0):
    """Retain global shape and source-point micro fluctuations >= 1%."""
    u = np.linspace(0, 1, N_GRID)
    excursion = np.maximum(np.ptp(values, axis=1), .05)
    z = (values - values[:, :1]) / excursion[:, None]
    coarse = np.array([np.interp(np.linspace(0, 1, N_PHASE), u, v) for v in z])
    dc = np.clip(np.gradient(coarse, axis=1) * (N_PHASE - 1) / 5, -3, 3)
    df = np.clip(np.gradient(z, axis=1) * (N_GRID - 1) / 5, -3, 3)
    low, high, event_bins, event_totals = [], [], [], []
    for (phase, y), scale, baseline, c in zip(raw_points, excursion, values[:, 0], coarse):
        rz = (y - baseline) / scale
        lo, hi = c.copy(), c.copy()
        for j in range(N_PHASE):
            in_bin = np.minimum((phase * N_PHASE).astype(int), N_PHASE - 1) == j
            if np.any(in_bin):
                lo[j] = np.min(rz[in_bin])
                hi[j] = np.max(rz[in_bin])
        peak, pp = find_peaks(y, prominence=.01)
        trough, tp = find_peaks(-y, prominence=.01)
        events = np.concatenate((peak, trough))
        counts = np.bincount(np.minimum((phase[events] * N_PHASE).astype(int), N_PHASE - 1), minlength=N_PHASE)
        low.append(lo)
        high.append(hi)
        event_bins.append(np.minimum(counts, 3) / 3)
        event_totals.append(len(events))
    # The source-point extrema and 1% event counts keep short oscillations in
    # the SOM distance even when they fall between the 96 common grid points.
    x = np.column_stack((.4 * z, coarse, derivative_weight * dc, .1 * df,
                         .2 * np.asarray(low), .2 * np.asarray(high),
                         .1 * np.asarray(event_bins)))
    if variation_weight:
        dz = np.diff(z, axis=1)
        signed_variation = np.column_stack([
            reducer(np.clip(dz[:, start:end], lower, upper), axis=1)
            for start, end in ((0, 24), (24, 48), (48, 72), (72, 95))
            for lower, upper, reducer in ((0, None, np.sum), (None, 0, np.sum))
        ])
        x = np.column_stack((x, variation_weight * np.clip(signed_variation, -2, 2)))
    if drawup_weight:
        # Distance above the lowest earlier observation is a directional
        # recovery descriptor. The source-point maximum in each phase bin
        # retains recoveries that the common interpolation grid misses.
        drawup = coarse - np.minimum.accumulate(coarse, axis=1)
        for i, ((phase, y), scale) in enumerate(zip(raw_points, excursion)):
            raw_drawup = (y - np.minimum.accumulate(y)) / scale
            bins = np.minimum((phase * N_PHASE).astype(int), N_PHASE - 1)
            for j in range(N_PHASE):
                in_bin = bins == j
                if np.any(in_bin):
                    drawup[i, j] = max(drawup[i, j], float(np.max(raw_drawup[in_bin])))
        x = np.column_stack((x, drawup_weight * np.clip(drawup, 0, 2)))
    # Route by the broad early trajectory. Short 1% source-point peaks still
    # enter the SOM through the envelope and event features above; they do
    # not abruptly move an otherwise unchanged curve to another map.
    branch = (np.max(values[:, :29], axis=1) - values[:, 0] >= .01).astype(int)
    return x, z, branch, np.asarray(event_totals)


def named_prototypes(codebook: np.ndarray, naming_shapes=None):
    """Interpret SOM neurons only after training; incomplete shapes stay unnamed."""
    u = np.linspace(0, 1, N_GRID)
    result = []
    for i, code in enumerate(codebook):
        shape = naming_shapes[i] if naming_shapes is not None else code
        curve = 1 + .5 * shape[:N_GRID] / .4
        m = run.morphology({"t": u, "t_hour": None, "y": curve}, run.DEFAULT)
        label = m["candidate_class"] if m and m["stages_complete"] else ""
        result.append(dict(neuron=i, label=label,
                           margin=m["score_margin"] if m else 0,
                           complete=bool(m and m["stages_complete"])))
    return result


def stage_sampling_factors(train_ix, branch_value, stage_labels, power):
    """Balance input-derived complete stages only in the early-gain branch."""
    if branch_value != 1 or not power:
        return np.ones(len(train_ix))
    if stage_labels is None:
        raise ValueError("stage labels required for early-stage balance")
    counts = Counter(stage_labels[i] for i in train_ix if stage_labels[i])
    if not counts:
        raise ValueError("no complete early-gain stages to balance")
    largest = max(counts.values())
    return np.array([
        min(15., (largest / counts[stage_labels[i]]) ** power)
        if stage_labels[i] else 1. for i in train_ix])


def sample_training_indices(train_ix, branch_ix, weights, seed, n, coupling="cdf"):
    """Draw weighted curves, optionally coupling draws across source holdouts."""
    rng = np.random.default_rng(seed)
    if coupling == "cdf":
        return train_ix[rng.choice(len(train_ix), n, p=weights)]
    if coupling != "gumbel":
        raise ValueError(f"unknown sampling coupling: {coupling}")
    # Each draw has one perturbation per curve in the full branch. Removing a
    # source preserves every remaining curve's perturbation at the same draw.
    columns = np.searchsorted(branch_ix, train_ix)
    if np.any(columns >= len(branch_ix)) or np.any(branch_ix[columns] != train_ix):
        raise ValueError("training curves must be a subset of the branch")
    perturbations = rng.gumbel(size=(n, len(branch_ix)))[:, columns]
    log_weights = np.full(len(weights), -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    return train_ix[np.argmax(perturbations + log_weights, axis=1)]


def train(x: np.ndarray, branch: np.ndarray, train_mask: np.ndarray, seed: int,
          side: int, iterations: int, density_power: float = 2,
          naming: str = "prototype", density_clip_quantile: float = .95,
          stage_labels=None, early_stage_balance_power: float = 0,
          sampling_coupling: str = "cdf"):
    models, neuron_rows = {}, []
    for b in (0, 1):
        train_ix = np.flatnonzero((branch == b) & train_mask)
        xx = x[train_ix]
        if len(xx) < 12:
            raise ValueError(f"Branch {b} has too few training curves")
        distances = cdist(xx, xx)
        neighbor = np.partition(distances, 10, axis=1)[:, 10]
        weights = np.minimum(neighbor, np.quantile(neighbor, density_clip_quantile)) ** density_power
        weights *= stage_sampling_factors(train_ix, b, stage_labels,
                                          early_stage_balance_power)
        weights = weights / weights.sum()
        branch_ix = np.flatnonzero(branch == b)
        sampled = x[sample_training_indices(train_ix, branch_ix, weights,
                                            seed + b, 5000, sampling_coupling)]
        som = MiniSom(side, side, x.shape[1], sigma=side / 4,
                      learning_rate=.1, random_seed=seed + b)
        som.random_weights_init(sampled)
        som.train_random(sampled, iterations)
        codebook = som.get_weights().reshape(side * side, -1)
        naming_shapes = xx[np.argmin(cdist(codebook, xx), axis=1)] if naming == "medoid" else None
        named = named_prototypes(codebook, naming_shapes)
        for item in named:
            neuron_rows.append(dict(branch="early_gain" if b else "no_early_gain", **item))
        models[b] = (codebook, named, train_ix)
    return models, neuron_rows


def assign(x, branch, metadata, events, models):
    output = []
    for b in (0, 1):
        subset = np.flatnonzero(branch == b)
        codebook, named, train_ix = models[b]
        d = cdist(x[subset], codebook)
        eligible = np.array([p["label"] != "" for p in named])
        if not np.any(eligible):
            raise ValueError(f"Branch {b} has no complete named prototype")
        qualified = np.where(eligible)[0]
        near = qualified[np.argmin(d[:, qualified], axis=1)]
        bm = np.argmin(d, axis=1)
        for local, i in enumerate(subset):
            candidate = int(near[local])
            bmu = int(bm[local])
            row = metadata[i]
            output.append(dict(file_id=row["file_id"], source_csv=row["source_csv"],
                               source_group=row["source_group"],
                               branch="early_gain" if b else "no_early_gain",
                               som_candidate_class=named[candidate]["label"],
                               som_candidate_neuron=candidate,
                               best_matching_neuron=bmu,
                               bmu_morphology=named[bmu]["label"],
                               som_direct_group=named[bmu]["label"] or "unresolved",
                               distance_to_candidate=float(d[local, candidate]),
                               quantization_distance=float(d[local, bmu]),
                               candidate_is_bmu=bool(candidate == bmu),
                               one_percent_source_extrema=int(events[i]),
                               source_sha256=row["source_sha256"],
                               analysis_sha256=row["analysis_sha256"],
                               trained_on_source=bool(i in train_ix)))
    return sorted(output, key=lambda r: r["file_id"])


def review_check(assignments, reviews):
    by_id = {r["file_id"]: r for r in assignments}
    detail = []
    for review in reviews:
        if review["source_verified"].lower() != "yes" or review["stages_verified"].lower() != "yes":
            continue
        row = by_id.get(review["file_id"])
        detail.append(dict(file_id=review["file_id"], reviewed_class=review["reviewed_class"],
                           som_candidate_class=row["som_candidate_class"] if row else "",
                           correct=bool(row and row["som_candidate_class"] == review["reviewed_class"]),
                           source_group=row["source_group"] if row else "",
                           best_matching_neuron=row["best_matching_neuron"] if row else "",
                           som_candidate_neuron=row["som_candidate_neuron"] if row else ""))
    return detail


def balanced_stage_agreement(assignments, stage_reference):
    """Unlabeled, class-balanced agreement with complete source-curve stages."""
    assigned = {r['file_id']: r['som_candidate_class'] for r in assignments}
    totals = Counter(label for label in stage_reference.values() if label)
    correct = Counter(label for file_id, label in stage_reference.items()
                      if label and assigned.get(file_id) == label)
    if any(totals[label] == 0 for label in CLASSES):
        raise ValueError('All four complete stage motifs are required for seed selection')
    score = float(np.mean([correct[label] / totals[label] for label in CLASSES]))
    return score, {label: dict(correct=correct[label], total=totals[label]) for label in CLASSES}


def select_seed(seeds, seed_quality, stage_scores, topology_tolerance=.005):
    """Choose a near-best topology with balanced coverage of four stage motifs."""
    best_topology = min(seed_quality[seed]['topographic_error'] for seed in seeds)
    eligible = [seed for seed in seeds
                if seed_quality[seed]['topographic_error'] <= best_topology + topology_tolerance]
    return min(eligible, key=lambda seed: (-stage_scores[seed],
                                           seed_quality[seed]['topographic_error'],
                                           seed_quality[seed]['quantization_error'], seed))


def consensus(seed_assignments, base_seed):
    """Keep a full four-way candidate while exposing random-seed instability."""
    by_seed = {seed: {r["file_id"]: r for r in rows}
               for seed, rows in seed_assignments.items()}
    result, votes = [], []
    for original in seed_assignments[base_seed]:
        file_id = original["file_id"]
        labels = [by_seed[seed][file_id]["som_candidate_class"] for seed in sorted(by_seed)]
        counts = Counter(labels)
        maximum = max(counts.values())
        winners = [k for k in CLASSES if counts[k] == maximum]
        chosen = original["som_candidate_class"] if original["som_candidate_class"] in winners else winners[0]
        row = dict(original)
        row["ensemble_majority_class"] = chosen
        row["ensemble_agreement"] = counts[row["som_candidate_class"]] / len(labels)
        row["som_stability"] = "stable" if counts[row["som_candidate_class"]] >= 4 else "variable"
        result.append(row)
        votes.extend(dict(file_id=file_id, seed=seed, som_candidate_class=by_seed[seed][file_id]["som_candidate_class"])
                     for seed in sorted(by_seed))
    return result, votes


def plot_prototypes(path: Path, models, assignments):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    u = np.linspace(0, 1, N_GRID)
    support = Counter((a["branch"], int(a["som_candidate_neuron"]))
                      for a in assignments)
    for ax, label in zip(axs.ravel(), CLASSES):
        samples = []
        for b in (0, 1):
            codebook, named, _ = models[b]
            branch_name = "early_gain" if b else "no_early_gain"
            samples.extend((1 + .5 * codebook[p["neuron"], :N_GRID] / .4,
                            p["margin"], support[branch_name, p["neuron"]])
                           for p in named if p["label"] == label)
        for shape, _, _ in samples:
            ax.plot(u, shape, color="#7899ab", alpha=.14, lw=.7)
        if samples:
            occupied = [item for item in samples if item[2] >= 3]
            if not occupied:
                occupied = [item for item in samples if item[2] > 0]
            representative = max(occupied or samples, key=lambda item: item[1])
            ax.plot(u, representative[0], color="#b42318", lw=2,
                    label=f"representative neuron ({representative[2]} curves)")
            ax.legend(loc="lower left", frameon=False, fontsize=8)
        ax.set_title(f"{label} | {sum(a['som_candidate_class']==label for a in assignments)} curves | {len(samples)} neurons")
        ax.set(xlabel="Fraction of observed duration", ylabel="Relative shape (display scale)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_verified_representatives(path: Path, root: Path, results: Path,
                                  assignments, metadata, reviews):
    """Show actual source observations in hours for all four reviewed shapes."""
    assigned = {r["file_id"]: r for r in assignments}
    sources = {r["file_id"]: r for r in metadata}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, label in zip(axs.ravel(), CLASSES):
        matches, misses = [], []
        for review in reviews:
            a = assigned.get(review["file_id"])
            source = sources.get(review["file_id"])
            if (a and source and review["source_verified"].lower() == "yes"
                    and review["stages_verified"].lower() == "yes"
                    and review["reviewed_class"] == label
                    and source["time_factor"]):
                (matches if a["som_candidate_class"] == label else misses).append((a, source))
        choices = matches or misses
        choices.sort(key=lambda pair: (pair[0]["som_stability"] != "stable",
                                       float(pair[0]["quantization_distance"])))
        if choices:
            a, source = choices[0]
            version = root / source["analysis_version"]
            if not version.exists() and source["analysis_version"] != source["source_csv"]:
                version = results / source["analysis_version"]
            t, y, _, _ = run.read_curve(version, source["folder"])
            t_hour = t * float(source["time_factor"])
            y = y / np.max(y)
            grid = np.linspace(t_hour[0], t_hour[-1], 300)
            ax.plot(grid, PchipInterpolator(t_hour, y, extrapolate=False)(grid),
                    color="#b42318", lw=1.5, label="PCHIP within observations")
            ax.scatter(t_hour, y, color="black", s=14, zorder=3, label="source points")
            if matches:
                title = f"{label} | {source['file_id']} | SOM agreement {float(a['ensemble_agreement']):.0%}"
            else:
                title = f"{label} | {source['file_id']} | source verified; SOM missed"
            ax.set_title(title)
            ax.legend(fontsize=7)
        else:
            ax.set_title(f"{label} | no source-verified SOM match")
        ax.set(xlabel="Time (h)", ylabel="Normalized PCE")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def map_quality(x, branch, models, fit_mask, side):
    """Euclidean quantization and 8-neighbor topographic errors."""
    quality = {}
    for name, mask in (("fit", fit_mask), ("heldout_source", ~fit_mask),
                       ("all_comparable", np.ones(len(x), bool))):
        qe, topo = [], []
        for b in (0, 1):
            ix = np.flatnonzero((branch == b) & mask)
            if not len(ix):
                continue
            codebook = models[b][0]
            d = cdist(x[ix], codebook)
            two = np.argpartition(d, 1, axis=1)[:, :2]
            first = two[np.arange(len(ix)), np.argmin(d[np.arange(len(ix))[:, None], two], axis=1)]
            second = two[np.arange(len(ix)), np.argmax(d[np.arange(len(ix))[:, None], two], axis=1)]
            qe.extend(d[np.arange(len(ix)), first])
            topo.extend(np.maximum(np.abs(first // side - second // side),
                                   np.abs(first % side - second % side)) > 1)
        quality[name] = dict(quantization_error=float(np.mean(qe)) if qe else None,
                             topographic_error=float(np.mean(topo)) if topo else None,
                             curves=len(qe))
    return quality


def motif_retrieval(x, branch, metadata, seeds, model_arrays, prototype_rows,
                    reviews, audit_findings=None):
    """Rank proximity to each post-hoc named SOM motif without forcing top-1."""
    distances = defaultdict(list)
    ratios = defaultdict(list)
    for seed in seeds:
        for b, branch_name in ((0, "no_early_gain"), (1, "early_gain")):
            ix = np.flatnonzero(branch == b)
            codebook = model_arrays[f"seed_{seed}_{branch_name}_codebook"]
            d = cdist(x[ix], codebook)
            named = [p for p in prototype_rows if p["seed"] == seed and
                     p["branch"] == branch_name and p["label"]]
            any_neuron = [p["neuron"] for p in named]
            if not any_neuron:
                continue
            closest_named = np.min(d[:, any_neuron], axis=1)
            for label in CLASSES:
                neurons = [p["neuron"] for p in named if p["label"] == label]
                if not neurons:
                    continue
                class_distance = np.min(d[:, neurons], axis=1)
                for local, i in enumerate(ix):
                    key = (metadata[i]["file_id"], label)
                    distances[key].append(float(class_distance[local]))
                    ratios[key].append(float(class_distance[local] / max(closest_named[local], 1e-9)))
    reviewed = {r["file_id"]: r for r in reviews}
    audited = {r["file_id"]: r for r in (audit_findings or [])}
    rows = []
    for r in metadata:
        for label in CLASSES:
            key = (r["file_id"], label)
            if key not in distances:
                continue
            review = reviewed.get(r["file_id"])
            audit = audited.get(r["file_id"])
            rows.append(dict(file_id=r["file_id"], source_group=r["source_group"],
                             source_csv=r["source_csv"], source_image=r["source_image"],
                             target_class=label, median_distance=float(np.median(distances[key])),
                             median_relative_distance=float(np.median(ratios[key])),
                             seeds_with_prototype=len(distances[key]),
                             quality_flags=r["quality_flags"],
                             analysis_version=r["analysis_version"],
                             reviewed_class=review["reviewed_class"] if review else "",
                             reviewed_stages_verified=review["stages_verified"] if review else "",
                             source_audit_issue=audit["issue"] if audit else "",
                             source_audit_resolution=audit["resolution_status"] if audit else "",
                             source_audit_rejected_class=audit.get("rejected_candidate_class", "") if audit else ""))
    shortlist = []
    for label in CLASSES:
        members = [r for r in rows if r["target_class"] == label]
        members.sort(key=lambda r: (r["median_relative_distance"],
                                    r["median_distance"], r["file_id"]))
        groups = set()
        for rank, r in enumerate(members, 1):
            r["rank_in_class"] = rank
            if len(groups) < 50 and r["source_group"] not in groups:
                shortlist.append(dict(r))
                groups.add(r["source_group"])
    return rows, shortlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "data_final")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "som_primary")
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density-power", type=float, default=2)
    parser.add_argument("--density-clip-quantile", type=float, default=.95)
    parser.add_argument("--naming", choices=("prototype", "medoid"), default="prototype")
    parser.add_argument("--derivative-weight", type=float, default=.3)
    parser.add_argument("--variation-weight", type=float, default=0)
    parser.add_argument("--drawup-weight", type=float, default=0)
    parser.add_argument("--ensemble-seeds", default="41,42,43,44,45")
    parser.add_argument("--topology-tolerance", type=float, default=.005)
    parser.add_argument("--early-stage-balance-power", type=float, default=0)
    parser.add_argument("--sampling-coupling", choices=("cdf", "gumbel"), default="cdf")
    args = parser.parse_args()
    if not 0 < args.density_clip_quantile < 1:
        parser.error("--density-clip-quantile must be between 0 and 1")
    if args.topology_tolerance < 0:
        parser.error("--topology-tolerance must be nonnegative")
    if args.early_stage_balance_power < 0:
        parser.error("--early-stage-balance-power must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    results = Path(__file__).resolve().parent / "results"
    canonical = run.rows(results / "canonical_curves.csv")
    reviews = run.rows(results / "source_reviews.csv")
    values, raw, metadata = input_curves(args.root, canonical)
    x, z, branch, events = features(values, raw, args.derivative_weight,
                                   args.variation_weight, args.drawup_weight)
    # These stages are computed from the input curves alone. When requested,
    # rare complete shapes receive more sampling opportunities in the
    # early-gain SOM; no source-review classes enter the fit.
    phase = np.linspace(0, 1, N_GRID)
    stage_reference = {}
    for value, row in zip(values, metadata):
        morphology = run.morphology({'t': phase, 't_hour': None, 'y': value}, run.DEFAULT)
        stage_reference[row['file_id']] = (morphology['candidate_class']
                                           if morphology['stages_complete'] else '')
    stage_labels = [stage_reference[row['file_id']] for row in metadata]
    seeds = sorted({int(s) for s in args.ensemble_seeds.split(",") if s.strip()} | {args.seed})
    # Fit from scratch with every reviewed DOI excluded. This checks behavior
    # on unseen literature sources, although these reviews informed exploration.
    reviewed_ids = {v["file_id"] for v in reviews}
    reviewed_groups = {r["source_group"] for r in metadata if r["file_id"] in reviewed_ids}
    train_mask = np.array([r["source_group"] not in reviewed_groups for r in metadata])
    full_by_seed, held_by_seed, full_models_by_seed, held_models_by_seed = {}, {}, {}, {}
    prototype_rows, seed_quality = [], {}
    model_arrays = {"feature_matrix": x, "file_ids": np.array([r["file_id"] for r in metadata])}
    per_seed_review, per_seed_primary_review = {}, {}
    for seed in seeds:
        full, protos = train(x, branch, np.ones(len(x), bool), seed, args.side,
                             args.iterations, args.density_power, args.naming,
                             args.density_clip_quantile, stage_labels,
                             args.early_stage_balance_power, args.sampling_coupling)
        full_models_by_seed[seed] = full
        full_by_seed[seed] = assign(x, branch, metadata, events, full)
        seed_quality[seed] = map_quality(x, branch, full, np.ones(len(x), bool),
                                        args.side)["fit"]
        for p in protos:
            prototype_rows.append(dict(seed=seed, **p))
        held_models, held_protos = train(x, branch, train_mask, seed, args.side,
                                        args.iterations, args.density_power, args.naming,
                                        args.density_clip_quantile, stage_labels,
                                        args.early_stage_balance_power, args.sampling_coupling)
        held_models_by_seed[seed] = held_models
        held_by_seed[seed] = assign(x, branch, metadata, events, held_models)
        per_seed_review[str(seed)] = {k: sum(r["correct"] for r in review_check(held_by_seed[seed], reviews)
                                             if r["reviewed_class"] == k) for k in CLASSES}
        per_seed_primary_review[str(seed)] = {k: sum(r["correct"] for r in review_check(full_by_seed[seed], reviews)
                                                     if r["reviewed_class"] == k) for k in CLASSES}
        for b, branch_name in ((0, "no_early_gain"), (1, "early_gain")):
            model_arrays[f"seed_{seed}_{branch_name}_codebook"] = full[b][0]
            model_arrays[f"heldout_seed_{seed}_{branch_name}_codebook"] = held_models[b][0]
    # Shape-stage agreement is computed from the same unlabeled input curves;
    # it balances rare Hill/Valley motifs and never reads source reviews.
    stage_scores, stage_detail = {}, {}
    for seed in seeds:
        stage_scores[seed], stage_detail[seed] = balanced_stage_agreement(
            full_by_seed[seed], stage_reference)
    selected_seed = select_seed(seeds, seed_quality, stage_scores,
                                args.topology_tolerance)
    base_model = full_models_by_seed[selected_seed]
    held_model = held_models_by_seed[selected_seed]
    assignments, seed_votes = consensus(full_by_seed, selected_seed)
    held_assignments, held_votes = consensus(held_by_seed, selected_seed)
    review_detail = review_check(held_assignments, reviews)
    primary_review = review_check(assignments, reviews)
    for review_rows, assigned_rows, votes in ((review_detail, held_assignments, held_votes),
                                              (primary_review, assignments, seed_votes)):
        by_id = {r["file_id"]: r for r in assigned_rows}
        by_vote = defaultdict(dict)
        for vote in votes:
            by_vote[vote["file_id"]][vote["seed"]] = vote["som_candidate_class"]
        for row in review_rows:
            match = by_id.get(row["file_id"])
            row["ensemble_agreement"] = match["ensemble_agreement"] if match else ""
            row["som_stability"] = match["som_stability"] if match else ""
            row["seed_votes"] = json.dumps(by_vote[row["file_id"]], sort_keys=True)
    for row in prototype_rows:
        row["assigned_curves"] = sum(a["branch"] == row["branch"] and
                                     a["best_matching_neuron"] == row["neuron"]
                                     for a in full_by_seed[row["seed"]])
    run.write_rows(args.output / "som_assignments.csv", assignments, list(assignments[0]))
    run.write_rows(args.output / "som_seed_votes.csv", seed_votes, list(seed_votes[0]))
    run.write_rows(args.output / "som_prototypes.csv", prototype_rows, list(prototype_rows[0]))
    run.write_rows(args.output / "source_heldout_review.csv", review_detail, list(review_detail[0]))
    run.write_rows(args.output / "source_primary_review.csv", primary_review, list(primary_review[0]))
    retrieval, shortlist = motif_retrieval(x, branch, metadata, seeds,
                                           model_arrays, prototype_rows, reviews,
                                           run.rows(results / "source_audit_findings.csv"))
    run.write_rows(args.output / "som_motif_retrieval.csv", retrieval, list(retrieval[0]))
    run.write_rows(args.output / "som_motif_shortlist.csv", shortlist, list(shortlist[0]))
    held_by_id = {r["file_id"]: r for r in held_assignments}
    fit_sensitivity = [dict(file_id=r["file_id"], primary_class=r["som_candidate_class"],
                            source_heldout_class=held_by_id[r["file_id"]]["som_candidate_class"],
                            changed=r["som_candidate_class"] != held_by_id[r["file_id"]]["som_candidate_class"])
                       for r in assignments]
    run.write_rows(args.output / "source_holdout_sensitivity.csv", fit_sensitivity, list(fit_sensitivity[0]))
    (args.output / "all_source_fit_sensitivity.csv").unlink(missing_ok=True)
    assigned_ids = {a["file_id"] for a in assignments}
    excluded = [dict(file_id=r["file_id"], source_csv=r["source_csv"],
                     reason=("response_identity_" + r["y_kind"] if r["y_kind"] != "pce"
                             else "nonpositive_maximum" if float(r["normalization_max"] or 0) <= 0
                             else "unusable_coordinates"))
                for r in canonical if r["file_id"] not in assigned_ids]
    run.write_rows(args.output / "excluded_curves.csv", excluded, ["file_id", "source_csv", "reason"])
    by_canonical = {r["file_id"]: r for r in assignments}
    excluded_by_id = {r["file_id"]: r["reason"] for r in excluded}
    all_files = []
    for source in run.rows(results / "input_manifest.csv"):
        a = by_canonical.get(source["canonical_id"])
        all_files.append(dict(file_id=source["file_id"], canonical_id=source["canonical_id"],
                              source_csv=source["source_csv"], source_sha256=source["source_sha256"],
                              y_kind=source["y_kind"],
                              som_candidate_class=a["som_candidate_class"] if a else "",
                              som_direct_group=a["som_direct_group"] if a else "not_comparable_PCE",
                              som_stability=a["som_stability"] if a else "not_comparable_PCE",
                              ensemble_agreement=a["ensemble_agreement"] if a else "",
                              one_percent_source_extrema=a["one_percent_source_extrema"] if a else "",
                              exclusion_reason="" if a else excluded_by_id.get(source["canonical_id"], "unusable_coordinates")))
    run.write_rows(args.output / "som_all_files.csv", all_files, list(all_files[0]))
    evidence_summary = som_evidence.build(results, args.output)
    source_by_id = {r["file_id"]: r for r in metadata}
    review_by_id = {r["file_id"]: r for r in reviews}
    followup = []
    for a in assignments:
        review = review_by_id.get(a["file_id"])
        mismatch = bool(review and review["source_verified"].lower() == "yes"
                        and review["stages_verified"].lower() == "yes"
                        and review["reviewed_class"] != a["som_candidate_class"])
        unstable = a["som_stability"] == "variable"
        if not (mismatch or unstable):
            continue
        source = source_by_id[a["file_id"]]
        priority = 0 if mismatch else (1 if a["som_candidate_class"] in ("hill", "valley") else 2)
        followup.append(dict(priority=priority, file_id=a["file_id"],
                             source_csv=a["source_csv"], source_group=a["source_group"],
                             source_image=source["source_image"],
                             som_candidate_class=a["som_candidate_class"],
                             reviewed_class=review["reviewed_class"] if review else "",
                             review_reason="source_verified_disagreement" if mismatch else "random_seed_instability",
                             ensemble_agreement=a["ensemble_agreement"],
                             quantization_distance=a["quantization_distance"],
                             one_percent_source_extrema=a["one_percent_source_extrema"],
                             quality_flags=source["quality_flags"]))
    followup.sort(key=lambda r: (r["priority"], -float(r["quantization_distance"])))
    run.write_rows(args.output / "som_followup_queue.csv", followup,
                   ["priority", "file_id", "source_csv", "source_group", "source_image",
                    "som_candidate_class", "reviewed_class", "review_reason",
                    "ensemble_agreement", "quantization_distance",
                    "one_percent_source_extrema", "quality_flags"])
    np.savez_compressed(args.output / "som_model.npz", **model_arrays)
    plot_prototypes(args.output / "four_som_motifs.png", base_model, assignments)
    plot_verified_representatives(args.output / "som_verified_representatives.png",
                                  args.root, results, assignments, metadata, reviews)
    counts = Counter(a["som_candidate_class"] for a in assignments)
    direct_counts = Counter(a["som_direct_group"] for a in assignments)
    held_counts = {k: dict(correct=sum(v["correct"] for v in review_detail if v["reviewed_class"] == k),
                           total=sum(v["reviewed_class"] == k for v in review_detail)) for k in CLASSES}
    primary_counts = {k: dict(correct=sum(v["correct"] for v in primary_review if v["reviewed_class"] == k),
                              total=sum(v["reviewed_class"] == k for v in primary_review)) for k in CLASSES}
    summary = dict(method=("hierarchical density-balanced SOM with input-stage-balanced early-gain sampling"
                           if args.early_stage_balance_power else
                           "hierarchical density-balanced SOM; post-hoc prototype morphology"),
                   source_curves=len(canonical), comparable_curves=len(metadata),
                   fitted_curves=len(metadata), source_heldout_fitted_curves=int(train_mask.sum()),
                   excluded_curves=len(excluded), direct_group_counts=dict(direct_counts),
                   source_heldout_groups=len(reviewed_groups),
                   branch_counts={"early_gain": int(np.sum(branch == 1)),
                                  "no_early_gain": int(np.sum(branch == 0))},
                   side=args.side, iterations=args.iterations, seed=selected_seed,
                   seed_selection="topology_within_tolerance_then_balanced_unlabeled_stage_agreement",
                   topology_tolerance=args.topology_tolerance,
                   stage_agreement_by_seed={str(k): stage_scores[k] for k in seeds},
                   stage_agreement_detail={str(k): stage_detail[k] for k in seeds},
                   seed_quality={str(k):v for k,v in seed_quality.items()},
                   ensemble_seeds=seeds,
                   density_power=args.density_power,
                   density_clip_quantile=args.density_clip_quantile,
                   early_stage_balance_power=args.early_stage_balance_power,
                   sampling_coupling=args.sampling_coupling,
                   naming=args.naming, derivative_weight=args.derivative_weight,
                   variation_weight=args.variation_weight,
                   drawup_weight=args.drawup_weight,
                   classes=dict(counts), source_primary_review=primary_counts,
                   source_heldout_review=held_counts,
                   source_primary_correct=sum(v["correct"] for v in primary_review),
                   source_primary_total=len(primary_review),
                   source_heldout_correct=sum(v["correct"] for v in review_detail),
                   source_heldout_total=len(review_detail),
                   primary_map_quality=seed_quality[selected_seed],
                   source_heldout_map_quality=map_quality(x, branch, held_model, train_mask, args.side),
                   per_seed_source_heldout_review=per_seed_review,
                   per_seed_source_primary_review=per_seed_primary_review,
                   stable_candidates=sum(a["som_stability"] == "stable" for a in assignments),
                   variable_candidates=sum(a["som_stability"] == "variable" for a in assignments),
                   followup_queue=len(followup),
                   motif_shortlist=len(shortlist),
                   all_input_files=len(all_files),
                   som_evidence_status_counts=evidence_summary["status_counts"],
                   source_holdout_changed_curves=sum(r["changed"] for r in fit_sensitivity),
                   one_percent_event_curves=int(np.sum(events > 0)),
                   prototype_counts=dict(Counter(p["label"] or "unnamed" for p in prototype_rows)),
                   notes=["The primary SOM is fitted on every comparable PCE curve without human-reviewed labels; it is a transductive discovery map.",
                          f"A separate sensitivity SOM excludes all {len(reviewed_groups)} reviewed DOI groups from fitting.",
                          "When stage balancing is enabled, stage names inferred from each input curve affect sampling weights; source reviews and templates do not enter training.",
                          "Prototype names are assigned after training with stage-shape rules.",
                          "A nearest named neuron gives a candidate, not verified class membership.",
                          "When the best matching neuron has no complete four-stage name, its direct group is unresolved rather than a forced target class.",
                          "Source-point normalized extrema of at least 1% are retained in features and counts.",
                          "Some source reviews informed representation exploration and some were selected from SOM retrieval; neither comparison is an independent accuracy estimate."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
