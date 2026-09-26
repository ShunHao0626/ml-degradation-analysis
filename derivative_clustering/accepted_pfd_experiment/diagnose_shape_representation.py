#!/usr/bin/env python3
"""Development-only diagnostic for locking a derivative shape representation.

The synthetic labels are never passed to the transformations or clustering;
they are used only to report recovery scores after clustering.  Seed 2026 is
the development set.  Later seeds are reserved for confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from run_injection_experiment import (
    CLASSES,
    GRID,
    HP_LAMBDA,
    generate_synthetic_pool,
    hp_filter,
    load_real_curves,
)


def _bin_summaries(signal: np.ndarray, bins: int) -> list[float]:
    result: list[float] = []
    for indexes in np.array_split(np.arange(signal.size), bins):
        part = signal[indexes]
        result.extend(
            [
                float(np.mean(part)),
                float(np.median(part)),
                float(np.mean(np.maximum(part, 0.0))),
                float(np.mean(np.minimum(part, 0.0))),
            ]
        )
    return result


def derivative_shape_features(values: np.ndarray) -> np.ndarray:
    """Generic multiscale trend descriptors, without class-specific rules."""
    features: list[list[float]] = []
    dt = float(GRID[1] - GRID[0])
    for curve in values:
        smooth = hp_filter(curve, smoothing=HP_LAMBDA)
        centered = smooth - np.median(smooth[:3])
        d1 = np.gradient(smooth, dt)
        d2 = np.gradient(d1, dt)
        row: list[float] = []
        # Coarse levels keep endpoint/amplitude information while the remaining
        # blocks encode the order and persistence of derivative signs.
        row.extend(np.interp(np.linspace(0, 200, 9), GRID, centered).tolist())
        for bins in (4, 8, 16):
            row.extend(_bin_summaries(d1, bins))
        for bins in (4, 8):
            row.extend(_bin_summaries(d2, bins))
        row.extend(
            [
                float(centered[-1]),
                float(np.min(centered)),
                float(np.max(centered)),
                float(GRID[int(np.argmin(centered))] / 200.0),
                float(GRID[int(np.argmax(centered))] / 200.0),
                float(np.sum(np.maximum(d1, 0.0)) * dt),
                float(np.sum(np.maximum(-d1, 0.0)) * dt),
                float(np.sum(np.abs(d2)) * dt),
            ]
        )
        features.append(row)
    return np.asarray(features, dtype=float)


def ordered_slope_features(values: np.ndarray, intervals: int = 16) -> np.ndarray:
    """Multiscale-compatible ordered slopes on equal time intervals.

    Unlike pooled derivative summaries, this deliberately preserves when a
    rise or fall occurs.  It remains class-agnostic: no target-specific rule,
    template or threshold is encoded.
    """
    sample_times = np.linspace(float(GRID[0]), float(GRID[-1]), intervals + 1)
    rows: list[np.ndarray] = []
    for curve in values:
        smooth = hp_filter(curve, smoothing=HP_LAMBDA)
        sampled = np.interp(sample_times, GRID, smooth)
        rows.append(np.diff(sampled) / np.diff(sample_times))
    return np.asarray(rows, dtype=float)


def evaluate(features: np.ndarray, truth: np.ndarray) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for scaler_name, scaler in (("standard", StandardScaler()), ("robust", RobustScaler())):
        transformed = scaler.fit_transform(features)
        for method, estimator in (
            ("ward", AgglomerativeClustering(n_clusters=4, linkage="ward")),
            ("average", AgglomerativeClustering(n_clusters=4, linkage="average")),
            ("kmeans", KMeans(n_clusters=4, n_init=50, random_state=17)),
        ):
            predicted = estimator.fit_predict(transformed)
            results.append(
                {
                    "scaler": scaler_name,
                    "method": method,
                    "ari": float(adjusted_rand_score(truth, predicted)),
                    "homogeneity": float(homogeneity_score(truth, predicted)),
                    "silhouette": float(silhouette_score(transformed, predicted)),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "01_hour_audit",
    )
    args = parser.parse_args()
    real_ids, real_values, raw_hour = load_real_curves(args.audit_dir)
    pool = generate_synthetic_pool(
        real_ids, real_values, raw_hour, per_class=args.per_class, seed=args.seed
    )
    values = np.vstack([item.grid_values for item in pool])
    truth = np.asarray([item.target_class for item in pool])
    report = {
        "seed": args.seed,
        "per_class": args.per_class,
        "results": evaluate(derivative_shape_features(values), truth),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
