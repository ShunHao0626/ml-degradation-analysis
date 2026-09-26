#!/usr/bin/env python3
"""Minimal, data-free demonstration of PFD-DTW-HC."""

from __future__ import annotations

import numpy as np

from pfd_dtw import pfd_dtw_hierarchical_clustering


def main() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 120)
    series = [
        np.sin(2 * np.pi * x) + rng.normal(0, 0.04, x.size),
        np.sin(2 * np.pi * (x + 0.03)) + rng.normal(0, 0.04, x.size),
        np.exp(-((x - 0.45) / 0.13) ** 2) + rng.normal(0, 0.04, x.size),
        np.exp(-((x - 0.50) / 0.13) ** 2) + rng.normal(0, 0.04, x.size),
    ]
    result = pfd_dtw_hierarchical_clustering(series, n_clusters=2)
    print("cluster labels:", result.labels.tolist())
    print("distance matrix:\n", np.round(result.distance_matrix, 3))


if __name__ == "__main__":
    main()
