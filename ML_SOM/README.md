# ML_SOM workspace

English · [中文](README_CN.md)

This directory contains upstream dataset construction, the broader 0–200 h SOM/morphology study, and a cross-experiment figure collector. The [separate early degradation repository](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) maintains the fixed 1,842-curve density-weighted K-means reproduction; its duplicate code and input copy have been removed from this directory.

| Directory | Purpose | Git status |
|---|---|---|
| [`target_shape_classification/`](target_shape_classification/CODE_MAP.md) | 0–200 h source audit, SOM, morphology, and upstream exploration | Code and methods public; raw inputs and per-curve results local |
| [`canonical_2246_full_dataset_20260925/`](canonical_2246_full_dataset_20260925/README.md) | Build and verify 2,246 canonical curves | Builder public; generated data local |
| [`experiment_cluster_figures_20260926/`](experiment_cluster_figures_20260926/README.md) | Collect cross-experiment figures with provenance | Collector public; copied gallery local |
| `data_final/`, `original_curves_2250/` | Upstream curves and source files | Local data, excluded from Git |
| `si_1842_supplementary/` | Unique historical SI text, figures, and plotting values | Local supplement, excluded from Git; use the separate repository for reproduction |

The maintained Hartono/PvkSOM code lives at the repository root in [`thesis/`](../thesis/README.md), and derivative clustering lives in [`derivative_clustering/`](../derivative_clustering/README.md). See the [repository README](../README.md) for the overall project boundary.
