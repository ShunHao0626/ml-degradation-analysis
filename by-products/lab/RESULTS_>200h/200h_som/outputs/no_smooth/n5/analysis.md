# NO-SMOOTH variant — # n=5 SOM analysis (1×5)

**Topology:** `1×5` (5 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:49:45 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 1.0966 |
| Topographic error | 0.0988 |
| Empty nodes | 0 |
| Largest cluster | 888 (49.01%) |
| Smallest cluster | 66 (3.64%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 66 (3.6%) | 0.9995 | 0.2637 | -0.7357 | -0.00203 | `initial_drop_rapid_degradation_steady_loss` |
| 1 | (0,1) | 142 (7.8%) | 0.9980 | 0.4897 | -0.5084 | -0.00206 | `initial_drop_rapid_degradation_steady_loss` |
| 2 | (0,2) | 888 (49.0%) | 0.9882 | 0.9512 | -0.0369 | -0.00025 | `no_initial_change_stable_decelerating_loss` |
| 3 | (0,3) | 445 (24.6%) | 0.9928 | 0.8358 | -0.1570 | -0.00077 | `initial_drop_moderate_degradation_decelerating_loss` |
| 4 | (0,4) | 271 (15.0%) | 0.9967 | 0.7029 | -0.2938 | -0.00122 | `initial_drop_moderate_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n5/``.
