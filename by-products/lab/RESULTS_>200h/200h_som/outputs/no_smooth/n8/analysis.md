# NO-SMOOTH variant — # n=8 SOM analysis (2×4)

**Topology:** `2×4` (8 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:49:54 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.8782 |
| Topographic error | 0.0000 |
| Empty nodes | 0 |
| Largest cluster | 583 (32.17%) |
| Smallest cluster | 20 (1.1%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 392 (21.6%) | 0.9942 | 0.9092 | -0.0850 | -0.00044 | `no_initial_change_stable_decelerating_loss` |
| 1 | (0,1) | 242 (13.4%) | 0.9957 | 0.7375 | -0.2582 | -0.00111 | `initial_drop_moderate_degradation_steady_loss` |
| 2 | (0,2) | 131 (7.2%) | 0.9995 | 0.6240 | -0.3756 | -0.00145 | `initial_drop_moderate_degradation_steady_loss` |
| 3 | (0,3) | 55 (3.0%) | 0.9994 | 0.3417 | -0.6576 | -0.00185 | `initial_drop_rapid_degradation_steady_loss` |
| 4 | (1,0) | 583 (32.2%) | 0.9846 | 0.9692 | -0.0154 | -0.00016 | `no_initial_change_stable_decelerating_loss` |
| 5 | (1,1) | 304 (16.8%) | 0.9928 | 0.8338 | -0.1590 | -0.00079 | `initial_drop_moderate_degradation_decelerating_loss` |
| 6 | (1,2) | 85 (4.7%) | 0.9967 | 0.4375 | -0.5592 | -0.00256 | `initial_drop_rapid_degradation_steady_loss` |
| 7 | (1,3) | 20 (1.1%) | 1.0000 | 0.1470 | -0.8530 | -0.00186 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n8/``.
