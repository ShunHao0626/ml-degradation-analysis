# NO-SMOOTH variant — # n=6 SOM analysis (2×3)

**Topology:** `2×3` (6 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:49:48 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 1.0218 |
| Topographic error | 0.0000 |
| Empty nodes | 0 |
| Largest cluster | 832 (45.92%) |
| Smallest cluster | 20 (1.1%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 448 (24.7%) | 0.9928 | 0.8509 | -0.1418 | -0.00071 | `initial_drop_moderate_degradation_decelerating_loss` |
| 1 | (0,1) | 142 (7.8%) | 0.9983 | 0.5326 | -0.4657 | -0.00196 | `initial_drop_rapid_degradation_steady_loss` |
| 2 | (0,2) | 74 (4.1%) | 0.9989 | 0.3437 | -0.6552 | -0.00214 | `initial_drop_rapid_degradation_steady_loss` |
| 3 | (1,0) | 832 (45.9%) | 0.9877 | 0.9549 | -0.0328 | -0.00023 | `no_initial_change_stable_decelerating_loss` |
| 4 | (1,1) | 296 (16.3%) | 0.9962 | 0.7243 | -0.2720 | -0.00115 | `initial_drop_moderate_degradation_steady_loss` |
| 5 | (1,2) | 20 (1.1%) | 1.0000 | 0.1470 | -0.8530 | -0.00186 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n6/``.
