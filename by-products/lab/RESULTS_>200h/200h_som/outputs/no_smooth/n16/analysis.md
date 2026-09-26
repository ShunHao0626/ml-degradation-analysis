# NO-SMOOTH variant — # n=16 SOM analysis (4×4)

**Topology:** `4×4` (16 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:50:08 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.6928 |
| Topographic error | 0.3449 |
| Empty nodes | 0 |
| Largest cluster | 364 (20.09%) |
| Smallest cluster | 4 (0.22%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 54 (3.0%) | 0.9984 | 0.4058 | -0.5925 | -0.00289 | `initial_drop_rapid_degradation_steady_loss` |
| 1 | (0,1) | 28 (1.5%) | 1.0000 | 0.5000 | -0.5000 | -0.00050 | `initial_drop_rapid_degradation_decelerating_loss` |
| 2 | (0,2) | 4 (0.2%) | 0.8133 | 0.9666 | 0.1533 | 0.00075 | `initial_drop_moderate_degradation_decelerating_loss` |
| 3 | (0,3) | 53 (2.9%) | 0.9871 | 0.7643 | -0.2227 | -0.00175 | `no_initial_change_moderate_degradation_steady_loss` |
| 4 | (1,0) | 41 (2.3%) | 0.9987 | 0.2638 | -0.7348 | -0.00280 | `initial_drop_rapid_degradation_steady_loss` |
| 5 | (1,1) | 18 (1.0%) | 1.0000 | 0.1501 | -0.8499 | -0.00165 | `initial_drop_rapid_degradation_steady_loss` |
| 6 | (1,2) | 103 (5.7%) | 0.9971 | 0.7214 | -0.2757 | -0.00079 | `initial_drop_moderate_degradation_decelerating_loss` |
| 7 | (1,3) | 245 (13.5%) | 0.9953 | 0.9063 | -0.0890 | -0.00046 | `no_initial_change_stable_decelerating_loss` |
| 8 | (2,0) | 71 (3.9%) | 0.9976 | 0.5964 | -0.4012 | -0.00125 | `initial_drop_rapid_degradation_steady_loss` |
| 9 | (2,1) | 53 (2.9%) | 0.9988 | 0.5643 | -0.4345 | -0.00255 | `initial_drop_rapid_degradation_steady_loss` |
| 10 | (2,2) | 190 (10.5%) | 0.9958 | 0.8655 | -0.1304 | -0.00058 | `initial_drop_moderate_degradation_decelerating_loss` |
| 11 | (2,3) | 364 (20.1%) | 0.9902 | 0.9802 | -0.0100 | -0.00010 | `no_initial_change_stable_decelerating_loss` |
| 12 | (3,0) | 130 (7.2%) | 0.9951 | 0.7351 | -0.2600 | -0.00127 | `initial_drop_moderate_degradation_steady_loss` |
| 13 | (3,1) | 138 (7.6%) | 0.9977 | 0.8149 | -0.1828 | -0.00071 | `initial_drop_moderate_degradation_decelerating_loss` |
| 14 | (3,2) | 298 (16.4%) | 0.9928 | 0.9432 | -0.0496 | -0.00031 | `no_initial_change_stable_decelerating_loss` |
| 15 | (3,3) | 22 (1.2%) | 0.8105 | 0.9688 | 0.1583 | 0.00002 | `initial_gain_moderate_degradation_decelerating_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n16/``.
