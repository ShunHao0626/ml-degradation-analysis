# NO-SMOOTH variant — # n=9 SOM analysis (3×3)

**Topology:** `3×3` (9 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:49:58 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.8571 |
| Topographic error | 0.1087 |
| Empty nodes | 0 |
| Largest cluster | 569 (31.4%) |
| Smallest cluster | 10 (0.55%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 291 (16.1%) | 0.9947 | 0.8375 | -0.1572 | -0.00079 | `initial_drop_moderate_degradation_decelerating_loss` |
| 1 | (0,1) | 140 (7.7%) | 0.9995 | 0.6326 | -0.3669 | -0.00159 | `initial_drop_moderate_degradation_steady_loss` |
| 2 | (0,2) | 63 (3.5%) | 0.9991 | 0.3101 | -0.6890 | -0.00239 | `initial_drop_rapid_degradation_steady_loss` |
| 3 | (1,0) | 385 (21.2%) | 0.9950 | 0.9118 | -0.0832 | -0.00043 | `no_initial_change_stable_decelerating_loss` |
| 4 | (1,1) | 235 (13.0%) | 0.9954 | 0.7509 | -0.2445 | -0.00105 | `initial_drop_moderate_degradation_steady_loss` |
| 5 | (1,2) | 18 (1.0%) | 1.0000 | 0.1501 | -0.8499 | -0.00165 | `initial_drop_rapid_degradation_steady_loss` |
| 6 | (2,0) | 569 (31.4%) | 0.9869 | 0.9697 | -0.0172 | -0.00016 | `no_initial_change_stable_decelerating_loss` |
| 7 | (2,1) | 10 (0.6%) | 0.7500 | 0.9765 | 0.2264 | 0.00060 | `initial_gain_moderate_degradation_decelerating_loss` |
| 8 | (2,2) | 101 (5.6%) | 0.9974 | 0.5002 | -0.4973 | -0.00189 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n9/``.
