# NO-SMOOTH variant — # n=10 SOM analysis (2×5)

**Topology:** `2×5` (10 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:50:03 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.8146 |
| Topographic error | 0.0226 |
| Empty nodes | 0 |
| Largest cluster | 572 (31.57%) |
| Smallest cluster | 16 (0.88%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 379 (20.9%) | 0.9949 | 0.9119 | -0.0830 | -0.00043 | `no_initial_change_stable_decelerating_loss` |
| 1 | (0,1) | 214 (11.8%) | 0.9962 | 0.7577 | -0.2386 | -0.00113 | `initial_drop_moderate_degradation_steady_loss` |
| 2 | (0,2) | 77 (4.2%) | 0.9967 | 0.5285 | -0.4682 | -0.00230 | `initial_drop_rapid_degradation_steady_loss` |
| 3 | (0,3) | 16 (0.9%) | 1.0000 | 0.1421 | -0.8579 | -0.00159 | `initial_drop_rapid_degradation_steady_loss` |
| 4 | (0,4) | 44 (2.4%) | 1.0000 | 0.3079 | -0.6921 | -0.00204 | `initial_drop_rapid_degradation_steady_loss` |
| 5 | (1,0) | 572 (31.6%) | 0.9844 | 0.9698 | -0.0146 | -0.00016 | `no_initial_change_stable_decelerating_loss` |
| 6 | (1,1) | 279 (15.4%) | 0.9910 | 0.8472 | -0.1438 | -0.00070 | `initial_drop_moderate_degradation_decelerating_loss` |
| 7 | (1,2) | 142 (7.8%) | 0.9973 | 0.6908 | -0.3065 | -0.00111 | `initial_drop_moderate_degradation_steady_loss` |
| 8 | (1,3) | 43 (2.4%) | 1.0000 | 0.5796 | -0.4204 | -0.00070 | `initial_drop_rapid_degradation_decelerating_loss` |
| 9 | (1,4) | 46 (2.5%) | 0.9982 | 0.3406 | -0.6576 | -0.00328 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n10/``.
