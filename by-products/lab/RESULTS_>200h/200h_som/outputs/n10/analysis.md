# n=10 SOM analysis (2×5)

**Topology:** `2×5` (10 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:37:21 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.8360 |
| Topographic error | 0.0006 |
| Empty nodes | 0 |
| Largest cluster | 569 (31.4%) |
| Smallest cluster | 10 (0.55%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 387 (21.4%) | 0.9950 | 0.9116 | -0.0834 | -0.00043 | `no_initial_change_stable_decelerating_loss` |
| 1 | (0,1) | 296 (16.3%) | 0.9948 | 0.8357 | -0.1591 | -0.00080 | `initial_drop_moderate_degradation_decelerating_loss` |
| 2 | (0,2) | 129 (7.1%) | 0.9995 | 0.6290 | -0.3705 | -0.00153 | `initial_drop_moderate_degradation_steady_loss` |
| 3 | (0,3) | 63 (3.5%) | 0.9956 | 0.4120 | -0.5836 | -0.00303 | `initial_drop_rapid_degradation_steady_loss` |
| 4 | (0,4) | 18 (1.0%) | 0.9998 | 0.1503 | -0.8496 | -0.00165 | `initial_drop_rapid_degradation_steady_loss` |
| 5 | (1,0) | 569 (31.4%) | 0.9869 | 0.9697 | -0.0172 | -0.00016 | `no_initial_change_stable_decelerating_loss` |
| 6 | (1,1) | 10 (0.6%) | 0.7501 | 0.9765 | 0.2265 | 0.00060 | `initial_gain_moderate_degradation_decelerating_loss` |
| 7 | (1,2) | 245 (13.5%) | 0.9956 | 0.7448 | -0.2508 | -0.00108 | `initial_drop_moderate_degradation_steady_loss` |
| 8 | (1,3) | 41 (2.3%) | 1.0002 | 0.5725 | -0.4277 | -0.00067 | `initial_drop_rapid_degradation_decelerating_loss` |
| 9 | (1,4) | 54 (3.0%) | 0.9995 | 0.3036 | -0.6959 | -0.00231 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).