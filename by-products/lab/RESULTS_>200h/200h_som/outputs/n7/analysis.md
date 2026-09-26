# n=7 SOM analysis (1×7)

**Topology:** `1×7` (7 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:37:09 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.9460 |
| Topographic error | 0.4161 |
| Empty nodes | 0 |
| Largest cluster | 749 (41.34%) |
| Smallest cluster | 20 (1.1%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 426 (23.5%) | 0.9938 | 0.8756 | -0.1182 | -0.00058 | `initial_drop_moderate_degradation_decelerating_loss` |
| 1 | (0,1) | 749 (41.3%) | 0.9867 | 0.9598 | -0.0269 | -0.00021 | `no_initial_change_stable_decelerating_loss` |
| 2 | (0,2) | 267 (14.7%) | 0.9941 | 0.7694 | -0.2247 | -0.00108 | `initial_drop_moderate_degradation_steady_loss` |
| 3 | (0,3) | 121 (6.7%) | 0.9980 | 0.4983 | -0.4997 | -0.00213 | `initial_drop_rapid_degradation_steady_loss` |
| 4 | (0,4) | 20 (1.1%) | 0.9998 | 0.1470 | -0.8528 | -0.00186 | `initial_drop_rapid_degradation_steady_loss` |
| 5 | (0,5) | 64 (3.5%) | 0.9993 | 0.3353 | -0.6640 | -0.00209 | `initial_drop_rapid_degradation_steady_loss` |
| 6 | (0,6) | 165 (9.1%) | 0.9975 | 0.6775 | -0.3199 | -0.00125 | `initial_drop_moderate_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).