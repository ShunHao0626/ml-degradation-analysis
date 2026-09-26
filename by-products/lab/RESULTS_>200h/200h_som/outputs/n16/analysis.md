# n=16 SOM analysis (4×4)

**Topology:** `4×4` (16 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:40:41 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 0.6927 |
| Topographic error | 0.3449 |
| Empty nodes | 0 |
| Largest cluster | 364 (20.09%) |
| Smallest cluster | 4 (0.22%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 54 (3.0%) | 0.9984 | 0.4051 | -0.5933 | -0.00290 | `initial_drop_rapid_degradation_steady_loss` |
| 1 | (0,1) | 28 (1.5%) | 1.0004 | 0.4999 | -0.5005 | -0.00050 | `initial_drop_rapid_degradation_decelerating_loss` |
| 2 | (0,2) | 4 (0.2%) | 0.8136 | 0.9665 | 0.1529 | 0.00075 | `initial_drop_moderate_degradation_decelerating_loss` |
| 3 | (0,3) | 53 (2.9%) | 0.9871 | 0.7638 | -0.2234 | -0.00175 | `no_initial_change_moderate_degradation_steady_loss` |
| 4 | (1,0) | 41 (2.3%) | 0.9986 | 0.2633 | -0.7354 | -0.00281 | `initial_drop_rapid_degradation_steady_loss` |
| 5 | (1,1) | 18 (1.0%) | 0.9998 | 0.1503 | -0.8496 | -0.00165 | `initial_drop_rapid_degradation_steady_loss` |
| 6 | (1,2) | 103 (5.7%) | 0.9969 | 0.7213 | -0.2757 | -0.00079 | `initial_drop_moderate_degradation_decelerating_loss` |
| 7 | (1,3) | 245 (13.5%) | 0.9953 | 0.9063 | -0.0891 | -0.00046 | `no_initial_change_stable_decelerating_loss` |
| 8 | (2,0) | 71 (3.9%) | 0.9977 | 0.5962 | -0.4016 | -0.00125 | `initial_drop_rapid_degradation_steady_loss` |
| 9 | (2,1) | 53 (2.9%) | 0.9988 | 0.5638 | -0.4350 | -0.00255 | `initial_drop_rapid_degradation_steady_loss` |
| 10 | (2,2) | 190 (10.5%) | 0.9959 | 0.8654 | -0.1305 | -0.00058 | `initial_drop_moderate_degradation_decelerating_loss` |
| 11 | (2,3) | 364 (20.1%) | 0.9902 | 0.9802 | -0.0100 | -0.00010 | `no_initial_change_stable_decelerating_loss` |
| 12 | (3,0) | 130 (7.2%) | 0.9951 | 0.7347 | -0.2604 | -0.00128 | `initial_drop_moderate_degradation_steady_loss` |
| 13 | (3,1) | 138 (7.6%) | 0.9977 | 0.8148 | -0.1829 | -0.00071 | `initial_drop_moderate_degradation_decelerating_loss` |
| 14 | (3,2) | 298 (16.4%) | 0.9928 | 0.9432 | -0.0496 | -0.00031 | `no_initial_change_stable_decelerating_loss` |
| 15 | (3,3) | 22 (1.2%) | 0.8103 | 0.9688 | 0.1585 | 0.00002 | `initial_gain_moderate_degradation_decelerating_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).