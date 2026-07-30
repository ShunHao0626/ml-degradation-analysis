# n=5 SOM analysis (1×5)

**Topology:** `1×5` (5 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:36:50 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 1.0965 |
| Topographic error | 0.0988 |
| Empty nodes | 0 |
| Largest cluster | 888 (49.01%) |
| Smallest cluster | 66 (3.64%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 66 (3.6%) | 0.9995 | 0.2635 | -0.7360 | -0.00204 | `initial_drop_rapid_degradation_steady_loss` |
| 1 | (0,1) | 142 (7.8%) | 0.9981 | 0.4892 | -0.5090 | -0.00207 | `initial_drop_rapid_degradation_steady_loss` |
| 2 | (0,2) | 888 (49.0%) | 0.9882 | 0.9512 | -0.0370 | -0.00025 | `no_initial_change_stable_decelerating_loss` |
| 3 | (0,3) | 445 (24.6%) | 0.9929 | 0.8357 | -0.1572 | -0.00078 | `initial_drop_moderate_degradation_decelerating_loss` |
| 4 | (0,4) | 271 (15.0%) | 0.9967 | 0.7027 | -0.2940 | -0.00122 | `initial_drop_moderate_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).