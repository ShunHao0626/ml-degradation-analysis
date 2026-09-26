# NO-SMOOTH variant — # n=4 SOM analysis (2×2)

**Topology:** `2×2` (4 nodes)
**σ:** 0.5  •  **learning_rate:** 0.1  •  **iterations:** 50000  •  **seed:** 42
**Generated:** 2026-07-30T17:49:42 UTC

## Headline metrics

| Metric | Value |
|--------|------:|
| Curves (n) | 1812 |
| Quantisation error | 1.2847 |
| Topographic error | 0.0000 |
| Empty nodes | 0 |
| Largest cluster | 1103 (60.87%) |
| Smallest cluster | 78 (4.3%) |

## Per-cluster summary

| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |
|-----------:|:----:|------:|------:|------:|------:|------:|------|
| 0 | (0,0) | 443 (24.4%) | 0.9943 | 0.7653 | -0.2289 | -0.00102 | `initial_drop_moderate_degradation_steady_loss` |
| 1 | (0,1) | 188 (10.4%) | 0.9983 | 0.5429 | -0.4554 | -0.00185 | `initial_drop_rapid_degradation_steady_loss` |
| 2 | (1,0) | 1103 (60.9%) | 0.9891 | 0.9353 | -0.0538 | -0.00032 | `no_initial_change_stable_decelerating_loss` |
| 3 | (1,1) | 78 (4.3%) | 0.9993 | 0.2731 | -0.7262 | -0.00217 | `initial_drop_rapid_degradation_steady_loss` |

## Notes

- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.
- "suggested name" is a post-hoc descriptor derived from metrics only.
- Empty-node check: 0 empty nodes (no auto-retry of seed; reported to user).

## No-smooth note

The feature matrix used here is the **Akima-interpolated, per-curve-normalised** curve **without** Savitzky–Golay smoothing. Every other step (QC, normalisation, SOM training, cluster labelling) is identical to the smoothed run in ``outputs/n4/``.
