# PvkSOM 200h Analysis Report

**Generated:** 2026-07-29T16:24:03.925396
**Run start:** 2026-07-29T16:23:31.676182
**Output directory:** /Users/shunhao/Desktop/ML/lab/results_all/A_result/outputs/200h_som

---

## 1. Dataset Overview

| Metric | Value |
|--------|------:|<br>
| Total curves loaded | 2,151 |
| Included (processed) | 1,453 |
| Excluded (QC failed) | 698 |
| Time window | 0–200 h |
| Grid interval | 10 min |
| Grid points per curve | 1201 |
| savgol window | 71 |
| savgol polyorder | 2 |
| Normalisation | Per-curve MaxAbsScaler |

## 2. Inclusion and Exclusion Criteria

### 2.1 Inclusion

A curve is included if ALL of the following hold:
- Valid curve ID, time, and PCE values
- At least 10 unique time points in [0, 200h]
- Max PCE > 0 in [0, 200h]
- No NaN/Inf after Akima interpolation

### 2.2 Exclusion

Top exclusion reasons:

- *Too few unique time points (8) in [0,200]h window (data spans 99% of window)* — 11 curves (1.6%)
- *Too few unique time points (9) in [0,200]h window (data spans 94% of window)* — 11 curves (1.6%)
- *Too few unique time points (9) in [0,200]h window (data spans 92% of window)* — 11 curves (1.6%)
- *Too few unique time points (8) in [0,200]h window (data spans 88% of window)* — 9 curves (1.3%)
- *Too few unique time points (8) in [0,200]h window (data spans 98% of window)* — 9 curves (1.3%)

## 3. SOM Configuration

| Parameter | Value |
|-----------|------:|<br>
| SOM shape | 2×2 |
| sigma | 0.5 |
| learning_rate | 0.1 |
| iterations | 50,000 |
| random_seed | 42 |
| weight init | random_weights_init |
| activation_distance | euclidean |
| neighborhood_function | gaussian |

**Quantisation Error:** 1.4558
**Topographic Error:** 0.0000

## 4. Cluster Results (2×2 SOM)

### Node (0,0) — no_initial_change_stable_steady_loss

| Metric | Value |
|--------|------:|<br>
| Curves | 860 (59.2%) |
| PCE@0h | 0.9888 |
| PCE@10h | 0.9879 |
| PCE@50h | 0.9782 |
| PCE@100h | 0.9614 |
| PCE@150h | 0.9427 |
| PCE@200h | 0.9227 |
| Mean BMU distance | 1.0940 |
| AUC | 0.9595 |
| Naming basis | PCE@200h=0.923, slope_0_10h=-0.00009/h, slope_100_200h=-0.00039/h, Δ200h=-0.066 |

### Node (0,1) — no_initial_change_moderate_degradation_steady_loss

| Metric | Value |
|--------|------:|<br>
| Curves | 157 (10.8%) |
| PCE@0h | 0.9941 |
| PCE@10h | 0.9586 |
| PCE@50h | 0.8331 |
| PCE@100h | 0.7089 |
| PCE@150h | 0.6013 |
| PCE@200h | 0.5055 |
| Mean BMU distance | 2.4747 |
| AUC | 0.7203 |
| Naming basis | PCE@200h=0.506, slope_0_10h=-0.00355/h, slope_100_200h=-0.00203/h, Δ200h=-0.489 |

### Node (1,0) — no_initial_change_moderate_degradation_steady_loss

| Metric | Value |
|--------|------:|<br>
| Curves | 371 (25.5%) |
| PCE@0h | 0.9934 |
| PCE@10h | 0.9792 |
| PCE@50h | 0.9193 |
| PCE@100h | 0.8566 |
| PCE@150h | 0.7980 |
| PCE@200h | 0.7455 |
| Mean BMU distance | 1.4495 |
| AUC | 0.8601 |
| Naming basis | PCE@200h=0.745, slope_0_10h=-0.00141/h, slope_100_200h=-0.00111/h, Δ200h=-0.248 |

### Node (1,1) — initial_drop_rapid_degradation_decelerating_loss

| Metric | Value |
|--------|------:|<br>
| Curves | 65 (4.5%) |
| PCE@0h | 0.9892 |
| PCE@10h | 0.9061 |
| PCE@50h | 0.6499 |
| PCE@100h | 0.4599 |
| PCE@150h | 0.3311 |
| PCE@200h | 0.2592 |
| Mean BMU distance | 3.8174 |
| AUC | 0.5077 |
| Naming basis | PCE@200h=0.259, slope_0_10h=-0.00831/h, slope_100_200h=-0.00201/h, Δ200h=-0.730 |

## 5. Quantisation Error Sweep (n = 2 … 10)

Main model (n=4, 2×2) QE: **1.4558**

| n | Topology | QE |
|---:|:--------:|------:|<br>
| 2 | 1.0×2.0 | 2.2622 |
| 3 | 1.0×3.0 | 1.7069 |
| 4 | 2.0×2.0 | 1.4558 ← main |
| 5 | 1.0×5.0 | 1.2369 |
| 6 | 1.0×6.0 | 1.1451 |
| 7 | 1.0×7.0 | 1.0386 |
| 8 | 1.0×8.0 | 1.0249 |
| 9 | 1.0×9.0 | 0.9468 |
| 10 | 1.0×10.0 | 0.9032 |

> ⚠️ **Topology for n ≠ 4:** 1×n rectangular (implementation assumption — not confirmed in author code).

## 6. Parameter Sensitivity

- **sigma0.3_lr0.1** (σ=0.3, lr=0.1): QE=1.4558
- **sigma0.3_lr0.1** (σ=0.3, lr=0.1): QE=1.4558
- **sigma0.5_lr0.3** (σ=0.5, lr=0.3): QE=1.6822
- **sigma0.5_lr0.3** (σ=0.5, lr=0.3): QE=1.6822

## 7. K-means Validation (k = 2 … 10)

**SOM vs K-means(k=4):** ARI = 0.9004, NMI = 0.8677

| k | WCSS |
|---:|------:|<br>
| 2 | 11356.02 |
| 3 | 6564.53 |
| 4 | 4708.46 ← k=4 |
| 5 | 3892.24 |
| 6 | 3250.07 |
| 7 | 2943.71 |
| 8 | 2666.95 |
| 9 | 2416.67 |
| 10 | 2219.65 |

## 8. Reproducibility

| Package | Version |
|---------|--------|<br>
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| scikit-learn | not installed |
| minisom | unknown |
| matplotlib | 3.9.4 |
| python | 3.9.6 |

## 9. Research Limitations

1. **Pre-specified cluster count:** This analysis uses a 2×2 SOM (4 nodes) as a fixed setting. It does not prove that 4 is the optimal or only valid number of clusters for this dataset.
2. **Topology assumption:** For n ≠ 4 in the QE sweep, the 1×n topology is an implementation assumption not confirmed from the author code.
3. **K-means vs author DTW:** The auxiliary k-means uses Euclidean distance rather than the author's DTW metric.
4. **Forward-fill for short curves:** Curves shorter than 200 h have their tails forward-filled with the last observed value. This does not extrapolate physically realistic behaviour.
5. **Descriptive names are post-hoc:** Cluster shape names (stable, slow_degradation, etc.) are assigned after visual inspection of the mean curves. They are not predetermined physical categories.

## 10. Suggested Next Steps

1. Investigate whether Node (0,0) and (0,1) cluster curves truly represent distinct physical degradation mechanisms or are artefacts of the preprocessing pipeline.
2. Cross-check with the author's original Zenodo dataset to validate preprocessing consistency.
3. Explore n=5 and n=6 clusters to determine whether additional structure exists beyond the 4-node model.
4. Incorporate device metadata (composition, architecture, illumination conditions) as secondary variables.

---
*End of analysis report.*