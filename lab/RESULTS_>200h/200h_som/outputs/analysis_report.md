# PvkSOM 200h Analysis Report

**Generated:** 2026-07-30T16:57:45 UTC
**Run start:** 2026-07-30T16:57:05
**Output directory:** `/Users/shunhao/Desktop/ML/lab/RESULTS_>200h/200h_som/outputs`

---

## 1. Dataset Overview

| Metric | Value |
|--------|------:|
| Total curves discovered | 1873 |
| Included (processed) | 1812 |
| Excluded (QC failed) | 61 |
| Time window | 0–200 h |
| Grid interval | 10 min |
| Grid points per curve | 1201 |
| Savitzky–Golay window | 71 |
| Savitzky–Golay polyorder | 2 |
| Per-curve normalization | divide by own max in 0–200 h |

## 2. Inclusion and Exclusion Criteria

### 2.1 Inclusion

A curve is included if ALL of the following hold:
- Valid sample_id, time, and PCE values
- ≥ 4 unique time points in [0, 200] h
- Curve spans ≥ 200 h after relative-time shift
- 0–200 h max PCE > 0 (finite, positive)
- Akima interpolation succeeds at exactly 1201 grid points
- All 1201 values are finite after normalisation + Savitzky–Golay

### 2.2 Exclusion

Top exclusion reasons:
- too few unique time points (3) in [0,200]h: 33 curves
- too few unique time points (2) in [0,200]h: 10 curves
- curve spans only 124.901 h (< 200 h): 1 curves
- curve spans only 192.682 h (< 200 h): 1 curves
- curve spans only 192.213 h (< 200 h): 1 curves
- too few unique time points (1) in [0,200]h: 1 curves
- curve spans only 120.469 h (< 200 h): 1 curves
- curve spans only 58.050 h (< 200 h): 1 curves
- curve spans only 143.776 h (< 200 h): 1 curves
- curve spans only 151.141 h (< 200 h): 1 curves

## 3. Preprocessing

Strict order:
1. `convert_time_to_hours`
2. `build_relative_ageing_time`
3. `restrict_to_200h`
4. `sort_by_time`
5. `resolve_duplicate_timestamps`
6. `akima_interpolate_to_grid`
7. `normalize_by_200h_max`
8. `apply_savgol_filter`

### 3.1 Resampling & Interpolation

- Time grid: `np.linspace(0, 200, 1201)` (10 min spacing)
- Interpolation: `scipy.interpolate.Akima1DInterpolator`
- No extrapolation beyond observed range

### 3.2 Normalisation & Smoothing

- Each curve is divided by its OWN max PCE in [0,200] h
- Then Savitzky-Golay smoothing (window=71, polyorder=2, mode='interp')
- Result: every smoothed curve has max ≈ 1.0 (small overshoot allowed)

## 4. SOM Configuration

| Parameter | Value |
|-----------|------:|
| shape | [2, 2] |
| sigma | 0.5 |
| learning_rate | 0.1 |
| activation_distance | euclidean |
| neighborhood_function | gaussian |
| topology | rectangular |
| initialization | random_weights_init |
| training_method | train |
| iterations | 50000 |
| random_seed | 42 |
| input_len | 1201 |

**Main model Quantisation Error:** 1.284634642495829
**Main model Topographic Error:** 0.0

## 5. Cluster Results (2×2 SOM)

### Node (0, 0)

| Metric | Value |
|--------|------:|
| Curves | 443 (24.4%) |
| pce_norm_0h_mean | 0.9943 |
| pce_norm_10h_mean | 0.9804 |
| pce_norm_50h_mean | 0.9265 |
| pce_norm_100h_mean | 0.8678 |
| pce_norm_150h_mean | 0.8128 |
| pce_norm_200h_mean | 0.7653 |
| mean_bmu_distance | 1.3723 |
| median_bmu_distance | 1.2960 |
| auc_200h | 174.1689 |

- **Suggested shape name (post-hoc):** `initial_drop_moderate_degradation_steady_loss`
- **Naming basis:** PCE@200h=0.765, slope_0_10h=-0.00138/h, slope_100_200h=-0.00102/h, Δ200h=-0.229, n=443

### Node (0, 1)

| Metric | Value |
|--------|------:|
| Curves | 188 (10.4%) |
| pce_norm_0h_mean | 0.9983 |
| pce_norm_10h_mean | 0.9627 |
| pce_norm_50h_mean | 0.8449 |
| pce_norm_100h_mean | 0.7281 |
| pce_norm_150h_mean | 0.6288 |
| pce_norm_200h_mean | 0.5429 |
| mean_bmu_distance | 2.3116 |
| median_bmu_distance | 2.2300 |
| auc_200h | 148.0102 |

- **Suggested shape name (post-hoc):** `initial_drop_rapid_degradation_steady_loss`
- **Naming basis:** PCE@200h=0.543, slope_0_10h=-0.00356/h, slope_100_200h=-0.00185/h, Δ200h=-0.455, n=188

### Node (1, 0)

| Metric | Value |
|--------|------:|
| Curves | 1103 (60.9%) |
| pce_norm_0h_mean | 0.9891 |
| pce_norm_10h_mean | 0.9886 |
| pce_norm_50h_mean | 0.9812 |
| pce_norm_100h_mean | 0.9671 |
| pce_norm_150h_mean | 0.9514 |
| pce_norm_200h_mean | 0.9353 |
| mean_bmu_distance | 0.9072 |
| median_bmu_distance | 0.8325 |
| auc_200h | 193.0910 |

- **Suggested shape name (post-hoc):** `no_initial_change_stable_decelerating_loss`
- **Naming basis:** PCE@200h=0.935, slope_0_10h=-0.00005/h, slope_100_200h=-0.00032/h, Δ200h=-0.054, n=1103

### Node (1, 1)

| Metric | Value |
|--------|------:|
| Curves | 78 (4.3%) |
| pce_norm_0h_mean | 0.9993 |
| pce_norm_10h_mean | 0.9142 |
| pce_norm_50h_mean | 0.6780 |
| pce_norm_100h_mean | 0.4906 |
| pce_norm_150h_mean | 0.3541 |
| pce_norm_200h_mean | 0.2731 |
| mean_bmu_distance | 3.6485 |
| median_bmu_distance | 3.3968 |
| auc_200h | 106.2238 |

- **Suggested shape name (post-hoc):** `initial_drop_rapid_degradation_steady_loss`
- **Naming basis:** PCE@200h=0.273, slope_0_10h=-0.00851/h, slope_100_200h=-0.00217/h, Δ200h=-0.726, n=78

## 6. Quantisation Error Sweep

Main model (n=4, 2×2) QE: **1.284634642495829**

| n | topology | QE | source |
|---:|:--------:|------:|:-------|
| 2 | 1x2 | 2.0016 | 1xn fallback (implementation assumption) |
| 3 | 1x3 | 1.5029 | 1xn fallback (implementation assumption) |
| 4 | 2x2 | 1.2846 | main (2x2) |
| 4 | 1x4 | 1.2846 | 1xn reference (for n=4 topology comparison) |
| 5 | 1x5 | 1.0965 | 1xn fallback (implementation assumption) |
| 6 | 1x6 | 1.0217 | 1xn fallback (implementation assumption) |
| 7 | 1x7 | 0.9460 | 1xn fallback (implementation assumption) |
| 8 | 1x8 | 0.8726 | 1xn fallback (implementation assumption) |
| 9 | 1x9 | 0.8443 | 1xn fallback (implementation assumption) |
| 10 | 1x10 | 0.8286 | 1xn fallback (implementation assumption) |

> Topology for n ≠ 4 is **1×n** in the QE sweep. This is an **implementation assumption** because the author code does not explicitly document a topology rule for arbitrary node counts.

## 7. Sensitivity Analysis

| Model | sigma | learning_rate | QE |
|-------|------:|--------------:|------:|
| sigma0.3_lr0.1 | 0.3 | 0.1 | 1.2846 |
| sigma0.5_lr0.3 | 0.5 | 0.3 | 1.2941 |

Cross-model ARI / NMI are reported in `sensitivity_summary.csv`. Clusters are aligned to the main model by Hungarian matching on the codebook vectors.

## 8. K-means Validation (k = 2 … 10)

| k | WCSS (inertia) |
|---:|---------------:|
| 2 | 12711.08 |
| 3 | 7101.77 |
| 4 | 5111.36 |
| 5 | 4171.30 |
| 6 | 3526.33 |
| 7 | 3177.18 |
| 8 | 2857.11 |
| 9 | 2626.90 |
| 10 | 2420.36 |

**SOM(main) vs KMeans(k=4):** ARI = 0.9472, NMI = 0.9249

## 9. Reproducibility

| Package | Version |
|---------|---------|
| python | 3.9.6 |
| platform | darwin |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| sklearn | 1.6.1 |
| minisom | unknown |
| matplotlib | 3.9.4 |
| run_start | 2026-07-30T16:57:05 |

## 10. Research Limitations

- Pre-specified 2×2 (4-node) topology: this analysis uses a 4-node SOM as a fixed setting for the main model; it does NOT prove 4 is optimal.
- For n ≠ 4, the QE-sweep topology is **1×n fallback** — an **implementation assumption** because the author code does not document a topology rule for arbitrary n.
- n=4 2×2 topology yields QE = 1.2846; the QE sweep shows strictly decreasing QE with more nodes, which is expected but does not imply physical meaning.
- Cluster descriptive names are post-hoc — assigned by inspecting the mean curves. They are not predetermined physical categories.
- Curves not reaching 200 h are excluded. No extrapolation to 200 h is performed.
- K-means validation uses Euclidean distance instead of the author's DTW metric. This is an implementation assumption.
- Empty nodes count: 0. Sensitivity models may show different empty-node patterns.

## 11. Conclusions (Exploratory)

- This is an exploratory 4-node SOM analysis. The 2×2 topology is a user-specified setting, not a statistically determined optimum.
- Cluster IDs do NOT have predetermined physical names. The numbers (0,1,2,3) reflect SOM node coordinates only.
- Different cluster numbers between models do not correspond one-to-one across runs; matching is performed via Hungarian assignment on codebook vectors.
- Findings should be cross-checked against quantile-based PCE grouping (Figure 3 in the paper) and sensitivity analyses before drawing degradation-mechanism conclusions.

## 12. Suggested Next Steps

- Compare QE between 2×2 (main) and 1×4 topologies to assess sensitivity of physical interpretation to topology choice.
- Incorporate device metadata (composition, architecture) as secondary variables.
- Run longer QE sweeps with different random seeds to assess stability.
- Compare with quantile regression of relative PCE change vs. max PCE bin.
- Examine cluster overlap by computing pairwise distance between cluster centroid curves.

---

*End of analysis report.*