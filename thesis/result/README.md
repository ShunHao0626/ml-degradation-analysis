# PvkSOM - Perovskite Solar Cell Degradation Analysis

**Associated Publication:** [Nature Communications 14, 4869 (2023)](https://doi.org/10.1038/s41467-023-40585-3)

**Authors:** Noor Titan Putri Hartono, Hans Köbler, Paolo Graniero, Mark Khenkin, Rutger Schlatmann, Carolin Ulbrich, Antonio Abate (Helmholtz-Zentrum-Berlin)

---

## Overview

PvkSOM uses **Self-Organizing Maps (SOM)** — an unsupervised machine learning method — to cluster perovskite solar cell degradation time-series data. The key research finding: **"Stability follows efficiency"** — higher-efficiency cells tend to degrade faster.

---

## Table of Contents

1. [Physical Background — Key Quantities](#physical-background--key-quantities)
2. [Input Files (Quick Reference)](#input-files) — contents overview
3. [Output Files — Detailed Physical Interpretation](#output-files--detailed-physical-interpretation) — what every PNG/HTML/JSON means physically
4. [SOM Results Summary & Cluster Physics](#som-results-summary) — the 4 degradation archetypes and their mechanisms
5. [Pipeline Steps](#pipeline-steps) — 11-step workflow
6. [Input Files — Detailed Physical Interpretation](#input-files--detailed-physical-interpretation) — what is actually inside each input file
7. [Preprocessing Steps — Physical Meaning](#preprocessing-steps--physical-meaning) — why each resample/normalize/smooth step exists
8. [Adjustable Parameters — Physical Rationale](#adjustable-parameters--physical-rationale) — what every parameter does and when to change it
9. [Use Case Quick Reference](#use-case-quick-reference) — which output for which purpose
10. [Adapting to Your Own Data](#adapting-to-your-own-data) — format requirements and pipeline
11. [Dependencies](#dependencies), [Usage](#usage), [Folder Structure](#folder-structure)
12. [Scientific Value of PvkSOM](#scientific-value-of-pvksom)

---

## Physical Background — Key Quantities

| Quantity | Symbol | Unit | Physical Meaning |
|----------|--------|------|------------------|
| **Power conversion efficiency** | PCE (η) | % | Fraction of incident solar energy converted to electricity — the **core performance metric** |
| **Maximum power** | Pmax | mW | Power output at the Maximum Power Point (MPP) |
| **Voltage at MPP** | Vmpp | V | Operating voltage at Pmax |
| **Current at MPP** | Impp | mA | Operating current at Pmax |
| **Open-circuit voltage** | Voc | V | Terminal voltage under zero current |
| **Short-circuit current** | Isc | mA | Output current under short-circuit |
| **Fill factor** | FF | % | Squareness of the I-V curve: FF = Pmax / (Voc × Isc) |
| **MPPT** | — | — | Maximum Power Point Tracking — real-time adjustment to stay at Pmax |
| **Normalized PCE** | PCE(t)/Pmax(0) | 0–1 | Efficiency relative to initial value — used for comparing **degradation shapes** independent of absolute scale |

---

## Input Files

Four files in `dataset/` are the pipeline input. For the physical meaning of each file (what is inside, why it is needed), see the **Input Files — Detailed Physical Interpretation** section below.

For the physical meaning of preprocessing steps (resample, truncate, Akima, MaxAbsScaler, Savitzky-Golay), see **Preprocessing Steps — Physical Meaning** below.

For the physical meaning and tuning rationale of every adjustable parameter (HOUR_LIMIT, SAGGOL_*, SOM_*), see **Adjustable Parameters — Physical Rationale** below.

| File | Type | Description |
|------|------|-------------|
| `dataset/PCE_df_grouping.csv` | CSV | 2,247 rows with columns: PCE_before, PCE_after, PCE_delta, PCE_before_ceil, etc. |
| `dataset/pkl_complete/20230303_mySeriesDrop.pkl` | Pickle | Raw time-series of MPPT efficiency measurements |
| `dataset/pkl_complete/20230303_mySeriesDropNorm.pkl` | Pickle | Normalized series (MaxAbsScaler) |
| `dataset/pkl_complete/20230303_mySeriesDrop_savgol.npy` | NumPy | Savitzky-Golay smoothed data (DIRECT INPUT TO SOM) |

---

## Output Files — Detailed Physical Interpretation

### A. SOM Training Results (Core Outputs)

| File | Physical Meaning (Perovskite Context) |
|------|---------------------------------------|
| **`som_summary.json`** | Training metadata: 2,245 time-series × 901 points → 2×2 SOM grid (4 archetypes). Reports `quantization_error` (= average shape-distance to nearest cluster center, lower = better representative curve) and `topographic_error` (= fraction of cells whose nearest SOM neighbor is NOT adjacent on the grid; 0.00 = perfect topology preservation). Also records cluster sizes per neuron. |
| **`som_umatrix.png`** | **U-Matrix (coolwarm color)**: color encodes the distance between adjacent SOM neurons. **Dark blue** = neighboring neurons represent very similar degradation shapes (same physical mechanism). **Red/light** = neurons represent distinct shapes → **cluster boundaries**. Used to identify whether the 4 archetypes are truly separate or there exist intermediate/transitional mechanisms (e.g., burn-in mixed with slow linear decay). |
| **`som_distance_map.png`** | Same U-Matrix data with **viridis colormap** (color-blind-friendly alternative). Same physical interpretation; preferred for publication figures. |
| **`som_clusters.png`** | **Most important output.** 2×2 panel grid showing all 2245 individual degradation curves (gray) + the **mean trajectory** (red) per cluster. The 4 panels reveal the **4 archetypal degradation modes**: (0,1) smooth monotonic drop (burn-in), (0,0) gradual linear decay (ion migration / interface degradation), (1,1) initial rise then fall (photo-induced defect healing), (1,0) sudden cliff-like failure (encapsulation breakdown / short circuit). |
| **`som_cluster_0.png`** … **`som_cluster_3.png`** | Single-cluster zoom-in: each curve drawn separately with the red mean. Used as **paper figures** to show within-cluster variance — reveals whether an archetype is homogeneous (tight curves) or hides sub-classes (wide spread). |
| **`som_cluster_X_interactive.html`** | Interactive Plotly version: **hover over any curve to see its individual data points** (sample number, PCE value, time). Enables inspection of specific devices during exploration without re-running the pipeline. |
| **`som_cluster_counts.png`** | Bar chart of cell count per neuron. Reveals the **population distribution of degradation modes** in the dataset. In this work: 1243 cells follow the burn-in archetype (mode (0,1)), while only 99 cells show catastrophic failure (mode (1,0)). This tells researchers which degradation mechanisms dominate their fabrication batches. |

### B. Validation & Comparison Outputs

| File | Physical Meaning (Perovskite Context) |
|------|---------------------------------------|
| **`som_vs_kmeans_comparison.png`** | Side-by-side bar charts: SOM cluster counts vs. K-Means cluster counts on the same data. If both produce **similar distributions**, the 4 archetypes are robust (not artifacts of one algorithm). Differences highlight cases where SOM's topology-preservation captures something K-Means misses. |
| **`pca_visualization.png`** | **2D scatter**: each cell projected onto PC1–PC2 (90%+ variance captured from 901-D space), colored by cluster. **Well-separated colors** → the 4 archetypes are physically distinct mechanisms. **Overlapping colors** → the mechanisms are continuous, not discrete categories. Essential for interpreting whether the clustering is meaningful. |
| **`overview_normalized.png`** | Grid of ~100 raw normalized curves. **Data quality check**: scan for anomalies such as sudden drops to zero (measurement interruption), negative spikes (sensor fault), or NaN gaps (missing data). |
| **`overview_savgol.png`** | Grid of ~100 Savitzky-Golay-smoothed curves. **Verify the smoothing preserved the physical features** of interest (burn-in knee, degradation plateau, recovery peaks). If key features are washed out → window too large; if noise still dominates → window too small. |

### C. PCE Efficiency vs. Degradation Analysis

| File | Physical Meaning (Perovskite Context) |
|------|---------------------------------------|
| **`pce_boxplot.html`** | **Interactive box plot** of PCE_delta (relative degradation %) grouped by initial efficiency bucket (5 groups: ~14%, ~17%, ~19%, ~21%, ~29%). Reveals the central finding: **higher initial efficiency correlates with larger relative degradation** — challenging the intuition that better devices are more stable. |
| **`pce_violin.html`** | **Interactive violin + box plot** — full distribution shape (density + quartiles + outliers). Shows whether degradation within each efficiency bucket is unimodal or has sub-populations (e.g., some 19%-efficient devices degrade as badly as 14% devices). |
| **`pce_violin.png`** | Static version of the violin plot — used directly as a **publication figure**. |

### D. Custom Data Outputs (`my_data/` — your 30 MPPT files)

| File | Physical Meaning (Perovskite Context) |
|------|---------------------------------------|
| **`my_data/my_som_summary.json`** | Same as `som_summary.json` but for **your 30 cells**. Quantization error = 0.54 (lower than the 2245-cell run because fewer cells = more homogeneous curves). Cluster sizes: (0,0)=1, (0,1)=5, (1,0)=9, (1,1)=15 — your 30 devices are spread across all 4 archetypes. |
| **`my_data/my_som_clusters.png`** | 2×2 grid of your 30 devices. The **red mean curves** show the average degradation trajectory of each archetype within your dataset. Compare with the 2245-cell reference archetypes to see whether your devices follow known mechanisms or exhibit novel behavior. |
| **`my_data/my_cluster_assignments.csv`** | Table mapping **filename → cluster (x, y)**. Lets you identify which physical mechanism each of your devices exhibits. For example, R403P15 → (0,0) → linear decay mode; S210P02 → (1,0) → catastrophic failure mode. |

### E. Input Metadata (`PCE_df_grouping.csv` columns)

| Column | Physical Meaning |
|--------|------------------|
| `PCE_before` | Average efficiency (%) over the **first 3 data points** at the start of the test |
| `PCE_after` | Average efficiency (%) over the **last 3 data points** at the end of the test |
| `PCE_delta` | **Relative degradation** = (PCE_before − PCE_after) / PCE_before × 100 (%) |
| `PCE_before_ceil` | `PCE_before` rounded up to nearest integer |
| `PCE_before_x` | Efficiency bucket index (1–5) — used to group cells by initial performance |
| `PCE_before_ceil_x` / `_median_x` / `_mean_x` | Alternative bucket definitions (ceil / median / mean thresholds) |

---

## SOM Results Summary

| Metric | Value | Physical Interpretation |
|--------|-------|-------------------------|
| Total series analyzed | 2,245 | Number of solar cell time-series |
| Series length | 900 | Data points per series (150 hours × 6/min) |
| Quantization error | 2.42 | Average shape distance from each curve to its cluster's mean — lower = tighter cluster fit |
| Topographic error | 0.0 | All cells' nearest neighbors on the SOM grid are adjacent — perfect topology preservation |

### Cluster Distribution & Physical Interpretation

| Cluster | Coordinates | Cell Count | Shape Feature | Inferred Physical Mechanism |
|---------|-------------|------------|---------------|----------------------------|
| 0 | (0,1) | 1,243 | Smooth monotonic drop, no inflection | **Burn-in degradation** — rapid initial efficiency loss driven by surface defect formation / halide segregation, then stabilization |
| 1 | (0,0) | 646 | Slow linear decline | **Linear degradation** — dominated by ion migration at interfaces / gradual electrode corrosion |
| 2 | (1,1) | 257 | Initial rise then fall ("hump") | **Light-soaking improvement + decay** — photo-induced ion rearrangement / defect passivation temporarily boosts efficiency, then degradation dominates |
| 3 | (1,0) | 99 | Sudden cliff-like collapse | **Catastrophic failure** — electrode delamination, encapsulation breach, or internal short circuit |

---

## Pipeline Steps

| Step | Function | Output |
|------|----------|--------|
| 1 | Load data | PCE_df, mySeriesDrop |
| 2 | Normalize | MaxAbsScaler normalized series |
| 3 | Smooth | Savitzky-Golay filtered series |
| 4 | Plot overview | Raw & smoothed data grids |
| 5 | Train SOM | MiniSom model (2×2, 50K iterations) |
| 6 | Visualize SOM | U-matrix, clusters, counts |
| 7 | K-Means comparison | Side-by-side cluster distribution |
| 8 | Time-Series K-Means | DTW-based clustering (optional) |
| 9 | PCA visualization | 2D projection colored by cluster |
| 10 | PCE analysis | Box plot & violin plot |
| 11 | Save summary | som_summary.json |

---

## Input Files — Detailed Physical Interpretation

### Raw Input Data (`dataset/`)

| File | Physical Meaning (What is Inside) |
|------|-----------------------------------|
| `dataset/PCE_df_grouping.csv` | **Per-device metadata + summary statistics table.** Originally derived from the nested JSON time-series in the source data. Each row corresponds to one **perovskite solar cell** (a single device × pixel). Contains precomputed degradation metrics: `PCE_before` (efficiency at test start), `PCE_after` (efficiency at test end), `PCE_delta` (relative degradation). Also contains efficiency-grouping labels derived from `PCE_before`. |
| `dataset/pkl_complete/20230303_mySeriesDrop.pkl` | **Full MPPT time-series before preprocessing** — pickled pandas Series. Each entry is one device's raw MPPT_EFF trace indexed by datetime. Sampling interval is ~10 min. Contains NaN values due to IV-scan interruptions and missing measurements. |
| `dataset/pkl_complete/20230303_mySeriesDropNorm.pkl` | **After step 4 (MaxAbsScaler normalization)**: each curve scaled to [0, 1] per device. Allows apples-to-apples comparison across devices with very different initial efficiencies (5% vs 22%). Cached to skip re-normalization. |
| `dataset/pkl_complete/20230303_mySeriesDrop_savgol.npy` | **After step 5 (Savitzky-Golay smoothing)**: window=71, order=2. This is the **direct input to the SOM** — NumPy array of shape (n_devices, 901). Each row is one cell's smoothed normalized efficiency trajectory over the 150-hour test window. |

### What Goes INTO the Pipeline (Step 1)

The `load_data()` function in `run_pipeline.py` reads:

1. **`PCE_df_grouping.csv`** → only used later in the PCE_analysis step (Step 10) for box/violin plots. Not used in the actual SOM training.
2. **`mySeriesDrop.pkl`** → **the actual input to the SOM pipeline**. Contains 2245 individual cell traces, each with columns `MPPT_t` (timestamp) and `MPPT_EFF` (efficiency value, 0-100% scale).

### What Comes OUT of the Pipeline (Step 5, the SOM input)

After preprocessing, the SOM sees a NumPy array of shape `(2245, 901)`:

```
data[i, t] = normalized + smoothed PCE efficiency of cell i at time t
```

- **Each row** = one perovskite cell's complete degradation trajectory
- **Each column** = one time point (10-min resolution over 150 h)
- **Values** in [0, 1] after MaxAbsScaler, then Savitzky-Golay filtered

---

## Preprocessing Steps — Physical Meaning

| Step | Method | Input → Output | Physical Purpose |
|------|--------|----------------|------------------|
| 1. **10-min resample** | `pd.resample('10min').mean()` | Irregular time stamps → uniformly spaced series at 10-min intervals | Unify sampling rate (raw MPPT may be 5, 10, or 15 min). 6 points/hour = enough resolution to capture degradation rates of 0.1–1%/h typical of perovskite cells, while averaging out measurement fluctuations. |
| 2. **Truncate to 150 h** | `.iloc[1:901]` | Variable-length series → fixed 901-point series | Standardize so all curves can be compared element-wise in a 901-D vector space. `150 × 6 + 1 = 901` points covers the maximum useful test duration (most aging mechanisms stabilize or saturate after ~6 days). |
| 3. **Akima interpolation** | `df.interpolate(method='akima')` | Series with NaN gaps → filled series | Bridge NaN gaps caused by IV-scan interruptions (~30 min every few hours), MPPT reset events, or sensor glitches. **Akima is chosen over linear interpolation because it preserves curvature without the overshooting artifacts of cubic splines** — critical for keeping the natural shape of burn-in or hump curves. |
| 4. **MaxAbsScaler normalization** | `MaxAbsScaler().fit_transform()` | PCE in [0, 100%] (or [0, 22%]) → [0, 1] | **Each curve independent** so 5%-efficient and 22%-efficient cells can be compared on their degradation SHAPE. Without this step, SOM would cluster by absolute efficiency rather than by degradation behavior. MinMaxScaler is an alternative but compresses near-zero values; MaxAbsScaler preserves them. |
| 5. **Savitzky-Golay smoothing** | `savgol_filter(arr, window=71, order=2)` | Noisy normalized series → smooth series | Remove measurement noise (~0.5–1% per-point fluctuation) while preserving features wider than ~12 h (window=71 points ≈ 11.8 h). Second-order polynomial fits the underlying physical trajectory (assumed continuous and mostly linear or quadratic over short intervals). |

---

## Adjustable Parameters — Physical Rationale

### A. Preprocessing Parameters

| Parameter | Current Value | Physical Meaning | Tuning Range | When to Adjust |
|-----------|---------------|------------------|--------------|----------------|
| **`HOUR_LIMIT`** | 150 | **Test duration window** (hours). The longer the window, the more degradation mechanisms become visible (e.g., 1000 h reveals electrode corrosion invisible in 150 h). However, more cells fail mid-test → fewer valid curves. | 50–300 | Short tests (<150 h available) → reduce. Long-duration stability tests → increase to capture late-stage mechanisms. |
| **`N_DATA_POINTS`** | 901 (= 150 × 6 + 1) | **Time resolution** (10-min intervals). Each point represents 10 minutes of physical degradation. | Computed: `HOUR_LIMIT × 6 + 1` | Not independently set — derived from HOUR_LIMIT. If you want finer time resolution, change the resample interval from `'10min'` to `'5min'` (then N_DATA_POINTS = 1801 for 150 h). |
| **`SAGGOL_WINDOW`** | 71 | **Smoothing window** (in time points). 71 points ≈ **11.8 hours** of time averaging. | 21, 31, 51, 71, 101 (must be odd) | **High noise** → larger window (101) to suppress fluctuations. **Want to preserve fine features** (e.g., 1-h light-soaking recovery peaks) → smaller window (31). Window must be **> polyorder** and **odd** for symmetric averaging. |
| **`SAGGOL_ORDER`** | 2 | **Polynomial order** for local fitting. Higher order fits more complex shapes per window but overfits noise. | 2 or 3 | **2** = safe default, captures linear + parabolic trends. **3** needed only for sharply humped curves where the parabola misses the peak. Stick with 2 if uncertain. |

### B. Normalization Parameter

| Parameter | Current Value | Physical Meaning | Alternatives |
|-----------|---------------|------------------|--------------|
| **Normalization method** | MaxAbsScaler | Each curve scaled to its own maximum. Non-negative PCE values → values in [0, 1]. | `MinMaxScaler` would map to [(min−max)/max, 1] which can be problematic if PCE ever goes negative (which shouldn't happen but is robust to). The choice of per-curve normalization is critical: without it, absolute PCE differences (5% vs 22%) would dominate clustering over shape. |

### C. SOM Training Parameters

| Parameter | Current Value | Physical Meaning | Tuning Range | When to Adjust |
|-----------|---------------|------------------|--------------|----------------|
| **`SOM_X`** × **`SOM_Y`** | 2 × 2 | **SOM grid size = number of archetype clusters**. 2×2 = 4 archetypes. | 2×2=4, 3×3=9, 4×4=16, 5×5=25 | **More archetypes** (3×3 = 9) if you suspect rare degradation modes exist beyond the 4 found here. **Fewer** (just 2 by using 1×2) for very coarse classification. The 2×2 choice reflects the prior that perovskite degradation typically has 3–5 dominant modes. |
| **`SOM_SIGMA`** | 0.5 | **Neighborhood radius** in the SOM grid. Larger sigma means each neuron's update affects more neighbors during training. | 0.1–2.0 | Higher sigma → smoother weight updates → better topology. Lower sigma → more localized adaptation → finer cluster boundaries. 0.5 is a balance for a 2×2 grid where whole-grid coupling is desired. |
| **`SOM_LR`** | 0.1 | **Learning rate** — step size for weight updates during training. | 0.01–1.0 | **Too low** → slow convergence, weights may not reach optimal positions. **Too high** → oscillations, unstable. 0.1 with exponential decay (built into MiniSom) is standard. |
| **`SOM_ITER`** | 50,000 | **Training iterations** — how many random samples drive the weight updates. MiniSom also implements exponential decay of LR over iterations. | 10,000–200,000 | **Loss plateaued early** → can reduce to 10,000–20,000 for speed. **QE still high** → increase to 100,000–200,000 for further refinement. 50,000 is enough for ~900-D input with 4 neurons. |
| **`random_seed`** | 42 | **Reproducibility seed**. Same seed + same params = identical cluster assignment. | any integer | Critical for paper reproducibility. Change only if you want to verify result stability across random initializations. |

### D. Clustering Validation Parameters

| Parameter | Current Value | Physical Meaning |
|-----------|---------------|------------------|
| `n_clusters` (K-Means) | 4 | Number of archetypes to extract via K-Means for comparison with SOM. Must equal SOM_X × SOM_Y for fair comparison. |
| `n_init` (K-Means) | 10 | How many times K-Means runs with different centroid seeds. Best (lowest inertia) is kept. Higher = more robust but slower. |
| `n_components` (PCA) | 2 | Number of principal components to project the 901-D curves into for visualization. PC1+PC2 typically explain 70–90% of variance for degradation data. |

---

## Use Case Quick Reference

| Use Case | Most Useful Outputs |
|----------|---------------------|
| Paper figures | `som_clusters.png`, `pce_violin.png`, `pca_visualization.png` |
| Interactive presentation | HTML files (hover to inspect individual curves) |
| Thesis defense / demo | HTML files + `som_cluster_counts.png` |
| Reproducibility | `som_summary.json` (re-run with same params) |
| Data quality check | `overview_savgol.png` |
| Method comparison | `som_vs_kmeans_comparison.png` |
| Thesis writing | `som_umatrix.png` + `pca_visualization.png` |

---

## Adapting to Your Own Data

### Data Format Requirements

Your MPPT CSV files must have one of these two formats:

**Format A** (standard):
```
Year,Month,Day,Hour,Minute,Pmax,Vmpp,Impp
2025,6,19,20,36,2.71,30.51,0.089
```

**Format B** (with Chinese metadata header + Step/Pmax/RecordTime):
```
样品名称=BZ15P02,,,,
...
Step,Pmax,PmaxV,PmaxI,RecordTime
MPPT追踪,8926.328,30375.428,293.867,2025/12/19 17:37
```

### Steps

1. Place CSV files in `原始数据/` (any nested sub-folders are picked up automatically).
2. Run: `python my_pipeline.py` (script at `abc/my_pipeline.py`).
3. Find your results in `result/my_data/`:
   - `my_som_clusters.png` — visual archetypes for your 30 cells
   - `my_cluster_assignments.csv` — filename → archetype mapping
   - `my_som_summary.json` — training metrics

The pipeline handles: format detection, encoding (UTF-8 / GBK), Chinese-header skipping, malformed number coercion, Akima interpolation, normalization, smoothing, and SOM training — all in one script.

---

## Dependencies

```
minisom           # SOM implementation
tslearn           # Time-series clustering (DTW)
sklearn           # K-Means, PCA, preprocessing
scipy             # Savitzky-Golay filter
numpy             # Array operations
pandas            # Data handling
matplotlib        # Static plots
seaborn            # Statistical visualization
plotly            # Interactive HTML plots
pickle5           # Load .pkl files
colorlover        # Color scales for Plotly
```

---

## Usage

```bash
# Setup environment
conda env create -f environment.yml          # Windows
conda env create -f environment_mac.yml    # Mac

# Activate
conda activate PvkSOM

# Run pipeline
python run_pipeline.py

# Or use Jupyter notebooks
jupyter notebook
# Open: 20230816_degradation_analysis_revision_11_cleaned.ipynb
```

---

## Folder Structure

```
abc/
├── run_pipeline.py                    # Main pipeline script
├── my_pipeline.py                     # Custom pipeline for your own data
├── *.ipynb                            # Jupyter notebooks for analysis
├── README.md                          # Documentation
├── LICENSE                            # BSD 2-Clause License
├── environment.yml                    # Conda env (Windows)
├── environment_mac.yml                # Conda env (Mac)
├── dataset/
│   ├── PCE_df_grouping.csv            # Input: PCE grouping data
│   └── pkl_complete/
│       ├── 20230303_mySeriesDrop.pkl
│       ├── 20230303_mySeriesDropNorm.pkl
│       └── 20230303_mySeriesDrop_savgol.npy
├── result/                            # Output files (this folder)
│   ├── README.md                      # This file
│   ├── som_summary.json
│   ├── som_*.png / .html
│   ├── pce_*.png / .html
│   └── my_data/                       # Outputs from your 30 files
│       ├── my_som_summary.json
│       ├── my_som_clusters.png
│       └── my_cluster_assignments.csv
└── result.zip                         # Archived results
```

---

## Scientific Value of PvkSOM

Using these outputs, perovskite researchers can:

1. **Rapidly classify** new device degradation curves into the 4 known archetypes.
2. **Infer physical mechanisms** from degradation shape (ion migration, interface degradation, encapsulation failure, defect passivation).
3. **Optimize fabrication**: if one archetype dominates → improve that process; if one is rare → replicate its favorable conditions.
4. **Predict reliability**: based on cluster assignment, estimate the long-term trajectory of new devices.
5. **Publish visualizations**: PNG/HTML files are directly usable as paper figures.