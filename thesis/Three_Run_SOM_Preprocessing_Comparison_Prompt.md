# Prompt: Three-Run SOM Preprocessing Comparison on the Hartono PSC Ageing Dataset

## Objective

Use the **original Hartono et al. PSC ageing dataset** and run exactly **three separate SOM clustering experiments** to test how preprocessing changes the resulting degradation-curve tendencies and shapes.

This task is **not a reproduction of the paper**. I have already reproduced the original paper workflow.

The only goal here is to compare the following three pipelines:

1. **Raw → SOM**
2. **Resample + Normalize → SOM**
3. **Resample + Normalize + Smooth → SOM**

After all three runs, generate the results for each test separately and then provide a direct comparison of the three outputs.

---

# Reference Dataset

Use the original dataset associated with:

**Hartono et al.**  
*Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset*  
Nature Communications (2023), 14:4869.

Dataset:
https://doi.org/10.5281/zenodo.8185883

Use the **same device subset and same analysis time range for all three tests** so that the only major difference between experiments is preprocessing.

Where possible, use the same subset used in the previously completed paper reproduction.

---

# General Rules

The following must remain fixed across all three experiments:

- same dataset
- same device/sample subset
- same analysis time range
- same SOM implementation
- same SOM topology / map size
- same initialization method
- same distance metric
- same training iterations
- same random seed for the primary run
- same method for assigning curves to SOM nodes/clusters
- same cluster-number selection procedure
- same plotting style and axis conventions where meaningful

Do **not** tune SOM parameters separately for each pipeline in order to obtain visually desired shapes.

Do **not** tune the analysis specifically to obtain Bridge / Hill / Slope / Valley.

The purpose is to observe what the data naturally produce under the three preprocessing levels.

---

# Important Note About the Raw Test

SOM requires each input sample to have the same dimensionality.

For **Test 1: Raw → SOM**, first inspect whether the original ageing curves can be represented as equal-length vectors without resampling.

- If the dataset is already stored on a common time grid, use the original PCE vectors directly.
- Do not normalize.
- Do not smooth.
- Do not interpolate.
- Do not resample.

If the raw curves genuinely have incompatible vector lengths or sampling locations such that direct SOM input is mathematically impossible:

1. clearly report this;
2. do not silently apply the preprocessing used in Tests 2 or 3;
3. use only the absolute minimum formatting necessary to construct equal-length vectors;
4. document exactly what was changed;
5. still label this run as **Near-Raw SOM** rather than claiming it is perfectly raw.

Do not hide this limitation.

---

# TEST 1 — Raw → SOM

## Goal

Determine what SOM learns when the original ageing-curve values are used with no intentional preprocessing.

## Pipeline

```text
Raw ageing curves
        ↓
SOM clustering
```

## Do Not Apply

- no resampling
- no interpolation
- no normalization
- no Savitzky–Golay smoothing
- no other denoising
- no derivative transformation
- no feature engineering

Preserve the original PCE magnitude.

## Required Analysis

Run SOM and determine the dominant cluster tendencies.

For every cluster, report:

- cluster ID
- number of curves
- percentage of dataset
- cluster mean / representative curve
- individual curves shown with low opacity
- initial PCE distribution
- maximum PCE distribution
- qualitative curve tendency

Examples of qualitative descriptions:

- initial increase
- slow decay
- medium decay
- rapid decay
- non-monotonic
- recovery-like
- mixed / unclear

Do not force these labels.

## Main Question

Does the raw-data SOM mainly cluster curves according to:

- absolute PCE magnitude,
- degradation rate,
- overall curve shape,
- or a mixture of these?

---

# TEST 2 — Resample + Normalize → SOM

## Goal

Remove sampling-grid differences and absolute PCE magnitude, but retain the unsmoothed curve morphology.

## Pipeline

```text
Raw ageing curves
        ↓
Resample to common time grid
        ↓
Interpolation as required
        ↓
MaxAbs normalization
        ↓
SOM clustering
```

## Resampling

Follow the reference paper where applicable:

```text
resampling interval = 10 minutes
```

Use the same common analysis time window for every curve.

Use **Akima interpolation** when interpolation is required.

## Normalization

Normalize each curve using its maximum absolute PCE within the analysed observation window:

\[
PCE_{\mathrm{norm}}(t)
=
\frac{PCE(t)}
{\max |PCE(t)|}
\]

Do **not** smooth the normalized curves.

## Do Not Apply

- no Savitzky–Golay smoothing
- no other denoising
- no derivative transformation
- no manual shape selection

## Required Analysis

Run SOM and report:

- cluster ID
- number of curves
- percentage of dataset
- cluster centre / mean curve
- individual normalized curves with low opacity
- qualitative curve tendency
- cluster-number evaluation
- quantization error

## Main Question

After removing PCE magnitude and aligning the time axis, does SOM recover clearer degradation-shape clusters?

---

# TEST 3 — Resample + Normalize + Smooth → SOM

## Goal

Evaluate the additional effect of smoothing after time alignment and normalization.

## Pipeline

```text
Raw ageing curves
        ↓
Resample to common time grid
        ↓
Akima interpolation
        ↓
MaxAbs normalization
        ↓
Savitzky–Golay smoothing
        ↓
SOM clustering
```

## Resampling

Use:

```text
10-minute interval
```

## Normalization

Use:

\[
PCE_{\mathrm{norm}}(t)
=
\frac{PCE(t)}
{\max |PCE(t)|}
\]

## Smoothing

Use the preprocessing condition reported in the reference study where compatible:

```text
Savitzky–Golay filter
window_length = 71
```

Use the same polynomial order as the existing successful reproduction code.

Do not change the smoothing parameters to make specific curve classes appear.

## Required Analysis

Run SOM and report:

- cluster ID
- number of curves
- percentage of dataset
- cluster centre / mean curve
- individual curves with low opacity
- qualitative curve tendency
- cluster-number evaluation
- quantization error

## Main Question

Does smoothing materially change the cluster morphology, or does it mainly reduce noise while preserving the same dominant tendencies?

---

# SOM Configuration

Reuse the SOM implementation and validated training configuration from the previous successful paper reproduction.

Where the original reference values are needed:

```text
sigma = 0.5
learning_rate = 0.1
```

Do not independently optimize `sigma` or `learning_rate` for the three tests.

The comparison must isolate **preprocessing**, not SOM hyperparameter tuning.

---

# Cluster Number

For each test, evaluate the natural cluster structure rather than automatically assuming four clusters.

Use the same cluster-number evaluation procedure for all three tests.

At minimum, evaluate:

```text
K = 2, 3, 4, 5, 6, 7, 8
```

Use SOM quantization error and elbow analysis.

Report:

- quantization error for each K
- selected K
- reason for selecting K

For direct visual comparison, also provide an additional **fixed K = 4 view** for each test.

Important:

> Fixed K = 4 is only a controlled comparison and must not be interpreted as evidence that four natural clusters exist.

---

# Required Figures

Generate separate figures for each experiment.

## Figure 1 — Raw SOM

Filename:

```text
01_raw_som_clusters.png
```

Show all resulting cluster-centre curves and the underlying member curves.

---

## Figure 2 — Resample + Normalize SOM

Filename:

```text
02_resample_normalize_som_clusters.png
```

---

## Figure 3 — Resample + Normalize + Smooth SOM

Filename:

```text
03_resample_normalize_smooth_som_clusters.png
```

---

# Direct Three-Test Comparison Figure

Generate one additional comparison figure:

```text
04_three_pipeline_comparison.png
```

Layout:

```text
(A) Raw → SOM

(B) Resample + Normalize → SOM

(C) Resample + Normalize + Smooth → SOM
```

The purpose is to make the changes in dominant curve tendency immediately visible.

Use comparable axis conventions wherever scientifically meaningful.

---

# Quantitative Comparison Between the Three Tests

Compare the clustering results across pipelines.

For the fixed-K comparison, calculate:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Comparisons:

```text
Test 1 vs Test 2
Test 2 vs Test 3
Test 1 vs Test 3
```

Because cluster labels are arbitrary, do not compare cluster IDs directly.

Also compare cluster-centre shapes using relevant curve characteristics:

- early-time slope
- late-time slope
- overall PCE change
- time of maximum
- time of minimum
- number of major turning points
- peak-to-final decline
- valley depth
- recovery amplitude

---

# Required Summary Table

Create:

| Test | Pipeline | Selected K | Main cluster tendencies | Main interpretation |
|---|---|---:|---|---|
| 1 | Raw → SOM | | | |
| 2 | Resample + Normalize → SOM | | | |
| 3 | Resample + Normalize + Smooth → SOM | | | |

Also create a cluster-level table:

| Test | Cluster | N | Fraction | Qualitative morphology | Key shape features |
|---|---:|---:|---:|---|---|

---

# Questions the Final Analysis Must Answer

## Question 1

How different is:

```text
Raw → SOM
```

from:

```text
Resample + Normalize → SOM
```

?

Specifically determine whether normalization changes SOM from mainly separating absolute PCE values to separating degradation tendencies.

---

## Question 2

How different is:

```text
Resample + Normalize → SOM
```

from:

```text
Resample + Normalize + Smooth → SOM
```

?

Determine whether smoothing:

- only suppresses noise,
- changes cluster membership,
- changes cluster-centre morphology,
- removes local peaks or valleys,
- or merges distinct shapes.

---

## Question 3

Across all three tests, are the major degradation tendencies broadly preserved?

Classify the conclusion as one of:

### A. Strongly robust

The major curve tendencies remain highly similar across all three pipelines.

### B. Moderately robust

The major tendencies remain recognizable, but preprocessing changes cluster boundaries or population sizes.

### C. Preprocessing-dependent

The dominant cluster shapes change substantially after normalization and/or smoothing.

### D. Unstable

No consistent cluster morphology is recovered across the three pipelines.

---

# Secondary Check: Bridge / Hill / Slope / Valley

Only after completing the three main tests, inspect whether any obtained clusters naturally resemble:

- Bridge
- Hill
- Slope
- Valley

For each test, create:

| Archetype | Clearly present | Weakly present | Rare | Not observed |
|---|---|---|---|---|
| Bridge | | | | |
| Hill | | | | |
| Slope | | | | |
| Valley | | | | |

Do not modify preprocessing or SOM parameters to make these four archetypes appear.

This is an observational check only.

---

# Robustness Across Random Seeds

After the primary run with one fixed seed, repeat each of the three pipelines using at least:

```text
10 random seeds
```

Keep all other settings fixed.

Report:

- mean quantization error
- standard deviation of quantization error
- clustering stability
- whether the qualitative cluster-centre shapes persist

The main conclusion should not depend on a single lucky SOM initialization.

---

# Output Directory

Save all results using the following structure:

```text
som_preprocessing_comparison/
│
├── test1_raw/
│   ├── cluster_assignments.csv
│   ├── cluster_summary.csv
│   ├── quantization_error.csv
│   └── 01_raw_som_clusters.png
│
├── test2_resample_normalize/
│   ├── cluster_assignments.csv
│   ├── cluster_summary.csv
│   ├── quantization_error.csv
│   └── 02_resample_normalize_som_clusters.png
│
├── test3_resample_normalize_smooth/
│   ├── cluster_assignments.csv
│   ├── cluster_summary.csv
│   ├── quantization_error.csv
│   └── 03_resample_normalize_smooth_som_clusters.png
│
├── comparison/
│   ├── 04_three_pipeline_comparison.png
│   ├── clustering_similarity_ARI_NMI.csv
│   ├── cluster_shape_features.csv
│   ├── three_test_summary.csv
│   └── bridge_hill_slope_valley_check.csv
│
└── final_analysis.md
```

---

# Final Analysis

The final `final_analysis.md` must contain:

1. Experimental objective
2. Exact definition of the three pipelines
3. Dataset/sample count used
4. Any unavoidable limitation in the Raw → SOM test
5. Natural K for each pipeline
6. Fixed-K comparison
7. Test 1 results
8. Test 2 results
9. Test 3 results
10. ARI/NMI comparison
11. Comparison of cluster-centre morphology
12. Effect of resampling + normalization
13. Additional effect of smoothing
14. Random-seed robustness
15. Bridge/Hill/Slope/Valley observational check
16. Final conclusion

The final conclusion should directly answer:

> **If the Hartono ageing curves are sent to SOM at three different preprocessing levels — raw, resampled+normalized, and resampled+normalized+smoothed — do we obtain broadly the same degradation tendencies and curve shapes, or are the SOM results strongly dependent on preprocessing?**

---

# Critical Constraints

- Run **exactly these three preprocessing pipelines** as the main experiments.
- Do not add extra preprocessing experiments.
- Do not reproduce the original paper as a separate fourth run.
- Do not tune each pipeline separately to obtain preferred shapes.
- Do not force Bridge/Hill/Slope/Valley.
- Do not remove inconvenient curves merely to improve visual clustering.
- Preserve identical experimental settings wherever possible.
- Clearly report when the raw-data format prevents a perfectly raw SOM input.
- Save every intermediate result required for reproducibility.
