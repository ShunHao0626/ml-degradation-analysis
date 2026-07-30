# Reference Parameter Audit: Hartono et al. Nature Communications 2023 (PvkSOM)

**Generated:** 2026-07-31
**Reference:** [10.1038/s41467-023-40585-3](https://doi.org/10.1038/s41467-023-40585-3)
**Author Code:** [PvkSOM](https://github.com/noortitan/PvkSOM)
**Main Notebook:** `20230816_degradation_analysis_revision_11_cleaned.ipynb`

---

## 1. SOM Parameters

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| SOM topology (x, y) | Fig 4a shows 2×2 grid | - | `som_x=2, som_y=2` | **2×2** |
| Number of clusters | 4 | - | 4 | 4 |
| sigma | Mentioned 0.5 | - | `sigma=0.5` | 0.5 |
| learning_rate | Mentioned 0.1 | - | `learning_rate=0.1` | 0.1 |
| activation_distance | "euclidean" | - | Not explicitly shown | euclidean |
| neighborhood_function | "gaussian" | - | MiniSom default | gaussian |
| Weight initialization | Not specified | - | `random_weights_init()` | random_weights_init |
| Training method | - | - | `train(data, 50000)` | train_random |
| Training iterations | 50,000 | - | 50,000 | 50,000 |
| random_seed | Not shown | - | Not set | 42 |
| Input length | 900 (150h × 6 pts/h) | - | `len(mySeriesDrop_savgol[0])` | 1201 (200h × 6 pts/h + 1) |

> **IMPORTANT NOTE ON TOPOLOGY:** The paper's Figure 4a displays 4 clusters in a 2×2 subplot layout. This is the **visualization arrangement**, not necessarily proof of the SOM topology. However, the author code explicitly sets `som_x=2, som_y=2`, confirming the 2×2 topology.

---

## 2. Preprocessing Parameters

### 2.1 Time Window

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| Analysis window | 150 hours | - | `hour_limit=150` | **200 hours** |
| Grid interval | 10 min | - | Not specified explicitly | 10 min (1/6 h) |
| Grid points | 900 | - | `np.linspace(0, 150, 900, endpoint=True)` | **1201** |

> **METHOD DIFFERENCE:** This study uses 200h window instead of 150h, as explicitly requested by the user.

### 2.2 Normalization

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| Normalization method | MaxAbsScaler | - | `MaxAbsScaler()` per curve | Per-curve max normalization |
| Normalization direction | Each curve divided by its max | - | `scaler.fit_transform()` per curve | Per-curve max = 1.0 |
| When normalized | Before smoothing | - | After smoothing? | Before smoothing |

> **NOTE:** The author code applies `MaxAbsScaler()` to each time series individually:
> ```python
> for i in range(len(mySeriesDrop_norm)):
>     scaler = MaxAbsScaler()
>     mySeriesDrop_norm[i] = MaxAbsScaler().fit_transform(mySeriesDrop_norm[i])
> ```
> This means each curve is scaled by its own maximum value, consistent with the user's requirement.

### 2.3 Smoothing (Savitzky-Golay Filter)

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| window_length | 71 | - | `n = 71` | 71 |
| polyorder | 2 | - | `savgol_filter(data, 71, 2)` | 2 |
| mode | Not shown | - | Default | **interp** (implementation assumption) |

> **Implementation Assumption:** The paper/author code does not explicitly specify the `mode` parameter for Savitzky-Golay filter. Using `mode='interp'` (default in scipy) as the implementation assumption.

### 2.4 Interpolation

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| Interpolation method | Not described | - | Not shown in main notebook | **Akima** |
| Implementation assumption | - | - | - | Akima from `scipy.interpolate.Akima1DInterpolator` |

> **Implementation Assumption:** The author code loads pre-processed data from pickle files and does not show the interpolation method used to generate those files. Using Akima interpolation as requested by the user.

---

## 3. Data Quality Parameters

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| Minimum data points | Not specified | - | Not shown | 10 unique points in window |
| Max PCE threshold | > 0 | - | Assumed | > 0 |
| Duplicate handling | Not shown | - | Not shown | Mean of duplicates |

---

## 4. Quantization Error Calculation

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| QE formula | Not shown | - | Not shown | MiniSom built-in |
| Node sweep range | 2-10 | - | `x = (2,3,4,5,6,7,8,9,10)` | 2-10 |

> **NOTE:** The author code hardcodes QE values for different node counts (lines 2509-2520), suggesting they pre-computed these. The exact QE calculation method (built-in MiniSom or custom) is not visible in the notebook.

---

## 5. Cluster Validation

### 5.1 K-means

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| k range | Not shown | - | Not shown | 2-10 |
| Algorithm | - | - | `TimeSeriesKMeans(metric="dtw")` | `sklearn.KMeans` |
| DTW usage | Yes | - | Yes | No (implementation assumption) |

> **Implementation Assumption:** The author uses `TimeSeriesKMeans` with DTW metric. This study uses standard `sklearn.KMeans` with Euclidean distance due to computational efficiency. This is a known difference that should be noted.

### 5.2 DTW Barycenter Averaging

| Parameter | Main Paper | Supplementary | Author Code | This Study |
|-----------|:----------:|:-------------:|:-----------:|:----------:|
| DBA method | Yes | - | `dtw_barycenter_averaging()` | Not used |

> **Implementation Assumption:** DTW barycenter averaging is not implemented in this study due to computational complexity and the use of standard k-means for validation.

---

## 6. Summary of Confirmed vs. Assumed Parameters

### Confirmed from Author Code:
- `som_x = 2, som_y = 2` (2×2 topology)
- `sigma = 0.5`
- `learning_rate = 0.1`
- `window_length = 71` for Savitzky-Golay
- `polyorder = 2` for Savitzky-Golay
- `random_weights_init()` for initialization
- `train(data, 50000)` for training
- Per-curve `MaxAbsScaler()` normalization
- QE sweep from n=2 to n=10

### Implementation Assumptions (NOT from paper/code):
- `mode='interp'` for Savitzky-Golay
- Akima interpolation for missing points
- Random seed = 42
- Minimum 10 unique data points requirement
- Mean handling for duplicate timestamps
- Standard K-means (not DTW-based) for validation

### Intentional Differences (User-Requested):
- Analysis window: **200h** (vs. 150h in paper)
- Grid points: **1201** (vs. 900 in paper)

---

## 7. References

1. Hartono et al., "Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset," *Nature Communications* 14, 4869 (2023)
2. PvkSOM GitHub: https://github.com/noortitan/PvkSOM
3. Zenodo data: https://doi.org/10.5281/zenodo.8185882
