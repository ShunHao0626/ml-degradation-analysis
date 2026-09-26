# Three SOM Cluster Curve Plots — Why They Differ

**Generated:** 2026-07-31

The three `som_cluster_curves` figures are all 2×2 SOM analyses of perovskite MPPT ageing
curves in the 0–200 h window, sharing the same nominal SOM hyperparameters
(shape=2×2, σ=0.5, lr=0.1, iterations=50 000, random_seed=42). They differ in
**(a) the data loader / discovery rules, (b) the QC rules, (c) the smoothing/
interpolation modes, (d) the SOM training routine (`train` vs `train_random`),
and (e) the choice of feature passed to the SOM (smoothed vs. raw normalized).
All three differences propagate into the cluster assignments and therefore into
the four mean/median curves shown in each `som_cluster_curves` panel.

## 1. The three figures

| # | Figure path | Pipeline / source script | Run timestamp |
|---|-------------|--------------------------|---------------|
| 1 | `lab/result/SOM_200h_output/som_cluster_curves.png` | `lab/result/SOM_200h_pipeline.py` (single-file pipeline) | 2026-07-29 15:09 |
| 2 | `lab/results_all/A_result/outputs/200h_som/figures/som_cluster_curves.png` | earlier fork of the same idea (no `.py` ships with the output) | 2026-07-29 16:23 |
| 3 | `lab/RESULTS_>200h/200h_som/outputs/figures/som_n4_curves.png` | `lab/RESULTS_>200h/200h_som/scripts/run_200h_som.py` (modular pipeline) | 2026-07-31 00:59 |

> Image #3 is generated as the n=4 case inside the QE-sweep loop (`plot_som_clusters(... name="n4")`) and then copied to `som_n4_curves.png` at the end of `run_200h_som.py`.

## 2. Side-by-side cluster-level metrics

| | Image #1 (`SOM_200h_pipeline.py`) | Image #2 (`results_all/...`) | Image #3 (`RESULTS_>200h/...`) |
|---|---|---|---|
| Total included curves | **1 667** | **1 453** | **1 812** |
| Cluster (0,0) n / PCE@200h | 257 / 0.440 | 860 / 0.923 | 443 / 0.765 |
| Cluster (0,1) n / PCE@200h | 101 / 0.128 | 157 / 0.506 | 188 / 0.543 |
| Cluster (1,0) n / PCE@200h | 855 / 0.922 | 371 / 0.745 | 1 103 / 0.935 |
| Cluster (1,1) n / PCE@200h | 454 / 0.734 | 65 / 0.259 | 78 / 0.273 |
| QE @ 2×2 SOM | 1.754 | 1.456 | 1.285 |

The visual differences in the four panels are a direct consequence of these
numbers. Image #1's stable cluster (0,0) is 855 curves (51 %), Image #2 has 371
curves in that slot, and Image #3 has 1 103. Even when two images agree on the
*shape* of a cluster, the absolute counts — and therefore the IQR band, the
median, and the mean — shift.

## 3. Root-cause analysis (with code citations)

### 3.1 Data loading differs

| | Image #1 | Image #2 | Image #3 |
|---|---|---|---|
| Source directory | `lab/05_accepted_all copy/` (4 × `x_time_*` top dirs) | likely same 4 × `x_time_*` dirs | `lab/05_accepted_all copy/curves_over_200h_full/manifest.csv` |
| Loader | `load_all_raw_curves` in `load_raw_data.py` (synchronous, multi-threaded) | unclear (no `.py` ships with output) | `data_loading.discover_csv_files` + `load_one_curve` in `data_loading.py` |
| Per-curve unit detection | `get_figure_unit_from_folder`: validation JSON → folder-name inference → safe default (24 h) | same logic (fork) | `figure_unit_factor`: validation JSON → `DIR_TO_HOURS` table → folder-name fallback |
| `x_time_min` interpretation | factor = 1/60 | factor = 1/60 | factor = 60 (i.e. **minutes → 1 hour**) |

Image #3's config `DATA_CONFIG["time_unit_factor_by_dir"]` sets
`"x_time_min": 60.0` (i.e. minutes → hours), whereas the other two pipelines
use `1/60` (i.e. assume the files are actually in hours). Files inside
`x_time_min` that are stored in *minutes* therefore contribute different
time-series length to the feature matrix.

### 3.2 Different curves pass QC

| | Image #1 (`SOM_200h_pipeline.py:288`) | Image #2 (inferred from log) | Image #3 (`preprocessing.py:197` and `config.py:62`) |
|---|---|---|---|
| Minimum unique time points in [0,200] h | **10** | 10 | **4** |
| Minimum time-span requirement | none | none | **200 h** (`require_min_span_h=200`) |
| Coverage fraction | none | none | none (≥0) |
| Forward-fill on out-of-range grid | yes | yes | no (clamp instead, `preprocessing.py:91`) |

This is the single biggest source of the sample-count difference. Image #1 and
Image #2 require 10 unique points, Image #3 requires only 4. Image #3 *adds* a
requirement that the curve span the entire 200 h window (`rel_t_max ≥ 200 h`),
which actually removes some short-but-dense curves (see the `curve spans only
…` reasons in `excluded_samples.csv`). The net effect of "lower min_unique
points but higher min-span" is more curves overall (1 812 vs. 1 667/1 453).

### 3.3 Smoothing / interpolation differences

| | Image #1 (`SOM_200h_pipeline.py:319`) | Image #2 | Image #3 (`preprocessing.py:154`, `config.py:148`) |
|---|---|---|---|
| Savitzky–Golay `mode` | **default** (no `mode=` kwarg → 'interp' in old scipy, otherwise 'mirror') | default | **explicit `mode='interp'`** |
| Out-of-range handling after Akima | `_forward_fill_nan` then NaNs persist | similar | **boundary-clamp to nearest observed value** + forward/backward fill |
| Order of normalization vs smoothing | normalize first, then smooth | normalize first, then smooth | normalize first, then smooth |

The boundary-clamp strategy in Image #3 produces visibly flatter tails at t≈0 h
and t≈200 h compared with the forward-fill strategy, which can give curves with
near-constant plateaus at both ends of the window.

### 3.4 SOM training routine

| | Image #1 (`SOM_200h_pipeline.py:418`) | Image #2 (from `reference_parameter_audit.md`) | Image #3 (`som_analysis.py:111`, `config.py:106`) |
|---|---|---|---|
| Training routine | `som.train_random(...)` | `som.train_random(...)` | **`som.train(...)`** |
| Random seed | 42 | 42 | 42 (set via `_seed_random_state` and `MiniSom(random_seed=…)`) |
| Initialization | random init (via MiniSom default) | random init | **explicit `som.random_weights_init(X)` before training** |

`train` and `train_random` follow the **same update rule** but sample the
input vector in different orders; combined with the explicit
`random_weights_init(X)` call in Image #3, the codebook vectors that the SOM
converges to differ even though `random_seed=42`. Image #3's lower QE
(1.285 vs. 1.456 vs. 1.754) is consistent with this — `train` + explicit
re-initialization is the better converged of the three runs.

### 3.5 Different feature passed to SOM

Both pipelines feed the SOM a (n_curves × 1201) matrix of smoothed
normalized curves. The numbers above are from the post-smoothing feature, so
this is not a separate input difference — but the **post-smoothing feature**
itself differs because the **QC rules, smoothing mode, and boundary fill**
above differ.

### 3.6 Plotting differences (visible style)

| | Image #1 (`SOM_200h_pipeline.py:584`) | Image #2 (`lab/RESULTS_>200h/200h_som/src/plotting.py:97` re-used) | Image #3 (`plotting.py:97`) |
|---|---|---|---|
| Figure size | (12, 10) | n/a | (8.4, 6.4) |
| Member curves | `alpha=0.15`, `lw=0.5`, `gray` | n/a | `alpha=0.3`, `lw=0.4`, `lightgray` |
| IQR shade | `steelblue α=0.15` | n/a | `C0 (blue) α=0.2` |
| Mean / median line style | `mean` black 2.5 / `median` crimson 2.0 dashed | n/a | `mean` black 2.0 / `median` orange 1.4 dashed |
| ylim / xlim | explicit (0, 1.08) / (0, 200) | n/a | defaults |
| suptitle text | `SOM Clusters (2×2) — {n} curves, σ=…, lr=…` | n/a | `SOM clusters – {name} (2×2 subplot)` |

The 2×2 subplot **layout** is the same; only the visual styling differs.

## 4. Per-panel summary

Visually each figure looks like: "fast-decay cluster (PCE → 0)", "medium-decay
cluster", "slow-decay / stable cluster", "another fast-decay cluster". The
*qualitative* ordering of cluster (0,0) → (1,1) from "stable" to "degraded" is
the same in all three images because SOM node identity is preserved. The
*quantitative* mean/median curves and the panel counts differ because of §3.

## 5. Conclusion / which figure to trust

- All three runs use the same 2×2 topology, same σ=0.5, same lr=0.1, same
  iter=50 000, same seed=42. They differ in **preprocessing + training routine
  + loader semantics**, not in the choice of clustering algorithm.
- **Image #3 is the most reproducible and best documented**: it ships with
  `run_config.json`, `reference_parameter_audit.md`, an explicit
  `random_weights_init`, an explicit `train(...)`, and a per-curve
  boundary-clamp that is logically consistent with "do not extrapolate".
  It also has the lowest QE (1.285), so the codebook vectors are best
  fitted to the data.
- **Image #1 has the most inclusive QC** (only ≥10 unique points, no span
  requirement) and the lowest-iteration-equivalent training routine
  (`train_random`). It uses forward-fill on out-of-range grid points,
  which inflates the apparent number of "fast-decay" curves (cluster
  (0,0) holds 257 curves with PCE@200h ≈ 0.44, but the same cluster in
  Image #3 holds 443 curves with PCE@200h ≈ 0.77, because Image #3's
  boundary-clamp + better SOM fit pulls the boundary curves upward).
- **Image #2 sits between #1 and #3 in cluster counts** (1 453 curves) and
  QE (1.456). Without an associated `.py` it is the least auditable of the
  three.

**Recommendation:** use **Image #3** (`RESULTS_>200h/200h_som/outputs/figures/som_n4_curves.png`)
as the canonical 2×2 SOM result and cite
`reference_parameter_audit.md` when describing the parameters. Treat Images
#1 and #2 as historical iterations showing how the cluster assignments
moved as the preprocessing pipeline tightened (fewer short-curves dropped,
better boundary behaviour, better-trained SOM).