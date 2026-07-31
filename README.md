# ML Degradation Analysis

> An auditable machine-learning workflow for grouping **perovskite solar-cell degradation trajectories** by their power-conversion-efficiency (PCE) evolution.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Method](https://img.shields.io/badge/Method-Self--Organizing%20Map-orange)](https://en.wikipedia.org/wiki/Self-organizing_map)

## What this project does

Perovskite solar cells can lose performance in very different ways: some remain stable, while others decay steadily or exhibit an early rapid drop. This repository collects digitised PCE-versus-time curves from published studies, harmonises their time axes, and uses unsupervised learning to identify recurring degradation shapes.

The main reproducible analysis focuses on the first **200 hours** of each curve. It:

1. loads accepted curves digitised from the literature;
2. converts minutes, hours, days, weeks, months, and years to hours;
3. quality-controls, interpolates, smooths, and normalises each trajectory;
4. trains a 2 × 2 Self-Organizing Map (SOM) on the curve shapes;
5. compares SOM assignments with k-means and evaluates alternative SOM sizes and hyperparameters; and
6. exports traceable tables, figures, trained models, and Markdown reports.

The project builds on the PvkSOM workflow associated with Hartono *et al.* (2023), while adding a structured 200-hour analysis pipeline, quality-control records, validation outputs, and analysis of literature-derived curves.

## Highlights

- **2,151** curves are loaded in the current 200-hour run.
- **1,453** curves pass preprocessing and are included in the SOM analysis; **698** are excluded with documented reasons.
- The main model is a **2 × 2 SOM** (4 nodes) trained for 50,000 iterations.
- The four nodes span stable trajectories through moderate loss to early, rapid degradation.
- Full intermediate data, cluster assignments, audit files, figures, trained SOM artifacts, and run logs are versioned for inspection.

These figures are descriptive clustering results, not proof of distinct physical degradation mechanisms.

## Repository map

```text
.
├── lab/
│   ├── data_all/                   # Current canonical literature-derived PCE–time curves
│   │   ├── x_time_h/               # Curves reported in hours
│   │   ├── x_time_day/             # Curves reported in days
│   │   ├── x_time_min/             # Curves reported in minutes
│   │   ├── x_time_week_month_year/ # Curves reported in longer time units
│   ├── 05_accepted_all copy/       # Legacy analysis reports and selected-figure manifest
│   │   └── curves_over_200h_full/  # Compact CSV/JSON index of qualifying figures
│   ├── NEW/som_200h_reanalysis/    # New 200 h reanalysis, reports, code, and SI outputs
│   ├── RESULTS_>200h/200h_som/     # Earlier/extended 200 h SOM results
│   └── results_all/A_result/       # Structured primary analysis pipeline and outputs
│       ├── scripts/run_200h_som.py # Pipeline entry point
│       ├── src/                    # Loading, preprocessing, SOM, validation, plotting
│       ├── tests/                  # Preprocessing and output tests
│       └── outputs/200h_som/       # Versioned reports, tables, figures, models
└── thesis/
    ├── dataset/                    # PvkSOM-compatible processed dataset
    ├── *.ipynb                     # Original/exploratory notebook workflows
    ├── run_pipeline.py             # Thesis-oriented pipeline
    └── README.md                   # Upstream workflow notes and attribution
```

## Data and provenance

The current canonical literature curves are organised by source DOI and figure/series under `lab/data_all/`. Each accepted curve is stored as a CSV, with associated source notes and figure assets where available. The source time unit is retained, then converted to hours by the analysis pipeline.

`lab/05_accepted_all copy/curves_over_200h_full/` contains only CSV/JSON manifests. It is a compact index of selected figures rather than a second physical copy of the source assets.

`thesis/dataset/` contains the processed dataset used by the PvkSOM-derived notebook workflow:

- `PCE_df_grouping.csv` — grouping/metadata table;
- `pkl_complete/20230303_mySeriesDrop.pkl` — processed curves;
- `pkl_complete/20230303_mySeriesDropNorm.pkl` — normalised curves; and
- `pkl_complete/20230303_mySeriesDrop_savgol.npy` — smoothed curve representation.

Please consult and cite the underlying articles when reusing digitised literature data. This repository does not claim ownership of the original experimental measurements.

## Analysis workflow

### 1. Load and harmonise curves

The loader finds accepted CSV curves in the four `x_time_*` folders, records source metadata (DOI, figure, series, unit), and converts all time values to hours.

### 2. Quality control and preprocessing

For the primary run, a curve is retained when it has valid identifiers, time and PCE values; at least 10 unique points within 0–200 h; a positive PCE maximum; and no NaN/Inf values after interpolation. The pipeline then:

- resamples on a 10-minute grid (1,201 points from 0 to 200 h);
- interpolates with Akima interpolation and forward-fills only within the selected window;
- applies a Savitzky–Golay filter (window 71, polynomial order 2); and
- normalises each curve by its own maximum PCE in the 0–200 h window.

By default, trajectories that do not reach 200 h are excluded. The `--include-short` option retains them by forward-filling the tail; use that option carefully because it is not a physical extrapolation.

### 3. SOM clustering and validation

The main SOM uses Euclidean distance, a Gaussian neighbourhood, `sigma=0.5`, learning rate `0.1`, a fixed random seed of `42`, and 50,000 training iterations. The pipeline additionally provides:

- a quantisation-error sweep for 2–10 nodes;
- two hyperparameter-sensitivity variants;
- k-means comparisons for *k*=2–10; and
- automatic checks of counts, matrix shape, normalisation, assignments, and empty nodes.

For non-four-node SOMs, the current sweep uses a 1 × *n* topology; this is an implementation choice rather than a confirmed setting from the upstream notebook.

## Current 200-hour results

The current complete report is [available here](<lab/results_all/A_result/outputs/200h_som/analysis_report.md>). The primary 2 × 2 SOM identifies the following mean trajectory groups:

| SOM node | Interpretation of mean shape | Curves | Mean normalised PCE at 200 h |
|---|---|---:|---:|
| (0, 0) | Stable with gradual late loss | 860 (59.2%) | 0.923 |
| (0, 1) | Moderate, steady degradation | 157 (10.8%) | 0.506 |
| (1, 0) | Moderate, steady degradation | 371 (25.5%) | 0.746 |
| (1, 1) | Initial drop followed by rapid, decelerating loss | 65 (4.5%) | 0.259 |

The SOM-versus-k-means (*k*=4) comparison gives ARI = 0.9004 and NMI = 0.8677 for this run. See the [cluster assignments](<lab/results_all/A_result/outputs/200h_som/cluster_assignments.csv>), [cluster summary](<lab/results_all/A_result/outputs/200h_som/cluster_summary.csv>), [quality-control report](<lab/results_all/A_result/outputs/200h_som/quality_control_report.csv>), and [run configuration](<lab/results_all/A_result/outputs/200h_som/run_config.json>) for the auditable record.

Key visual outputs include the [cluster curves](<lab/results_all/A_result/outputs/200h_som/figures/som_cluster_curves.png>), [U-matrix](<lab/results_all/A_result/outputs/200h_som/figures/som_u_matrix.png>), [hit map](<lab/results_all/A_result/outputs/200h_som/figures/som_hit_map.png>), and [quantisation-error elbow](<lab/results_all/A_result/outputs/200h_som/figures/quantisation_error_elbow.png>).

## Run the primary pipeline

### Requirements

The original PvkSOM environment is provided in `thesis/environment.yml`. The structured 200-hour pipeline requires Python plus NumPy, pandas, SciPy, scikit-learn, MiniSom, Matplotlib, and the packages imported by the modules in `lab/results_all/A_result/src/`.

```bash
git clone https://github.com/ShunHao0626/ml-degradation-analysis.git
cd ml-degradation-analysis

conda env create -f thesis/environment.yml
conda activate PvkSOM
```

### Configure paths

Before running, open `lab/results_all/A_result/src/config.py` and update `OUT_ROOT` and the `DATA_DIRS` entries for your clone. They currently contain the original macOS absolute paths. In the current repository layout, point the four `DATA_DIRS` entries to `lab/data_all/x_time_h`, `lab/data_all/x_time_day`, `lab/data_all/x_time_min`, and `lab/data_all/x_time_week_month_year`.

The earlier pipeline in `lab/RESULTS_>200h/200h_som/` similarly uses a manifest under `lab/05_accepted_all copy/`; when running it against the migrated local layout, set its `DATASET_ROOT` to `lab/data_all` while retaining the manifest path.

### Execute

```bash
cd lab/results_all/A_result
python scripts/run_200h_som.py

# Optional: retain curves shorter than 200 h by forward-filling their tails
python scripts/run_200h_som.py --include-short
```

Outputs are written to `lab/results_all/A_result/outputs/200h_som/`. The run produces a reproducibility report, CSV tables, PNG/PDF figures, serialised models, processed arrays, and a log file.

To run the included tests:

```bash
cd lab/results_all/A_result
python -m pytest tests
```

## Important limitations

- SOM clusters are shape-based, unsupervised groupings. They should not automatically be interpreted as unique physical failure mechanisms.
- Curves come from different publications and experimental conditions; metadata coverage and measurement protocols vary.
- The four-node layout is a pre-specified main configuration, not a demonstrated global optimum.
- The k-means comparison uses Euclidean distance. It is an auxiliary validation and differs from the DTW-based comparison described in the upstream work.
- Some parameters (including the Savitzky–Golay edge mode and non-four-node sweep topology) are documented implementation assumptions.

## Relationship to upstream work

This repository includes/adapts materials from **PvkSOM**, the codebase accompanying:

> Hartono, N. T. P. *et al.* “Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset.” *Nature Communications* **14**, 4869 (2023). https://doi.org/10.1038/s41467-023-40585-3

Upstream repository: [noortitan/PvkSOM](https://github.com/noortitan/PvkSOM) · Dataset release: https://doi.org/10.5281/zenodo.8185882

Please acknowledge both the upstream work and the original source studies when building upon this project.

## License and reuse

The upstream `thesis/` materials include a BSD 2-Clause license. Review that license and the terms of the underlying publications before redistributing code or data. No additional repository-wide licence is declared at the root level.

## Contributing

Contributions are welcome, especially improvements to data provenance, portable configuration, metadata integration, preprocessing validation, and physically informed interpretation. Please open an issue or pull request with a concise description, and do not commit copyrighted source figures or data without permission.
