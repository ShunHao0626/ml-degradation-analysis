# ML degradation analysis

Unsupervised shape analysis of perovskite solar-cell (PSC) ageing curves: power-conversion efficiency (PCE) versus time. The project reproduces the Self-Organizing Map (SOM) workflow of Hartono *et al.* (2023, PvkSOM) and then asks, with a series of independent experiments, which degradation-curve shapes emerge from literature-digitised curves without supervision, and whether four target morphologies (IFO-Bridge / Hill / Slope / Valley) form natural clusters.

**No dataset is published in this repository.** See [Data policy](#data-policy).

## Contents

- [Research question](#research-question)
- [Data policy](#data-policy)
- [Repository map](#repository-map)
- [Experiment index](#experiment-index)
- [Current findings](#current-findings)
- [Reproducing an experiment](#reproducing-an-experiment)
- [Provenance and attribution](#provenance-and-attribution)

## Research question

1. Applied to PCE–time curves digitised from published figures, does the PvkSOM workflow (10-minute resampling, per-curve max normalisation, Savitzky–Golay smoothing, MiniSom SOM, quantisation-error elbow for the cluster count) give a stable partition, and what do its clusters look like?
2. Do the four target morphologies below appear as separate, stable clusters under fully unsupervised training, or only as individual curves?

| Target morphology | Description (first 200 h, then later) |
|---|---|
| IFO-Bridge | rapid rise, then slow decay |
| IFO-Hill | rapid rise, rapid fall, then slow decay |
| IFO-Slope | rapid fall, then slow decay |
| IFO-Valley | rapid fall, recovery, then slow decay |

These names come from internal project material; the literature searches in `lab-v2/literature_k_som_search/08_recent_literature_review/` and `pce_hour_curve_taxonomy/08_literature_review/` did not find them as established published terms, and the expansion of "IFO" is not defined by any primary source.

## Data policy

Datasets are kept local and are excluded by `.gitignore`. The repository contains code, configuration, run logs, reports, summary tables (cluster assignments, metrics, centroids) and cluster-level figures only. Excluded in every form, including copies produced inside experiment folders:

| Local path (not in git) | Content |
|---|---|
| `lab/data_all/` | 2,151 PCE–time curves digitised from 1,373 publications (DOI / figure / series), with the source figures, axis-validation JSON and paper full texts. Organised by source time unit: `x_time_h/`, `x_time_day/`, `x_time_min/`, `x_time_week_month_year/`. |
| `lab-v2/som_references/accepted/samples_test/` (and identical copies in `DTW/samples_test/`, `HDB-scan/samples_test/`) | 218-curve test subset (99 figures). |
| `thesis/dataset/` | Hartono *et al.* processed dataset (2,245 curves), Zenodo record [10.5281/zenodo.8185882](https://doi.org/10.5281/zenodo.8185882). |
| any `accepted/`, `raw_curves/`, `raw/`, `aligned/`, `*_input/`, `data/` folder inside an experiment; `preprocessed_*curves.csv`, `merged_all_curves.csv`, `included_samples.csv`, `*_samples.csv`; all `*.npy` / `*.npz` / `*.pkl`; interactive `*.html` | per-curve copies, resampled matrices, serialised curves and trained models |
| `*.pdf`, `*.xlsx`, paper full texts, rendered pages, `oa_pdfs/`, `open_access_papers/`, `fulltext_*` | third-party publications gathered during literature review |

Cluster assignment tables keep the DOI, figure and series identifiers of each curve so that results can be traced to the original publications without redistributing the digitised data.

Rerunning an experiment therefore requires placing the data at the paths above. Recorded absolute paths in `run_config.json`, `environment*.json` and `*.log` files refer to the machine on which the run was made.

## Repository map

Directory names are the working names used during the project and are kept unchanged.

```text
.
├── thesis/                                   # PvkSOM fork (Hartono et al. 2023): notebooks, reproduction, preprocessing comparison
├── lab/                                      # literature-curve database and the 200 h SOM reanalysis line
│   ├── data_all/                             # dataset (local only)
│   ├── 05_accepted_all copy/                 # legacy curve index and over/under-200 h reports
│   ├── result/                               # 200 h SOM, first version (Jul 2026)
│   ├── results_all/A_result/                 # 200 h SOM, structured pipeline with tests (Jul 2026)
│   ├── RESULTS_>200h/200h_som/               # 200 h SOM, n = 2..16 with and without smoothing (Jul 2026)
│   └── NEW/som_200h_reanalysis/              # canonical 200 h reanalysis + shape-sensitive and 36-node sub-studies
├── lab-v2/                                   # IFO four-shape hypothesis (Aug–Sep 2026)
│   ├── som_references/                       # 218-curve subset (local only), reference plots and slides
│   ├── som_200h_paper_exact/
│   ├── ifo_four_shape_discovery/
│   ├── ifo_four_shape_data_all_discovery/
│   ├── ifo_synthetic_four_class_paper_som/
│   ├── pure_unsupervised_paper_som_window_search/
│   ├── ifo_four_type_matching_curves_export/
│   ├── literature_k_som_search/
│   └── data_all_ifo_shape_groups/
├── pce_hour_curve_taxonomy/                  # literature-whitelisted parameters + literature review (Sep 2026)
├── pce_curve_pattern_discovery/              # paper parameters, K frozen before IFO check (Sep 2026)
├── HDB-scan/                                 # change-point features + HDBSCAN, no smoothing (Sep 2026)
├── DTW/                                      # derivative-aware DTW + hierarchical clustering, no smoothing (Sep 2026)
├── curve_discovery_unsupervised_20260906_01/ # variable-length curves, no smoothing (Sep 2026)
├── pce_som_no_smoothing_20260907_01/         # paper SOM without smoothing, 500 h window (Sep 2026)
├── pce_ifo_unsupervised_detail_search_20260907_01/ # large label-free SOM search, no smoothing (Sep 2026)
├── environment.yml                           # conda environment used for most runs (PvkSOM)
└── README.md
```

Every experiment folder is self-contained: a `README*.md` or `REPORT*.md`, numbered stage folders (`00_…`, `01_…`), `run_config.json`, a run log, `checksums.sha256` and a `verify_*.py` script that checks the archived outputs without retraining. Several folders vendor MiniSom 2.2.9 (`vendor/`) to match the paper's version.

## Experiment index

Chronological. "Input" is the local dataset used; counts are curves entering the final model.

| # | Directory | Period | Question | Input | Method | Outcome | Entry point |
|---|---|---|---|---|---|---|---|
| 1 | `thesis/` (`run_pipeline.py`, `result/`, `20230816_run_revision_excN2/`) | Jun–Jul 2026 | Reproduce the published 150 h workflow | Hartono 2,245 | 2×2 SOM, k-means comparison | Paper workflow reproduced | `thesis/README.md` |
| 2 | `lab/result/` | Jul 2026 | First 200 h SOM on the literature curves | data_all, 1,667 | 2×2 SOM | Superseded by #5 | `lab/result/SOM_200h_pipeline.py` |
| 3 | `lab/results_all/A_result/` | Jul 2026 | Structured, tested pipeline | data_all, 1,453 | 2×2 SOM, QE sweep, sensitivity, k-means | 4 nodes; ARI vs k-means 0.90 | `outputs/200h_som/analysis_report.md` |
| 4 | `lab/RESULTS_>200h/200h_som/` | Jul 2026 | Larger node counts, smoothing on/off | data_all, 1,812 | SOM n = 2..16 | Superseded by #5 | `scripts/run_200h_som.py` |
| 5 | `lab/NEW/som_200h_reanalysis/` | Jul–Aug 2026 | Canonical 200 h reanalysis with the SI cluster-count rule | data_all, 1,442 (≥10 points) + 1,785 sensitivity | SOM n = 2..10, seeds, centroid-overlap rule | n = 4: stable 57.6 %, moderate loss 27.3 %, rapid loss 11.0 %, initial drop + rapid loss 4.1 %; k-means ARI 0.97 | `REPORT.md`, `FILE_INDEX.md` |
| 5a | `…/10_shape_sensitive_reanalysis/` | Aug 2026 | Shape-sensitive representations | same | derivative / time-weighted features + SOM | Sub-study of #5 | `REPORT_CN.md` |
| 5b | `…/11_paper200h_tuned_rise_fall_som/` | Aug 2026 | Can a larger SOM isolate rise-then-fall curves? | 1,442 | 6×6 SOM, 99 configurations × 5 seeds, post-hoc RTF diagnostic | One node: precision 88.9 %, recall 14.3 % | `README_REPRODUCE_CN.md` |
| 6 | `lab-v2/som_200h_paper_exact/` | Aug 2026 | Paper parameters, 200 h window, 218 subset | samples_test, 114 | 2×2 SOM (MiniSom 2.2.9) | Nodes: 3 / 54 / 22 / 35; DTW k-means ARI 0.81 | `REPORT_CN.md` |
| 7 | `lab-v2/ifo_four_shape_discovery/` | Aug 2026 | Find the four IFO shapes in the 218 subset | samples_test, 130 | SOM parameter search, then phase-aware rules | SOM alone does not separate four shapes; strict rules: 61 curves | `REPORT_CN.md` |
| 8 | `lab-v2/ifo_four_shape_data_all_discovery/` | Aug 2026 | Same on the full database | data_all, 1,635 | as #7, windows 200–1000 h | Strict rules: 1,003 curves, 226 unresolved | `REPORT_CN.md` |
| 9 | `lab-v2/ifo_synthetic_four_class_paper_som/` | Aug 2026 | Can the paper SOM recover four clean shapes at all? | synthetic | 2×2 SOM on balanced synthetic classes | 100 % recovery on synthetic data | `REPORT_CN.md` |
| 10 | `lab-v2/pure_unsupervised_paper_som_window_search/` | Aug 2026 | Tune only variables the paper exposes | samples_test | 54 candidates (window, sigma, learning rate) | 300 h selected; clusters still ordered by decay strength | `REPORT_CN.md` |
| 11 | `lab-v2/ifo_four_type_matching_curves_export/` | Aug 2026 | Export strict rule matches | samples_test | rule matching | Bridge 3, Hill 3, Slope 51, Valley 4 | `README_CN.md` |
| 12 | `lab-v2/literature_k_som_search/` | Sep 2026 | Literature-backed cluster-count rule; where do the IFO names come from? | samples_test | SOM n = 2..10, three paper (sigma, lr) pairs; OpenAlex/Crossref review | K = 4 frozen before IFO check; names not found in literature | `REPORT_CN.md`, `K_SELECTION_DECISION_CN.md` |
| 13 | `pce_hour_curve_taxonomy/` | Sep 2026 | Parameter whitelist with evidence registry, 150/300/500 h | samples_test, 103 / 94 / 86 | SOM k = 2..10, n = 16 diagnostic | k = 5; only Slope-like forms a stable cluster | `reports/final_report.md` |
| 14 | `pce_curve_pattern_discovery/` | Sep 2026 | 27 paper-parameter candidates, K frozen before IFO check | samples_test, 109 | SOM, QE elbow | K = 4; Bridge / Hill / Valley not natural clusters | `REPORT_CN.md` |
| 15 | `lab-v2/data_all_ifo_shape_groups/` | Sep 2026 | Rule-based grouping of all curves (not clustering) | data_all, 2,151 | phase-aware rules, 500 h | Clear: Bridge 23, Hill 21, Slope 433, Valley 10; 516 boundary | `README_CN.md` |
| 16 | `HDB-scan/change_point_hdbscan_20260905/` | Sep 2026 | Label-free alternative without smoothing | samples_test, 79 (300 h) | change points + kinetic descriptors + HDBSCAN | 2 clusters + 8 noise; bootstrap ARI 0.52, not stable | `研究结果与方法报告.md` |
| 17 | `DTW/` (`derivative_dtw_results_20260906/` = fit, `result_derivative_dtw/` = export) | Sep 2026 | Derivative-aware DTW distance | samples_test, 91 | multivariate DTW + average-linkage hierarchical | K = 2 (84 / 7) | `result_derivative_dtw/README.md` |
| 18 | `curve_discovery_unsupervised_20260906_01/` | Sep 2026 | Keep variable-length curves, no smoothing | samples_test, 218 | SOM on relative-progress axis, QE elbow | 4 main clusters (46 / 23 / 89 / 60); IFO shapes exist as individuals only | `README.md` |
| 19 | `pce_som_no_smoothing_20260907_01/` | Sep 2026 | Paper SOM with smoothing disabled, 500 h primary | data_all | SOM, QE elbow, 10 seeds | K = 4 (371 / 586 / 56 / 189), seed ARI 1.0; all centres Slope-like | `FINAL_DECISION_CN.md` |
| 20 | `pce_ifo_unsupervised_detail_search_20260907_01/` | Sep 2026 | Exhaustive label-free search over windows, representations and SOM parameters | data_all, 1,590 (300 h) | 4,032 grid + 9,072 extended SOM runs | K = 4 auto-selected; no run yields four IFO nodes at once, at most three | `FINAL_DECISION_CN.md` |
| 21 | `thesis/som_preprocessing_comparison/` (`run_som_preprocessing_comparison.py`) | Sep 2026 | Effect of preprocessing level on the Hartono data | Hartono 2,245 | raw / resample+normalise / +smooth → SOM | Natural K: 4 / 5 / 5 | `final_analysis.md` |

## Current findings

- On the literature-digitised curves, unsupervised clustering repeatedly selects about four clusters, but they are ordered by decay strength and speed (stable, moderate, rapid, initial-drop-then-rapid), not by the four target topologies.
- Curves with Bridge, Hill, Slope and Valley appearance exist and can be picked out by explicit rules, but they do not form four separate, seed-stable clusters under SOM, HDBSCAN or DTW-based clustering, with or without smoothing.
- The paper's SOM does recover four classes on synthetic data built from those four shapes (#9), so the negative result is a property of the data, not of the method's capacity.
- Selecting a run because it "looks like four classes" would turn the target shapes into a model-selection signal; the reports therefore separate unsupervised training from any post-hoc morphology assessment.

All results are descriptive clustering outcomes on heterogeneous, figure-digitised data; they are not evidence of distinct physical degradation mechanisms.

## Reproducing an experiment

Most runs used the conda environment `PvkSOM` (Python 3.9, NumPy 1.26, SciPy 1.13, pandas 1.5, scikit-learn 1.6, Matplotlib 3.9; MiniSom 2.2.9 vendored where the paper version matters). `environment.yml` lists it. `HDB-scan/…/requirements-lock.txt` and `curve_discovery_unsupervised_20260906_01/requirements.txt` record the environments of the runs that used the system Python 3.9 instead.

```bash
conda env create -f environment.yml
conda activate PvkSOM

# typical experiment folder
cd lab-v2/ifo_four_shape_data_all_discovery
MPLCONFIGDIR=/tmp/mpl python run_pipeline.py     # retrain (needs the local dataset)
python verify_outputs.py                         # check archived outputs only
```

Each folder's own README/REPORT gives the exact commands. Verification scripts that compare against the local dataset (`verify_outputs.py` in #8 and #16, `verify_results.py` in #17) need the data present at the paths in [Data policy](#data-policy).

## Provenance and attribution

- Method and reference dataset: N. T. P. Hartono, H. Köbler, P. Graniero, M. Khenkin, R. Schlatmann, C. Ulbrich, A. Abate, "Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset", *Nat. Commun.* 14, 4869 (2023), [10.1038/s41467-023-40585-3](https://doi.org/10.1038/s41467-023-40585-3). Code: PvkSOM, BSD 2-Clause (`thesis/LICENSE`).
- SOM implementation: MiniSom (G. Vettigli), version 2.2.9 vendored in several experiments.
- Literature curves were digitised from published figures for analysis only. Every cluster table carries the source DOI, figure and series; consult and cite the original articles when reusing any result.
- Literature-review metadata came from OpenAlex and Crossref; screening tables are included, raw API dumps and downloaded papers are not.
