# ML degradation analysis

English · [中文](README_CN.md)

This repository studies power conversion efficiency (PCE) versus time in perovskite solar cells. It contains the upstream curve work, SOM and shape analyses, method comparisons, and historical experiments. Raw curves, third-party papers, and per-curve images used by these projects are normally kept outside Git.

## Start here

| Task | Entry point | Scope |
|---|---|---|
| Current 0–200 h shape analysis | [`ML_SOM/target_shape_classification/`](ML_SOM/target_shape_classification/README.md) | Input audit, SOM, shape interpretation, source review, and tests |
| Fixed 1,842-curve four-cluster reproduction | [Perovskite early degradation clusters](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) | Fixed inputs, density-weighted K-means, verification, and full results; maintained in its own repository |
| Build the 2,246-curve canonical dataset | [`ML_SOM/canonical_2246_full_dataset_20260925/`](ML_SOM/canonical_2246_full_dataset_20260925/README.md) | Deduplication, provenance, and integrity checks; generated data stay local |
| Reproduce PFD–DTW | [`derivative_clustering/`](derivative_clustering/README.md) | Derivative features, DTW, hierarchical clustering, and tests |
| Reproduce the Hartono PvkSOM workflow | [`thesis/`](thesis/README.md) | Original paper workflow and preprocessing comparison |
| Other derivative DTW experiments | [`DTW/`](DTW/result_derivative_dtw/code/REPRODUCE.md) | Earlier independent experiments and exports |
| Historical SOM, HDBSCAN, and parameter searches | [`by-products/`](by-products/README.md) | Research archive indexed by question |

`environment.yml` describes the earlier PvkSOM environment. For newer projects, use the environment and commands documented in each project directory. The [code map](ML_SOM/target_shape_classification/CODE_MAP.md) shows the dependencies within the current 0–200 h analysis.

## Repository boundary

This repository owns upstream curve curation, the broader 0–200 h SOM and morphology study, derivative/DTW comparisons, and the historical research record. The [separate early degradation repository](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) is the canonical public source for fitting four clusters to the **fixed set of 1,842 curves**. Its input selection came from an earlier exploration here, so that upstream work remains for provenance. The duplicate reproduction package is no longer maintained in this repository.

## Interpretation

The broader question is whether Bridge, Hill, Slope, and Valley patterns form distinct groups without labels. Historical SOM, HDBSCAN, and DTW analyses often divide the heterogeneous literature curves by degradation magnitude and speed. A cluster ID is not a validated pattern label. The separate 1,842-curve analysis uses density-weighted K-means, **not SOM**; earlier provisional shape candidates influenced its input selection, so its four groups are exploratory.

## Data and publication policy

This repository publishes code, configuration, methods, and selected aggregate tables and figures. Local-only material includes `ML_SOM/data_final/`, `ML_SOM/original_curves_2250/`, generated canonical datasets, historical experiment inputs, paper full texts, and per-curve galleries. The fixed 1,842-curve input and complete reproduction results are owned by the separate repository. Identifiers such as DOI, figure, and curve ID in summaries support provenance; they do not imply that the original files are present here.

Unique historical SI text, figures, and per-curve plotting values are retained locally at `ML_SOM/si_1842_supplementary/` and are excluded from this Git repository. Its historical cluster numbering may differ from the separate repository's display numbering.

## Attribution

The PvkSOM method and reference dataset come from Hartono et al., *Nature Communications* 14, 4869 (2023), [DOI: 10.1038/s41467-023-40585-3](https://doi.org/10.1038/s41467-023-40585-3). See `thesis/LICENSE` for the corresponding code license. Consult and cite the source articles for literature-digitized curves.
