# Historical experiments

English · [中文](README_CN.md)

This directory archives exploratory projects formerly at the repository root. Their internal structure, configuration, reports, and aggregate outputs remain as research records. Start new work from the [main README](../README.md). Raw curves, paper full texts, models, and large intermediate matrices remain local.

| Area | Directory | Main question |
|---|---|---|
| Literature curves and 200 h SOM | [`lab/`](lab/) | Initial through canonical 200 h reanalyses |
| Four-shape SOM search | [`lab-v2/`](lab-v2/) | Unsupervised searches on a 218-curve subset and the full literature collection |
| Literature parameters and cluster count | [`pce_hour_curve_taxonomy/`](pce_hour_curve_taxonomy/), [`pce_curve_pattern_discovery/`](pce_curve_pattern_discovery/) | Parameter evidence, K selection, and robustness |
| Change points and density clustering | [`HDB-scan/`](HDB-scan/) | HDBSCAN comparison without smoothing |
| Variable-length curves | [`curve_discovery_unsupervised_20260906_01/`](curve_discovery_unsupervised_20260906_01/) | Unsupervised analysis on a relative-progress axis |
| No-smoothing and broad searches | [`pce_som_no_smoothing_20260907_01/`](pce_som_no_smoothing_20260907_01/), [`pce_ifo_unsupervised_detail_search_20260907_01/`](pce_ifo_unsupervised_detail_search_20260907_01/) | 500 h baseline and window/representation sensitivity |

Many archived scripts assumed these directories were directly under the repository root. After the move, scripts with hard-coded relative paths may require their original layout or path adjustments; the archived results themselves are unchanged.
