# Code map for the 0–200 h analysis

English · [中文](CODE_MAP_CN.md)

This directory contains several research rounds. The table describes the current analysis path. Commands run from `ML_SOM/` and require the local `data_final/` directory. Public Git does not contain the curve data or source-review records in `results/` and `review_inputs/`.

| Stage | Main modules | Input → output |
|---|---|---|
| Source audit and canonical curves | `run.py` | Original CSVs, axis metadata, and hash-checked reviews → manifests, canonical curves, and morphology comparisons |
| SOM features and training | `unsupervised_som.py`, `som_evidence.py` | Canonical curves → features retaining observed points, anonymous neurons, stability, and post-training interpretation |
| Fixed 0–200 h classification | `classify_200h.py` | Observations within the window and SOM candidates → file-level classifications, Valley boundary review candidates, and exclusion reasons |
| Source review | `redigitize.py`, `audit_review_workload.py`, `delivery_coverage.py` | Confirmed source figures and versioned redigitization → review tasks and traceable coverage tables |
| Sensitivity and diagnostics | `input_version_ablation.py`, `recovery_ablation.py`, `hill_heldout_diagnostic.py`, etc. | Frozen inputs/results → representation, recovery-stage, and source-holdout diagnostics |
| Earlier 200 h exploration | `research_early_200h/` | Separate configurations and scripts kept for provenance |

`run.py` supplies input loading, axis conversion, normalization, and morphology functions used by later modules. `classify_200h.py` also imports `unsupervised_som.py`. Source review labels support interpretation and auditing after the unsupervised fit; they are not ground truth for the SOM.

The [separate early degradation repository](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) owns the final fixed 1,842-curve four-cluster reproduction. `research_early_200h/` here retains its upstream input-curation and exploration history.

## Minimal test

```bash
cd ML_SOM
python3 -m unittest discover -s target_shape_classification -p 'test_*.py' -q
```

See the directory [README](README.md) for full-run commands and data requirements. A new run without the local axis overrides and hash-checked source-review inputs is not directly comparable to the archived research counts.
