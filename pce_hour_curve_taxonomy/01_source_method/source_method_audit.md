# Source-method audit

## Authoritative workflow

The authoritative source is Hartono et al., *Nature Communications* 14, 4869
(2023), DOI `10.1038/s41467-023-40585-3`, together with its Supplementary
Information and released notebook in `thesis`.

Published preprocessing is:

1. resample MPPT PCE at 10-minute intervals;
2. fill missing points with Akima interpolation;
3. retain a fixed ageing window (150 h in the main SOM analysis);
4. normalize each trajectory to its maximum with MaxAbsScaler;
5. smooth with Savitzky-Golay, window length 71 and polynomial order 2 in the
   released notebook;
6. train MiniSom for 50,000 iterations in the released notebook.

Only the normalized PCE trajectory is supplied to SOM. Device materials,
temperature, architecture, literature labels, legend names, and target IFO names
must not be SOM inputs.

## Cluster-count rule

The paper does not justify forcing four clusters. It evaluates quantization error
for `n=2..10`, identifies an elbow/candidate region, then inspects whether fewer
clusters miss distinct main shapes or extra clusters overlap. For the original
2,245-curve dataset, it considered `n=4..6` plausible and selected `n=4` because:

- at `n=2`, exponential-decay curves could still be split into distinct shapes;
- at `n=5`, two clusters began to overlap;
- at `n=6`, overlap became more pronounced.

For the new data, the same decision process must be rerun; the original answer of
four is not inherited.

## Permitted parameter configurations

The paper visually reports exactly three SOM hyperparameter configurations:

- baseline: `sigma=0.5`, `learning_rate=0.1`;
- sensitivity A: `sigma=0.3`, `learning_rate=0.1`;
- sensitivity B: `sigma=0.5`, `learning_rate=0.3`.

The undocumented pair `(0.3, 0.3)` is not admitted. Preprocessing method,
resampling interval, normalization, Savitzky-Golay order/window, and training
iterations stay fixed. The released notebook considered a smoothing window of
201 but explicitly selected 71; this project therefore does not tune the
smoothing window to seek desired shapes.

The main SOM window is 150 h. The Supplementary Information also reports 300 h
and 500 h cutoff-time analyses. Because the requested IFO archetypes distinguish
behaviour after 200 h, 300 h and 500 h are admitted only as documented
time-window sensitivity branches. No ad hoc 200 h window is introduced.

## Reproducibility control

The released notebook does not set a random seed. The local convenience script
sets `random_seed=42`, which changes reproducibility but not the data or feature
space. This project will pre-register seeds and report stability across them.
Seeds are never searched or selected based on resemblance to IFO targets.

## Local implementation differences and risks

- `thesis/run_pipeline.py` hard-codes a 2-by-2 grid and therefore cannot be used
  to decide the number of clusters independently.
- It writes inside `thesis` when run. It will remain unexecuted so that the source
  tree stays read-only.
- The local `thesis/result/README.md` contains interpretations and tuning ranges
  that are broader than the paper/notebook evidence and contains a statement
  about efficiency degrading faster that conflicts with the paper's stated
  finding. It is not used as authority for the whitelist.
- The original dataset was homogeneous, in-house MPPT data with dense sampling.
  The supplied dataset consists of sparse curves digitized from figures and has
  no DOI/figure provenance. Any result must explicitly carry this domain and
  traceability limitation.

## Post-clustering morphology assessment

IFO-Bridge, IFO-Hill, IFO-Slope, and IFO-Valley are not training labels. After a
model is selected independently, cluster centroids and all member curves will be
plotted over physical time. Morphology names may then be assigned as an
interpretive audit. A target match cannot retroactively make a configuration or
cluster count preferred.
