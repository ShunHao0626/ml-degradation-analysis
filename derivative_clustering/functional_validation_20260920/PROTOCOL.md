# Frozen functional validation protocol

Written before running this experiment. This is an extension of the existing
development experiment, not an unbiased benchmark chosen without prior work.
No parameters will be selected using the new batches' class labels.

## Question

Can derivative representations help recover the user's IFO Bridge, Hill, Slope,
Valley shapes, and can they be integrated into SOM? This tests software and
synthetic shape recovery, not real-data generalization or physical mechanisms.

## Sources

- `md/SOM.pdf`, pp. 4–5: initial gain and slow/medium/fast exponential decay;
  these are DIFFERENT from the user's IFO target taxonomy. Main analysis uses
  150 h, 10 min resampling, Akima, per-curve max normalization, SG window 71;
  SOM sigma 0.5, learning rate 0.1. SI pp. 16–17 discusses QE and 4–6 clusters.
- `md/导数聚类-1-s2.0-S0020025524008533-main.pdf`, pp. 3, 5–6, 10:
  HP lambda 1000, global polynomial chosen by R², D1/D2/D3, weighted DTW, HC.
  Degree ceiling, numerical weights and linkage are not specified.
- User's attached IFO diagram supplies the four desired shape definitions;
  `eg` supplies qualitative variability, not quantitative population estimates.

## Fixed choices

- Normalize each curve by its observed maximum absolute value. Keep physical
  hours on disk; comparison coordinate u=(t-t0)/(t_last-t0).
- Use existing preprocessing: local Hampel threshold 5, PCHIP to 48 points,
  SG window 7 / degree 3. Derivatives are with respect to u, NOT hour.
- Robust channel scaling across each unlabeled batch, as in prior development.
- SOM: 2x2, sigma 0.5, learning rate 0.1, random sample initialization,
  10,000 random updates, seeds 11/22/33, arithmetic prototypes and Euclidean
  feature distance. One model uses level only; another concatenates
  sqrt(0.5)*level, sqrt(0.375)*D1, sqrt(0.125)*D2 after scaling.
  Report all seeds, not the best. This is a SOM adaptation, not a literal
  reproduction of the 150 h source study.
- DTW: radius 6, minimum cumulative local weighted Euclidean cost, divided by
  length of the selected path. This does NOT optimize mean cost directly.
  Compare level, D1 only, D1+D2, shared level+D1+D2, separate 50:50 fusion.
  Graph: self-tuned symmetric 10-neighbor affinity, spectral cluster_qr, k=4.
- Paper-style comparator: PCHIP raw max-normalized level to 48 points, HP=1000,
  degree chosen by largest R² among 1..20, derivatives in sample-index units,
  weights 3:2:1, unrestricted cumulative DTW, average and single linkage.
  Assumptions are explicit; this cannot be called the authors' exact setup.
- Known k=4 is a functional-test condition, NOT evidence that unknown real
  data has four natural clusters. Inspect k=2..8 silhouettes without using
  class labels; do not change the main k on the basis of these results.

## Batches

1. Existing 400 CSVs (development, already inspected previously).
2. Same-family generator, new seeds 31001/31002/31003, 60 per class (240/batch).
3. Independent piecewise PCHIP anchor generator, same three seed numbers but
   independently specified functions, 60 per class, 15–80 observations.
4. Independent generator with only eight retained observed points per curve.
5. Independent generator observed only through 55% of its original duration;
   retained labels describe the full latent curve. This is deliberately an
   identifiability stress test, not an expectation that hidden stages can be
   reconstructed.
6. Independent 240 targets plus 120 non-target curves (flat, increasing,
   repeated peaks, abrupt failure), main k=4 plus k=8 diagnostic. No claim of
   open-set recognition; quantify background contamination after matching.

Duration strata matched across target classes: 10% 0.5–2 h, 35% 8–80 h,
35% 120–1000 h, 20% 1200–4500 h. Metadata/labels/outlier flags never enter the
fitting API. Randomly permute each batch and assign opaque IDs. Persist actual
observations, seed, generator parameters and input/code hashes. Do not reject
curves based on target-shape QC.

## Evaluation

ARI, Hungarian-matched accuracy and macro recall, per-class recall, duration
stratum ARI; SOM repeat-seed mean and range; independent-batch mean and range.
Hungarian naming is post-hoc evaluation, not a deployable name classifier.
For mixed batches, match clusters using target rows, then count ALL background
rows in those clusters when calculating target precision. k=8 may fragment a
class: report target-only ARI as well as recall of one matched cluster/class.
Inspect representative raw observations and clustered curves. Verify compiled
DTW against existing Python implementations, derivative invariance to physical
time scaling, duplicate-time handling, rejection of invalid data, and exact
invariance to metadata columns. Run existing mathematical tests as well.

## Success and interpretation

The deliverable is a runnable pipeline, independently regenerated tests and an
honest comparison answering whether derivatives help and under which limits.
Perfect accuracy and a positive derivative effect are NOT completion gates.
No future-real-data performance claim can be verified before that data exists.
