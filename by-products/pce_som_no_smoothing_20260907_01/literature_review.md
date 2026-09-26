# Method literature and decision record

## Anchor paper

Hartono et al., *Stability follows efficiency based on the analysis of a large
perovskite solar cells ageing dataset*, Nature Communications 14, 4869 (2023),
DOI: 10.1038/s41467-023-40585-3.

Primary method: per-curve normalized PCE trajectories, MiniSom, main parameters
`sigma=0.5`, `learning_rate=0.1`, 50,000 updates. The supplementary material
defines the optimum as the **minimum** number of clusters that captures the
distinct main shapes: use the quantization-error elbow, reject too-small K when
one centre still contains separable shapes, and reject too-large K when centres
start to overlap/become indistinguishable. The paper checks K-means as a second
algorithm and reports sensitivity at `(0.3,0.1)` and `(0.5,0.3)`.

- Article: https://www.nature.com/articles/s41467-023-40585-3
- Archived local text: `../thesis/paper/work.md`
- Archived local supplement: `../thesis/paper/Supplementary.md`

## Candidate improvements reviewed

1. Paparrizos & Gravano, *k-Shape: Efficient and Accurate Clustering of Time
   Series*, SIGMOD 2015, DOI 10.1145/2723372.2737793. It uses normalized
   cross-correlation and is intentionally invariant to amplitude and phase.
   This is useful for many domains, but phase invariance can erase whether a
   peak/trough occurs before or after the requested 200 h boundary. It is not
   used for the final decision.
   https://www.cs.columbia.edu/~gravano/Papers/2015/sigmod2015.pdf

2. Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series*,
   ICML 2017. DTW handles temporal shifts and different speeds; soft-DTW makes
   averaging/clustering differentiable. Unrestricted time warping can align an
   early event with a late event and weaken the physical meaning of elapsed
   hours. It is retained as a future constrained-distance option, not used to
   choose K here.
   https://proceedings.mlr.press/v70/cuturi17a.html

3. Blondel, Mensch & Vert, *Differentiable Divergences Between Time Series*,
   AISTATS 2021. It corrects the entropic bias of soft-DTW. This is relevant if
   a future analysis replaces Euclidean SOM distance with a time-warped
   divergence.
   https://proceedings.mlr.press/v130/blondel21a.html

4. Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI
   2022, DOI 10.1609/AAAI.V36I8.20881. It learns multi-scale representations
   without labels. The present curves are sparsely digitized from figures
   (rather than dense homogeneous MPPT streams), so augmentation assumptions
   and representation capacity would add more uncontrolled choices than the
   paper-based SOM. It is not used in the final model.
   https://ojs.aaai.org/index.php/AAAI/article/view/20881

5. Duan et al., *MF-CLR: Multi-Frequency Contrastive Learning Representation
   for Time Series*, ICML 2024. It addresses multi-frequency channels, while
   the current task is a single PCE channel converted to elapsed hours. It is
   reviewed as a frontier method but does not match the present observation
   structure.
   https://proceedings.mlr.press/v235/duan24b.html

## Implemented optimization

The final pipeline stays within the anchor paper's model family and improves
the evidence around it:

- paper parameters plus wider-neighbourhood candidates;
- sequential order (paper-compatible) and randomized update order;
- five independent initializations for screening and ten for the final fit;
- QE elbow, centre separation/overlap, occupied-node checks, silhouette, and
  adjusted-Rand stability;
- a K-means elbow and agreement check, as in the anchor paper;
- frozen-model post-hoc morphology audit;
- 150 h sensitivity to distinguish conclusions caused by the longer 500 h
  window.

No external label or target IFO template enters any score.

## Implemented shape-sensitive sensitivity analysis

The k-Shape review motivated a narrower change that does not introduce phase
invariance: each unsmoothed trajectory is z-normalized, then the same SOM and
the same K-selection protocol are rerun. No cross-correlation, time warping,
target curve, or class label is used. This analysis independently selected
K=4. Its frozen centres were two Slope-like, one Bridge-like, and one
Valley-like; it did not produce an independent Hill-like centre. Its silhouette
when assignments are evaluated back in the common MaxAbs trajectory space was
-0.1643, versus 0.5054 for the primary paper representation. It is therefore a
sensitivity result, not the selected primary model.
