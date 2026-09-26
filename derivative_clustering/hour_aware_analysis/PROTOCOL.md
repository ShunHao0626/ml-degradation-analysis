# Hour-aware shape and kinetic analysis

Written before new experiments. User requirement: preserve meaningful time in
hours while identifying the same morphological families at different rates.

## Interpretation

Two different questions require two outputs: a shape family may be invariant to
time dilation; kinetic equivalence must not be. Default output is a hierarchy,
`shape_cluster` plus an observed-response-timescale subgroup and measured hour
parameters. This is not a claim that equal shape means equal lifetime.

The user confirmed both choices: same morphology with hour-based subgroups,
and unsupervised clustering followed by post-hoc semantic interpretation.
No predefined hour windows (including 200 h) are assumed.

## Unchanged shape branch

Use the previously tested max-normalized PCHIP/SG 48-point representation and
shared-path DTW weights level/D1/D2 = .5/.375/.125, radius 6, 10-neighbor graph,
four shape clusters for the synthetic functional experiment. This branch still
uses relative observed progress; it is NOT advertised as hour-sensitive.
Inputs containing other shapes can request more clusters. Cluster IDs remain
anonymous; generator labels never enter fitting.

## New physical-time branch

- Process the original irregular hour samples (not 48 normalized-time samples).
  Duplicate times use median response. Vertical scaling is per-curve observed
  maximum absolute response, explicitly retained in the output.
- Fit a quadratic local polynomial to seven nearest physical-time observations,
  using tricube weights and two Huber reweighting passes. Center and scale the
  regression locally for conditioning, then analytically convert derivatives
  back to `dy/dhour`, `d2y/dhour2`. No derivative is standardized separately for
  each curve in this physical output.
- Record first sustained absolute excursion of 0.05, 0.10 and 0.20 from the
  estimated starting level, in max-normalized response units. Record direction,
  interpolated hour, actual bracketing observation hours, and bracket width.
  The interpolation bracket is not a statistical confidence interval.
- Primary kinetics feature is elapsed hour to a 0.10 response excursion. This
  measures response onset, not a universal decay constant, T80 or full lifetime.
  Brackets wider than half the estimated elapsed crossing time are unresolved.
  No crossing observed is censored at the last observed elapsed hour; do not
  replace it by that endpoint. Flat/low-amplitude curves therefore need not
  receive a timescale subgroup.
- Within each inferred shape cluster, use BIC to choose 1..4 Gaussian mixture
  components on log10(t_excursion_0.10_h / 1 h), with at least eight members per
  component. Report actual median response hours and order subgroups from short
  to long timescale. Small groups and unresolved crossings remain undivided or
  unassigned; report them. All choices are label-blind.

## Comparisons and acceptance

1. Four morphology families with identical sampled response paths reproduced at
   1x, 10x, 100x clock scale, 12 independent curves per family and three seeds.
   Same-shape replicas should keep shape grouping; physical derivatives must
   scale inversely and response times proportionally. Evaluate subgroup scale
   recovery post-hoc, not by feeding scale labels.
2. Native-time single-partition diagnostic: shape DTW with an added raw-hour
   channel in a common 100 h unit; compare predeclared time weights .01/.1/1.
   This illustrates the conflict, not a tuned replacement or a TWED replica.
3. Existing 400 curves, plus the prior same-family and independent 31001 batches;
   report shape recovery, kinetic output coverage, inferred subgroups, and raw
   hour figures. Existing observations must remain unchanged.
4. Same physical process with extra late observations: test that response time
   and early per-hour derivatives do not become total-duration surrogates.
5. Short observation with no detected excursion, sparse crossing, identical
   observed prefix with different future: return ambiguity/censoring honestly;
   no extrapolation, flat padding or synthetic late data.

Outputs: unchanged raw-hour observations, shape and kinetic assignments,
physical channels, landmark brackets/censoring, subgroup models, metrics, input
hashes, native-hour cluster galleries and a Chinese explanation. A cross-scale
cluster mean must not be drawn by pretending normalized progress is hour.

All plots use physical hour coordinates. Shape-only group galleries use a
linear-hour overview plus hour-labeled log/symlog detail when needed. Subgroups
are plotted separately on their own linear-hour axes. No real-data performance
guarantee is a completion condition; this is a concrete resolution of the
representation conflict, with its limitations visible.

## Related primary literature (not claimed as implemented algorithms)

- Marron et al., Functional Data Analysis of Amplitude and Phase Variation,
  https://arxiv.org/abs/1512.03216 : separating horizontal and amplitude variation.
- Marteau, Time Warp Edit Distance, https://arxiv.org/abs/0802.3522 : explicit
  timestamps and a temporal stiffness penalty. Our diagnostic is not TWED.

## Logged numerical amendment after the first run

The initial ARPACK run is retained in `initial_arpack_outputs/`. Reloading the
independent_31001 CSV changed inputs by at most 4.55e-13 and DTW distances by
2.76e-14, but ARPACK/QR shape ARI changed from 1 to .6323. Its graph had three
connected components. The third replica batch also failed despite a graph of
four disconnected 36-member components. This is numerical solver sensitivity,
not evidence for a new shape weight. Use dense symmetric `scipy.linalg.eigh`
on the same normalized graph Laplacian, followed by the same QR partitioning
algorithm. Distance, weights, graph, cluster count and kinetic procedure remain
unchanged. Report eigenvalue residuals and regression checks. This amendment
was made after seeing results and is not a preregistered independent validation.

The late-observation invariance test holds for its decaying process, whose
observed maximum is unchanged. In general a newly observed higher peak changes
the chosen vertical reference and may change normalized excursion parameters;
a fixed externally calibrated response reference is needed to prevent that.
