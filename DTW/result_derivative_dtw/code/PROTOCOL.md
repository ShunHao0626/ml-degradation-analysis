# Analysis protocol (selection rules implemented in the main script before execution)

The supplied `02_derivative_dtw_clustering.md` governs this run. Original source files are never edited. No curve templates, manual curve classes, smoothing, Ward linkage, or fixed K are used. Hierarchical clustering is the selected G1 branch; the optional DTW-SOM branch is not implemented.

## QC and time

Require explicitly identifiable hour/day time axis and PCE/efficiency response, >=6 finite unique points, positive response. Days convert to hours. Exclude cycles, uncertain axes, conflicting same-time values, and extreme duration records above median(log10 duration)+5*1.4826*MAD(log10 duration). This last rule flags possible calibration problems rather than repairing them. Identical duplicates can be removed; no averaging over neighboring times is performed. Missing rows are removed with audit. Percent-relative PCE is permitted even above 100.

Main time origin is each curve's first recorded observation. Keep the original absolute timestamps separately. This is elapsed observed time, not guaranteed true stress-onset time. Generate endpoint windows at observed duration order statistics providing >=100%,90%,80%,70%,60%,50%,40%,30%,20% coverage. Keep only curves covering the entire window with >=6 ORIGINAL points within it. Append a linearly interpolated endpoint at T when required; never extrapolate. Nominal coverage and usable coverage have different denominators/meaning and are both reported. The candidate expansion below 80% is necessary because the common early interval can contain very few points for long-duration traces.

Use variable-length original sampling, without a common grid. All four normalizations use the window's retained curve: y/y0, y/ymax, (y-mean)/population SD, (y-median)/IQR. If a scale is exactly zero, use scale=1 and preserve a constant signal; audit such cases. Derivative is diff(normalized y)/diff(real elapsed hours), attached to the right endpoint; each DTW sequence has N-1 entries for matched support across the three methods. The original first y observation remains in all raw/normalized outputs. Derivative channel is NOT independently standardized. Lambda therefore has an hour-scale interpretation and can be numerically weak for slow curves; report this explicitly.

## Search and distances

For each window and normalization calculate classic PCE-only, derivative-only, and [PCE, lambda*derivative] with lambda .25,.5,1,2,4. Use radius .02,.05,.10,.15,.20 plus unrestricted control. For unequal length sequences the band is |i/(n-1)-j/(m-1)| <= max(radius, .5/(n-1)+.5/(m-1)). This explicitly defines a scaled-diagonal Sakoe-Chiba adaptation, including unavoidable sample discretization widening. A sample-index band does not ensure a physical-time band: audit actual time displacements.

Optimize the standard sum of squared Euclidean local costs over monotone DTW paths; diagonal wins ties. Main distance is sqrt(sum/path length) **on that minimum-sum path**, a path-length correction for variable sampling counts. This is not a minimum-average-cost-path optimization. Conventional sqrt(sum) is an explicit final sensitivity control, because raw accumulated distance can depend strongly on trace length. Exact Python DP and accelerated C++ DP are cross-checked.

Average, complete and weighted linkage; K=2..10. Save every distance matrix, every label vector, silhouette and Dunn (minimum pairwise intercluster / maximum pairwise intracluster distance). Flag solutions with smallest cluster <max(3,ceil(.05*N)) or largest >95% as insufficient for the primary population-level conclusion; their results remain available. This is an occupancy safeguard, not a requested phenotype count.

## Two-stage stability and selection

Stage 1: exhaustively evaluate all internal metrics. Within each window rank .75*silhouette percentile + .25*Dunn percentile. Shortlist the best two DISTINCT partitions per multivariate normalization, the best multivariate candidate for EACH lambda, plus the best classic and derivative-only control. Only constrained candidates with acceptable occupancy are shortlisted when available. This is screening, not exhaustive resampling of every grid point. Lambda is represented across the stability shortlist but not every lambda-radius-linkage-K combination receives resampling.

Stage 2: each shortlisted candidate receives 50 independent repetitions of each: 90% sample-size bootstrap WITH replacement, 95% sample-size bootstrap WITH replacement, deletion of 5% interior time points, deletion of 10% interior time points. Endpoints remain to preserve coverage and baseline. Deletion count is rounded and at least one; actual fraction is saved, since sparse traces cannot realize exactly 5%. Recompute normalizations, finite differences and DTW after deletion. Bootstrap ARI compares unique observed curves to the baseline partition, counting each once; draw duplicates still affect fitting. Bootstrap90/95 are sample SIZES, not 90/95% unique coverage. Consensus uses co-observation-specific denominators; never observed pairs are NaN. No confidence interval is claimed from replicate percentile ranges.

Selection score, fixed before outcomes:
0.45*minimum(mean ARI across four perturbations)
+0.25*(silhouette+1)/2
+0.10*Dunn/(1+Dunn)
+0.20*usable coverage among QC-eligible curves
-0.20*mean physical time displacement / T.
Additional penalty .15 when 95th percentile maximum displacement >.35*T or mean derivative-turn-collapse fraction >.25. Select highest-score multivariate candidate with acceptable occupancy and without extreme warp; if none exists report the best multivariate candidate as exploratory. Classic/derivative controls cannot silently replace the requested multivariate main analysis. Unrestricted controls are not primary candidates. Interpret curve morphology only AFTER writing frozen_parameters.json.

Report: raw curves per cluster, actual medoids, derivatives, raw extrema (including boundary extrema), turning-time distributions without smoothing, within-cluster distances, representative paths, consensus, window comparisons on common sample IDs. Add grouped bootstrap by source figure (source figure is not a verified paper/device identifier), absolute-origin, conventional distance, three-point local linear slope, radius and normalization sensitivity for the frozen model. These controls evaluate the frozen result and are not used to retune toward a desired class count.

## Limits

These are observational digitized curves with uncertain independence and heterogeneous stress protocols. Internal selection gives a conditional exploratory recommendation, not independent validation of a unique physical taxonomy. Local extrema can be digitization noise. Late windows induce a duration-selected cohort; selection score weights are pragmatic and require sensitivity reporting. All labels and metrics are descriptive, without mechanistic claims or classification accuracy.
