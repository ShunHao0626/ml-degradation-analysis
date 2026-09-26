# Scope and decision record

Created: 2026-09-02 (Asia/Shanghai)

## Research question

Can the PCE–hour trajectories in the initial sample collection yield clusters
whose post hoc morphology corresponds to IFO-Bridge, IFO-Hill, IFO-Slope, and
IFO-Valley when the published unsupervised workflow is followed and only
literature-supported parameters are varied?

The required number of clusters is not assumed to be four. Fewer or more
clusters are acceptable when selected by the source method's documented rule.

## Frozen exclusions

- No forced `k=4` or forced 2-by-2 SOM map for the purpose of matching the target.
- No target-shape labels used in training, tuning, or cluster-count selection.
- No extra variables beyond the published PCE–time representation.
- No undocumented feature engineering or algorithm substitution.
- No selection of a configuration solely because it visually resembles the four
  requested archetypes.

## Current gates

- [x] Create independent project.
- [ ] Verify which initial records contain usable PCE–hour series.
- [ ] Identify the exact source-paper method and its implementation differences.
- [ ] Approve a parameter whitelist from textual evidence.
- [ ] Identify the literature-backed cluster-count rule.
- [ ] Freeze and run baseline.
- [ ] Run only whitelisted configurations.
- [ ] Interpret morphology after unsupervised selection.
- [ ] Run literature review only if the experimental branch remains unresolved.

