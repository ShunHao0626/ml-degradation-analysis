# Initial dataset audit

Source: `/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted/samples_test`

## Inventory

- 99 sample directories
- 218 digitized CSV curves
- every CSV has numeric `x,y` columns, at least 4 distinct points, and monotonic x
- 65/99 validation records have zero legend-to-series mappings even though CSVs
  may be present
- all 99 records use placeholder DOI `samples_test`
- all 99 records have blank figure and panel identifiers

The missing DOI/figure identifiers prevent tracing most curves back to the
publication from which they were digitized. Filenames and sample-directory names
remain the only available provenance.

## Strict PCE-hour eligibility

The automated first pass admits only explicit time-in-hours (or explicit days,
converted by the unit identity `1 day = 24 h`) paired with a y-axis labelled PCE
or efficiency.

- 142/218 curves pass the strict axis test, from 67/99 sample directories.
- 117 of those reach at least 150 h by their recorded maximum x value.
- 76 curves are held out before modelling:
  - 17 use a cycle-count x-axis;
  - 15 use a power/MPPT proxy rather than explicit PCE/efficiency;
  - 44 have missing x and/or y metadata.

These exclusions are data-eligibility checks, not manual shape selection.

## Quality issues requiring a frozen rule before preprocessing

- The curves are sparse digitizations (strict-pool median: 38 points), unlike the
  source paper's dense in-house MPPT measurements.
- Several digitized x minima are slightly negative and several start after zero.
- Record `图片140` has an impossible hour scale of roughly 41-42 million hours
  and must be excluded as a calibration failure before any model fitting.
- Some y-axes are absolute PCE percentages while others are already normalized;
  the paper's per-curve MaxAbs normalization makes their shapes comparable without
  adding a variable.
- Eligibility for 150/300/500 h still needs the paper's duration and
  maximum-after-cutoff rule applied to the full digitized curve.

Machine-readable details are in `curve_manifest.csv` and
`dataset_audit_summary.json`.
