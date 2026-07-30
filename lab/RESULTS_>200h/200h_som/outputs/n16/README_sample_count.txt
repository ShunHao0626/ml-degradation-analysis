Note on sample counts
======================

This n=16 (4x4) sub-analysis reuses the **same 1812-sample preprocessed
feature matrix** (``outputs/X_preprocessed.npy``) that the main 2x2
SOM was trained on. Only the SOM itself is retrained with a 4x4 grid.

Concretely:
- ``sample_id`` ordering in ``cluster_assignments.csv`` matches
  ``outputs/included_samples.csv``.
- The total number of rows in ``cluster_assignments.csv`` is exactly 1812.
- The 16 cluster-size values reported in ``analysis.md`` and
  ``cluster_summary.csv`` are the partition of those 1812 curves into
  16 groups; their sum is 1812.

The two smallest clusters (c2 with 4 curves, c15 with 22 curves)
both exhibit ``initial_gain`` behaviour: PCE@0h < PCE@200h after
per-curve normalisation, suggesting a small sub-population whose
performance appears to improve over the first 200 hours.
