Note on sample counts
======================

This n=8 sub-analysis reuses the **same 1812-sample preprocessed
feature matrix** (``outputs/X_preprocessed.npy``) that the main 2x2
SOM was trained on. Only the SOM itself is retrained.

Concretely:
- ``sample_id`` ordering in this file matches ``outputs/included_samples.csv``.
- The total number of rows in ``cluster_assignments.csv`` is exactly 1812.
- The cluster-size values reported in ``analysis.md`` and
  ``cluster_summary.csv`` are the partition of those 1812 curves into
  8 groups; their sum is 1812.

The "cluster size" numbers vary between n because they describe the
**partition** of the same set of curves; the underlying population
does not change.
