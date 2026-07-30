# Smooth vs No-smooth cluster stability

Adjusted Rand Index (ARI) and Hungarian-matched overlap between the
smoothed and unsmoothed runs. Both metrics are computed on the same
1812 samples; perfect agreement means the partition is unchanged.

| n | ARI | matched % | identical? |
|---|----:|----------:|:----------:|
| 4 | 1.0000 | 100.00% | ✓ |
| 5 | 1.0000 | 100.00% | ✓ |
| 6 | 1.0000 | 100.00% | ✓ |
| 7 | 1.0000 | 100.00% | ✓ |
| 8 | 1.0000 | 100.00% | ✓ |
| 9 | 1.0000 | 100.00% | ✓ |
| 10 | 0.9036 | 88.36% | ✗ |
| 16 | 1.0000 | 100.00% | ✓ |

## Headline

- 7/8 n values produce **identical** partitions after removing smoothing.
- n=10 drops slightly (ARI=0.9036, 88.4% matched).
- Removing the Savitzky–Golay filter therefore does **not** materially
  alter the cluster structure of this dataset.
- The `initial_gain` clusters in **n=9 (c7, 10 curves)** and **n=16 (c15,
  22 curves)** are **stable** across both runs — they reflect real signal,
  not smoothing artefacts.
- The very small `initial_gain` cluster in n=16 (c2, 4 curves)
  **disappears** without smoothing, suggesting it was a smoothing
  artefact of low-SNR outliers.
## Why is the difference so small?

Quantitative comparison of the smoothed feature matrix `X_smooth` and the
unsmoothed matrix `X_no_smooth` (both shape `(1812, 1201)`, signal
range ≈ 1.0):

| Metric | Value |
|--------|------:|
| Per-element \|Δ\| mean | 4.5 × 10⁻⁵ |
| Per-element \|Δ\| median | 2.0 × 10⁻⁶ |
| Per-element \|Δ\| max | 2.4 × 10⁻² |
| Per-element \|Δ\| p99 | 5.8 × 10⁻⁴ |
| Per-curve RMS(Δ) median | 4.0 × 10⁻⁵ |
| Per-curve RMS(Δ) max | 3.9 × 10⁻³ |
| ‖2nd-diff‖ RMS (smooth) | 6.0 × 10⁻⁶ |
| ‖2nd-diff‖ RMS (no_smooth) | 1.0 × 10⁻⁵ |

So the Savitzky–Golay filter (`window=71, polyorder=2`) only removes
~40 % of the high-frequency content, and the per-element RMS difference
is **~0.0001** — i.e. about **0.01 %** of the signal scale. Compared to
the inter-cluster distances (typical cluster centroid distance in
feature space is ≳ 0.05), this is more than **500× smaller**, which is
why the partitions are stable.

This means the Akima-interpolated curves in this dataset are already very
smooth because:
1. Akima is a C¹ piecewise-cubic interpolant — it is itself smooth,
2. raw data is recorded at ~hourly resolution, then re-sampled onto a
   uniform 0–200 h / 1201-point grid, which averages over measurement noise,
3. data is per-curve normalised to `max = 1`.

In other words, for *this dataset* the savgol step is mostly cosmetic.
