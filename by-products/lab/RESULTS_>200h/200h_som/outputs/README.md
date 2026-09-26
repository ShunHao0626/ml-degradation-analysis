# outputs/

Results from the **200h SOM analysis pipeline**.

> **👉 Main entry point: [`REPORT.md`](REPORT.md)** — a self-contained
> markdown report covering n=4..16 cluster scans, the smoothed-vs-no-smooth
> stability experiment, and the rationale for choosing n=10 as the primary
> result.

## Layout

```
outputs/
├── REPORT.md                                  ← main report (READ THIS FIRST)
├── analysis_report.md                         ← original n=4 analysis
├── all_n_summary.csv                          ← smoothed QE/sizes per n
├── cluster_assignments.csv                    ← 1812 curves × {n=4 BMU}
├── cluster_summary.csv
├── cluster_shape_metrics.csv
├── preprocessed_curves.csv
├── X_preprocessed.npy                         ← (1812, 1201) savgol-smoothed features
├── som_model.pkl
├── som_weights.npy
├── som_n4_centroids.csv
├── figures/                                   ← n=4 figures
├── n5/ ... n16/                               ← per-n subdirectories
│
└── no_smooth/                                 ← no-savgol variant
    ├── stability_report.md                    ← ARI / Hungarian overlap
    ├── cluster_shape_master.csv               ← all clusters across n
    ├── X_no_smooth.npy                        ← (1812, 1201) Akima-only features
    ├── n4/ ... n16/                           ← per-n subdirectories
    └── figures/
        ├── qe_smooth_vs_no_smooth.png
        ├── report_cross_n_overview.png        ← used in REPORT §2
        ├── report_stability_bars.png          ← used in REPORT §4.2
        ├── report_centroid_distance_matrices.png  ← used in REPORT §5
        └── n{n}_cluster_size_distribution.png × 8
```

## Quick numbers

| n | topology | QE (smooth) | QE (no_smooth) | ARI | matched % |
|---|---:|---:|---:|---:|---:|
| 4 | 2×2 | 1.2846 | 1.2847 | 1.0000 | 100.00 % |
| 5 | 1×5 | 1.0965 | 1.0966 | 1.0000 | 100.00 % |
| 6 | 2×3 | 1.0217 | 1.0218 | 1.0000 | 100.00 % |
| 7 | 1×7 | 0.9460 | 0.9461 | 1.0000 | 100.00 % |
| 8 | 2×4 | 0.8781 | 0.8782 | 1.0000 | 100.00 % |
| 9 | 3×3 | 0.8570 | 0.8571 | 1.0000 | 100.00 % |
| 10 | 2×5 | 0.8360 | 0.8146 | **0.9036** | 88.36 % |
| 16 | 4×4 | 0.6927 | 0.6928 | 1.0000 | 100.00 % |

See [`REPORT.md`](REPORT.md) for the full discussion.
