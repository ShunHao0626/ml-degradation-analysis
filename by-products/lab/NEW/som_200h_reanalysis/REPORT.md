# 200 h SOM reanalysis of the literature-mined ageing dataset

## Headline result

- Raw curves discovered: **2,151**
- Duration ≥200 h: **1,862**
- Duration ≥200 h and maximum PCE reached by 200 h: **1,823**
- Paper-aligned dataset (≥4 unique points): **1,785**
- Main high-quality dataset (≥10 unique points): **1,442**
- Selected main SOM node count: **n=4**
- Selected paper-sensitivity node count: **n=4**

The primary result uses the ≥10-point dataset because the source is literature-digitised rather than dense in-house MPPT data. The paper-aligned ≥4-point dataset is retained as a sensitivity analysis.

## Preprocessing

1. Convert source time units to hours.
2. Shift every curve to relative time t=0.
3. Require duration ≥200 h.
4. Require the global maximum PCE to occur by 200 h.
5. Retain 0–200 h plus the first observed point after 200 h.
6. Average duplicate timestamps.
7. Akima-interpolate to 1,201 points at 10-minute spacing.
8. Divide each curve by its own maximum PCE in 0–200 h.
9. Apply Savitzky–Golay smoothing (window 71, polynomial order 2).

The first point after 200 h brackets the endpoint; no flat tail fill and no extrapolation is used.

## Cluster-number decision

Main n=4: mean QE=1.3755, mean seed ARI=1.0000, max centroid correlation=0.9989.

Paper sensitivity n=4: mean QE=1.2811, mean seed ARI=1.0000, max centroid correlation=0.9994.

The n=4 solution is the smallest SI-range model with stable seeds, no empty or <1% class, and no centroid pair meeting both near-duplicate conditions (correlation ≥0.995 and RMSE <0.10). In n=5 and n=6, the closest centroid RMSE falls below 0.10, indicating subdivision of an existing shape rather than a clearly new pattern.

See `05_cluster_number_decision/SI_METHOD_APPLICATION.md` for the complete SI-based decision logic and n=2–10 tables.

## Dataset-selection sensitivity

Both the 1,442-curve main dataset and the 1,785-curve paper-aligned dataset select n=4. On their 1,442 common curves, label agreement is ARI=0.7884 and NMI=0.7638.

## Final main clusters

| class_label | raw_cluster_id | n_curves | fraction_percent | pce_norm_0h_mean | pce_norm_200h_mean | auc_0_200h | suggested_shape_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| class_01 | 0 | 831 | 57.6283 | 0.9899 | 0.9204 | 192.0966 | no_initial_change_stable |
| class_03 | 1 | 159 | 11.0264 | 0.9982 | 0.4878 | 142.393 | initial_drop_rapid_loss |
| class_02 | 2 | 393 | 27.2538 | 0.9959 | 0.7414 | 172.1333 | no_initial_change_moderate_loss |
| class_04 | 3 | 59 | 4.0915 | 0.9994 | 0.2329 | 99.6273 | initial_drop_rapid_loss |

Class labels are ordered from higher to lower normalized PCE at 200 h. Suggested shape names are descriptive post-hoc labels, not physical degradation mechanisms.

## K-means reference

- ARI: 0.9742
- NMI: 0.9388
- Hungarian-aligned matched fraction: 98.54%

K-means here uses Euclidean distance on the same 1,201-dimensional curves. The reference paper reports DTW K-means, which is not available in the installed environment; this difference is explicitly retained as an implementation limitation.

## Per-curve traceability

`06_final_model/final_curve_classification.csv` contains one row per curve with DOI, figure, series, absolute original CSV path, selected database path information, SOM node, raw cluster, ordered class, suggested shape label, BMU distance, and K-means reference cluster.

Each `06_final_model/clusters/class_XX/` directory contains:

- `members.csv` — all metadata and classification fields;
- `raw_curves/` — browsable raw CSV files assigned to that class.

## Limitations

- The source curves are literature-mined and combine laboratories and ageing conditions.
- The article's original dataset was homogeneous, in-house MPPT data.
- A curve-shape cluster must not be interpreted as a unique physical mechanism without metadata analysis.
- SOM node identities are arbitrary; ordered class IDs are a reporting convention.

## Reproduction

From this project directory:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python3 code/run_pipeline.py
python3 code/verify_outputs.py
```
