# Application of the Supplementary Information cluster-number method

## What the SI does

The SI scans SOM node counts n=2–10 using quantisation error. It identifies n=4–6 as the plausible range, then inspects whether additional clusters overlap or become indistinguishable. It selects the smallest cluster count that still captures distinct main shapes. K-means is used as a reference validation.

Source: `thesis/paper/Supplementary.md`, especially Supplementary Figs. 6 and 13–17 and the SOM Quantisation Error section.

## Quantitative translation used here

For every n=2–10, this project stores:

- QE and topographic error for five random seeds;
- mean/min pairwise ARI across seeds;
- maximum pairwise centroid correlation and minimum centroid RMSE;
- empty nodes and the minimum cluster fraction;
- PCA-based silhouette, Davies–Bouldin, and Calinski–Harabasz indices;
- K-means WCSS and validation indices.

The SI-supported candidate interval remains n=4–6. The decision rule chooses the smallest n in that range satisfying all of: no empty nodes, no cluster below 1%, mean seed ARI ≥0.80, and no near-duplicate centroid pair. A pair is considered near-duplicate only when its correlation is ≥0.995 and its RMSE is <0.10. Using both quantities avoids treating all smooth monotonic degradation profiles as identical. If none pass, a documented fallback score is used.

## Main high-quality dataset — selected n=4

| n_nodes | topology | qe_mean_across_seeds | qe_std_across_seeds | mean_pairwise_seed_ARI | max_centroid_correlation | min_cluster_fraction | silhouette_pca20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1x2 | 2.2896 | 0.0 | 1.0 | 0.99 | 0.1546 | 0.686 |
| 3 | 1x3 | 1.6214 | 0.0 | 1.0 | 0.9936 | 0.0832 | 0.5829 |
| 4 | 2x2 | 1.3755 | 0.0 | 1.0 | 0.9989 | 0.0409 | 0.5436 |
| 5 | 1x5 | 1.177 | 0.0 | 1.0 | 0.9993 | 0.034 | 0.4672 |
| 6 | 2x3 | 1.0875 | 0.0 | 1.0 | 0.9984 | 0.018 | 0.4546 |
| 7 | 1x7 | 1.001 | 0.0 | 1.0 | 0.9992 | 0.0153 | 0.4063 |
| 8 | 2x4 | 0.976 | 0.0004 | 0.9993 | 0.9997 | 0.0153 | 0.4105 |
| 9 | 3x3 | 0.9075 | 0.0 | 1.0 | 0.9999 | 0.0153 | 0.3671 |
| 10 | 2x5 | 0.8769 | 0.0168 | 0.8059 | 0.9997 | 0.0118 | 0.3608 |

## Paper-aligned sensitivity dataset — selected n=4

| n_nodes | topology | qe_mean_across_seeds | qe_std_across_seeds | mean_pairwise_seed_ARI | max_centroid_correlation | min_cluster_fraction | silhouette_pca20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1x2 | 2.075 | 0.0 | 1.0 | 0.9869 | 0.1524 | 0.6919 |
| 3 | 1x3 | 1.5 | 0.0 | 1.0 | 0.991 | 0.0762 | 0.5982 |
| 4 | 2x2 | 1.2811 | 0.0 | 1.0 | 0.9994 | 0.0437 | 0.5478 |
| 5 | 1x5 | 1.1096 | 0.0 | 1.0 | 0.9997 | 0.037 | 0.4879 |
| 6 | 2x3 | 1.037 | 0.0 | 1.0 | 0.9993 | 0.0134 | 0.4813 |
| 7 | 1x7 | 0.9495 | 0.0 | 1.0 | 0.9995 | 0.0123 | 0.426 |
| 8 | 2x4 | 0.8843 | 0.0024 | 0.9755 | 0.9995 | 0.0118 | 0.3885 |
| 9 | 3x3 | 0.8559 | 0.0069 | 0.9414 | 0.9997 | 0.0095 | 0.3917 |
| 10 | 2x5 | 0.8192 | 0.0002 | 0.9987 | 0.9999 | 0.0095 | 0.386 |

## Rule audit

Main decision table: `main_high_quality_min10/cluster_number_decision.csv`

Paper-sensitivity decision table: `paper_aligned_min4/cluster_number_decision.csv`

The full decision is intentionally not based on QE alone because QE almost always decreases when more nodes are added. This follows the SI's instruction to inspect overlap and interpretability after the 4–6 elbow range has been identified.
