# Four-cluster results / 四簇结果

These are the included results for the fixed set of 1,842 early (0–200 h) curves. Cluster 1–4 contain **1,710 / 44 / 21 / 67** curves. The fit uses density-weighted K-means, not SOM.

这些是固定的 1842 条早期曲线的复现结果。四簇数量依次为 **1710／44／21／67**；拟合方法为密度加权 K-means，而非 SOM。

| File | Contents |
| --- | --- |
| `cluster_assignments.csv` | Cluster for every curve / 逐曲线簇号 |
| `representatives.csv` | Representative curves / 代表曲线 |
| `model.npz` | Model inputs, features, weights, and centers / 模型数组 |
| `summary.json` | Counts and input/config/code hashes / 数量与哈希 |
| `cluster_four_groups.png`, `figures/` | Plots / 图 |
| `dashboard.html` | Offline curve and source-figure browser / 离线浏览页面 |
| `cluster_1/`–`cluster_4/` | Copied source DOI folders; `members.csv` gives the exact cluster membership / 来源文件副本，以 `members.csv` 为准 |

To regenerate, follow the commands in the repository's root [README](../README.md) or [中文 README](../README_CN.md). The run replaces generated cluster folders in this directory. `environment.json` records the original runtime versions.
