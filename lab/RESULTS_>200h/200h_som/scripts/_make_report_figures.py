"""Generate the cross-n comparison figures used in REPORT.md.

Produces a single PNG with 4 panels:
  (A) Quantisation Error vs n — smooth vs no_smooth vs k=1..10 k-means WCSS reference
  (B) Cluster size distribution across n (stacked bar)
  (C) Largest vs smallest cluster as % of total — illustrates imbalance
  (D) Topographic Error vs n (smooth vs no_smooth)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Users/shunhao/Desktop/ML/lab/RESULTS_>200h/200h_som")
OUT = ROOT / "outputs" / "no_smooth" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load data ------------------------------------------------------------
smooth = pd.read_csv(ROOT / "outputs" / "all_n_summary.csv")
no_smooth = pd.read_csv(ROOT / "outputs" / "no_smooth" / "all_n_no_smooth_summary.csv")
kmeans = pd.read_csv(ROOT / "outputs" / "kmeans_wcss.csv")
stab = pd.read_csv(ROOT / "outputs" / "no_smooth" / "stability_report.csv")

# Parse cluster_sizes JSON
smooth["sizes"] = smooth["cluster_sizes"].apply(
    lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else []
)
no_smooth["sizes"] = no_smooth["cluster_sizes"].apply(
    lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else []
)


def get_sizes(row, prefer_main=False):
    if row["sizes"]:
        return row["sizes"]
    # Main 2x2 cluster sizes known from earlier (443, 188, 1103, 78)
    return [443, 188, 1103, 78]


smooth["sizes_list"] = smooth.apply(get_sizes, axis=1)
no_smooth["sizes_list"] = no_smooth.apply(get_sizes, axis=1)

# ---- Figure ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# (A) QE
ax = axes[0, 0]
n_vals = sorted(set(smooth["n_nodes"].dropna().astype(int).tolist()))
qe_s = [smooth[smooth["n_nodes"] == n]["quantisation_error"].values[0] for n in n_vals]
qe_n = [no_smooth[no_smooth["n_nodes"] == n]["quantisation_error"].values[0] for n in n_vals]
ax.plot(n_vals, qe_s, "o-", color="C0", lw=2, label="SOM (smoothed)")
ax.plot(n_vals, qe_n, "s--", color="C3", lw=2, label="SOM (no smooth)")
ax.set_xlabel("# clusters (n)")
ax.set_ylabel("Quantisation Error")
ax.set_title("(A) Quantisation error across n")
ax.set_xticks(n_vals)
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
# Annotate the elbow
for n, qe in zip(n_vals, qe_s):
    ax.annotate(f"{qe:.3f}", (n, qe), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="C0")

# (B) Cluster size distribution
ax = axes[0, 1]
# Use smooth sizes
n_show = n_vals
all_sizes = []
for n in n_show:
    s = smooth[smooth["n_nodes"] == n].iloc[0]["sizes_list"]
    all_sizes.append(sorted(s, reverse=True))
x = np.arange(len(n_show))
bottom = np.zeros(len(n_show))
colors = plt.cm.tab20(np.linspace(0, 1, max(len(s) for s in all_sizes)))
max_clusters = max(len(s) for s in all_sizes)
for k in range(max_clusters):
    h = []
    for s in all_sizes:
        h.append(s[k] if k < len(s) else 0)
    ax.bar(x, h, bottom=bottom, color=colors[k], label=f"cluster {k}" if k < 5 else None)
    bottom += np.array(h)
ax.set_xticks(x)
ax.set_xticklabels([f"n={n}" for n in n_show])
ax.set_ylabel("# curves in cluster")
ax.set_title("(B) Cluster size distribution (sorted, stacked)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# (C) Cluster imbalance (largest / smallest as % of total)
ax = axes[1, 0]
largest_pct = []
smallest_pct = []
for n in n_show:
    s = smooth[smooth["n_nodes"] == n].iloc[0]["sizes_list"]
    largest_pct.append(max(s) / 1812 * 100)
    smallest_pct.append(min(s) / 1812 * 100)
ax.plot(n_show, largest_pct, "o-", color="C0", label="largest cluster")
ax.plot(n_show, smallest_pct, "s-", color="C3", label="smallest cluster")
ax.set_xlabel("# clusters (n)")
ax.set_ylabel("% of total samples")
ax.set_title("(C) Cluster imbalance (max / min %)")
ax.set_xticks(n_show)
ax.grid(alpha=0.3)
ax.legend()

# (D) TE
ax = axes[1, 1]
te_n = []
for n in n_vals:
    v = no_smooth[no_smooth["n_nodes"] == n]["topographic_error"].values
    te_n.append(float(v[0]) if len(v) and not np.isnan(v[0]) else np.nan)
ax.plot(n_vals, te_n, "s--", color="C3", lw=2, label="SOM (no smooth)")
ax.set_xlabel("# clusters (n)")
ax.set_ylabel("Topographic Error")
ax.set_title("(D) Topographic error (no_smooth)")
ax.set_xticks(n_vals)
ax.grid(alpha=0.3)
ax.legend()

fig.suptitle("Cross-n SOM comparison — smoothed vs no-smooth",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = OUT / "report_cross_n_overview.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
