"""Stability / cluster-distance plots used in REPORT.md."""

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

stab = pd.read_csv(ROOT / "outputs" / "no_smooth" / "stability_report.csv")
smooth = pd.read_csv(ROOT / "outputs" / "all_n_summary.csv")
no_smooth = pd.read_csv(ROOT / "outputs" / "no_smooth" / "all_n_no_smooth_summary.csv")
smooth["sizes"] = smooth["cluster_sizes"].apply(
    lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else []
)
no_smooth["sizes"] = no_smooth["cluster_sizes"].apply(
    lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else []
)
smooth["sizes_list"] = smooth["sizes"].apply(lambda v: v if v else [443, 188, 1103, 78])
no_smooth["sizes_list"] = no_smooth["sizes"].apply(lambda v: v if v else [443, 188, 1103, 78])

# ---- Cluster-centroid distance matrices for n=4, 8, 10, 16 --------------
def centroid_distance_matrix(out_dir: Path) -> np.ndarray | None:
    csv = out_dir / "cluster_centroid_distances.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, index_col=0).to_numpy()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (A) ARI vs n
ax = axes[0]
colors = ["C2" if v == 1.0 else "C3" for v in stab["ARI_smooth_vs_no_smooth"]]
ax.bar(stab["n"].astype(str), stab["ARI_smooth_vs_no_smooth"], color=colors)
ax.axhline(1.0, color="k", ls="--", alpha=0.4, lw=1)
ax.set_ylim(0.85, 1.01)
ax.set_xlabel("# clusters (n)")
ax.set_ylabel("ARI (smooth vs no_smooth)")
ax.set_title("(A) Cluster stability — Adjusted Rand Index")
for i, (n, ari) in enumerate(zip(stab["n"], stab["ARI_smooth_vs_no_smooth"])):
    ax.text(i, ari + 0.005, f"{ari:.3f}", ha="center", fontsize=9)

# (B) Hungarian-matched fraction
ax = axes[1]
ax.bar(stab["n"].astype(str), stab["Hungarian_matched_fraction"] * 100,
       color=colors)
ax.axhline(100, color="k", ls="--", alpha=0.4, lw=1)
ax.set_ylim(85, 101)
ax.set_xlabel("# clusters (n)")
ax.set_ylabel("Hungarian-matched %")
ax.set_title("(B) Cluster stability — Hungarian overlap")
for i, (n, m) in enumerate(zip(stab["n"], stab["Hungarian_matched_fraction"])):
    ax.text(i, m * 100 + 0.5, f"{m*100:.1f}%", ha="center", fontsize=9)

fig.suptitle("Smooth vs No-Smooth partition stability",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = OUT / "report_stability_bars.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")

# ---- Centroid distance heatmaps -------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, n in zip(axes.flat, [4, 8, 10, 16]):
    sub = smooth[smooth["n_nodes"] == n]
    if sub.empty:
        ax.set_visible(False)
        continue
    sizes = sub.iloc[0]["sizes_list"]
    if n == 4:
        M = centroid_distance_matrix(ROOT / "outputs")
    else:
        M = centroid_distance_matrix(ROOT / "outputs" / f"n{n}")
    if M is None:
        ax.set_visible(False)
        continue
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_title(f"n={n} centroid distances  (largest={max(sizes)}, smallest={min(sizes)})")
    ax.set_xlabel("cluster")
    ax.set_ylabel("cluster")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Cluster-centroid distance matrices",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = OUT / "report_centroid_distance_matrices.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
