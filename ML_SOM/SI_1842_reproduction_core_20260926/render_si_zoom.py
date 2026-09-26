"""Redraw SI Fig. S1 with display-only smoothing of cluster 0's black trend."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "SI_1842_curves_20260926"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
DEFAULT_RESULTS = (HERE / CONFIG["output_dir"]).resolve()


def display_sigma(cluster: int) -> float:
    # Visualization only. No change to the feature encoder or KMeans fit.
    return 3.0 if cluster == 0 else 1.2


def render_zoom(results_dir: Path | None = None) -> None:
    results = results_dir or DEFAULT_RESULTS
    with np.load(results / "model.npz") as model:
        raw = model["raw_rank"]
        ids = model["curve_id"].tolist()
    with (results / "anonymous_assignments.csv").open(newline="", encoding="utf-8") as stream:
        assignments = list(csv.DictReader(stream))
    assert ids == [row["curve_id"] for row in assignments]
    labels = np.array([int(row["anonymous_cluster"]) for row in assignments])
    rank = np.linspace(0, 1, raw.shape[1])
    colors = {0: "#719b82", 1: "#416987", 2: "#ad945e", 3: "#907192"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6), sharex=True)
    for ax, cluster in zip(axes.flat, range(4)):
        members = np.flatnonzero(labels == cluster)
        ax.set_facecolor("#eaf1fb")
        ax.set_axisbelow(True)
        segments = [np.column_stack((rank, raw[i])) for i in members]
        alpha = .018 if len(members) > 500 else .12 if len(members) > 100 else .19
        ax.add_collection(LineCollection(segments, colors=colors[cluster],
                                         linewidths=.7, alpha=alpha, zorder=1))
        median = gaussian_filter1d(np.median(raw[members], axis=0),
                                   sigma=display_sigma(cluster), mode="nearest")
        span = np.ptp(median)
        pad = max(.02, .5 * span)
        lo = max(-.05, float(median.min() - pad))
        hi = min(1.05, float(median.max() + pad))
        line_style = dict(solid_capstyle="round", solid_joinstyle="round") if cluster == 0 else {}
        ax.plot(rank, median, color="#171b20", linewidth=2.0, zorder=3,
                **line_style)
        ax.set_title(f"Anonymous cluster {cluster} · {len(members)} curves",
                     fontsize=12, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        ax.grid(color="white", linewidth=1.3)
        ax.tick_params(length=0, colors="#3c4b58")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel("Normalized PCE")
    for ax in axes[-1]:
        ax.set_xlabel("Relative observation rank (0–1)")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    folder = results / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "Supplementary_Fig_S1_zoom.png", dpi=180)
    fig.savefig(folder / "Supplementary_Fig_S1_zoom.svg",
                metadata={"Date": "2026-09-26"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    render_zoom(args.results_dir)
