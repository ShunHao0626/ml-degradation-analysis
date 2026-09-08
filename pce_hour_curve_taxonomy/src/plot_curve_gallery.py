#!/usr/bin/env python3
"""Plot every model-input trajectory for post-clustering visual interpretation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-page", type=int, default=24)
    args = parser.parse_args()

    bundle = np.load(args.bundle)
    data = bundle["data"]
    time_axis = bundle["time_hours"]
    metadata = read_csv(args.metadata)
    assignments = read_csv(args.assignments)
    if not (len(data) == len(metadata) == len(assignments)):
        raise ValueError("data/metadata/assignment length mismatch")
    cluster_by_id = {row["curve_id"]: int(row["cluster_display"]) for row in assignments}
    order = sorted(
        range(len(metadata)),
        key=lambda index: (cluster_by_id[metadata[index]["curve_id"]], metadata[index]["sample"], metadata[index]["curve_id"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    page_count = math.ceil(len(order) / args.per_page)
    for page in range(page_count):
        selected = order[page * args.per_page : (page + 1) * args.per_page]
        ncols = 4
        nrows = math.ceil(len(selected) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.0 * nrows), sharex=True, sharey=True)
        axes_array = np.atleast_1d(axes).ravel()
        for ax, index in zip(axes_array, selected):
            row = metadata[index]
            cluster = cluster_by_id[row["curve_id"]]
            ax.plot(time_axis, data[index], color="#1f77b4", linewidth=1.5)
            if time_axis[-1] >= 200:
                ax.axvline(200, color="#9e9e9e", linewidth=0.8, linestyle="--")
            sample_label = row["sample"].replace("图片", "img")
            ax.set_title(f"{sample_label} | cluster {cluster}", fontsize=9)
            ax.grid(alpha=0.18)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Normalized PCE")
        for ax in axes_array[len(selected):]:
            ax.axis("off")
        fig.suptitle(
            f"All model-input trajectories, page {page + 1}/{page_count} (post-clustering audit)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(args.output / f"curve_gallery_page_{page + 1:02d}.png", dpi=190)
        plt.close(fig)


if __name__ == "__main__":
    main()
