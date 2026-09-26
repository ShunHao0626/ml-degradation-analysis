#!/usr/bin/env python3
"""Make source-image/0–250 h comparison pages for pending boundary reviews."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import classify_200h as current
import run


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parent / "data_final")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--per-page", type=int, default=4)
    args = parser.parse_args()
    if args.per_page < 1:
        parser.error("--per-page must be positive")
    reviews = [r for r in run.rows(args.results / "som_200h" / "review_priority_200h.csv")
               if "boundary_valley_candidate" in r["reasons"].split(";")]
    canonical = {r["file_id"]: r for r in run.rows(args.results / "canonical_curves.csv")}
    output = args.results / "som_200h" / "boundary_review_pages"
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("page_*.png"):
        stale.unlink()
    index = []
    for start in range(0, len(reviews), args.per_page):
        page = start // args.per_page + 1
        batch = reviews[start:start + args.per_page]
        fig, axes = plt.subplots(len(batch), 2, figsize=(15, 3.6 * len(batch)),
                                 squeeze=False)
        for pos, review in enumerate(batch):
            fid = review["file_id"]
            row = canonical[fid]
            source = args.root / row["source_image"]
            axes[pos, 0].imshow(Image.open(source))
            axes[pos, 0].set_axis_off()
            axes[pos, 0].set_title(f"{fid}  ·  {row['source_group']}  ·  source panel", fontsize=10)
            observation, reason = current.window_observations(args.root, row)
            if observation is None:
                raise ValueError(f"{fid}: {reason}")
            t, y = observation["near_t"], observation["near_y"]
            axes[pos, 1].plot(t, y, ".-", color="#23658a", markersize=3, linewidth=1.2)
            axes[pos, 1].axvline(200, color="#bd3445", linestyle="--", linewidth=1)
            axes[pos, 1].set_xlim(0, 250)
            axes[pos, 1].set_xlabel("Time (h)")
            axes[pos, 1].set_ylabel("Normalized PCE")
            axes[pos, 1].grid(alpha=.15)
            axes[pos, 1].set_title(
                f"{fid}  ·  {review['final_class']} candidate  ·  "
                f"trough {float(review['boundary_valley_trough_h']):.1f} h  ·  "
                f"recovery {float(review['boundary_valley_recovery_by_250h']):.3f}",
                fontsize=10)
            index.append(dict(file_id=fid, page=f"page_{page:02d}.png",
                              source_csv=row["source_csv"], source_image=row["source_image"],
                              source_group=row["source_group"]))
        fig.tight_layout()
        fig.savefig(output / f"page_{page:02d}.png", dpi=150)
        plt.close(fig)
    run.write_rows(output / "index.csv", index,
                   ["file_id", "page", "source_csv", "source_image", "source_group"])
    print(f"{len(reviews)} pending boundary candidates on {page if reviews else 0} pages")


if __name__ == "__main__":
    main()
