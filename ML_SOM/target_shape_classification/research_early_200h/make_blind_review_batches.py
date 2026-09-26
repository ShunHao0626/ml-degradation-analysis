#!/usr/bin/env python3
"""Prepare neutral, model-hidden review forms; never generates human labels."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from early_data import HERE, prepare, rows, write_rows


def choose(pool, count, rng):
    return rng.sample(pool, min(count, len(pool)))


def plot_neutral(path, item, boundary=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.7), dpi=110)
    real = [(h, y) for h, y, _ in item["points"] if 0 <= h <= 200]
    ax.plot(item["t"], item["y"], color="#285f77", lw=1)
    ax.scatter([p[0] for p in real], [p[1] for p in real], s=8, color="#285f77")
    if boundary and item["boundary"]:
        ax.axvspan(200, 250, color="#e8b778", alpha=.25)
        ax.scatter([p[0] for p in item["boundary"]], [p[1] for p in item["boundary"]],
                   s=22, color="#ac7132")
    ax.axvline(200, color="#a74747", ls="--", lw=1)
    ax.set_xlim(0, 250 if boundary else 205)
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Normalized PCE")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    index = rows(HERE / "delivery/class_index.csv")
    priority = {r["curve_id"]: int(r["priority"]) for r in rows(HERE / "reviews/review_queue.csv")}
    canonical = {r["file_id"]: r for r in rows(Path(cfg["legacy_results_root"]) / "canonical_curves.csv")}
    rng = random.Random(cfg["split_seed"] + 1)
    by_split = {s: [r for r in index if r["split"] == s] for s in ("train", "validation", "test")}
    train_random = choose(by_split["train"], 60, rng)
    taken = {r["curve_id"] for r in train_random}
    train_priority = sorted((r for r in by_split["train"] if r["curve_id"] not in taken),
                            key=lambda r: (-priority.get(r["curve_id"], 0), r["curve_id"]))[:60]
    selected = [(r, "train_random") for r in train_random]
    selected += [(r, "train_priority") for r in train_priority]
    selected += [(r, "validation_random") for r in choose(by_split["validation"], 100, rng)]
    selected += [(r, "test_natural_random") for r in choose(by_split["test"], 300, rng)]
    review_dir = HERE / "reviews"
    strict_dir, boundary_dir = review_dir / "blind_strict_charts", review_dir / "blind_boundary_charts"
    strict_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    strict, boundary = [], []
    for row, reason in selected:
        fid = row["curve_id"]
        item, error = prepare(canonical[fid], cfg["source_root"], cfg)
        if error:
            raise ValueError(fid + ": " + error)
        strict_png = f"blind_strict_charts/{fid}.png"
        plot_neutral(review_dir / strict_png, item)
        base = dict(curve_id=fid, sampling_reason=reason, source_group=row["source_group"],
                    strict_chart=strict_png, expert_id="", primary_sequence="",
                    class_or_nonconforming="", evidence_status="", turn_times_h="",
                    turn_amplitudes="", observed_stage_support="", notes="", review_date="")
        strict.append(base)
        boundary_png = ""
        if item["boundary"]:
            boundary_png = f"blind_boundary_charts/{fid}.png"
            plot_neutral(review_dir / boundary_png, item, boundary=True)
        boundary.append(dict(curve_id=fid, sampling_reason=reason,
                             strict_opinion_locked_sha256="", boundary_chart=boundary_png,
                             source_image=row["source_image"], expert_id="",
                             boundary_class_or_nonconforming="", boundary_evidence_status="",
                             observation_end_used_h="", source_marker_support="",
                             notes="", review_date=""))
    write_rows(review_dir / "blind_strict_form.csv", strict, list(strict[0]))
    write_rows(review_dir / "blind_boundary_form.csv", boundary, list(boundary[0]))
    info = dict(train_random=60, train_priority=60, validation_random=100,
                test_natural_random=300, total=len(selected),
                instruction="Two experts must fill independent copies; lock strict opinions before revealing boundary form/source image. No labels are prefilled.")
    (review_dir / "batch_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
