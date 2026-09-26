#!/usr/bin/env python3
"""Post-freeze raw-point examples of four requested morphologies; not clusters."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import run_pipeline as pipeline

OUT = PROJECT / "07_posthoc_ifo"


def scores(y: np.ndarray, grid: np.ndarray) -> dict:
    peak_i, trough_i = int(np.argmax(y)), int(np.argmin(y))
    start, end, peak, trough = y[0], y[-1], y[peak_i], y[trough_i]
    amplitude = max(float(np.ptp(y)), 1e-12)
    rise = max(0.0, peak - start); drop = max(0.0, peak - end)
    peak_valid = 1.0 if 0 < grid[peak_i] <= min(200, .45 * grid[-1]) else .02
    base = max(start, end)
    level = base + .55 * max(0.0, peak - base)
    width = float(np.mean(y >= level)) if peak > base else 0.0
    recovery = float(np.max(y[trough_i:]) - trough) if trough_i < len(y) - 1 else 0.0
    valley_valid = 1.0 if 0 < grid[trough_i] <= min(200, .45 * grid[-1]) else .02
    monotone_decay = float(np.mean(np.diff(y) <= 0))
    return {
        "Bridge": min(rise, drop) / amplitude * width * peak_valid,
        "Hill": min(rise, drop) / amplitude * (1 - width) * peak_valid,
        "Slope": max(0.0, start - end) / amplitude * monotone_decay,
        "Valley": min(max(0.0, start - trough), recovery) / amplitude * valley_valid,
        "peak_time_h": float(grid[peak_i]), "trough_time_h": float(grid[trough_i]),
        "peak_width_fraction": width, "recovery": recovery,
    }


def main() -> None:
    if not (PROJECT / "05_frozen_selection/frozen_selection.json").exists():
        raise SystemExit("Primary result must be frozen first")
    # 500 h detail cohort observes both the requested early and late phases.
    _, display, grid, meta = pipeline.load_variant("w0500_detail_maxabs_level")
    rows = []
    for i, y in enumerate(display):
        rows.append({"row": i, "curve_id": meta.iloc[i]["curve_id"], "source_file": meta.iloc[i]["source_file"],
                     "legend": meta.iloc[i]["legend"], **scores(y, grid)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "individual_ifo_candidate_scores.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    provenance = []
    for kind, ax in zip(("Bridge", "Hill", "Slope", "Valley"), axes.flat):
        chosen = frame.sort_values(kind, ascending=False).head(8)
        for rank, (_, r) in enumerate(chosen.iterrows(), 1):
            i = int(r["row"])
            x, y = common.read_xy(Path(r["source_file"]))
            t = (x - x[0]) * float(meta.iloc[i]["time_factor_to_hours"])
            mask = t <= 500 + 1e-9; scale = np.max(np.abs(y[mask]))
            ax.plot(t[mask], y[mask] / scale, marker=".", ms=3, lw=.8, alpha=.75, label=r["curve_id"] if rank <= 4 else None)
            provenance.append({"ifo_candidate": kind, "rank": rank, "score": r[kind], "curve_id": r["curve_id"],
                               "source_file": r["source_file"], "legend": r["legend"]})
        ax.axvline(200, color="grey", ls="--", lw=.8)
        ax.set_title(f"Post-hoc {kind}-like individual candidates"); ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.supxlabel("Elapsed time (h)"); fig.supylabel("Raw-point PCE / in-window MaxAbs")
    fig.suptitle("Four requested appearances exist as individual raw curves (not cluster labels; smoothing OFF)")
    fig.tight_layout(); fig.savefig(OUT / "individual_ifo_raw_point_gallery.png", dpi=240); plt.close(fig)
    pd.DataFrame(provenance).to_csv(OUT / "individual_ifo_candidate_provenance.csv", index=False)


if __name__ == "__main__":
    main()
