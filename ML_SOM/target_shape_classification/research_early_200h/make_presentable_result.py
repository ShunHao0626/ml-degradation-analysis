"""Make a thesis/result-style presentation of the locked 0–200 h candidates.

This reads the integrated delivery and strict numeric evidence view. It does not
change classifications, retrain SOM, or turn rule candidates into references.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.interpolate import PchipInterpolator

from make_layered_four_shapes import make_layered_four_shapes


ROOT = Path(__file__).resolve().parent
INTEGRATED = ROOT / "final_integrated_20260925"
STRICT = ROOT / "evaluation/strict_numeric_coverage_20260925/strict_200h_class_index.csv"
OUT = INTEGRATED / "result"
CLASSES = ("bridge", "hill", "slope", "valley")
COLORS = {"bridge": "#168477", "hill": "#d88424", "slope": "#326bb1", "valley": "#9b5da5"}
# These are illustrative observed curves selected after visual inspection. They
# are not prototypes, validation references, or a criterion used for modeling.
EXAMPLE_IDS = {"bridge": "F00953", "hill": "F01221", "slope": "F02188", "valley": "F02161"}
STAGE_EXAMPLE_IDS = {"bridge": "F00953", "hill": "F01477", "slope": "F02237", "valley": "F02161"}
STATUS_COLORS = {
    "stage_candidate": "#388874",
    "insufficient_evidence": "#e9b44c",
    "pattern_conflict": "#d16b63",
}
# Exploratory presentation subset, defined on locked automatic features. The
# all-candidate distribution remains available separately and is not replaced.
GROUP_MIN_AMPLITUDE = {"bridge": 0.05, "hill": 0.04, "slope": 0.20, "valley": 0.10}
GROUP_MIN_REVERSAL = {"hill": 0.05, "valley": 0.05}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path)
    xy = []
    for row in rows:
        try:
            t, y = float(row["hour"]), float(row["normalized_pce"])
        except (ValueError, KeyError):
            continue
        if 0 <= t <= 200 and np.isfinite(y):
            xy.append((t, y))
    if len(xy) < 2:
        raise ValueError(f"Fewer than two strict numeric points: {path}")
    xy.sort()
    arr = np.asarray(xy, dtype=float)
    t_unique, inverse = np.unique(arr[:, 0], return_inverse=True)
    if len(t_unique) < 2:
        raise ValueError(f"Fewer than two distinct strict hours: {path}")
    if len(t_unique) != len(arr):
        sums = np.bincount(inverse, weights=arr[:, 1])
        counts = np.bincount(inverse)
        return t_unique, sums / counts
    return arr[:, 0], arr[:, 1]


def read_full_curve(path: Path, end_h: float = 500,
                    require_post_200: bool = True) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in read_rows(path):
        try:
            t, y = float(row["hour"]), float(row["normalized_pce"])
        except (ValueError, KeyError):
            continue
        if 0 <= t <= end_h and np.isfinite(y):
            points.append((t, y))
    points.sort()
    if len(points) < 2 or (require_post_200 and not any(t > 200 for t, _ in points)):
        raise ValueError(f"Curve lacks required measured observations: {path}")
    arr = np.asarray(points, dtype=float)
    return arr[:, 0], arr[:, 1]


def make_stage_examples(index_by_id: dict[str, dict[str, str]]) -> None:
    """Plot observed examples on a common hour axis with the 200 h division."""
    callouts = {
        "bridge": [("rapid increase", 42, (65, 0.58)), ("slow decay", 330, (325, 0.72))],
        "hill": [("rapid increase", 48, (12, 0.58)), ("rapid decay", 155, (123, 0.45)),
                 ("slow decay", 350, (355, 0.60))],
        "slope": [("rapid decay", 100, (70, 0.55)), ("slow decay", 355, (345, 0.55))],
        "valley": [("rapid decay", 50, (48, 0.47)), ("power recovery", 150, (155, 0.50)),
                   ("slow decay", 440, (365, 0.55))],
    }
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 8.6), constrained_layout=True)
    for ax, name in zip(axs.flat, CLASSES):
        fid = STAGE_EXAMPLE_IDS[name]
        row = index_by_id[fid]
        if (row["integrated_class"] != name or row["evidence_status"] != "stage_candidate"
                or row["post_200_matches_diagram"] != "True"):
            raise ValueError(f"Selected stage example does not meet locked status: {fid}")
        t, y = read_full_curve(INTEGRATED / row["curve_csv"])
        ax.axvspan(0, 200, color="#e9f3e8", zorder=0)
        ax.axvspan(200, 500, color="#fff0d7", zorder=0)
        ax.plot(t, y, color=COLORS[name], linewidth=2.3, zorder=3)
        ax.scatter(t, y, s=3 if len(t) > 200 else 10, color=COLORS[name], zorder=4)
        ax.axvline(200, color="#a16565", linestyle="--", linewidth=1.1, zorder=2)
        for label, target_t, xytext in callouts[name]:
            target_y = float(np.interp(target_t, t, y))
            ax.annotate(label, xy=(target_t, target_y), xytext=xytext,
                        color="#224b39", fontsize=9.5, weight="bold",
                        arrowprops={"arrowstyle": "->", "color": "#89959a", "lw": 1.0},
                        bbox={"boxstyle": "round,pad=.18", "fc": "white", "ec": "none", "alpha": .76})
        ax.text(100, 0.015, "0–200 h", ha="center", va="bottom", fontsize=9, color="#56715e", weight="bold")
        ax.text(350, 0.015, "After 200 h", ha="center", va="bottom", fontsize=9, color="#9b7044", weight="bold")
        ax.set(xlim=(0, 500), ylim=(-0.04, 1.065), xlabel="Time (h)", ylabel="Normalized PCE")
        ax.set_title(f"{name.title()}  ·  observed curve {fid}", fontsize=12, weight="bold")
        ax.grid(alpha=.14, linewidth=.6)
    fig.suptitle("Four early PCE shapes · measured trajectories across the 200 h boundary",
                 fontsize=16, weight="bold")
    fig.savefig(OUT / "four_stage_measured_examples.png", dpi=165)
    fig.savefig(OUT / "four_stage_measured_examples.svg")
    plt.close(fig)


def median_with_support(curves: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    grid = np.arange(0.0, 201.0, 2.0)
    matrix = np.full((len(curves), len(grid)), np.nan)
    for i, curve in enumerate(curves):
        t, y = curve["hour"], curve["pce"]
        valid = (grid >= t[0]) & (grid <= t[-1])
        matrix[i, valid] = np.interp(grid[valid], t, y)
    support = np.sum(np.isfinite(matrix), axis=0)
    median = np.full(len(grid), np.nan)
    for j, count in enumerate(support):
        if count:
            median[j] = np.nanmedian(matrix[:, j])
    minimum = max(10, int(np.ceil(0.10 * len(curves))))
    median[support < minimum] = np.nan
    return grid, median, support, minimum


def plot_curves(ax, curves: list[dict], name: str, median: tuple) -> None:
    segments = [np.column_stack((c["hour"], c["pce"])) for c in curves]
    alpha = 0.045 if len(curves) > 500 else 0.14
    ax.add_collection(LineCollection(segments, colors=[(0.30, 0.38, 0.46, alpha)], linewidths=0.65))
    grid, med, support, minimum = median
    ax.plot(grid, med, color=COLORS[name], linewidth=2.7, label="Median at supported hours")
    ax.axvline(200, color="#bd4050", linestyle="--", linewidth=1.0)
    ax.set(xlim=(0, 202), ylim=(-0.055, 1.055), xlabel="Time (h)", ylabel="Normalized PCE")
    ax.set_title(f"{name.title()}  ·  {len(curves)} numeric candidates", fontsize=12, weight="bold")
    ax.grid(alpha=0.17, linewidth=0.6)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.text(0.99, 0.02, f"Median shown at n ≥ {minimum}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#576777")


def write_pngs(by_class: dict[str, list[dict]], medians: dict[str, tuple],
               integrated_counts: Counter, statuses: dict[str, Counter]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, name in zip(axs.flat, CLASSES):
        plot_curves(ax, by_class[name], name, medians[name])
    fig.suptitle("PCE shapes in the first 200 h · observed numeric curves", fontsize=17, weight="bold")
    fig.savefig(OUT / "four_class_overview.png", dpi=165)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, name in zip(axs.flat, CLASSES):
        curve = next(c for c in by_class[name] if c["id"] == EXAMPLE_IDS[name])
        ax.plot(curve["hour"], curve["pce"], "o-", color=COLORS[name],
                linewidth=2.0, markersize=2.8)
        ax.axvline(200, color="#bd4050", linestyle="--", linewidth=1.0)
        ax.set(xlim=(0, 202), ylim=(-0.055, 1.055), xlabel="Time (h)", ylabel="Normalized PCE")
        ax.set_title(f"{name.title()}  ·  {curve['id']}  ·  observed example", weight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Four early shapes · one illustrative measured curve per class", fontsize=16, weight="bold")
    fig.savefig(OUT / "illustrative_examples.png", dpi=165)
    plt.close(fig)
    for name in CLASSES:
        fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        plot_curves(ax, by_class[name], name, medians[name])
        fig.savefig(OUT / f"class_{name}.png", dpi=165)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    numeric = [len(by_class[c]) for c in CLASSES]
    image_only = [integrated_counts[c] - numeric[i] for i, c in enumerate(CLASSES)]
    x = np.arange(4)
    ax.bar(x, numeric, color=[COLORS[c] for c in CLASSES], label="Numeric 0–200 h candidates")
    ax.bar(x, image_only, bottom=numeric, color="#e7e2d8", edgecolor="#62666a", hatch="///",
           label="Image opinion only; numeric stage unresolved")
    for i, (n, m) in enumerate(zip(numeric, image_only)):
        ax.text(i, n + m + 12, str(n + m), ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x, [c.title() for c in CLASSES])
    ax.set_ylabel("Canonical curves")
    ax.set_title("Four-class delivery: 2,216 candidate records")
    ax.set_ylim(0, max(integrated_counts.values()) * 1.09)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "class_counts.png", dpi=165)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    bottom = np.zeros(4, dtype=int)
    for status in STATUS_COLORS:
        values = np.array([statuses[c][status] for c in CLASSES])
        ax.bar(x, values, bottom=bottom, label=status.replace("_", " "), color=STATUS_COLORS[status])
        bottom += values
    ax.set_xticks(x, [c.title() for c in CLASSES])
    ax.set_ylabel("Numeric candidates")
    ax.set_title("Evidence status remains separate from class")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "evidence_status.png", dpi=165)
    plt.close(fig)


def write_all_numeric_full_overview(by_class: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Show every numeric candidate through 400 h without selecting exemplars."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    median_rows = []
    coverage = {}
    grid = np.arange(0.0, 401.0, 2.0)
    for ax, name in zip(axs.flat, CLASSES):
        curves = by_class[name]
        n = len(curves)
        segments = [np.column_stack((c["full_hour"], c["full_pce"])) for c in curves]
        alpha = .016 if n > 500 else (.07 if n > 100 else .12)
        ax.axvspan(0, 200, color="#edf5ec", zorder=0)
        ax.axvspan(200, 400, color="#fff4e2", zorder=0)
        ax.add_collection(LineCollection(segments, colors=[(.30, .38, .46, alpha)],
                                         linewidths=.65, zorder=2))
        matrix = np.full((n, len(grid)), np.nan)
        post_observed = 0
        reached_400 = 0
        for i, curve in enumerate(curves):
            t, y = curve["full_hour"], curve["full_pce"]
            if np.any((t > 200) & (t <= 400)):
                post_observed += 1
            if t[-1] >= 400:
                reached_400 += 1
            valid = (grid >= t[0]) & (grid <= t[-1])
            matrix[i, valid] = np.interp(grid[valid], t, y)
        support = np.sum(np.isfinite(matrix), axis=0)
        minimum = max(10, int(np.ceil(.10 * n)))
        med = np.full(len(grid), np.nan)
        for j, count in enumerate(support):
            if count >= minimum:
                med[j] = float(np.nanmedian(matrix[:, j]))
            median_rows.append({"class": name, "hour": int(grid[j]),
                                "pointwise_median_normalized_pce": "" if not np.isfinite(med[j]) else f"{med[j]:.6f}",
                                "curve_span_support": int(count), "minimum_display_support": minimum,
                                "class_numeric_count": n})
        ax.plot(grid, med, color="#d6282d", linewidth=2.5, zorder=4,
                label="All-candidate pointwise median")
        ax.axvline(200, color="#84696a", linestyle="--", linewidth=1.0, zorder=3)
        ax.set(xlim=(0, 400), ylim=(-.05, 1.20), xlabel="Hour (h)",
               ylabel="Normalized PCE")
        ax.set_title(f"{name.title()} · all {n} numeric candidates", weight="bold")
        ax.text(.98, .06, f"Observed after 200 h: {post_observed}/{n}\nReached 400 h: {reached_400}/{n}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#364955", bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none"})
        ax.grid(alpha=.15, linewidth=.6)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=.85)
        coverage[name] = {"numeric_candidates": n, "observed_after_200_h_to_400_h": post_observed,
                          "reached_400_h": reached_400, "median_minimum_support": minimum}
    fig.suptitle("All 2,207 numeric candidates · measured trajectories (0–400 h)",
                 fontsize=16, weight="bold")
    fig.savefig(OUT / "all_numeric_2207_full_400h.png", dpi=165)
    fig.savefig(OUT / "all_numeric_2207_full_400h.svg")
    plt.close(fig)
    with (OUT / "all_numeric_2207_pointwise_median.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(median_rows[0]))
        writer.writeheader()
        writer.writerows(median_rows)
    return coverage


def write_group_trends(integrated: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Overlay a coherent measured subset and summarize event landmarks.

    Individual gray lines retain actual hours and normalized PCE. The red line
    is a *derived display trend*: a shape-preserving interpolation through
    medians of observed event times and PCE values, never a rescaled input.
    """
    cohorts: dict[str, list[dict]] = {}
    all_landmarks: dict[str, list[tuple[str, float, float]]] = {}
    cohort_rows = []
    landmark_rows = []
    trend_rows = []
    expected = {"bridge": 12, "hill": 14, "slope": 218, "valley": 7}
    for name in CLASSES:
        members = []
        for row in integrated:
            if (row["integrated_class"] != name or row["evidence_status"] != "stage_candidate"
                    or row["post_200_matches_diagram"] != "True"):
                continue
            amplitude = float(row["initial_amplitude"] or "nan")
            reversal = float(row["reversal_amplitude"] or "nan")
            turn = float(row["primary_turn_h"] or "nan")
            if not np.isfinite(amplitude) or amplitude < GROUP_MIN_AMPLITUDE[name]:
                continue
            if name in GROUP_MIN_REVERSAL and (not np.isfinite(reversal)
                                               or reversal < GROUP_MIN_REVERSAL[name]
                                               or not 15 <= turn <= 145):
                continue
            t, y = read_full_curve(INTEGRATED / row["curve_csv"], require_post_200=False)
            pre = t <= 200
            post = t > 200
            if (pre.sum() < 5 or post.sum() < 5 or t[0] > 40 or t[-1] < 400
                    or t[pre][-1] < 160 or t[post][0] > 250):
                continue
            members.append({"id": row["curve_id"], "source": row["source_group"],
                            "t": t, "y": y, "turn": turn,
                            "amplitude": amplitude, "reversal": reversal})
            cohort_rows.append({
                "class": name, "curve_id": row["curve_id"], "source_group": row["source_group"],
                "initial_amplitude": row["initial_amplitude"],
                "reversal_amplitude": row["reversal_amplitude"],
                "primary_turn_h": row["primary_turn_h"],
                "first_observed_h": f"{t[0]:.6f}", "last_observed_h": f"{t[-1]:.6f}",
                "strict_point_count": int(pre.sum()), "post_point_count_to_500h": int(post.sum()),
                "source_curve_csv": row["curve_csv"],
            })
        if len(members) != expected[name]:
            raise ValueError(f"Group trend cohort changed for {name}: {len(members)}")
        cohorts[name] = members

        def median_at(hour: float) -> float:
            return float(np.median([np.interp(hour, m["t"], m["y"]) for m in members]))

        anchors = [("first observed", float(np.median([m["t"][0] for m in members])),
                    float(np.median([m["y"][0] for m in members])))]
        if name in ("bridge", "hill", "valley"):
            median_turn_h = float(np.median([m["turn"] for m in members]))
            median_turn_y = float(np.median([np.interp(m["turn"], m["t"], m["y"])
                                             for m in members]))
            anchors.append(("initial peak" if name in ("bridge", "hill") else "initial trough",
                            median_turn_h, median_turn_y))
        else:
            anchors.append(("100 h", 100.0, median_at(100)))
        anchors.append(("200 h", 200.0, median_at(200)))
        if name == "valley":
            peaks = []
            for m in members:
                in_post = (m["t"] >= 200) & (m["t"] <= 400)
                peak_i = int(np.argmax(m["y"][in_post]))
                peaks.append((float(m["t"][in_post][peak_i]), float(m["y"][in_post][peak_i])))
            anchors.append(("post-200 recovery peak", float(np.median([p[0] for p in peaks])),
                            float(np.median([p[1] for p in peaks]))))
        anchors.append(("400 h", 400.0, median_at(400)))
        hours = np.asarray([a[1] for a in anchors])
        values = np.asarray([a[2] for a in anchors])
        if not np.all(np.diff(hours) > 0):
            raise ValueError(f"Nonmonotonic display landmarks for {name}")
        all_landmarks[name] = anchors
        for stage, hour, value in anchors:
            landmark_rows.append({"class": name, "stage": stage, "median_hour": f"{hour:.6f}",
                                  "median_normalized_pce": f"{value:.6f}",
                                  "cohort_size": len(members)})
        grid = np.r_[hours[0], np.arange(np.ceil(hours[0] / 2) * 2, 400, 2), 400.0]
        trend = PchipInterpolator(hours, values)(grid)
        for hour, value in zip(grid, trend):
            trend_rows.append({"class": name, "hour": f"{hour:.6f}",
                               "normalized_pce_display_trend": f"{value:.6f}",
                               "cohort_size": len(members)})

    def write_table(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_table(OUT / "group_trend_cohort.csv", cohort_rows)
    write_table(OUT / "group_trend_landmarks.csv", landmark_rows)
    write_table(OUT / "group_trend_display_curve.csv", trend_rows)

    for focused in (False, True):
        fig, axs = plt.subplots(2, 2, figsize=(14, 8.8), constrained_layout=True)
        for ax, name in zip(axs.flat, CLASSES):
            members = cohorts[name]
            ax.axvspan(0, 200, color="#edf5ec", zorder=0)
            ax.axvspan(200, 400, color="#fff4e2", zorder=0)
            for i, m in enumerate(members):
                ax.plot(m["t"], m["y"], color="#667580",
                        alpha=.17 if len(members) < 30 else .045, linewidth=.8, zorder=2,
                        label="Measured curves" if i == 0 else "_nolegend_")
            anchor = all_landmarks[name]
            hour = np.asarray([a[1] for a in anchor])
            value = np.asarray([a[2] for a in anchor])
            grid = np.linspace(hour[0], 400, 400)
            ax.plot(grid, PchipInterpolator(hour, value)(grid), color="#d6282d",
                    linewidth=3.0, zorder=4, label="Median event-landmark trend")
            ax.scatter(hour, value, color="#d6282d", s=15, zorder=5)
            ax.axvline(200, color="#84696a", linestyle="--", linewidth=1.0)
            ax.set_xlim(0, 400)
            if focused:
                lower = (0.0 if name == "slope" else
                         max(0.0, np.floor((min(value) - .20) * 10) / 10))
                ax.set_ylim(lower, 1.04)
            else:
                ax.set_ylim(0, 1.04)
            ax.set_title(f"{name.title()} · {len(members)} measured curves", weight="bold")
            ax.set_xlabel("Hour (h)")
            ax.set_ylabel("Normalized PCE")
            ax.grid(alpha=.15, linewidth=.6)
            ax.legend(loc="lower left", framealpha=.85, fontsize=7.5)
        fig.suptitle("Four class-group trends from measured trajectories (0–400 h)",
                     fontsize=16, weight="bold")
        basename = "group_trends_focused" if focused else "group_trends_shared_scale"
        fig.savefig(OUT / f"{basename}.png", dpi=165)
        fig.savefig(OUT / f"{basename}.svg")
        plt.close(fig)

    return {name: {"curves": len(cohorts[name]),
                   "source_groups": len({m["source"] for m in cohorts[name]})} for name in CLASSES}


def html_page(name: str, curves: list[dict], median: tuple) -> str:
    grid, med, support, minimum = median
    payload = {
        "name": name.title(), "color": COLORS[name], "minimumSupport": minimum,
        "defaultId": STAGE_EXAMPLE_IDS[name],
        "median": [[round(float(t), 4), round(float(y), 6)] for t, y in zip(grid, med) if np.isfinite(y)],
        "curves": [{
            "id": c["id"], "status": c["status"], "origin": c["origin"], "source": c["source"],
            "post_pattern": c["post_pattern"], "post_match": c["post_match"],
            "xy": [[round(float(t), 5), round(float(y), 6)] for t, y in zip(c["full_hour"], c["full_pce"])],
            "csv": f"../classes/{name}/{c['id']}.csv", "image": f"../classes/{name}/{c['id']}.svg",
        } for c in curves],
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLASS · 200 h 前后曲线</title><style>
body{font-family:system-ui,-apple-system,sans-serif;color:#253342;background:#f5f7f8;margin:0}header,main{max-width:1160px;margin:auto;padding:18px 24px}
header{background:#17364d;color:white;max-width:none;padding-left:max(24px,calc((100vw - 1112px)/2))}h1{margin:0;font-size:24px}p{line-height:1.5}
.card{background:white;border:1px solid #dbe3e9;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 7px #213d540b}
canvas{width:100%;height:520px;display:block}input,select{padding:9px;border:1px solid #bdc9d2;border-radius:6px;font-size:15px}input[type=text]{width:min(360px,90%)}input[type=checkbox]{width:auto}
table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #e1e8ec;padding:8px}button{border:0;background:none;color:#125d87;cursor:pointer;font:inherit}
a{color:#126184}.small{color:#526372;font-size:13px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
</style></head><body><header><h1>CLASS · 200 h 前后逐曲线</h1><div>数值规则候选 · 与 SOM 节点和人工参考标签分开记录</div></header>
<main><p><a href="index.html">← 结果首页</a> · <a href="../all_curves_index.html">完整逐曲线图库</a></p>
<div class="card"><div class="toolbar"><label>观察范围 <select id="view"><option value="500">0–500 h（含后段）</option><option value="200">0–200 h（严格初期）</option></select></label></div><canvas id="plot"></canvas><p id="cursor" class="small">灰线连接各曲线真实观测点；彩线仅为 0–200 h 的群组中位数。绿色为初期，黄色为后段。</p></div>
<div class="card"><div class="toolbar"><input type="text" id="search" placeholder="按曲线 ID 或来源搜索"><select id="status"><option value="">全部证据状态</option><option>stage_candidate</option><option>insufficient_evidence</option><option>pattern_conflict</option></select><label><input type="checkbox" id="strong">只看初期强候选且后段吻合</label><span id="match" class="small"></span></div>
<p id="selected">点击下方曲线编号，可在上图高亮。</p><table><thead><tr><th>曲线</th><th>证据状态</th><th>200 h 后模式</th><th>来源组</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="small">展示线连接已观测点，不外推。中位数只为初期视觉概括，由各曲线观测范围内的线性插值计算，至少 MINIMUM 条曲线支持才显示；峰谷时刻不同会被抹平。“初期强候选且后段吻合”是既有自动规则状态，不等于独立人工验证。</p>
</main><script id="dataset" type="application/json">DATA</script><script>
const d=JSON.parse(document.getElementById('dataset').textContent), canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
let selected=null, selectedCurve=null,maxH=500;const search=document.getElementById('search'),status=document.getElementById('status'),view=document.getElementById('view'),strongFilter=document.getElementById('strong');
const left=56,right=18,top=18,bottom=42;let W,H,pw,ph;
function coords(t,y){return [left+t/maxH*pw,top+(1.05-y)/1.105*ph]}
function resize(){const rect=canvas.getBoundingClientRect(),pixelRatio=window.devicePixelRatio||1;W=rect.width;H=rect.height;canvas.width=Math.round(W*pixelRatio);canvas.height=Math.round(H*pixelRatio);ctx.setTransform(pixelRatio,0,0,pixelRatio,0,0);pw=W-left-right;ph=H-top-bottom;draw()}
function line(points,color,width){ctx.beginPath();for(let i=0;i<points.length;i++){const [x,y]=coords(points[i][0],points[i][1]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){ctx.clearRect(0,0,W,H);ctx.fillStyle='white';ctx.fillRect(0,0,W,H);ctx.fillStyle='#fff1db';ctx.fillRect(left,top,pw,ph);ctx.fillStyle='#eaf4e9';ctx.fillRect(left,top,Math.min(pw,coords(200,0)[0]-left),ph);ctx.font='12px system-ui';ctx.lineWidth=1;
for(let t=0;t<=maxH;t+=(maxH===500?50:25)){let [x]=coords(t,0);ctx.strokeStyle='#dce6e8';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+ph);ctx.stroke();ctx.fillStyle='#435466';ctx.fillText(String(t),x-8,H-19)}
for(let y=0;y<=1.001;y+=.2){let yy=coords(0,y)[1];ctx.strokeStyle='#e7edf1';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(left+pw,yy);ctx.stroke();ctx.fillStyle='#435466';ctx.fillText(y.toFixed(1),10,yy+4)}
ctx.save();ctx.beginPath();ctx.rect(left,top,pw,ph);ctx.clip();for(const c of d.curves){if(status.value&&c.status!==status.value)continue;if(strongFilter.checked&&!(c.status==='stage_candidate'&&c.post_match==='True'))continue;line(c.xy,d.curves.length>500?'rgba(65,85,100,.034)':'rgba(65,85,100,.12)',.75)}
line(d.median,d.color,3);if(selectedCurve){line(selectedCurve.xy,'#c3233b',2.7);for(const p of selectedCurve.xy){const [x,y]=coords(...p);ctx.fillStyle='#c3233b';ctx.beginPath();ctx.arc(x,y,2.4,0,Math.PI*2);ctx.fill()}}ctx.restore();
ctx.strokeStyle='#a86971';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(coords(200,0)[0],top);ctx.lineTo(coords(200,0)[0],top+ph);ctx.stroke();ctx.setLineDash([]);
ctx.strokeStyle='#526372';ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,top+ph);ctx.lineTo(left+pw,top+ph);ctx.stroke();ctx.fillStyle='#253342';ctx.fillText('Time (h)',W/2-26,H-3);ctx.save();ctx.translate(13,H/2+42);ctx.rotate(-Math.PI/2);ctx.fillText('Normalized PCE',0,0);ctx.restore()}
function selectCurve(id){selected=id;selectedCurve=d.curves.find(c=>c.id===id);document.getElementById('selected').innerHTML='';let p=document.getElementById('selected');
const strong=document.createElement('strong');strong.textContent=id+' · '+selectedCurve.status+' · 200 h 后 '+selectedCurve.post_pattern+' · '+selectedCurve.origin;p.appendChild(strong);p.append('　');
for(const [label,url] of [['曲线图',selectedCurve.image],['逐点 CSV',selectedCurve.csv]]){const a=document.createElement('a');a.href=url;a.target='_blank';a.textContent=label;p.appendChild(a);p.append('　')}draw()}
function renderList(){const term=search.value.trim().toLowerCase(),v=status.value;const found=d.curves.filter(c=>(!v||c.status===v)&&(!strongFilter.checked||(c.status==='stage_candidate'&&c.post_match==='True'))&&(!term||c.id.toLowerCase().includes(term)||c.source.toLowerCase().includes(term)));
document.getElementById('match').textContent=`显示前 ${Math.min(80,found.length)} / ${found.length} 条`;const tbody=document.getElementById('rows');tbody.replaceChildren();for(const c of found.slice(0,80)){const tr=document.createElement('tr'),td1=document.createElement('td'),b=document.createElement('button');b.textContent=c.id;b.onclick=()=>selectCurve(c.id);td1.appendChild(b);tr.appendChild(td1);for(const value of [c.status,c.post_pattern,c.source]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}tbody.appendChild(tr)}}
search.addEventListener('input',renderList);status.addEventListener('change',()=>{renderList();draw()});strongFilter.addEventListener('change',()=>{renderList();draw()});view.addEventListener('change',()=>{maxH=Number(view.value);draw()});canvas.addEventListener('mousemove',e=>{if(!selectedCurve)return;const t=Math.max(0,Math.min(maxH,(e.offsetX-left)/pw*maxH)),shown=selectedCurve.xy.filter(p=>p[0]<=maxH);if(!shown.length)return;const nearest=shown.reduce((a,b)=>Math.abs(a[0]-t)<Math.abs(b[0]-t)?a:b);document.getElementById('cursor').textContent=`${selectedCurve.id} · 最近真实点：${nearest[0].toFixed(2)} h, normalized PCE ${nearest[1].toFixed(4)}`});
window.addEventListener('resize',resize);renderList();resize();selectCurve(d.defaultId);
</script></body></html>""".replace("CLASS", name.title()).replace("MINIMUM", str(minimum)).replace("DATA", data)


def write_index(summary: dict) -> None:
    cards = "".join(
        f'<a class="card" href="class_{c}_interactive.html"><img src="class_{c}.png" alt="{c.title()} curves"><b>{c.title()}</b><span>{summary["strict_numeric_candidates"][c]} numeric candidates · {summary["integrated_counts"][c]} integrated</span></a>'
        for c in CLASSES
    )
    count_rows = "".join(
        f'<tr><th>{name.title()}</th><td>{summary["original_file_class_counts"][name]}</td>'
        f'<td>{summary["integrated_counts"][name]}</td>'
        f'<td>{summary["strict_numeric_candidates"][name]}</td>'
        f'<td>{summary["som_posthoc_class_counts"][name]}</td></tr>'
        for name in ("slope", "bridge", "hill", "valley")
    )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCE 0–200 h · 四类结果</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;color:#233544;background:#f4f7f8;margin:0}}header{{background:#18374c;color:white;padding:28px max(24px,calc((100vw - 1150px)/2))}}h1{{margin:0 0 8px}}main{{max-width:1150px;margin:auto;padding:20px 24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}}.card{{display:flex;flex-direction:column;background:white;border:1px solid #dce4e8;border-radius:12px;overflow:hidden;text-decoration:none;color:inherit}}.card img{{width:100%;height:190px;object-fit:contain}}.card b,.card span{{padding:0 12px 8px}}.card span{{color:#526373;font-size:13px}}.feature{{background:white;border:1px solid #dce4e8;border-radius:12px;padding:15px;margin:16px 0}}.feature img{{width:100%;max-height:650px;object-fit:contain}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid #dce4e8}}th:first-child{{text-align:left}}a{{color:#126184}}p{{line-height:1.6}}
</style></head><body><header><h1>PCE 初期四类 · 结果总览</h1><div>0–200 h | 可追溯候选曲线库 | 2026-09-25</div></header><main>
<p><b>2,216 条四类整合记录</b>，其中 2,207 条有严格初期数值支持，9 条仅有原图判读意见。首图以密度呈现全部 2,207 条，并叠加 908 条自动规则一致候选及其形态趋势；四类名称是当前分段规则候选，不是 SOM 节点名称。</p>
<div class="feature"><h2>先分清数量口径</h2><table><thead><tr><th>类别</th><th>原文件映射<br>2250 个 CSV</th><th>去重整合<br>2246 条曲线</th><th>本页数值图<br>2207 条</th><th>SOM 节点后验命名<br>2205 条</th></tr></thead><tbody>{count_rows}</tbody></table><p>原文件含 4 个同曲线别名；去重整合的四类中另有 9 条仅原图意见，无法画成严格初期数值曲线。冻结 SOM 实际输出 100 个匿名节点，最后一列是依据训练代表曲线的规则对节点事后取名，<b>不是 SOM 自发得到四个已验证类别</b>。本页四类曲线图采用“本页数值图”一列。<a href="../../delivery/class_index.csv">查看 SOM 逐曲线后验字段</a>。</p></div>
<div class="feature"><h2>四类形态放大图：2,207 + 908 条</h2><img src="four_shapes_layered_shape_zoom.png" alt="四类按局部纵轴放大的曲线形态"><p>为看清小幅峰谷，四类面板分别放大 normalized PCE 的局部 y 轴：Bridge 0.93–1.02、Hill 0.83–1.02、Slope 0.74–1.02、Valley 0.83–1.02；<b>范围外的曲线部分会被裁切，原数值和小时未改变</b>。灰色密度由全部 2,207 条数值曲线计算；彩色细线是其中 908 条规则一致的实测曲线（Bridge 24、Hill 66、Slope 768、Valley 50）；深色线是不同来源的实测代表；红线是 908 条的事件关键点合成趋势。下方为逐时支持比例。<a href="four_shapes_layered_focused.png">较宽纵轴版</a> · <a href="four_shapes_layered_shared_scale.png">统一纵轴版</a> · <a href="layered_shape_manifest.csv">入图 ID 与层级</a>。</p></div>
<div class="feature"><h2>全量原始曲线叠加：0–400 h</h2><img src="all_numeric_2207_full_400h.png" alt="全部2207条数值曲线按四类叠加"><p>四类分别为 Bridge 87、Hill 176、Slope 1,868、Valley 76 条；红线是全量逐时中位数，峰谷发生时刻不同会使它变平。<a href="all_numeric_2207_full_400h.svg">下载矢量图</a> · <a href="all_numeric_2207_pointwise_median.csv">逐时中位数和支持数量</a>。</p></div>
<div class="feature"><h2>额外严格筛选的形态展示子集</h2><img src="group_trends_focused.png" alt="四类形态展示子集的多曲线叠加"><p>这张旧图只包含进一步满足完整覆盖和可见幅度要求的 Bridge {summary['group_trend_cohort']['bridge']['curves']}、Hill {summary['group_trend_cohort']['hill']['curves']}、Slope {summary['group_trend_cohort']['slope']['curves']}、Valley {summary['group_trend_cohort']['valley']['curves']} 条，不能代表全量。<a href="group_trends_shared_scale.png">统一纵轴版本</a> · <a href="group_trend_cohort.csv">入图 ID</a>。</p></div>
<div class="feature"><h2>200 h 前后四类形态：真实曲线</h2><img src="four_stage_measured_examples.png" alt="四类曲线的0–200 h和200 h后阶段"><p>绿色背景为 0–200 h，黄色背景为 200 h 后；四条线均连接真实观测点，未按示意图绘造。<a href="four_stage_measured_examples.svg">下载矢量图</a>。</p><p>自动规则同时标为“初期强候选”和“后段与图式吻合”的曲线：Bridge {summary['full_pattern_support']['bridge']}、Hill {summary['full_pattern_support']['hill']}、Slope {summary['full_pattern_support']['slope']}、Valley {summary['full_pattern_support']['valley']}。可在下方各类交互页勾选查看。</p></div>
<div class="feature"><h2>每类一条真实观测示例</h2><img src="illustrative_examples.png" alt="四类真实曲线示例"><p>示例便于辨认四类形态，经过人工挑选，不是 SOM 原型或独立验证参考。</p></div>
<div class="feature"><h2>只看严格初期 0–200 h</h2><img src="four_class_overview.png" alt="四类初期叠加图"><p>这也是全部 2,207 条数值曲线，但只显示 0–200 h；彩色逐时中位数会掩盖不同曲线峰谷发生时间的差异。</p></div><div class="grid">{cards}</div>
<div class="grid"><div class="feature"><a href="class_counts.png"><img src="class_counts.png" alt="四类数量"></a></div><div class="feature"><a href="evidence_status.png"><img src="evidence_status.png" alt="证据状态"></a></div></div>
<div class="feature"><h2>无标签 SOM 结构</h2><img src="som_structure.png" alt="冻结 SOM U-matrix 与节点占用"><p>这是冻结的 10×10 匿名 SOM 地图。节点并不等于 Bridge / Hill / Slope / Valley。<a href="../../evaluation/som_rule_alignment_20260925/README.md">查看节点与规则候选的对应审计</a>。</p></div>
<p><a href="../all_curves_index.html">逐曲线完整图库</a> · <a href="../class_index.csv">四类索引 CSV</a> · <a href="README.md">方法与限制</a> · <a href="../../CURRENT_RESEARCH_STATUS.md">当前研究状态</a></p>
</main></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_readme(summary: dict) -> None:
    lines = [
        "# PCE 初期四类结果展示", "",
        "此目录按 `thesis/result` 的图文入口方式展示本研究现有结果，使用的是当前 `original_curves_2250` 路线及 0–200 h 协议。**它是既有候选结果的展示，不重新分类，也不把 SOM 节点当作四类。**",
        "", "## 打开顺序", "",
        "1. [图文总览](index.html)：首页首图合并全量分布、规则一致的实测曲线和四类阶段趋势。",
        "2. [四类形态放大图](four_shapes_layered_shape_zoom.png)（[SVG](four_shapes_layered_shape_zoom.svg)）：灰色密度为全部 2207 条数值候选，彩线为其中 908 条自动规则一致候选，深色为不同来源的真实代表，红线为 908 条的事件关键点合成趋势。按类将 y 轴放大到 Bridge 0.93–1.02、Hill 0.83–1.02、Slope 0.74–1.02、Valley 0.83–1.02；范围外的成员片段会被裁切。另有[较宽纵轴版](four_shapes_layered_focused.png)和[统一纵轴版](four_shapes_layered_shared_scale.png)。",
        "3. [全量四类原始叠加图](all_numeric_2207_full_400h.png)（[SVG](all_numeric_2207_full_400h.svg)）：Bridge 87、Hill 176、Slope 1868、Valley 76 条灰线全部入图；红线为逐时中位数。[中位数与支持数](all_numeric_2207_pointwise_median.csv)可复核。",
        "4. [旧版进一步筛选的形态子集](group_trends_focused.png)：为显出峰谷，另要求完整覆盖和可见幅度；仅含 251 条，不代表全量。",
        "5. [200 h 前后四类单曲线示例](four_stage_measured_examples.png)：每类一条自动规则强候选且后段与图示相符的完整实测曲线。",
        "6. [全量严格初期叠加图](four_class_overview.png)：全部 2207 条数值候选的 0–200 h 视图。",
        "7. 分别查看 [Bridge](class_bridge_interactive.html)、[Hill](class_hill_interactive.html)、[Slope](class_slope_interactive.html)、[Valley](class_valley_interactive.html) 的交互页，可切换 0–200 h / 0–500 h、筛选初期强候选且后段吻合、按 ID／来源搜索，并打开逐曲线 CSV 和 SVG。",
        "8. [完整图库](../all_curves_index.html)包含全部整合记录及原图意见；[索引 CSV](../class_index.csv)保留路径、来源与证据状态。",
        "", "## 数量口径", "",
        "| 类别 | 原文件映射（2250） | 去重整合（2246） | 严格数值（2207） | SOM 节点后验命名（2205） |", "|---|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        lines.append(f"| {c.title()} | {summary['original_file_class_counts'][c]} | {summary['integrated_counts'][c]} | {summary['strict_numeric_candidates'][c]} | {summary['som_posthoc_class_counts'][c]} |")
    lines += [
        "", "2250 个原文件中有 4 个是同曲线别名；去重后四类整合记录 2216 条，另有 30 条未入四类。严格数值候选合计 2207；另有 9 条只依据原图给出四类意见，现有指定 CSV 在 0–200 h 只有 0 或 1 个有效时间点，**不画成数值曲线**。30 条未入四类者为 6 条用户判为不符合四类、24 条不属于目标实际小时 PCE 输入。SOM 最后一列是 10×10 匿名节点依据训练代表曲线规则作的**后验命名**，不是纯无监督四类真值；其数量与本页规则分类数量不能混用。规则一致候选数为 Bridge 24、Hill 66、Slope 768、Valley 50，旧版额外严格展示子集数为 12、14、218、7。",
        "", "## 图的计算约定", "",
        "- x 轴为 Hour (h)，y 轴为 Normalized PCE。叠加灰线连接每条曲线的真实点，不平滑或补造初期观测。图中 200 h 虚线标明观察边界。",
        "- 新版分层图的灰色密度以全部 2207 条数值候选计算：每条曲线在 2 h 网格的每个可支持位置最多贡献一次，按同类该小时支持数归一，再仅对显示密度作轻度高斯平滑；不同面板色深独立缩放，不可据颜色跨类比较数量。插值只发生在两端真实点相隔不超过 100 h 的区间；4 条短曲线完全落在相邻网格点之间时，以其真实点入最近 2 h 网格箱，未构造连接段。全部 ID、层级、入箱数量和来源在 [分层名单](layered_shape_manifest.csv)，未平滑密度整数矩阵在 [NPZ](layered_shape_density.npz)。",
        "- 彩色细线连接 908 条同时满足 `stage_candidate` 与 `post_200_matches_diagram=True` 的实测点；超过 100 h 的观测空档断线。深色线为每类自动选取的 5 条不同来源、最接近组内事件中位特征的真实曲线。红线取规则一致组的首点、初期峰／谷、200 h、400 h 等事件中位时间与中位 PCE，再作形状保持插值，是合成展示趋势；灰／彩／深色单曲线的小时坐标均未拉伸。Valley 的回升峰对多数候选发生在 200 h 前，新版红线如实呈现这一点。",
        "- 放大图只改变每个面板的显示 y 范围，不重新归一化、不平移、不拉伸任何一条曲线的 PCE 或 hour；裁切掉的曲线段仍参与原有密度、计数及支持表计算。灰色全量密度在放大图中以低透明度作背景，彩色实测曲线和红色趋势较醒目；绘图透明度只影响视觉，不改变数据或入图数量。需要比较真实变化幅度和全体离散程度时看较宽纵轴或统一纵轴版。",
        "- 每面板下方两条支持曲线分别是全量和规则一致候选在每个 2 h 时点、且相邻观测间隔不超过 100 h 的覆盖比例；主图另写出 200／400 h 的实际支持条数。见[逐时支持数](layered_shape_hourly_support.csv)、[事件关键点](layered_shape_landmarks.csv)、[合成红线坐标](layered_shape_event_trend.csv)及[摘要](layered_shape_summary.json)。这些是自动规则候选的展示，未进行独立人工准确率验证。",
        "- 全量 0–400 h 图没有按形态强弱或后段完整性筛选：2207 条数值候选全部画成灰线。其红线在每个 2 h 时点取观测跨度覆盖该时点的曲线之中位数，至少有该类 10% 且不少于 10 条支持才显示；不外推超过单条曲线的首末观测时间。后段有实测点的数量按 200–400 h 范围统计，图内另列能到达 400 h 的数量；逐类数据见 `summary.json` 的 `all_numeric_400h_coverage`。点间线段与逐时中位数插值只用于可视化，不等于其间有实测点。",
        "- 200 h 前后示例选用 Bridge F00953、Hill F01477、Slope F02237、Valley F02161。它们的 `evidence_status=stage_candidate`，且自动后段检查为 `post_200_matches_diagram=True`。示例线只连接源 CSV 中 0–500 h 的真实点，不做平均或拟合；箭头文字解释阶段，不是新标签。",
        "- 在 2207 条数值候选中，同时满足自动初期 `stage_candidate` 和 `post_200_matches_diagram=True` 的各类数量：Bridge 24、Hill 66、Slope 768、Valley 50。其余曲线可能证据不足、缺少后段观测、后段模式不明或与图示冲突；类别与证据状态须一起阅读。",
        "- 旧版额外筛选的形态展示子集图进一步选取记录充分且变化可见的曲线：严格窗口与后段各至少 5 个真实点，首点不晚于 40 h、末点不早于 400 h，200 h 前最后点不早于 160 h、后段首点不晚于 250 h；并要求初期幅度 Bridge ≥0.05、Hill ≥0.04 且下降幅度 ≥0.05、Slope ≥0.20、Valley ≥0.10 且回升幅度 ≥0.05 normalized PCE。Hill / Valley 主转折在 15–145 h。筛选仅服务于可视化，不改变全库类别或评价样本；完整入图 ID 在 [`group_trend_cohort.csv`](group_trend_cohort.csv)。",
        "- 群体趋势红线并非逐时平均或某一真实曲线：每类取入图曲线的首点、主转折、200 h 和 400 h 的中位时间／中位 PCE；Slope 在 100 h 增加一个中位点，Valley 增加 200 h 后回升峰。用形状保持插值连接这些中位关键点，仅用于展示，不参与分类或准确率计算。[关键点表](group_trend_landmarks.csv)和[红线数据](group_trend_display_curve.csv)可复核；灰线不拉伸小时轴，也不重归一化。",
        "- 彩色中位数只供展示：在每条曲线的**已观测时间范围内**按 2 h 网格线性插值；某时点至少有该类 10% 的曲线且不少于 10 条支持才显示。中位数没有用于分类，也不代表个体曲线必有同样峰谷。",
        "- 类别计数图包含 9 条原图意见；四类叠加图和交互页只包含 2207 条严格数值候选。证据状态图也只对 2207 条数值候选计数。",
        "- [SOM 结构图](som_structure.png)复制自冻结地图的审计结果，展示 10×10 匿名节点、训练和验证占用；与四类候选的关系见[独立对应审计](../../evaluation/som_rule_alignment_20260925/README.md)。",
        "", "## 不能由这些图得出的结论", "",
        "自动四类名称是规则候选，非独立盲判真值。不能从四个面板推断 SOM 天然分出恰好四簇，不能用图中类内中位数估计分类准确率，也不能据此声称特定物理退化机制。仍缺独立随机人工参考及 E3/E4 同预算比较。参见[当前研究状态](../../CURRENT_RESEARCH_STATUS.md)。",
        "", "此目录由 [`make_presentable_result.py`](../../make_presentable_result.py)从锁定的整合索引与逐曲线 CSV 生成；`summary.json` 记录输入哈希和实际数量。旧 `thesis/result` 结果没有作为输入。", "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    integrated = read_rows(INTEGRATED / "class_index.csv")
    original_files = read_rows(INTEGRATED / "all_files.csv")
    strict = read_rows(STRICT)
    som_rows = read_rows(ROOT / "delivery/class_index.csv")
    integrated_counts = Counter(row["integrated_class"] for row in integrated)
    original_counts = Counter(row["integrated_class"] for row in original_files)
    som_counts = Counter(row["som_posthoc_class"] for row in som_rows)
    if (sum(integrated_counts.values()) != 2216 or len(strict) != 2207
            or len(original_files) != 2250 or len(som_rows) != 2205
            or sum(som_counts.values()) != 2205):
        raise ValueError("Locked integrated / strict candidate totals have changed")
    by_class: dict[str, list[dict]] = defaultdict(list)
    statuses: dict[str, Counter] = defaultdict(Counter)
    ids: set[str] = set()
    for row in strict:
        name = row["strict_200h_candidate_class"]
        if name not in CLASSES or row["strict_200h_coverage_status"] != "numeric_rule_candidate_unvalidated":
            raise ValueError(f"Unexpected strict candidate status: {row['curve_id']}")
        fid = row["curve_id"]
        if fid in ids:
            raise ValueError(f"Duplicate curve: {fid}")
        ids.add(fid)
        path = INTEGRATED / row["curve_csv"]
        t, y = read_curve(path)
        full_t, full_y = read_full_curve(path, require_post_200=False)
        by_class[name].append({"id": fid, "hour": t, "pce": y,
                               "full_hour": full_t, "full_pce": full_y,
                               "status": row["evidence_status"],
                               "origin": row["classification_origin"],
                               "source": row["source_group"],
                               "post_pattern": row["post_200_pattern"],
                               "post_match": row["post_200_matches_diagram"]})
        statuses[name][row["evidence_status"]] += 1
    assert sum(len(v) for v in by_class.values()) == 2207
    assert all(set(statuses[name]) <= set(STATUS_COLORS) for name in CLASSES)
    medians = {name: median_with_support(by_class[name]) for name in CLASSES}
    write_pngs(by_class, medians, integrated_counts, statuses)
    all_numeric_coverage = write_all_numeric_full_overview(by_class)
    group_summary = write_group_trends(integrated)
    layered_summary = make_layered_four_shapes(INTEGRATED, STRICT, OUT)
    shutil.copyfile(ROOT / "evaluation/som_map.png", OUT / "som_structure.png")
    summary = {
        "protocol": "early200h-v2-extrema-2026-09-24",
        "presentation_source": "final_integrated_20260925 + strict_numeric_coverage_20260925",
        "integrated_counts": {name: integrated_counts[name] for name in CLASSES},
        "original_file_class_counts": {name: original_counts[name] for name in CLASSES},
        "som_posthoc_class_counts": {name: som_counts[name] for name in CLASSES},
        "strict_numeric_candidates": {name: len(by_class[name]) for name in CLASSES},
        "image_opinions_without_numeric_stage": 9,
        "illustrative_example_ids": EXAMPLE_IDS,
        "full_stage_example_ids": STAGE_EXAMPLE_IDS,
        "full_pattern_support": {name: sum(c["status"] == "stage_candidate" and c["post_match"] == "True"
                                           for c in by_class[name]) for name in CLASSES},
        "all_numeric_400h_coverage": all_numeric_coverage,
        "group_trend_cohort": group_summary,
        "layered_four_shapes": layered_summary,
        "nonconforming": 6,
        "out_of_scope": 24,
        "evidence_status_by_class": {name: dict(statuses[name]) for name in CLASSES},
        "class_index_sha256": sha256(INTEGRATED / "class_index.csv"),
        "original_files_index_sha256": sha256(INTEGRATED / "all_files.csv"),
        "som_posthoc_index_sha256": sha256(ROOT / "delivery/class_index.csv"),
        "strict_index_sha256": sha256(STRICT),
        "claim": "Presentation of rule-based candidate classes; no independent accuracy estimate",
    }
    for name in CLASSES:
        (OUT / f"class_{name}_interactive.html").write_text(
            html_page(name, by_class[name], medians[name]), encoding="utf-8")
    make_stage_examples({row["curve_id"]: row for row in integrated})
    write_index(summary)
    write_readme(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "numeric": sum(map(len, by_class.values())),
                      "integrated": sum(integrated_counts.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
