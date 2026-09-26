"""Render all numeric PCE candidates and the rule-consistent shape layer.

The gray density counts each candidate once per supported hour. Individual
trajectories and representative examples retain actual hours. The red line is
an explicitly synthetic event-landmark summary of rule-consistent candidates.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter


CLASSES = ("bridge", "hill", "slope", "valley")
COLORS = {"bridge": "#168477", "hill": "#d88424", "slope": "#326bb1", "valley": "#9b5da5"}
EXPECTED_ALL = {"bridge": 87, "hill": 176, "slope": 1868, "valley": 76}
EXPECTED_MATCHING = {"bridge": 24, "hill": 66, "slope": 768, "valley": 50}
GRID = np.arange(0.0, 401.0, 2.0)
Y_EDGES = np.linspace(-.05, 1.20, 126)
MAX_INTERPOLATION_GAP_H = 100.0
SHAPE_ZOOM_Y_LIMITS = {
    "bridge": (.93, 1.02),
    "hill": (.83, 1.02),
    "slope": (.74, 1.02),
    "valley": (.83, 1.02),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measured_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in read_rows(path):
        try:
            hour, pce = float(row["hour"]), float(row["normalized_pce"])
        except (KeyError, ValueError):
            continue
        if 0 <= hour <= 500 and np.isfinite(pce):
            points.append((hour, pce))
    points.sort()
    if len(points) < 2:
        raise ValueError(f"Too few numeric points: {path}")
    arr = np.asarray(points)
    hours, inverse = np.unique(arr[:, 0], return_inverse=True)
    values = np.bincount(inverse, weights=arr[:, 1]) / np.bincount(inverse)
    if len(hours) < 2:
        raise ValueError(f"Too few distinct hours: {path}")
    return hours, values


def supported_values(hours: np.ndarray, values: np.ndarray,
                     query: np.ndarray) -> np.ndarray:
    """Interpolate within measured spans whose bounding observations are <=100 h apart."""
    out = np.full(len(query), np.nan)
    j = np.searchsorted(hours, query, side="left")
    exact = (j < len(hours)) & (hours[np.minimum(j, len(hours) - 1)] == query)
    between = (j > 0) & (j < len(hours))
    acceptable = np.zeros(len(query), dtype=bool)
    acceptable[between] = (hours[j[between]] - hours[j[between] - 1]
                           <= MAX_INTERPOLATION_GAP_H)
    valid = exact | acceptable
    out[valid] = np.interp(query[valid], hours, values)
    return out


def at(curve: dict, hour: float) -> float:
    return float(supported_values(curve["hours"], curve["pce"],
                                  np.array([hour]))[0])


def observed_segments(curve: dict) -> list[np.ndarray]:
    """Leave data gaps above 100 h disconnected in the display."""
    hours, values = curve["hours"], curve["pce"]
    cut = np.flatnonzero(np.diff(hours) > MAX_INTERPOLATION_GAP_H) + 1
    return [a for a in np.split(np.column_stack((hours, values)), cut) if len(a) >= 2]


def event_landmarks(name: str, curves: list[dict]) -> list[dict]:
    def median_at(hour: float) -> tuple[float, int]:
        values = np.asarray([at(c, hour) for c in curves])
        valid = np.isfinite(values)
        if valid.sum() < max(5, int(np.ceil(.10 * len(curves)))):
            raise ValueError(f"Insufficient event support: {name} at {hour} h")
        return float(np.median(values[valid])), int(valid.sum())

    def observed_event(field: str, label: str) -> dict:
        pairs = []
        for curve in curves:
            raw = curve["index"].get(field, "")
            if not raw:
                continue
            hour = float(raw)
            pce = at(curve, hour)
            if np.isfinite(pce):
                pairs.append((hour, pce))
        if len(pairs) != len(curves):
            raise ValueError(f"Missing observed event: {name} {field}")
        return {"stage": label, "hour": float(np.median([p[0] for p in pairs])),
                "pce": float(np.median([p[1] for p in pairs])), "support": len(pairs)}

    first = {"stage": "first observed", "hour": float(np.median([c["hours"][0] for c in curves])),
             "pce": float(np.median([c["pce"][0] for c in curves])), "support": len(curves)}
    landmarks = [first]
    if name in ("bridge", "hill", "valley"):
        landmarks.append(observed_event("primary_turn_h", "initial trough" if name == "valley"
                                        else "initial peak"))
    if name == "valley":
        landmarks.append(observed_event("secondary_turn_h", "recovery crest in 0-200 h"))
    if name == "slope":
        pce, support = median_at(100)
        landmarks.append({"stage": "100 h", "hour": 100.0, "pce": pce,
                          "support": support})
    for hour in (200.0, 400.0):
        pce, support = median_at(hour)
        landmarks.append({"stage": f"{hour:g} h", "hour": hour, "pce": pce,
                          "support": support})
    times = np.array([p["hour"] for p in landmarks])
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"Nonmonotonic event times for {name}: {times}")
    return landmarks


def representative_ids(name: str, curves: list[dict]) -> set[str]:
    """Choose five source-diverse actual curves near the median event profile."""
    candidates = []
    for curve in curves:
        p200, p400 = at(curve, 200), at(curve, 400)
        if not (np.isfinite(p200) and np.isfinite(p400)):
            continue
        row = curve["index"]
        values = [float(curve["pce"][0])]
        floors = [.05]
        if name in ("bridge", "hill", "valley"):
            turn = float(row["primary_turn_h"])
            values.extend([turn, at(curve, turn)])
            floors.extend([20.0, .05])
        if name == "valley":
            crest = float(row["secondary_turn_h"])
            values.extend([crest, at(curve, crest)])
            floors.extend([20.0, .05])
        if name == "slope":
            values.append(at(curve, 100))
            floors.append(.05)
        values.extend([p200, p400])
        floors.extend([.05, .05])
        if all(np.isfinite(values)):
            candidates.append((curve, values, floors))
    features = np.asarray([x[1] for x in candidates])
    middle = np.median(features, axis=0)
    spread = np.maximum(np.quantile(features, .75, axis=0)
                        - np.quantile(features, .25, axis=0), candidates[0][2])
    scores = np.mean(((features - middle) / spread) ** 2, axis=1)
    selected, sources = set(), set()
    for i in np.argsort(scores):
        curve = candidates[int(i)][0]
        if curve["source"] not in sources:
            selected.add(curve["id"])
            sources.add(curve["source"])
        if len(selected) == 5:
            break
    if len(selected) < 5:
        raise ValueError(f"Too few source-diverse representatives: {name}")
    return selected


def make_layered_four_shapes(integrated_dir: Path, strict_index: Path,
                             output_dir: Path) -> dict[str, dict]:
    integrated = {row["curve_id"]: row for row in read_rows(integrated_dir / "class_index.csv")}
    strict = read_rows(strict_index)
    cohorts: dict[str, dict] = {}
    manifest_rows, landmark_rows, trend_rows, support_rows = [], [], [], []
    for name in CLASSES:
        curves = []
        for row in strict:
            if row["strict_200h_candidate_class"] != name:
                continue
            fid = row["curve_id"]
            source_row = integrated[fid]
            hours, values = measured_xy(integrated_dir / row["curve_csv"])
            matching = (row["evidence_status"] == "stage_candidate"
                        and row["post_200_matches_diagram"] == "True")
            curves.append({"id": fid, "source": row["source_group"], "hours": hours,
                           "pce": values, "index": source_row, "matching": matching,
                           "csv": row["curve_csv"]})
        matching_curves = [c for c in curves if c["matching"]]
        if len(curves) != EXPECTED_ALL[name] or len(matching_curves) != EXPECTED_MATCHING[name]:
            raise ValueError(f"Changed locked cohort: {name} {len(curves)}/{len(matching_curves)}")
        representatives = representative_ids(name, matching_curves)
        landmarks = event_landmarks(name, matching_curves)
        for point in landmarks:
            landmark_rows.append({"class": name, "stage": point["stage"],
                                  "median_hour": f"{point['hour']:.6f}",
                                  "median_normalized_pce": f"{point['pce']:.6f}",
                                  "supporting_curves": point["support"],
                                  "matching_cohort_size": len(matching_curves)})
        times = np.asarray([p["hour"] for p in landmarks])
        pces = np.asarray([p["pce"] for p in landmarks])
        display_hours = np.r_[times[0], np.arange(np.ceil(times[0] / 2) * 2, 400, 2), 400.0]
        display_pces = PchipInterpolator(times, pces)(display_hours)
        for hour, pce in zip(display_hours, display_pces):
            trend_rows.append({"class": name, "hour": f"{hour:.6f}",
                               "synthetic_event_trend_normalized_pce": f"{pce:.6f}",
                               "matching_cohort_size": len(matching_curves)})
        density_count = np.zeros((len(Y_EDGES) - 1, len(GRID)), dtype=int)
        support_all = np.zeros(len(GRID), dtype=int)
        support_matching = np.zeros(len(GRID), dtype=int)
        for curve in curves:
            y_grid = supported_values(curve["hours"], curve["pce"], GRID)
            fallback = not np.isfinite(y_grid).any()
            if fallback:
                # Four very short curves fall between adjacent 2 h grid positions.
                # Add their measured points to the nearest bins so every input
                # contributes to the density without inventing a connecting span.
                for hour, pce in zip(curve["hours"], curve["pce"]):
                    if hour <= 400:
                        y_grid[int(np.argmin(np.abs(GRID - hour)))] = pce
            valid = np.isfinite(y_grid)
            if not valid.any():
                raise ValueError(f"No density contribution: {curve['id']}")
            support_all += valid.astype(int)
            if curve["matching"]:
                support_matching += valid.astype(int)
            columns = np.flatnonzero(valid)
            bins = np.searchsorted(Y_EDGES, y_grid[valid], side="right") - 1
            in_range = (bins >= 0) & (bins < len(Y_EDGES) - 1)
            np.add.at(density_count, (bins[in_range], columns[in_range]), 1)
            manifest_rows.append({
                "class": name, "curve_id": curve["id"], "source_group": curve["source"],
                "evidence_status": curve["index"]["evidence_status"],
                "post_200_matches_diagram": curve["index"]["post_200_matches_diagram"],
                "rule_consistent_layer": str(curve["matching"]),
                "source_diverse_representative": str(curve["id"] in representatives),
                "observed_after_200_to_400": str(bool(np.any((curve["hours"] > 200)
                                                              & (curve["hours"] <= 400)))),
                "density_nearest_bin_fallback": str(fallback),
                "density_grid_hour_count": int(valid.sum()),
                "last_observed_hour_to_500": f"{curve['hours'][-1]:.6f}",
                "source_curve_csv": curve["csv"],
            })
        for j, hour in enumerate(GRID):
            support_rows.append({"class": name, "hour": int(hour),
                                 "all_candidate_support": int(support_all[j]),
                                 "rule_consistent_support": int(support_matching[j]),
                                 "all_candidate_total": len(curves),
                                 "rule_consistent_total": len(matching_curves)})
        cohorts[name] = {"all": curves, "matching": matching_curves,
                         "representatives": representatives, "landmarks": landmarks,
                         "density": density_count, "support_all": support_all,
                         "support_matching": support_matching,
                         "source_groups": len({c["source"] for c in matching_curves})}

    write_rows(output_dir / "layered_shape_manifest.csv", manifest_rows)
    write_rows(output_dir / "layered_shape_landmarks.csv", landmark_rows)
    write_rows(output_dir / "layered_shape_event_trend.csv", trend_rows)
    write_rows(output_dir / "layered_shape_hourly_support.csv", support_rows)
    np.savez_compressed(output_dir / "layered_shape_density.npz", hour=GRID, pce_edges=Y_EDGES,
                        **{f"{name}_counts": cohorts[name]["density"] for name in CLASSES})

    for variant in ("shape_zoom", "focused", "shared_scale"):
        fig = plt.figure(figsize=(15, 10.7))
        outer = fig.add_gridspec(2, 2, left=.065, right=.985, top=.91,
                                 bottom=.135, wspace=.16, hspace=.21)
        for i, name in enumerate(CLASSES):
            layer = cohorts[name]
            inner = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[4.2, .75], hspace=.02)
            ax = fig.add_subplot(inner[0])
            cov = fig.add_subplot(inner[1], sharex=ax)
            ax.axvspan(0, 200, color="#edf5ec", zorder=0)
            ax.axvspan(200, 400, color="#fff4e2", zorder=0)
            density = gaussian_filter(layer["density"].astype(float), sigma=(1.0, .8))
            density /= np.maximum(1, layer["support_all"])[None, :]
            positive = density[density > 0]
            vmax = float(np.quantile(positive, .995)) if len(positive) else 1.0
            ax.imshow(density, extent=(-1, 401, Y_EDGES[0], Y_EDGES[-1]), origin="lower",
                      aspect="auto", cmap="Greys", vmin=0, vmax=max(vmax, .01),
                      alpha=.10 if variant == "shape_zoom" else .27,
                      interpolation="bilinear", zorder=1)
            matching_segments = [segment for curve in layer["matching"]
                                 for segment in observed_segments(curve)]
            if variant == "shape_zoom":
                alpha = (.038 if len(layer["matching"]) > 500 else
                         .115 if len(layer["matching"]) > 50 else .22)
            else:
                alpha = (.028 if len(layer["matching"]) > 500 else
                         .085 if len(layer["matching"]) > 50 else .17)
            ax.add_collection(LineCollection(matching_segments, colors=COLORS[name],
                                             linewidths=.7, alpha=alpha, zorder=2))
            for curve in layer["matching"]:
                if curve["id"] in layer["representatives"]:
                    for segment in observed_segments(curve):
                        ax.plot(segment[:, 0], segment[:, 1], color="#233d55", linewidth=1.1,
                                alpha=.72, zorder=3)
            landmarks = layer["landmarks"]
            event_t = np.asarray([p["hour"] for p in landmarks])
            event_y = np.asarray([p["pce"] for p in landmarks])
            line_t = np.linspace(event_t[0], 400, 600)
            ax.plot(line_t, PchipInterpolator(event_t, event_y)(line_t),
                    color="#d6282d", linewidth=3.1, zorder=5)
            ax.scatter(event_t, event_y, color="#d6282d", s=16, zorder=6)
            ax.axvline(200, color="#80676b", linestyle="--", linewidth=1, zorder=4)
            all_n, match_n = len(layer["all"]), len(layer["matching"])
            ax.set_title(f"{name.title()} · all {all_n} · rule-consistent {match_n}",
                         weight="bold", fontsize=11)
            ax.set_ylabel("Normalized PCE")
            ax.set_xlim(0, 400)
            if variant == "shape_zoom":
                ax.set_ylim(*SHAPE_ZOOM_Y_LIMITS[name])
                ax.text(.02, .03, "Y-axis magnified", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=7.3, color="#334453",
                        bbox={"facecolor": "white", "alpha": .83,
                              "edgecolor": "none"}, zorder=7)
            elif variant == "focused":
                lower = {"bridge": .70, "hill": .55, "slope": -.05, "valley": .35}[name]
                ax.set_ylim(lower, 1.10)
            else:
                ax.set_ylim(-.05, 1.20)
            ax.grid(alpha=.12, linewidth=.5)
            s_all = layer["support_all"]
            s_match = layer["support_matching"]
            ax.text(.98, .04,
                    f"Support at 200 / 400 h: all {s_all[100]} / {s_all[200]}"
                    f" · consistent {s_match[100]} / {s_match[200]}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.3,
                    color="#334453", bbox={"facecolor": "white", "alpha": .83,
                                             "edgecolor": "none"}, zorder=7)
            cov.plot(GRID, s_all / all_n * 100, color="#666e74", linewidth=1.1)
            cov.plot(GRID, s_match / match_n * 100, color=COLORS[name], linewidth=1.5)
            cov.axvline(200, color="#80676b", linestyle="--", linewidth=.8)
            cov.set(ylim=(0, 105), yticks=[0, 100], ylabel="Coverage %", xlabel="Hour (h)")
            cov.tick_params(axis="both", labelsize=7)
            cov.grid(alpha=.12, linewidth=.5)
            plt.setp(ax.get_xticklabels(), visible=False)
        title = ("Four PCE shapes · magnified normalized PCE axes" if variant == "shape_zoom"
                 else "Four PCE shapes: all 2,207 candidates and 908 rule-consistent trajectories")
        fig.suptitle(title,
                     fontsize=15, weight="bold", y=.975)
        if variant == "shape_zoom":
            fig.text(.5, .945,
                     "Each panel zooms its y-axis to reveal shape; trajectories outside the range are clipped",
                     ha="center", fontsize=9, color="#4d5964")
        fig.legend(handles=[
            Patch(facecolor="#9ca5ab", alpha=.5, label="All-candidate density"),
            Line2D([0], [0], color="#168477", linewidth=1.5, label="Rule-consistent measured curves"),
            Line2D([0], [0], color="#233d55", linewidth=1.5, label="Source-diverse actual examples"),
            Line2D([0], [0], color="#d6282d", linewidth=3, label="Synthetic median-event trend"),
        ], loc="lower center", ncol=4, frameon=False, fontsize=8,
            bbox_to_anchor=(.5, .032))
        stem = f"four_shapes_layered_{variant}"
        fig.savefig(output_dir / f"{stem}.png", dpi=170, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    summary = {
        name: {"all_numeric_candidates": len(cohorts[name]["all"]),
               "rule_consistent_candidates": len(cohorts[name]["matching"]),
               "rule_consistent_source_groups": cohorts[name]["source_groups"],
               "source_diverse_example_ids": sorted(cohorts[name]["representatives"]),
               "all_support_200_h": int(cohorts[name]["support_all"][100]),
               "all_support_400_h": int(cohorts[name]["support_all"][200]),
               "rule_consistent_support_200_h": int(cohorts[name]["support_matching"][100]),
               "rule_consistent_support_400_h": int(cohorts[name]["support_matching"][200])}
        for name in CLASSES
    }
    (output_dir / "layered_shape_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
