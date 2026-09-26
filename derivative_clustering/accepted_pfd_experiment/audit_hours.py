#!/usr/bin/env python3
"""Audit accepted curves, convert defensible time axes to hours, and integrate them.

The source dataset is never modified.  Outputs include a curve-level manifest,
a long-format hour dataset, a 0--200 h primary grid, summary JSON, and QA plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


PRIMARY_END_H = 200.0
PRIMARY_GRID_POINTS = 81
MIN_POINTS_PRIMARY = 4
MAX_PLAUSIBLE_DURATION_H = 20_000.0


def classify_time_axis(name: str | None, unit: str | None) -> tuple[str, float | None, str]:
    text = f"{name or ''} {unit or ''}".strip().lower()
    if not text:
        return "unknown", None, "missing axis name and unit"
    if "cycle" in text or "numbers" in text:
        return "cycles", None, "cycle count has no documented hours-per-cycle conversion"
    if "(d)" in text or re.search(r"\bday(s)?\b", text):
        return "days", 24.0, "days multiplied by 24"
    hour_tokens = (
        "time",
        "duration",
        "hour",
        "(h",
        " hr",
        "heating time",
        "storage time",
        "damp heat hour",
        "uv exposure time",
    )
    if any(token in text for token in hour_tokens):
        return "hours", 1.0, "already expressed in hours"
    return "unknown", None, "axis cannot be converted to hours without an external conversion"


def classify_y_axis(name: str | None, unit: str | None) -> str:
    text = f"{name or ''} {unit or ''}".strip().lower()
    if not text:
        return "unknown"
    if "norm" in text or "relative" in text or "a.u." in text:
        return "normalized_or_relative"
    if "%" in text or "pce" in text or "efficiency" in text:
        return "absolute_percent_like"
    return "other"


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [(float(row["x"]), float(row["y"])) for row in reader]
    if not rows:
        raise ValueError(f"empty curve: {path}")
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values: {path}")
    return values[:, 0], values[:, 1]


def combine_duplicate_times(time_h: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for t, value in zip(time_h, y):
        grouped[float(t)].append(float(value))
    ordered = sorted(grouped)
    combined_y = np.asarray([np.median(grouped[t]) for t in ordered], dtype=float)
    duplicates = int(sum(len(grouped[t]) - 1 for t in ordered))
    return np.asarray(ordered, dtype=float), combined_y, duplicates


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(source: Path, output: Path) -> dict[str, object]:
    manifest: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    primary_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    grid = np.linspace(0.0, PRIMARY_END_H, PRIMARY_GRID_POINTS)

    json_paths = sorted(source.rglob("validation_result.json"))
    for json_path in json_paths:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        axis = metadata.get("axis", {})
        x_axis = axis.get("x", {}) or {}
        y_axis = axis.get("y", {}) or {}
        time_class, factor, conversion = classify_time_axis(x_axis.get("name"), x_axis.get("unit"))
        y_class = classify_y_axis(y_axis.get("name"), y_axis.get("unit"))
        curve_paths = sorted((json_path.parent / "accepted").glob("*.csv"))

        for curve_path in curve_paths:
            x_raw, y_raw = read_curve(curve_path)
            curve_id = curve_path.stem
            row: dict[str, object] = {
                "curve_id": curve_id,
                "source_image": json_path.parent.name,
                "source_csv": str(curve_path.relative_to(source)),
                "record_id": metadata.get("record_id", ""),
                "x_name": x_axis.get("name") or "",
                "x_unit": x_axis.get("unit") or "",
                "time_class": time_class,
                "conversion_rule": conversion,
                "y_name": y_axis.get("name") or "",
                "y_unit": y_axis.get("unit") or "",
                "y_axis_type": y_class,
                "raw_points": int(x_raw.size),
                "duplicate_time_points": "",
                "duration_h": "",
                "points_0_200h": "",
                "initial_reference": "",
                "primary_included": False,
                "confidence": "excluded",
                "exclusion_reason": "",
            }

            if factor is None:
                row["exclusion_reason"] = conversion
                manifest.append(row)
                continue

            time_h = factor * (x_raw - np.min(x_raw))
            order = np.argsort(time_h, kind="mergesort")
            time_h, y_sorted = time_h[order], y_raw[order]
            time_h, y_clean, duplicate_count = combine_duplicate_times(time_h, y_sorted)
            duration = float(time_h[-1])
            points_primary = int(np.count_nonzero(time_h <= PRIMARY_END_H))
            reference_count = min(3, y_clean.size)
            y_reference = float(np.median(y_clean[:reference_count]))
            y_relative = y_clean / y_reference if y_reference > 0 else np.full_like(y_clean, np.nan)

            row.update(
                {
                    "duplicate_time_points": duplicate_count,
                    "duration_h": duration,
                    "points_0_200h": points_primary,
                    "initial_reference": y_reference,
                }
            )

            reasons: list[str] = []
            if duration > MAX_PLAUSIBLE_DURATION_H:
                reasons.append(f"implausible duration > {MAX_PLAUSIBLE_DURATION_H:g} h")
            if duration < PRIMARY_END_H:
                reasons.append(f"does not cover {PRIMARY_END_H:g} h")
            if points_primary < MIN_POINTS_PRIMARY:
                reasons.append(f"fewer than {MIN_POINTS_PRIMARY} observed points by {PRIMARY_END_H:g} h")
            if y_reference <= 0:
                reasons.append("non-positive initial reference")

            for t, raw, relative in zip(time_h, y_clean, y_relative):
                long_rows.append(
                    {
                        "curve_id": curve_id,
                        "source_image": json_path.parent.name,
                        "time_h": float(t),
                        "y_raw": float(raw),
                        "y_relative": float(relative),
                        "original_x_name": x_axis.get("name") or "",
                        "original_x_unit": x_axis.get("unit") or "",
                        "conversion_rule": conversion,
                        "y_axis_type": y_class,
                    }
                )

            if reasons:
                row["exclusion_reason"] = "; ".join(reasons)
                row["confidence"] = "excluded"
            else:
                interpolation = PchipInterpolator(time_h, y_relative, extrapolate=False)
                interpolated = np.asarray(interpolation(grid), dtype=float)
                if not np.all(np.isfinite(interpolated)):
                    row["exclusion_reason"] = "interpolation produced missing values"
                else:
                    row["primary_included"] = True
                    row["confidence"] = "medium" if points_primary < 6 or y_class == "unknown" else "high"
                    for t, value in zip(grid, interpolated):
                        primary_rows.append(
                            {"curve_id": curve_id, "time_h": float(t), "y_relative": float(value)}
                        )
                    primary_curves.append((curve_id, grid.copy(), interpolated))
            manifest.append(row)

    manifest_fields = list(manifest[0])
    write_csv(output / "curve_manifest.csv", manifest_fields, manifest)
    write_csv(
        output / "integrated_hours_long.csv",
        [
            "curve_id",
            "source_image",
            "time_h",
            "y_raw",
            "y_relative",
            "original_x_name",
            "original_x_unit",
            "conversion_rule",
            "y_axis_type",
        ],
        long_rows,
    )
    write_csv(output / "primary_0_200h_long.csv", ["curve_id", "time_h", "y_relative"], primary_rows)

    status_counts = Counter(str(row["time_class"]) for row in manifest)
    exclusion_counts = Counter(
        str(row["exclusion_reason"]) for row in manifest if not bool(row["primary_included"])
    )
    durations = [float(row["duration_h"]) for row in manifest if row["duration_h"] != ""]
    summary: dict[str, object] = {
        "source_figures": len(json_paths),
        "total_curves": len(manifest),
        "time_axis_curve_counts": dict(status_counts),
        "hour_convertible_curves": len(long_rows) and len({str(row["curve_id"]) for row in long_rows}),
        "primary_window_h": PRIMARY_END_H,
        "primary_grid_points": PRIMARY_GRID_POINTS,
        "primary_included_curves": len(primary_curves),
        "primary_high_confidence": sum(row["confidence"] == "high" for row in manifest),
        "primary_medium_confidence": sum(row["confidence"] == "medium" for row in manifest),
        "excluded_curves": sum(not bool(row["primary_included"]) for row in manifest),
        "exclusion_reason_counts": dict(exclusion_counts),
        "duration_h_median_convertible": float(np.median(durations)) if durations else None,
        "parameters": {
            "max_plausible_duration_h": MAX_PLAUSIBLE_DURATION_H,
            "min_observed_points_0_200h": MIN_POINTS_PRIMARY,
            "initial_reference": "median of first up to three observed y values",
            "duplicate_time_rule": "median y at identical converted time_h",
            "interpolation": "PCHIP",
        },
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(status_counts.keys(), status_counts.values(), color="#3264a8")
    axes[0, 0].set_title("Time-axis audit (218 curves)")
    axes[0, 0].set_ylabel("curve count")
    axes[0, 0].tick_params(axis="x", rotation=20)

    plausible = [value for value in durations if value <= MAX_PLAUSIBLE_DURATION_H]
    axes[0, 1].hist(plausible, bins=30, color="#4c9f70", edgecolor="white")
    axes[0, 1].axvline(PRIMARY_END_H, color="#b3261e", linestyle="--", label="200 h")
    axes[0, 1].set_title("Convertible duration distribution")
    axes[0, 1].set_xlabel("duration (h)")
    axes[0, 1].legend()

    coverage_grid = np.asarray([50, 100, 150, 200, 300, 500, 750, 1000], dtype=float)
    coverage = [sum(value >= point and value <= MAX_PLAUSIBLE_DURATION_H for value in durations) for point in coverage_grid]
    axes[1, 0].plot(coverage_grid, coverage, marker="o", color="#7a3e9d")
    axes[1, 0].set_title("Curve coverage by required duration")
    axes[1, 0].set_xlabel("required coverage (h)")
    axes[1, 0].set_ylabel("eligible curves")
    axes[1, 0].grid(alpha=0.25)

    for _, t, values in primary_curves:
        axes[1, 1].plot(t, values, color="#3977b8", alpha=0.10, linewidth=0.8)
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_title(f"Primary integrated curves (n={len(primary_curves)})")
    axes[1, 1].set_xlabel("time (h)")
    axes[1, 1].set_ylabel("relative performance")
    axes[1, 1].set_ylim(bottom=min(-0.1, axes[1, 1].get_ylim()[0]))
    figure.tight_layout()
    figure.savefig(output / "audit_overview.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "accepted" / "samples_test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "01_hour_audit",
    )
    args = parser.parse_args()
    summary = audit(args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
