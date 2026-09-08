#!/usr/bin/env python3
"""Create paper-aligned fixed-length curve matrices from the audited CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from sklearn.preprocessing import MaxAbsScaler


def load_xy(path: Path, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    values: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values.append((float(row["x"]) * multiplier, float(row["y"])))
    values.sort()
    grouped: dict[float, list[float]] = {}
    for x_value, y_value in values:
        grouped.setdefault(x_value, []).append(y_value)
    x = np.array(sorted(grouped), dtype=float)
    y = np.array([np.mean(grouped[value]) for value in x], dtype=float)
    x = x - x[0]
    return x, y


def curve_id(sample: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{sample}_{digest}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))

    for window in config["windows_hours"]:
        out_dir = args.output_root / f"{window}h"
        out_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[np.ndarray] = []
        included: list[dict[str, object]] = []
        statuses: list[dict[str, object]] = []
        grid = np.arange(1, window * 6 + 1, dtype=float) / 6.0

        for audit in audit_rows:
            path_text = audit["csv_path"]
            status = {
                "sample": audit["sample"],
                "curve_id": "",
                "csv_path": path_text,
                "window_hours": window,
                "included": False,
                "reason": "",
                "duration_hours": "",
                "first_global_max_time_hours": "",
                "raw_points": audit["numeric_rows"],
            }
            if not path_text:
                status["reason"] = "missing_csv"
                statuses.append(status)
                continue
            path = Path(path_text)
            status["curve_id"] = curve_id(audit["sample"], path)
            if audit["strict_pce_hour_eligible"] != "True":
                status["reason"] = f"axis_or_csv:{audit['eligibility_reason']}"
                statuses.append(status)
                continue

            multiplier = float(audit["hour_multiplier"])
            x, y = load_xy(path, multiplier)
            duration = float(x[-1])
            status["duration_hours"] = duration
            if duration > float(config["quality_rules"]["max_plausible_duration_hours"]):
                status["reason"] = "implausible_duration_over_100000h"
                statuses.append(status)
                continue
            if duration + 1e-9 < window:
                status["reason"] = "shorter_than_window"
                statuses.append(status)
                continue

            first_max_index = int(np.flatnonzero(y == np.max(y))[0])
            first_max_time = float(x[first_max_index])
            status["first_global_max_time_hours"] = first_max_time
            if first_max_time > window + 1e-9:
                status["reason"] = "global_pce_max_after_window"
                statuses.append(status)
                continue

            interpolator = Akima1DInterpolator(x, y)
            interpolated = np.asarray(interpolator(grid), dtype=float)
            if not np.all(np.isfinite(interpolated)):
                status["reason"] = "akima_nonfinite_within_window"
                statuses.append(status)
                continue
            normalized = MaxAbsScaler().fit_transform(interpolated.reshape(-1, 1)).ravel()
            smoothed = savgol_filter(
                normalized,
                window_length=int(config["savgol_window_points"]),
                polyorder=int(config["savgol_polyorder"]),
            )
            prepared.append(smoothed)
            status["included"] = True
            status["reason"] = "included"
            statuses.append(status)
            included.append(
                {
                    **status,
                    "x_name": audit["x_name"],
                    "x_unit": audit["x_unit"],
                    "y_name": audit["y_name"],
                    "y_unit": audit["y_unit"],
                    "mapping_count": audit["mapping_count"],
                    "normalized_min_before_smoothing": float(np.min(normalized)),
                    "normalized_max_before_smoothing": float(np.max(normalized)),
                    "smoothed_min": float(np.min(smoothed)),
                    "smoothed_max": float(np.max(smoothed)),
                }
            )

        matrix = np.vstack(prepared) if prepared else np.empty((0, len(grid)))
        np.savez_compressed(out_dir / "preprocessed_curves.npz", data=matrix, time_hours=grid)
        write_csv(out_dir / "curve_metadata.csv", included, list(included[0]) if included else list(statuses[0]))
        write_csv(out_dir / "all_curve_status.csv", statuses, list(statuses[0]))
        reason_counts: dict[str, int] = {}
        for row in statuses:
            reason_counts[str(row["reason"])] = reason_counts.get(str(row["reason"]), 0) + 1
        summary = {
            "window_hours": window,
            "input_manifest_rows": len(audit_rows),
            "included_curves": len(included),
            "matrix_shape": list(matrix.shape),
            "time_grid_start_hours": float(grid[0]),
            "time_grid_end_hours": float(grid[-1]),
            "reason_counts": reason_counts,
        }
        (out_dir / "preprocessing_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
