#!/usr/bin/env python3
"""Audit digitized curve records without modifying the source dataset.

The audit is deliberately conservative: only a time axis expressed in hours
(or days with an explicit unit conversion) and a PCE/efficiency y-axis qualify
for the strict PCE-hour pool. Power proxies and missing axis metadata are kept
for manual review but are not silently treated as PCE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def classify_x(name: object, unit: object) -> tuple[str, float | None]:
    name_text = clean(name)
    unit_text = clean(unit).strip("() ")
    combined = f"{name_text} {unit_text}".strip()
    if "cycle" in combined:
        return "non_time_cycle", None
    if unit_text in {"d", "day", "days"} or "time (d)" in name_text:
        return "time_days", 24.0
    hour_markers = ("hour", " hr", "(h)", " h)", "time h", "duration")
    if unit_text in {"h", "hr", "hrs", "hour", "hours"} or any(
        marker in f" {combined}" for marker in hour_markers
    ):
        return "time_hours", 1.0
    if "time" in name_text:
        return "time_unit_missing", None
    if not combined:
        return "missing", None
    return "not_time", None


def classify_y(name: object, unit: object) -> str:
    combined = f"{clean(name)} {clean(unit)}".strip()
    if "pce" in combined or "efficiency" in combined:
        return "pce_or_efficiency"
    if any(token in combined for token in ("pmax", "mppt", "output power", "spo")):
        return "power_proxy"
    if not combined:
        return "missing"
    return "not_pce"


def read_curve(path: Path) -> dict[str, object]:
    x_values: list[float] = []
    y_values: list[float] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {clean(field): field for field in (reader.fieldnames or [])}
        x_field = fields.get("x")
        y_field = fields.get("y")
        if x_field is None or y_field is None:
            return {
                "numeric_rows": 0,
                "malformed_rows": 0,
                "x_min": "",
                "x_max": "",
                "y_min": "",
                "y_max": "",
                "x_unique": 0,
                "x_monotonic": False,
                "csv_status": "missing_x_y_columns",
            }
        for row in reader:
            try:
                x = float(row[x_field])
                y = float(row[y_field])
            except (TypeError, ValueError):
                malformed += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                malformed += 1
                continue
            x_values.append(x)
            y_values.append(y)
    monotonic = all(b >= a for a, b in zip(x_values, x_values[1:]))
    status = "ok" if len(x_values) >= 4 and len(set(x_values)) >= 4 else "too_few_points"
    if x_values and max(x_values) <= min(x_values):
        status = "zero_x_span"
    if not monotonic:
        status = "x_not_monotonic"
    return {
        "numeric_rows": len(x_values),
        "malformed_rows": malformed,
        "x_min": min(x_values) if x_values else "",
        "x_max": max(x_values) if x_values else "",
        "y_min": min(y_values) if y_values else "",
        "y_max": max(y_values) if y_values else "",
        "x_unique": len(set(x_values)),
        "x_monotonic": monotonic,
        "csv_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    sample_dirs = sorted(
        (path for path in args.input.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for sample_dir in sample_dirs:
        result_path = sample_dir / "validation_result.json"
        metadata: dict[str, object] = {}
        metadata_status = "ok"
        try:
            metadata = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            metadata_status = "missing_or_invalid"

        axis = metadata.get("axis") if isinstance(metadata.get("axis"), dict) else {}
        x_axis = axis.get("x") if isinstance(axis.get("x"), dict) else {}
        y_axis = axis.get("y") if isinstance(axis.get("y"), dict) else {}
        x_name, x_unit = x_axis.get("name"), x_axis.get("unit")
        y_name, y_unit = y_axis.get("name"), y_axis.get("unit")
        x_class, hour_multiplier = classify_x(x_name, x_unit)
        y_class = classify_y(y_name, y_unit)
        mappings = metadata.get("series_mappings")
        mapping_count = len(mappings) if isinstance(mappings, list) else 0
        csv_paths = sorted(sample_dir.glob("accepted/*.csv"))

        if not csv_paths:
            rows.append(
                {
                    "sample": sample_dir.name,
                    "record_id": metadata.get("record_id", ""),
                    "csv_path": "",
                    "x_name": x_name or "",
                    "x_unit": x_unit or "",
                    "y_name": y_name or "",
                    "y_unit": y_unit or "",
                    "x_class": x_class,
                    "y_class": y_class,
                    "hour_multiplier": hour_multiplier or "",
                    "mapping_count": mapping_count,
                    "csv_count_in_sample": 0,
                    "metadata_status": metadata_status,
                    "numeric_rows": 0,
                    "malformed_rows": 0,
                    "x_min": "",
                    "x_max": "",
                    "x_max_hours": "",
                    "y_min": "",
                    "y_max": "",
                    "x_unique": 0,
                    "x_monotonic": False,
                    "csv_status": "missing_csv",
                    "strict_pce_hour_eligible": False,
                    "eligibility_reason": "missing_csv",
                }
            )
            continue

        for csv_path in csv_paths:
            curve = read_curve(csv_path)
            eligible = (
                x_class in {"time_hours", "time_days"}
                and y_class == "pce_or_efficiency"
                and curve["csv_status"] == "ok"
            )
            reasons: list[str] = []
            if x_class not in {"time_hours", "time_days"}:
                reasons.append(x_class)
            if y_class != "pce_or_efficiency":
                reasons.append(y_class)
            if curve["csv_status"] != "ok":
                reasons.append(str(curve["csv_status"]))
            x_max_hours = ""
            if curve["x_max"] != "" and hour_multiplier is not None:
                x_max_hours = float(curve["x_max"]) * hour_multiplier
            rows.append(
                {
                    "sample": sample_dir.name,
                    "record_id": metadata.get("record_id", ""),
                    "csv_path": str(csv_path.resolve()),
                    "x_name": x_name or "",
                    "x_unit": x_unit or "",
                    "y_name": y_name or "",
                    "y_unit": y_unit or "",
                    "x_class": x_class,
                    "y_class": y_class,
                    "hour_multiplier": hour_multiplier or "",
                    "mapping_count": mapping_count,
                    "csv_count_in_sample": len(csv_paths),
                    "metadata_status": metadata_status,
                    **curve,
                    "x_max_hours": x_max_hours,
                    "strict_pce_hour_eligible": eligible,
                    "eligibility_reason": "eligible" if eligible else ";".join(reasons),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "curve_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(args.input.resolve()),
        "sample_directories": len(sample_dirs),
        "manifest_rows": len(rows),
        "csv_curves": sum(bool(row["csv_path"]) for row in rows),
        "strict_pce_hour_curves": sum(bool(row["strict_pce_hour_eligible"]) for row in rows),
        "strict_pce_hour_samples": len(
            {row["sample"] for row in rows if row["strict_pce_hour_eligible"]}
        ),
        "curves_reaching_150h": sum(
            bool(row["strict_pce_hour_eligible"])
            and row["x_max_hours"] != ""
            and float(row["x_max_hours"]) >= 150.0
            for row in rows
        ),
        "x_class_counts": dict(Counter(str(row["x_class"]) for row in rows)),
        "y_class_counts": dict(Counter(str(row["y_class"]) for row in rows)),
        "csv_status_counts": dict(Counter(str(row["csv_status"]) for row in rows)),
        "ineligibility_reason_counts": dict(
            Counter(str(row["eligibility_reason"]) for row in rows if not row["strict_pce_hour_eligible"])
        ),
    }
    (args.output / "dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
