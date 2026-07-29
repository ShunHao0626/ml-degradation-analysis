#!/usr/bin/env python3
"""Scan all x_time_* directories and find curves whose max time does NOT exceed 200h."""

import os
import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict

BASE_DIR = Path("/Users/shunhao/Desktop/ML/lab/05_accepted_all copy")

from typing import Optional, Tuple

# Map directory -> base multiply factor to convert x-axis to hours
DIR_UNIT_FACTOR = {
    "x_time_h": 1.0,
    "x_time_day": 24.0,
    "x_time_min": 1.0 / 60.0,
    "x_time_week_month_year": None,  # parsed from unit hints below
}

UNIT_TO_HOURS = OrderedDict([
    ("seconds", 1.0 / 3600.0),
    ("s", 1.0 / 3600.0),
    ("sec", 1.0 / 3600.0),
    ("minutes", 1.0 / 60.0),
    ("min", 1.0 / 60.0),
    ("hours", 1.0),
    ("hour", 1.0),
    ("h", 1.0),
    ("days", 24.0),
    ("day", 24.0),
    ("d", 24.0),
    ("weeks", 24.0 * 7),
    ("week", 24.0 * 7),
    ("w", 24.0 * 7),
    ("months", 24.0 * 30),
    ("month", 24.0 * 30),
    ("years", 24.0 * 365),
    ("year", 24.0 * 365),
    ("y", 24.0 * 365),
])


def doi_from_dirname(dirname: str) -> str:
    m = re.match(r"^(\d+\.\d+)_([^/]+)$", dirname)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return dirname


def figure_from_folder(folder_name: str) -> str:
    """Extract 'Fig5_D' style panel label.

    Examples:
        'Fig5_D_stability_performance_curve_pce' -> 'Fig5_D'
        'Fig5' -> 'Fig5'
        'Fig23_whole_stability' -> 'Fig23'
        'Fig3' -> 'Fig3'
    Takes the first two underscore-separated tokens starting with 'Fig'.
    """
    parts = folder_name.split("_")
    if parts and re.match(r"^[Ff]ig\d+$", parts[0]):
        if len(parts) >= 2 and re.match(r"^[a-zA-Z]$", parts[1]):
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    return folder_name


def _norm_unit_string(s: str) -> str:
    """Strip non-alphabetic chars and lower-case."""
    return re.sub(r"[^a-z]", "", s.lower())


# Map from common alias tokens to canonical unit keys
UNIT_ALIASES = {
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "h": "hours",
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "w": "weeks", "wk": "weeks", "wks": "weeks",
    "month": "months", "months": "months", "m": "months", "mo": "months",
    "year": "years", "years": "years", "yr": "years", "yrs": "years", "y": "years",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds", "s": "seconds",
}


def _resolve_unit_tokens(text: str) -> Tuple[Optional[float], str]:
    """Try to match against any of the canonical unit tokens.

    Uses word-boundary matching after normalization so 'h' won't match 'hrs'
    and 's' won't match 'seconds'.
    """
    if not text:
        return None, ""
    raw = _norm_unit_string(text)
    if not raw:
        return None, ""
    # Match longest aliases first via regex word boundaries
    by_len = sorted(UNIT_ALIASES.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(a) for a in by_len) + r")\b"
    m = re.search(pattern, raw)
    if not m:
        return None, ""
    alias = m.group(1)
    canonical = UNIT_ALIASES[alias]
    return UNIT_TO_HOURS[canonical], canonical


def get_validation_unit(fig_dir: Path) -> Tuple[Optional[float], str]:
    """Read validation_result.json to get x-axis unit + label, return (hours_factor, label)."""
    vj = fig_dir / "validation_result.json"
    if not vj.exists():
        return None, ""
    try:
        with open(vj, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, ""
    x = data.get("axis", {}).get("x", {})
    unit = x.get("unit")
    name = (x.get("name") or "") if isinstance(x.get("name"), str) else ""
    # Try unit field first
    if unit is not None and isinstance(unit, str):
        factor, label = _resolve_unit_tokens(unit)
        if factor is not None:
            return factor, label
    # Fall back to the axis name
    factor, label = _resolve_unit_tokens(name)
    return factor, label


def get_replot_or_validation_unit(folder_path: Path) -> Tuple[float, str]:
    """Determine the (hours factor, unit label) for a given figure folder."""
    # Try validation json first
    factor, label = get_validation_unit(folder_path)
    if factor is not None:
        return factor, label
    # Otherwise infer from parent dir
    parts = folder_path.parts
    for p in parts:
        if p.startswith("x_time_"):
            if p == "x_time_min":
                return 1.0 / 60.0, "min"
            if p == "x_time_h":
                return 1.0, "h"
            if p == "x_time_day":
                return 24.0, "day"
            if p == "x_time_week_month_year":
                # Fall back to figure name hint
                f = folder_path.name.lower()
                if "week" in f:
                    return 24.0 * 7, "week"
                if "month" in f:
                    return 24.0 * 30, "month"
                if "year" in f:
                    return 24.0 * 365, "year"
                # default to hour
                return 1.0, "h"
    return 1.0, "h"


def read_x_values_from_csv(csv_path: Path) -> list[float]:
    xs: list[float] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header
        except StopIteration:
            return xs
        for row in reader:
            if not row:
                continue
            raw = row[0].strip()
            if not raw:
                continue
            try:
                xs.append(float(raw))
            except ValueError:
                continue
    return xs


def main():
    all_curves = []

    for top in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        top_path = BASE_DIR / top
        if not top_path.exists():
            print(f"[!] Missing: {top_path}", file=sys.stderr)
            continue

        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_from_dirname(doi_dir.name)

            for fig_dir in doi_dir.iterdir():
                if not fig_dir.is_dir():
                    continue
                if fig_dir.name in ("image",):
                    continue

                fig_label = figure_from_folder(fig_dir.name)
                h_factor, unit_label = get_replot_or_validation_unit(fig_dir)

                # Pick CSVs: prefer accepted subfolder; fall back to fig_dir
                accepted_dir = fig_dir / "accepted"
                csv_files: list[Path] = []
                if accepted_dir.is_dir():
                    csv_files = sorted(accepted_dir.glob("*.csv"))
                if not csv_files:
                    csv_files = sorted(fig_dir.glob("*.csv"))

                for csv_file in csv_files:
                    xs = read_x_values_from_csv(csv_file)
                    if not xs:
                        continue
                    max_x = max(xs)
                    max_x_h = max_x * h_factor
                    all_curves.append({
                        "top_dir": top,
                        "doi": doi,
                        "doi_dir": doi_dir.name,
                        "figure_folder": fig_dir.name,
                        "figure_label": fig_label,
                        "csv_file": str(csv_file.relative_to(BASE_DIR)),
                        "unit_label": unit_label or "?",
                        "max_x_raw": max_x,
                        "max_x_hours": max_x_h,
                        "n_points": len(xs),
                    })

    under200 = [c for c in all_curves if c["max_x_hours"] < 200.0]
    over200 = [c for c in all_curves if c["max_x_hours"] >= 200.0]
    under200.sort(key=lambda r: r["max_x_hours"])

    out_dir = BASE_DIR / "analysis_report"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "curves_under_200h_detailed.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["top_dir", "doi", "figure_label", "figure_folder", "unit_label",
                         "max_x_raw", "max_x_hours", "n_points", "csv_file"])
        for r in under200:
            writer.writerow([r["top_dir"], r["doi"], r["figure_label"], r["figure_folder"],
                             r["unit_label"], round(r["max_x_raw"], 3),
                             round(r["max_x_hours"], 3),
                             r["n_points"], r["csv_file"]])

    print(f"# Total curves scanned: {len(all_curves)}")
    print(f"# Curves < 200h: {len(under200)}")
    print(f"# Curves >= 200h: {len(over200)}")

    by_top = defaultdict(lambda: [0, 0])
    for c in all_curves:
        by_top[c["top_dir"]][1] += 1
    for c in under200:
        by_top[c["top_dir"]][0] += 1

    print("\ntop_dir | <200h | total")
    for k in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        v = by_top[k]
        print(f"{k} | {v[0]} | {v[1]}")

    dois_under = sorted({c["doi"] for c in under200})
    print(f"\n# unique DOIs under 200h: {len(dois_under)}")

    pairs = {(c["doi"], c["figure_label"]) for c in under200}
    print(f"# unique (DOI, Figure) pairs under 200h: {len(pairs)}")

    # Build a per-DOI "Figure list" with min/max hours observed
    by_doi = defaultdict(lambda: {"hours": [], "figures": set(), "folders": set()})
    for r in under200:
        d = r["doi"]
        by_doi[d]["hours"].append(r["max_x_hours"])
        by_doi[d]["figures"].add(r["figure_label"])
        by_doi[d]["folders"].add(r["figure_folder"])

    # Save comprehensive dictionary to JSON for use in report rendering
    summary_json = out_dir / "curves_under_200h_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_curves": len(all_curves),
            "under_200h": len(under200),
            "over_200h": len(over200),
            "by_top": {k: {"under_200h": v[0], "total": v[1]} for k, v in by_top.items()},
            "per_doi": {
                d: {
                    "min_hours": round(min(v["hours"]), 2),
                    "max_hours": round(max(v["hours"]), 2),
                    "figures": sorted(v["figures"]),
                    "folders": sorted(v["folders"]),
                } for d, v in by_doi.items()
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
