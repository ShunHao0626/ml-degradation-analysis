#!/usr/bin/env python3
"""
load_raw_data.py
================
Load all raw CSV files from the four x_time_* directories.

Handles:
  x_time_h/       → x in hours
  x_time_day/     → x in days
  x_time_min/     → x in minutes (files are actually in hours, corrected)
  x_time_week_month_year/ → x in weeks/months, determined from validation JSON

Each curve → one row in a DataFrame with columns:
  csv_file, top_dir, doi, figure_folder, series_id,
  unit_label, max_x_raw, max_x_hours, n_points, x, y

The function mimics the logic of the existing build_over_200h_csv.py
script so the output schema is compatible with the SOM pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("load_raw")


# ------------------------------------------------------------------
# Unit conversion
# ------------------------------------------------------------------

UNIT_ALIASES: dict[str, str] = {
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "h": "hours",
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "w": "weeks", "wk": "weeks", "wks": "weeks",
    "month": "months", "months": "months", "m": "months", "mo": "months",
    "year": "years", "years": "years", "yr": "years", "yrs": "years", "y": "years",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds", "s": "seconds",
}

UNIT_TO_HOURS: dict[str, float] = {
    "seconds": 1.0 / 3600.0,
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 24.0 * 7,
    "months": 24.0 * 30,
    "years": 24.0 * 365,
}

DIR_UNIT_FACTOR: dict[str, float] = {
    "x_time_h": 1.0,               # hours
    "x_time_day": 24.0,           # days → hours
    "x_time_min": 1.0 / 60.0,    # minutes → hours
    "x_time_week_month_year": None,  # auto-detect per figure
}


# ------------------------------------------------------------------
# DOI / figure name utilities
# ------------------------------------------------------------------

def doi_from_dirname(dirname: str) -> str:
    m = re.match(r"^(\d+\.\d+)_([^/]+)$", dirname)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return dirname


def figure_from_folder(folder_name: str) -> str:
    """Extract 'Fig5_D' from folder name like 'Fig5_D_stability_performance_curve_pce'."""
    parts = folder_name.split("_")
    if parts and re.match(r"^[Ff]ig\d+$", parts[0]):
        if len(parts) >= 2 and re.match(r"^[a-zA-Z]$", parts[1]):
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    return folder_name


def series_id_from_csv_name(csv_name: str) -> str:
    m = re.search(r"__Series[_\s]+(\d+)", csv_name, re.IGNORECASE)
    if m:
        return f"Series_{m.group(1)}"
    return ""


# ------------------------------------------------------------------
# Unit detection from validation JSON
# ------------------------------------------------------------------

def _norm_unit_string(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def get_validation_unit(fig_dir: Path) -> tuple[Optional[float], str]:
    """Read validation_result.json to get x-axis unit."""
    vj = fig_dir / "validation_result.json"
    if not vj.exists():
        return None, ""
    try:
        with open(vj, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, ""

    x = data.get("axis", {}).get("x", {})
    unit = x.get("unit")
    name = x.get("name", "") or ""

    for src in [unit, name]:
        if not isinstance(src, str):
            continue
        norm = _norm_unit_string(src)
        for alias, canonical in UNIT_ALIASES.items():
            if norm == alias or norm == _norm_unit_string(alias):
                factor = UNIT_TO_HOURS[canonical]
                logger.debug("  validation: '%s' → %s (%.4f)", src, canonical, factor)
                return factor, canonical
    return None, ""


def get_figure_unit_from_folder(folder_path: Path) -> tuple[float, str]:
    """Determine (hours_factor, unit_label) for a figure folder.

    Strategy:
      1. Read validation_result.json in the figure folder.
      2. Fall back to directory-name inference.
    """
    # 1. From validation JSON
    factor, label = get_validation_unit(folder_path)
    if factor is not None:
        return factor, label

    # 2. From parent directory name
    for p in folder_path.parts:
        if p.startswith("x_time_"):
            if p == "x_time_h":
                return 1.0, "hours"
            if p == "x_time_day":
                return 24.0, "days"
            if p == "x_time_min":
                return 1.0 / 60.0, "minutes"
            if p == "x_time_week_month_year":
                # Infer from figure folder name
                f = folder_path.name.lower()
                if "week" in f:
                    return 24.0 * 7, "weeks"
                if "month" in f:
                    return 24.0 * 30, "months"
                if "year" in f:
                    return 24.0 * 365, "years"
                return 24.0, "days"   # safe default
    return 1.0, "hours"


# ------------------------------------------------------------------
# CSV reader
# ------------------------------------------------------------------

def read_xy_from_csv(csv_path: Path) -> tuple[list[float], list[float]]:
    """Read (x, y) pairs from a raw CSV. Returns (xs, ys)."""
    xs: list[float] = []
    ys: list[float] = []
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            next(reader)   # skip header
        except StopIteration:
            return [], []
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                xi = float(row[0].strip())
                yi = float(row[1].strip())
                xs.append(xi)
                ys.append(yi)
            except ValueError:
                continue
    return xs, ys


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_all_raw_curves(
    base_dir: Path,
    max_workers: int = 0,  # not used; synchronous for safety
) -> pd.DataFrame:
    """
    Scan all four x_time_* directories and load every CSV curve.

    Returns a DataFrame with one row per data point:
      csv_file, top_dir, doi, figure_folder, series_id,
      unit_label, max_x_raw, max_x_hours, n_points, x, y
    """
    rows: list[dict] = []

    for top in ["x_time_h", "x_time_day", "x_time_min", "x_time_week_month_year"]:
        top_path = base_dir / top
        if not top_path.exists():
            logger.warning("Skipping missing: %s", top_path)
            continue

        logger.info("Scanning: %s", top)

        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_from_dirname(doi_dir.name)

            for fig_dir in sorted(doi_dir.iterdir()):
                if not fig_dir.is_dir() or fig_dir.name in ("image",):
                    continue

                fig_label = figure_from_folder(fig_dir.name)
                h_factor, unit_label = get_figure_unit_from_folder(fig_dir)

                # Prefer accepted/ subfolder
                accepted_dir = fig_dir / "accepted"
                if accepted_dir.is_dir():
                    csv_files = sorted(accepted_dir.glob("*.csv"))
                else:
                    csv_files = sorted(fig_dir.glob("*.csv"))

                for csv_file in csv_files:
                    xs, ys = read_xy_from_csv(csv_file)
                    if not xs:
                        continue

                    series_id = series_id_from_csv_name(csv_file.name)
                    max_x = max(xs)
                    max_x_h = max_x * h_factor
                    n_pts = len(xs)

                    for xi, yi in zip(xs, ys):
                        rows.append({
                            "csv_file": str(csv_file.relative_to(base_dir)),
                            "top_dir": top,
                            "doi": doi,
                            "figure_folder": fig_dir.name,
                            "figure_label": fig_label,
                            "series_id": series_id,
                            "unit_label": unit_label,
                            "max_x_raw": max_x,
                            "max_x_hours": max_x_h,
                            "n_points": n_pts,
                            "x": xi,
                            "y": yi,
                        })

    df = pd.DataFrame(rows)
    n_curves = df["csv_file"].nunique()
    logger.info(
        "Loaded %d rows, %d unique curves from %s",
        len(df), n_curves, base_dir,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base = Path("/Users/shunhao/Desktop/ML/lab/05_accepted_all copy/")
    df = load_all_raw_curves(base)
    print(df.head())
    print(df["unit_label"].value_counts())
