"""
src/data_loading.py
===================
Load raw CSV curves from the four x_time_* directories and return a
consolidated DataFrame with one row per curve.

Each row contains:
  csv_file      — relative path, used as unique curve ID
  top_dir       — which x_time_* directory
  doi           — extracted from directory name
  figure_label  — extracted from folder name
  series_id     — extracted from CSV file name
  unit_label    — time unit (hours/days/minutes/weeks/months)
  max_x_raw     — raw maximum time value
  max_x_hours   — maximum time in hours
  n_points      — number of data points
  x             — list of time values
  y             — list of PCE values

References:
  - Author notebook: 20230816_degradation_analysis_revision_11_cleaned.ipynb
  - Author data: Zenodo DOI 10.5281/zenodo.8185883
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIRS, DATA_CONFIG, UNIT_TO_HOURS, UNIT_ALIASES, logger

# =============================================================================
# DOI / figure / series utilities
# =============================================================================

def doi_from_dirname(dirname: str) -> str:
    """Extract 'doi/suffix' from directory name like '10.1039_d0tc05455k'."""
    m = re.match(r"^(\d+\.\d+)_(.+)$", dirname)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return dirname


def figure_from_folder(folder_name: str) -> str:
    """Extract short label like 'Fig5_D' from folder name.

    Input:  'Fig5_D_stability_performance_curve_pce'
    Output: 'Fig5_D'
    """
    parts = folder_name.split("_")
    if parts:
        first = parts[0]
        # Match FigNN or figNN
        m = re.match(r"^([Ff]ig\d+)$", first)
        if m:
            if len(parts) >= 2 and re.match(r"^[a-zA-Z]$", parts[1]):
                return f"{first}_{parts[1]}"
            return first
    return folder_name


def series_id_from_csv_name(csv_name: str) -> str:
    """Extract 'Series_N' from CSV file name.

    Input:  'doi__Fig5__A__pce__MAPbI3__Series_1.csv'
    Output: 'Series_1'
    """
    m = re.search(r"[Ss]eries[_\s]+(\d+)", csv_name)
    if m:
        return f"Series_{m.group(1)}"
    return ""


# =============================================================================
# Unit detection
# =============================================================================

def _norm_unit(s: str) -> str:
    """Normalise a unit string to a canonical key."""
    return re.sub(r"[^a-z]", "", s.lower())


def get_unit_from_validation(fig_dir: Path) -> tuple[Optional[float], str]:
    """Read validation_result.json to determine the x-axis time unit.

    Returns:
        (hours_factor, unit_label) or (None, "") if not found.
    """
    vj = fig_dir / "validation_result.json"
    if not vj.exists():
        return None, ""

    try:
        with open(vj, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, ""

    for src in [data.get("axis", {}).get("x", {}).get("unit", ""),
                data.get("axis", {}).get("x", {}).get("name", "")]:
        if not isinstance(src, str):
            continue
        norm = _norm_unit(src)
        for alias, canonical in UNIT_ALIASES.items():
            if norm == alias or norm == _norm_unit(alias):
                factor = UNIT_TO_HOURS[canonical]
                logger.debug("  validation: '%s' → %s (×%.4f h)", src, canonical, factor)
                return factor, canonical

    return None, ""


def get_unit_from_dir(top_dir_name: str) -> tuple[float, str]:
    """Determine (hours_factor, unit_label) from directory name fallback."""
    if top_dir_name == "x_time_h":
        return 1.0, "hours"
    if top_dir_name == "x_time_day":
        return 24.0, "days"
    if top_dir_name == "x_time_min":
        return 1.0 / 60.0, "minutes"
    if top_dir_name == "x_time_week_month_year":
        return 24.0, "days"   # default; overridden per figure from validation JSON
    return 1.0, "hours"


def get_unit_from_figure_folder(folder_name: str, default_factor: float) -> float:
    """Infer hours factor from figure folder name keywords (for week/month/year)."""
    name_lower = folder_name.lower()
    if "week" in name_lower:
        return 24.0 * 7
    if "month" in name_lower:
        return 24.0 * 30
    if "year" in name_lower:
        return 24.0 * 365
    return default_factor


# =============================================================================
# CSV reader
# =============================================================================

def read_xy_from_csv(csv_path: Path) -> tuple[list[float], list[float]]:
    """Read (x, y) pairs from a raw CSV with 'x,y' header."""
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


# =============================================================================
# Main loader
# =============================================================================

def load_all_curves(use_cache: bool = True) -> pd.DataFrame:
    """
    Scan all four x_time_* directories and load every accepted CSV curve.

    Args:
        use_cache: If True and CACHE_CSV exists, load from cache instead of
                   re-scanning directories.

    Returns:
        DataFrame with one row per curve. Key columns:
          csv_file, top_dir, doi, figure_label, series_id,
          unit_label, max_x_raw, max_x_hours, n_points, x, y

    Note:
        x and y are stored as JSON-encoded lists in the DataFrame cells.
        Use pd.DataFrame.to_dict("list") or .apply() to extract arrays.
    """
    from .config import CACHE_CSV

    # ── Try cache ─────────────────────────────────────────────────────────────
    if use_cache and CACHE_CSV.exists() and CACHE_CSV.stat().st_size > 1000:
        logger.info("Loading cached merged data: %s", CACHE_CSV)
        df = pd.read_csv(CACHE_CSV)
        logger.info("  Loaded %d rows (%d curves) from cache",
                    len(df), df["csv_file"].nunique())
        return df

    logger.info("Scanning raw data directories (no cache found)...")

    # ── Scan directories ──────────────────────────────────────────────────────
    records: list[dict] = []

    for top_name, top_path in sorted(DATA_DIRS.items()):
        if not top_path.exists():
            logger.warning("Directory not found, skipping: %s", top_path)
            continue

        logger.info("  Scanning: %s", top_name)

        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_from_dirname(doi_dir.name)

            for fig_dir in sorted(doi_dir.iterdir()):
                if not fig_dir.is_dir():
                    continue
                fig_label = figure_from_folder(fig_dir.name)

                # Determine time unit: validation JSON > dir name > folder keyword
                v_factor, v_unit = get_unit_from_validation(fig_dir)
                if v_factor is not None:
                    h_factor = v_factor
                    unit_label = v_unit
                else:
                    fallback_factor, fallback_label = get_unit_from_dir(top_name)
                    if top_name == "x_time_week_month_year":
                        h_factor = get_unit_from_figure_folder(fig_dir.name, fallback_factor)
                        unit_label = v_unit or fallback_label
                    else:
                        h_factor = fallback_factor
                        unit_label = fallback_label

                # Prefer accepted/ subfolder
                accepted_dir = fig_dir / "accepted"
                if accepted_dir.is_dir():
                    csv_files = sorted(f for f in accepted_dir.glob("*.csv")
                                      if f.is_file())
                else:
                    csv_files = sorted(f for f in fig_dir.glob("*.csv")
                                      if f.is_file() and f.name != fig_dir.name + ".csv")

                for csv_file in csv_files:
                    xs, ys = read_xy_from_csv(csv_file)
                    if not xs:
                        continue

                    series_id = series_id_from_csv_name(csv_file.name)
                    max_x = max(xs)
                    max_x_h = max_x * h_factor
                    n_pts = len(xs)

                    records.append({
                        "csv_file":    str(Path(top_name) / Path(doi_dir.name) /
                                           Path(fig_dir.name) / csv_file.name),
                        "top_dir":     top_name,
                        "doi":         doi,
                        "figure_label": fig_label,
                        "series_id":   series_id,
                        "unit_label":  unit_label,
                        "max_x_raw":   max_x,
                        "max_x_hours": max_x_h,
                        "n_points":    n_pts,
                        # Store as JSON for compact CSV serialisation
                        "x":           json.dumps(xs),
                        "y":           json.dumps(ys),
                    })

    df = pd.DataFrame(records)
    n_curves = df["csv_file"].nunique()
    logger.info("Loaded %d rows, %d unique curves", len(df), n_curves)
    logger.info("Unit distribution:\n%s",
                df["unit_label"].value_counts().to_string())

    # ── Save cache ────────────────────────────────────────────────────────────
    CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    logger.info("Cached to: %s", CACHE_CSV)

    return df


def curves_to_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deserialise the 'x' and 'y' JSON columns back to numpy arrays.

    Adds 'x_arr' and 'y_arr' columns to df (drops the JSON strings).
    """
    df = df.copy()
    df["x_arr"] = df["x"].apply(lambda s: json.loads(s))
    df["y_arr"] = df["y"].apply(lambda s: json.loads(s))
    return df


if __name__ == "__main__":
    df = load_all_curves(use_cache=True)
    print(df.head())
    print(df["unit_label"].value_counts())
