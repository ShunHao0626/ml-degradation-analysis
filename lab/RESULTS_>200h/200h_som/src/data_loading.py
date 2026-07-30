"""Data loading utilities.

Builds a long-format DataFrame where each row is one observation, plus
per-curve summary columns. Re-uses `load_raw_data.load_all_raw_curves`
where possible to keep behaviour consistent with the rest of the project.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Unit detection (matches load_raw_data.DIR_UNIT_FACTOR)
# ----------------------------------------------------------------------------
UNIT_ALIASES = {
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "h": "hours",
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "w": "weeks",
    "month": "months", "months": "months", "m": "months", "mo": "months",
    "year": "years", "years": "years", "y": "years",
    "minute": "minutes", "minutes": "minutes", "min": "minutes",
    "second": "seconds", "seconds": "seconds", "s": "seconds",
}
UNIT_TO_HOURS = {
    "seconds": 1.0 / 3600.0,
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 24.0 * 7,
    "months": 24.0 * 30,
    "years": 24.0 * 365,
}
DIR_TO_HOURS = {
    "x_time_h": 1.0,
    "x_time_day": 24.0,
    "x_time_min": 1.0 / 60.0,
}


def _norm_unit(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _factor_from_validation_json(fig_dir: Path) -> Optional[float]:
    """If a validation_result.json exists, infer hours-factor from it."""
    vj = fig_dir / "validation_result.json"
    if not vj.exists():
        return None
    try:
        with open(vj, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    x = data.get("axis", {}).get("x", {})
    for k in ("unit", "name"):
        v = x.get(k)
        if not isinstance(v, str):
            continue
        norm = _norm_unit(v)
        for alias, canonical in UNIT_ALIASES.items():
            if norm == alias or norm == _norm_unit(alias):
                return UNIT_TO_HOURS[canonical]
    return None


def figure_unit_factor(top_dir: str, fig_dir: Path) -> float:
    """Determine hour factor for a figure folder."""
    factor = _factor_from_validation_json(fig_dir)
    if factor is not None:
        return factor
    if top_dir in DIR_TO_HOURS:
        return DIR_TO_HOURS[top_dir]
    # x_time_week_month_year: derive from folder name
    name = fig_dir.name.lower()
    if "week" in name:
        return 24.0 * 7
    if "month" in name:
        return 24.0 * 30
    if "year" in name:
        return 24.0 * 365
    return 24.0  # safe default for week/month/year


# ----------------------------------------------------------------------------
# Manifest construction
# ----------------------------------------------------------------------------
def discover_csv_files() -> pd.DataFrame:
    """Walk dataset_root and produce a manifest of every CSV curve.

    The returned DataFrame has one row per CSV file with columns:
        sample_id, top_dir, doi, doi_dir, figure_folder, csv_file, unit_factor
    """
    rows = []
    base = config.DATASET_ROOT
    for top in sorted(base.iterdir()):
        if not top.is_dir():
            continue
        if not top.name.startswith("x_time_"):
            continue
        for doi_dir in sorted(top.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_dir.name.replace("_", "/", 1)
            for fig_dir in sorted(doi_dir.iterdir()):
                if not fig_dir.is_dir() or fig_dir.name == "image":
                    continue
                accepted = fig_dir / "accepted"
                if accepted.is_dir():
                    csv_paths = sorted(accepted.glob("*.csv"))
                else:
                    csv_paths = sorted(fig_dir.glob("*.csv"))
                unit_factor = figure_unit_factor(top.name, fig_dir)
                for csv_p in csv_paths:
                    sample_id = f"{doi}|{fig_dir.name}|{csv_p.stem}"
                    rows.append({
                        "sample_id": sample_id,
                        "top_dir": top.name,
                        "doi_dir": doi_dir.name,
                        "doi": doi,
                        "figure_folder": fig_dir.name,
                        "csv_file": str(csv_p),
                        "rel_csv_file": str(csv_p.relative_to(base)),
                        "unit_factor": unit_factor,
                    })
    df = pd.DataFrame(rows)
    logger.info("Discovered %d candidate CSV curves", len(df))
    return df


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
def _read_xy(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, usecols=[0, 1], names=["x", "y"], header=0)
    except Exception:
        try:
            df = pd.read_csv(csv_path, header=None)
            df.columns = ["x", "y"] + list(range(2, df.shape[1]))
            df = df[["x", "y"]]
        except Exception:
            return pd.DataFrame(columns=["x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    return df


def load_one_curve(csv_path: str | Path, unit_factor: float) -> pd.DataFrame:
    """Load a single curve, convert time to hours, sort."""
    df = _read_xy(Path(csv_path))
    if df.empty:
        return df
    df["x_hours"] = df["x"].astype(float) * float(unit_factor)
    df["pce"] = df["y"].astype(float)
    df = df[["x_hours", "pce"]]
    df = df.sort_values("x_hours").reset_index(drop=True)
    return df


def load_all_curves(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return long-format df with one row per observation.

    Required columns: sample_id, x_hours, pce, doi, top_dir, figure_folder.
    """
    out = []
    for _, r in manifest.iterrows():
        d = load_one_curve(r["csv_file"], r["unit_factor"])
        if d.empty:
            continue
        d["sample_id"] = r["sample_id"]
        d["doi"] = r["doi"]
        d["top_dir"] = r["top_dir"]
        d["figure_folder"] = r["figure_folder"]
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    logger.info("Loaded %d observation rows from %d curves", len(df), df["sample_id"].nunique())
    return df


# ----------------------------------------------------------------------------
# Per-curve summaries
# ----------------------------------------------------------------------------
def per_curve_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("sample_id")
    summary = g["x_hours"].agg(["min", "max", "count"]).rename(
        columns={"min": "start_time_h", "max": "end_time_h", "count": "n_points_raw"}
    )
    summary["n_unique_times"] = g["x_hours"].nunique()
    summary["has_nan_pce"] = g["pce"].apply(lambda s: bool(s.isna().any()))
    summary["has_inf_pce"] = g["pce"].apply(lambda s: bool(np.isinf(s).any()))
    summary["max_pce_raw"] = g["pce"].max()
    summary = summary.reset_index()
    return summary
