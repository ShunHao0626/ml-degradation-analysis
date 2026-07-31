"""Dataset discovery, unit conversion, curve auditing, and selected-database export."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig


UNIT_TO_HOURS = {
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "secs": 1.0 / 3600.0,
    "s": 1.0 / 3600.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "mins": 1.0 / 60.0,
    "hour": 1.0,
    "hours": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "h": 1.0,
    "day": 24.0,
    "days": 24.0,
    "d": 24.0,
    "week": 24.0 * 7.0,
    "weeks": 24.0 * 7.0,
    "wk": 24.0 * 7.0,
    "wks": 24.0 * 7.0,
    "w": 24.0 * 7.0,
    "month": 24.0 * 30.0,
    "months": 24.0 * 30.0,
    "mo": 24.0 * 30.0,
    "year": 24.0 * 365.0,
    "years": 24.0 * 365.0,
    "yr": 24.0 * 365.0,
    "yrs": 24.0 * 365.0,
    "y": 24.0 * 365.0,
}

TOP_DIR_FALLBACK = {
    "x_time_h": (1.0, "hours"),
    "x_time_day": (24.0, "days"),
    "x_time_min": (1.0 / 60.0, "minutes"),
    "x_time_week_month_year": (24.0, "days"),
}


@dataclass
class RawCurve:
    curve_id: str
    sample_id: str
    source_path: Path
    source_relative_path: str
    top_dir: str
    doi_dir: str
    doi: str
    figure_folder: str
    series_id: str
    unit_factor_to_hours: float
    unit_label: str
    unit_source: str
    y_axis_name: str
    y_axis_unit: str
    x_hours_absolute: np.ndarray
    x_hours_relative: np.ndarray
    y: np.ndarray


def _normalise_unit(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _read_validation(fig_dir: Path) -> dict:
    path = fig_dir / "validation_result.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_time_unit(fig_dir: Path, top_dir: str) -> tuple[float, str, str]:
    validation = _read_validation(fig_dir)
    axis_x = validation.get("axis", {}).get("x", {})
    for field_name in ("unit", "name"):
        value = axis_x.get(field_name)
        if not isinstance(value, str):
            continue
        key = _normalise_unit(value)
        if key in UNIT_TO_HOURS:
            return UNIT_TO_HOURS[key], key, f"validation.axis.x.{field_name}"

    folder_lower = fig_dir.name.lower()
    if top_dir == "x_time_week_month_year":
        for token, factor, label in (
            ("week", 24.0 * 7.0, "weeks"),
            ("month", 24.0 * 30.0, "months"),
            ("year", 24.0 * 365.0, "years"),
        ):
            if token in folder_lower:
                return factor, label, "figure_folder_name"

    factor, label = TOP_DIR_FALLBACK[top_dir]
    return factor, label, "top_directory_fallback"


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _series_id_from_name(name: str) -> str:
    match = re.search(r"[Ss]eries[_\s]+(\d+)", name)
    return f"Series_{match.group(1)}" if match else ""


def discover_curves(config: AnalysisConfig) -> list[RawCurve]:
    curves: list[RawCurve] = []
    for top_dir in ("x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"):
        top_path = config.dataset_root / top_dir
        if not top_path.is_dir():
            continue
        for doi_dir in sorted(path for path in top_path.iterdir() if path.is_dir()):
            doi = doi_dir.name.replace("_", "/", 1)
            for fig_dir in sorted(path for path in doi_dir.iterdir() if path.is_dir()):
                accepted = fig_dir / "accepted"
                csv_paths = (
                    sorted(accepted.glob("*.csv"))
                    if accepted.is_dir()
                    else sorted(fig_dir.glob("*.csv"))
                )
                if not csv_paths:
                    continue
                factor, unit_label, unit_source = infer_time_unit(fig_dir, top_dir)
                validation = _read_validation(fig_dir)
                axis_y = validation.get("axis", {}).get("y", {})
                for csv_path in csv_paths:
                    x_raw, y = _read_xy(csv_path)
                    if x_raw.size == 0:
                        x_abs = np.asarray([], dtype=float)
                        x_rel = np.asarray([], dtype=float)
                    else:
                        x_abs = x_raw * factor
                        x_rel = x_abs - float(np.min(x_abs))
                    rel = str(csv_path.relative_to(config.dataset_root))
                    curve_id = rel
                    sample_id = (
                        f"{doi}|{fig_dir.name}|{csv_path.stem}"
                    )
                    curves.append(
                        RawCurve(
                            curve_id=curve_id,
                            sample_id=sample_id,
                            source_path=csv_path.resolve(),
                            source_relative_path=rel,
                            top_dir=top_dir,
                            doi_dir=doi_dir.name,
                            doi=doi,
                            figure_folder=fig_dir.name,
                            series_id=_series_id_from_name(csv_path.name),
                            unit_factor_to_hours=float(factor),
                            unit_label=unit_label,
                            unit_source=unit_source,
                            y_axis_name=str(axis_y.get("name") or ""),
                            y_axis_unit=str(axis_y.get("unit") or ""),
                            x_hours_absolute=x_abs,
                            x_hours_relative=x_rel,
                            y=y,
                        )
                    )
    return curves


def audit_curves(curves: Iterable[RawCurve], config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict] = []
    limit = config.window_hours
    for curve in curves:
        x = curve.x_hours_relative
        y = curve.y
        reasons: list[str] = []
        if x.size == 0 or y.size == 0:
            span = math.nan
            n_unique = 0
            global_max_time = math.nan
            max_y = math.nan
            reasons.append("no_valid_xy")
        else:
            span = float(np.max(x) - np.min(x))
            within = (x >= 0.0) & (x <= limit + 1e-9)
            n_unique = int(np.unique(x[within]).size)
            max_y = float(np.max(y))
            max_indices = np.flatnonzero(
                np.isclose(y, max_y, rtol=1e-12, atol=max(1e-12, abs(max_y) * 1e-12))
            )
            global_max_time = float(np.min(x[max_indices]))
            if span < limit - 1e-9:
                reasons.append("duration_lt_200h")
            if global_max_time > limit + 1e-9:
                reasons.append("global_max_pce_after_200h")
            if not np.isfinite(max_y) or max_y <= 0:
                reasons.append("nonpositive_or_invalid_max_pce")
            if n_unique < config.paper_min_unique_points:
                reasons.append("unique_points_lt_4")

        paper_selected = (
            x.size > 0
            and span >= limit - 1e-9
            and global_max_time <= limit + 1e-9
            and np.isfinite(max_y)
            and max_y > 0
            and n_unique >= config.paper_min_unique_points
        )
        main_selected = paper_selected and n_unique >= config.main_min_unique_points
        if paper_selected and not main_selected:
            reasons.append("unique_points_4_to_9_main_qc")

        rows.append(
            {
                "curve_id": curve.curve_id,
                "sample_id": curve.sample_id,
                "source_absolute_path": str(curve.source_path),
                "source_relative_path": curve.source_relative_path,
                "top_dir": curve.top_dir,
                "doi_dir": curve.doi_dir,
                "doi": curve.doi,
                "figure_folder": curve.figure_folder,
                "series_id": curve.series_id,
                "unit_factor_to_hours": curve.unit_factor_to_hours,
                "unit_label": curve.unit_label,
                "unit_source": curve.unit_source,
                "y_axis_name": curve.y_axis_name,
                "y_axis_unit": curve.y_axis_unit,
                "n_raw_points": int(x.size),
                "n_unique_points_0_200h": n_unique,
                "start_time_absolute_h": float(np.min(curve.x_hours_absolute))
                if curve.x_hours_absolute.size
                else math.nan,
                "end_time_absolute_h": float(np.max(curve.x_hours_absolute))
                if curve.x_hours_absolute.size
                else math.nan,
                "duration_h": span,
                "global_max_pce_time_relative_h": global_max_time,
                "global_max_pce_value": max_y,
                "duration_ge_200h": bool(np.isfinite(span) and span >= limit - 1e-9),
                "global_max_pce_by_200h": bool(
                    np.isfinite(global_max_time) and global_max_time <= limit + 1e-9
                ),
                "paper_selected_min4": bool(paper_selected),
                "main_selected_min10": bool(main_selected),
                "selection_or_exclusion_reasons": ";".join(reasons) if reasons else "included",
            }
        )
    audit = pd.DataFrame(rows).sort_values("curve_id").reset_index(drop=True)
    return audit


def save_selection_outputs(audit: pd.DataFrame, config: AnalysisConfig) -> None:
    out = config.output_root / "01_data_selection"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "all_curves_audit.csv", index=False)
    audit[audit["main_selected_min10"]].to_csv(
        out / "main_selected_curves.csv", index=False
    )
    audit[audit["paper_selected_min4"]].to_csv(
        out / "paper_sensitivity_selected_curves.csv", index=False
    )
    audit[~audit["main_selected_min10"]].to_csv(
        out / "main_excluded_curves.csv", index=False
    )

    counts = (
        audit["selection_or_exclusion_reasons"]
        .str.split(";")
        .explode()
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="n_curves")
    )
    counts.to_csv(out / "selection_reason_counts.csv", index=False)

    flow = pd.DataFrame(
        [
            ("all_discovered", len(audit)),
            ("duration_ge_200h", int(audit["duration_ge_200h"].sum())),
            (
                "duration_ge_200h_and_global_max_by_200h",
                int((audit["duration_ge_200h"] & audit["global_max_pce_by_200h"]).sum()),
            ),
            ("paper_selected_min4", int(audit["paper_selected_min4"].sum())),
            ("main_selected_min10", int(audit["main_selected_min10"].sum())),
        ],
        columns=["selection_stage", "n_curves"],
    )
    flow.to_csv(out / "selection_flow.csv", index=False)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_selected_database(
    audit: pd.DataFrame,
    config: AnalysisConfig,
    selection_column: str,
    folder_name: str,
) -> None:
    """Export raw selected CSVs and per-figure validation JSONs.

    Hard links are used when possible so the two selected databases remain
    independently browsable without needlessly duplicating bytes.
    """
    root = config.output_root / "02_selected_curve_database" / folder_name
    manifest = audit[audit[selection_column]].copy()
    copied_paths: list[str] = []
    for row in manifest.itertuples(index=False):
        source = Path(row.source_absolute_path)
        destination = root / row.source_relative_path
        _copy_or_link(source, destination)
        copied_paths.append(str(destination.resolve()))

        fig_source = source.parent.parent if source.parent.name == "accepted" else source.parent
        validation = fig_source / "validation_result.json"
        if validation.is_file():
            fig_rel = Path(row.source_relative_path).parent
            if fig_rel.name == "accepted":
                fig_rel = fig_rel.parent
            _copy_or_link(validation, root / fig_rel / "validation_result.json")

    manifest["selected_database_absolute_path"] = copied_paths
    manifest.to_csv(root / "manifest.csv", index=False)
    (root / "README.md").write_text(
        "# Selected curve database\n\n"
        f"Selection column: `{selection_column}`\n\n"
        f"Curves: **{len(manifest)}**\n\n"
        "Raw CSV paths preserve the hierarchy from `lab/data_all`. "
        "`validation_result.json` is copied once per figure where available.\n",
        encoding="utf-8",
    )
