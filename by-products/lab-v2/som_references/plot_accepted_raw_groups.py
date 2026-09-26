#!/usr/bin/env python3
"""Overlay all accepted raw curves in endpoint-compatible groups.

The output directory intentionally contains PNG figures only.  CSV samples are
plotted exactly as stored: no interpolation, smoothing, normalization, x-shift,
sorting, or duplicate-x aggregation is applied.
"""

from __future__ import annotations

import colorsys
import csv
import itertools
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_accepted_raw_groups")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "accepted" / "samples_test"
OUTPUT_DIR = ROOT / "accepted" / "figures"

# A group may span at most this ratio of raw x maxima.  This keeps, for
# example, roughly 200 h and 300 h together while separating 200 h and 1000 h.
MAX_ENDPOINT_RATIO = 1.60

EXPECTED_CURVES = 218
EXPECTED_POINTS = 9_674

X_ORDER = {
    "time_hours": 0,
    "time_days": 1,
    "cycles": 2,
    "unknown_x": 3,
}
Y_ORDER = {
    "normalized": 0,
    "percent": 1,
    "relative_percent": 2,
    "other_y": 3,
}

X_LABELS = {
    "time_hours": "Time (h)",
    "time_days": "Time (d)",
    "cycles": "Cycle count",
    "unknown_x": "Raw x value (axis metadata unavailable)",
}
Y_LABELS = {
    "normalized": "Normalized output (raw)",
    "percent": "PCE / efficiency (%) (raw)",
    "relative_percent": "Relative output (%) (raw)",
    "other_y": "Raw y value (anomalous / unmatched scale)",
}

X_TITLES = {
    "time_hours": "time in hours",
    "time_days": "time in days",
    "cycles": "cycle count",
    "unknown_x": "unknown x-axis metadata",
}
Y_TITLES = {
    "normalized": "normalized output",
    "percent": "absolute PCE / efficiency",
    "relative_percent": "relative percentage",
    "other_y": "anomalous / unmatched y scale",
}

X_FILENAME = {
    "time_hours": "time-hours",
    "time_days": "time-days",
    "cycles": "cycles",
    "unknown_x": "unknown-x",
}
Y_FILENAME = {
    "normalized": "normalized",
    "percent": "percent",
    "relative_percent": "relative-percent",
    "other_y": "other-y",
}


@dataclass(frozen=True)
class Curve:
    path: Path
    record: str
    record_number: int
    series_id: str
    legend_name: str
    x_family: str
    y_family: str
    x: np.ndarray
    y: np.ndarray
    x_max: float


@dataclass
class AtomicUnit:
    """Curves that should stay together when their endpoints are compatible."""

    semantic_key: tuple[str, str]
    curves: list[Curve]
    endpoint_min: float
    endpoint_max: float


def compact_text(value: object) -> str:
    return str(value or "").strip()


def axis_text(metadata: dict, axis_name: str) -> str:
    axis = (metadata.get("axis", {}).get(axis_name) or {})
    return " ".join(
        compact_text(axis.get(field)).lower() for field in ("name", "unit")
    )


def classify_x(metadata: dict) -> str:
    text = axis_text(metadata, "x")
    if "cycle" in text:
        return "cycles"
    if re.search(r"\(d\)|\bdays?\b", text):
        return "time_days"
    if re.search(
        r"time|duration|hour|\bhr\b|damp heat|storage|\(h\)", text
    ):
        return "time_hours"
    return "unknown_x"


def percentile(sorted_values: list[float], fraction: float) -> float:
    index = round(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def classify_y(metadata: dict, y_values: np.ndarray) -> str:
    """Separate incompatible y scales, using values only when labels are absent.

    Four source figures label fraction-scale data as percent.  The q95 check
    keeps those curves with the other normalized traces so that a 0--1 curve is
    not flattened below true 10--25% PCE curves.
    """

    text = axis_text(metadata, "y")
    values = sorted(float(value) for value in y_values)
    median = statistics.median(values)
    q95 = percentile(values, 0.95)

    if "relative" in text:
        return "relative_percent"
    if re.search(r"norm|a\.u\.?", text):
        if median < -2 or median > 5:
            return "other_y"
        return "normalized"
    if "%" in text:
        if median >= -1 and q95 <= 1.5:
            return "normalized"
        return "percent"
    if re.search(r"pce|efficien|power|pmax|mppt|\bspo\b", text):
        if median >= -1 and q95 <= 2:
            return "normalized"
        return "percent"
    if -1 <= median <= 2 and q95 <= 2:
        return "normalized"
    if -5 < median < 40:
        return "percent"
    return "other_y"


def record_number(record: str) -> int:
    match = re.search(r"(\d+)", record)
    return int(match.group(1)) if match else 10**9


def normalized_series_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def series_label(csv_path: Path, metadata: dict) -> tuple[str, str]:
    parts = csv_path.stem.split("__")
    series_id = parts[-1].replace("_", " ") if parts else "Series"
    fallback = parts[-2] if len(parts) >= 2 else series_id

    wanted = normalized_series_id(series_id)
    legend_name = ""
    for mapping in metadata.get("series_mappings", []) or []:
        mapped_id = compact_text(mapping.get("series_id"))
        if normalized_series_id(mapped_id) == wanted:
            legend_name = compact_text(mapping.get("legend_name"))
            break
    legend_name = (legend_name or fallback).lstrip("。．. ")
    return series_id, legend_name


def read_csv_exact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            x_value = float(row["x"])
            y_value = float(row["y"])
            if not (math.isfinite(x_value) and math.isfinite(y_value)):
                raise ValueError(f"Non-finite raw point in {path}")
            x_values.append(x_value)
            y_values.append(y_value)
    if not x_values:
        raise ValueError(f"No points in {path}")
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def load_curves() -> list[Curve]:
    curves: list[Curve] = []
    for metadata_path in sorted(
        INPUT_DIR.glob("*/validation_result.json"),
        key=lambda path: record_number(path.parent.name),
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        x_family = classify_x(metadata)
        record = metadata_path.parent.name
        for csv_path in sorted((metadata_path.parent / "accepted").glob("*.csv")):
            x_values, y_values = read_csv_exact(csv_path)
            series_id, legend_name = series_label(csv_path, metadata)
            curves.append(
                Curve(
                    path=csv_path,
                    record=record,
                    record_number=record_number(record),
                    series_id=series_id,
                    legend_name=legend_name,
                    x_family=x_family,
                    y_family=classify_y(metadata, y_values),
                    x=x_values,
                    y=y_values,
                    x_max=float(np.max(x_values)),
                )
            )
    return curves


def endpoint_ratio(endpoint_min: float, endpoint_max: float) -> float:
    if endpoint_min <= 0:
        return math.inf
    return endpoint_max / endpoint_min


def make_atomic_units(curves: list[Curve]) -> list[AtomicUnit]:
    """Keep source-related curves together unless their endpoints differ a lot."""

    units: list[AtomicUnit] = []
    sort_key = lambda curve: (
        curve.x_family,
        curve.y_family,
        curve.record_number,
        curve.record,
        curve.x_max,
        curve.path.name,
    )
    group_key = lambda curve: (curve.x_family, curve.y_family, curve.record)

    for (_, _, _), record_curves_iter in itertools.groupby(
        sorted(curves, key=sort_key), key=group_key
    ):
        record_curves = list(record_curves_iter)
        endpoint_min = min(curve.x_max for curve in record_curves)
        endpoint_max = max(curve.x_max for curve in record_curves)
        if endpoint_ratio(endpoint_min, endpoint_max) <= MAX_ENDPOINT_RATIO:
            units.append(
                AtomicUnit(
                    semantic_key=(
                        record_curves[0].x_family,
                        record_curves[0].y_family,
                    ),
                    curves=record_curves,
                    endpoint_min=endpoint_min,
                    endpoint_max=endpoint_max,
                )
            )
        else:
            for curve in record_curves:
                units.append(
                    AtomicUnit(
                        semantic_key=(curve.x_family, curve.y_family),
                        curves=[curve],
                        endpoint_min=curve.x_max,
                        endpoint_max=curve.x_max,
                    )
                )
    return units


def group_curves(curves: list[Curve]) -> list[list[Curve]]:
    """Greedily form semantic groups with complete x-max ratio <= 1.6."""

    units = make_atomic_units(curves)
    units.sort(
        key=lambda unit: (
            X_ORDER[unit.semantic_key[0]],
            Y_ORDER[unit.semantic_key[1]],
            unit.endpoint_min,
            unit.endpoint_max,
            unit.curves[0].record_number,
        )
    )

    groups: list[list[Curve]] = []
    semantic_key = lambda unit: unit.semantic_key
    for _, semantic_units_iter in itertools.groupby(units, key=semantic_key):
        current_units: list[AtomicUnit] = []
        current_min = math.inf
        current_max = -math.inf
        for unit in semantic_units_iter:
            candidate_min = min(current_min, unit.endpoint_min)
            candidate_max = max(current_max, unit.endpoint_max)
            compatible = (
                endpoint_ratio(candidate_min, candidate_max) <= MAX_ENDPOINT_RATIO
            )
            if current_units and not compatible:
                groups.append(
                    [curve for item in current_units for curve in item.curves]
                )
                current_units = [unit]
                current_min = unit.endpoint_min
                current_max = unit.endpoint_max
            else:
                current_units.append(unit)
                current_min = candidate_min
                current_max = candidate_max
        if current_units:
            groups.append([curve for item in current_units for curve in item.curves])

    for group in groups:
        group.sort(
            key=lambda curve: (
                curve.record_number,
                curve.record,
                normalized_series_id(curve.series_id),
                curve.path.name,
            )
        )
    groups.sort(
        key=lambda group: (
            X_ORDER[group[0].x_family],
            Y_ORDER[group[0].y_family],
            min(curve.x_max for curve in group),
        )
    )
    return groups


def format_number(value: float, *, filename: bool = False) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        text = f"{value / 1_000_000:.1f}M"
    elif absolute >= 10_000:
        text = f"{value:,.0f}"
    elif absolute >= 1_000:
        text = f"{value:,.0f}"
    elif absolute >= 100:
        text = f"{value:.0f}"
    elif absolute >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".") if "." in text and "M" not in text else text
    if filename:
        return text.replace(",", "").replace(".", "p")
    return text


def series_short_name(series_id: str) -> str:
    match = re.search(r"(\d+)", series_id)
    return f"S{match.group(1)}" if match else series_id


def legend_label(curve: Curve) -> str:
    series_short = series_short_name(curve.series_id)
    if normalized_series_id(curve.legend_name) == normalized_series_id(
        curve.series_id
    ):
        identity = series_short
    else:
        identity = f"{curve.legend_name} · {series_short}"
    unit = {
        "time_hours": "h",
        "time_days": "d",
        "cycles": "cycles",
        "unknown_x": "x",
    }[curve.x_family]
    return (
        f"Fig {curve.record_number:g} · {identity} · "
        f"max={format_number(curve.x_max)} {unit}"
    )


def source_color(index: int) -> tuple[float, float, float]:
    # Golden-angle spacing gives locally distinct colors for up to ~30 sources.
    hue = (0.08 + index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.76)


def line_style(series_id: str) -> tuple[object, str]:
    match = re.search(r"(\d+)", series_id)
    index = (int(match.group(1)) - 1) if match else 0
    styles: list[object] = [
        "-",
        (0, (5, 2)),
        (0, (2, 1)),
        (0, (6, 1.5, 1.5, 1.5)),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return styles[index % len(styles)], markers[index % len(markers)]


def group_filename(index: int, group: list[Curve]) -> str:
    endpoint_min = min(curve.x_max for curve in group)
    endpoint_max = max(curve.x_max for curve in group)
    return (
        f"{index:02d}_{X_FILENAME[group[0].x_family]}_"
        f"{Y_FILENAME[group[0].y_family]}_xmax-"
        f"{format_number(endpoint_min, filename=True)}-"
        f"{format_number(endpoint_max, filename=True)}_n{len(group):03d}.png"
    )


def render_group(index: int, group: list[Curve]) -> Path:
    x_family = group[0].x_family
    y_family = group[0].y_family
    endpoint_min = min(curve.x_max for curve in group)
    endpoint_max = max(curve.x_max for curve in group)
    sources = sorted(
        {curve.record for curve in group}, key=lambda item: (record_number(item), item)
    )
    source_colors = {
        record: source_color(source_index)
        for source_index, record in enumerate(sources)
    }

    if len(group) <= 18:
        legend_columns = 1
        figure_width = 16.0
        right_edge = 0.70
    elif len(group) <= 40:
        legend_columns = 2
        figure_width = 20.0
        right_edge = 0.61
    else:
        legend_columns = 3
        figure_width = 24.0
        right_edge = 0.53
    legend_rows = math.ceil(len(group) / legend_columns)
    figure_height = max(8.8, min(13.5, 7.8 + 0.20 * legend_rows))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), facecolor="white")
    fig.subplots_adjust(left=0.075, right=right_edge, bottom=0.12, top=0.84)

    line_width = 1.15 if len(group) <= 20 else 0.95
    marker_size = 3.0 if len(group) <= 20 else 2.4
    alpha = 0.90 if len(group) <= 20 else 0.80

    for curve in group:
        style, marker = line_style(curve.series_id)
        ax.plot(
            curve.x,
            curve.y,
            color=source_colors[curve.record],
            linestyle=style,
            linewidth=line_width,
            marker=marker,
            markersize=marker_size,
            markeredgewidth=0.35,
            alpha=alpha,
            label=legend_label(curve),
            zorder=2,
        )

    if y_family == "normalized":
        ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.55, zorder=1)

    unit = {
        "time_hours": "h",
        "time_days": "d",
        "cycles": "cycles",
        "unknown_x": "raw x units",
    }[x_family]
    fig.suptitle(
        f"Raw curves — {X_TITLES[x_family]}, {Y_TITLES[y_family]}",
        fontsize=17,
        fontweight="semibold",
        y=0.965,
    )
    ax.set_title(
        "Grouped raw x maxima: "
        f"{format_number(endpoint_min)}–{format_number(endpoint_max)} {unit}  |  "
        f"{len(group)} curves from {len(sources)} source figures",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel(X_LABELS[x_family], fontsize=12)
    ax.set_ylabel(Y_LABELS[y_family], fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#d2d2d2", linewidth=0.65, alpha=0.72)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=5))
    scalar_formatter = ScalarFormatter(useMathText=False)
    scalar_formatter.set_powerlimits((-4, 5))
    ax.xaxis.set_major_formatter(scalar_formatter)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.025, y=0.065)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        ncol=legend_columns,
        fontsize=7.3,
        title="Source · curve · raw x maximum",
        title_fontsize=8.2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.2,
        labelspacing=0.58,
        borderaxespad=0.0,
    )
    fig.text(
        0.075,
        0.035,
        "Original CSV samples connected in stored order; no smoothing, interpolation, "
        "normalization, x-shift, or duplicate-x removal.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    output_path = OUTPUT_DIR / group_filename(index, group)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def verify(curves: list[Curve], groups: list[list[Curve]], outputs: list[Path]) -> None:
    if len(curves) != EXPECTED_CURVES:
        raise RuntimeError(f"Expected {EXPECTED_CURVES} curves, found {len(curves)}")
    point_count = sum(len(curve.x) for curve in curves)
    if point_count != EXPECTED_POINTS:
        raise RuntimeError(f"Expected {EXPECTED_POINTS} points, found {point_count}")

    source_paths = [curve.path.resolve() for curve in curves]
    grouped_paths = [curve.path.resolve() for group in groups for curve in group]
    if len(grouped_paths) != len(set(grouped_paths)):
        raise RuntimeError("At least one curve was assigned to more than one group")
    if set(grouped_paths) != set(source_paths):
        raise RuntimeError("Grouped curves do not exactly match the source CSV set")

    for group in groups:
        semantic_keys = {(curve.x_family, curve.y_family) for curve in group}
        if len(semantic_keys) != 1:
            raise RuntimeError(f"Mixed semantic axes in group: {semantic_keys}")
        endpoint_min = min(curve.x_max for curve in group)
        endpoint_max = max(curve.x_max for curve in group)
        if endpoint_ratio(endpoint_min, endpoint_max) > MAX_ENDPOINT_RATIO + 1e-12:
            raise RuntimeError("Endpoint ratio exceeded the configured limit")

    if len(outputs) != len(groups) or len(set(outputs)) != len(outputs):
        raise RuntimeError("Unexpected output figure count or duplicate output names")
    for output in outputs:
        if output.suffix.lower() != ".png" or output.stat().st_size <= 10_000:
            raise RuntimeError(f"Invalid output figure: {output}")

    output_entries = list(OUTPUT_DIR.iterdir())
    if set(output_entries) != set(outputs):
        extras = sorted(str(path.name) for path in set(output_entries) - set(outputs))
        raise RuntimeError(f"Output folder contains non-result or stale files: {extras}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "text.parse_math": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.16,
        }
    )

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Refusing to mix or overwrite files in non-empty output folder: {OUTPUT_DIR}"
        )

    curves = load_curves()
    groups = group_curves(curves)
    outputs = [
        render_group(index, group) for index, group in enumerate(groups, start=1)
    ]
    verify(curves, groups, outputs)

    print(f"Loaded {len(curves)} raw curves with {sum(len(c.x) for c in curves)} points")
    print(f"Rendered and verified {len(outputs)} PNG figures in {OUTPUT_DIR}")
    for output, group in zip(outputs, groups):
        endpoint_min = min(curve.x_max for curve in group)
        endpoint_max = max(curve.x_max for curve in group)
        print(
            f"{output.name}\t{len(group)} curves\t"
            f"xmax {endpoint_min:.6g}..{endpoint_max:.6g}"
        )


if __name__ == "__main__":
    main()
