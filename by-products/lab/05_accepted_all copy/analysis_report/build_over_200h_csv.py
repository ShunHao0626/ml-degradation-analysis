#!/usr/bin/env python3
"""Consolidate every curve whose max time is >= 200 hours into a single CSV.

Strategy:
  - Reuse the scanner used for the under-200h report to enumerate every curve.
  - For each qualifying curve (max_x_hours >= 200), read all its (x, y)
    data points and emit one row per data point.
  - Output a single CSV:
      curves_over_200h_consolidated.csv

Row schema:
  top_dir, doi, figure_folder, figure_label, figure_short, panel_letter,
  series_id (inferred from filename), unit_label, max_x_raw, max_x_hours,
  n_points, csv_file, x, y
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from find_under_200h import (  # type: ignore
    get_replot_or_validation_unit,
    read_x_values_from_csv,
    doi_from_dirname,
    figure_from_folder,
)

BASE_DIR = Path("/Users/shunhao/Desktop/ML/lab/05_accepted_all copy")
OUT_PATH = BASE_DIR / "analysis_report" / "curves_over_200h_consolidated.csv"
THRESHOLD_H = 200.0


def short_figure(figure_label: str) -> tuple[str, str]:
    """Split figure_label into (FigN, PanelLetter like '_D')."""
    parts = figure_label.split("_", 1)
    if len(parts) == 1:
        return parts[0], ""
    head, tail = parts
    if len(tail) == 1 and tail.isalpha():
        return head, f"_{tail}"
    return head, ""


def series_id_from_csv_name(csv_name: str) -> str:
    m = re.search(r"__Series[_\s]+(\d+)", csv_name)
    if m:
        return f"Series_{m.group(1)}"
    return ""


def main():
    out_dir = BASE_DIR / "analysis_report"
    out_dir.mkdir(exist_ok=True)

    total_curves = 0
    kept_curves = 0
    skipped_curves = 0
    points_written = 0
    rows_buffer: list[dict] = []

    for top in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        top_path = BASE_DIR / top
        if not top_path.exists():
            print(f"[!] missing {top_path}", file=sys.stderr)
            continue

        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_from_dirname(doi_dir.name)

            for fig_dir in sorted(doi_dir.iterdir()):
                if not fig_dir.is_dir() or fig_dir.name in ("image",):
                    continue
                fig_label = figure_from_folder(fig_dir.name)
                fig_short, panel_letter = short_figure(fig_label)
                h_factor, unit_label = get_replot_or_validation_unit(fig_dir)

                # Select CSV files: prefer accepted/, fall back to fig_dir
                accepted_dir = fig_dir / "accepted"
                csv_files = sorted(accepted_dir.glob("*.csv")) if accepted_dir.is_dir() else []
                if not csv_files:
                    csv_files = sorted(fig_dir.glob("*.csv"))

                for csv_file in csv_files:
                    total_curves += 1
                    series_id = series_id_from_csv_name(csv_file.stem)

                    xs = read_x_values_from_csv(csv_file)
                    if not xs:
                        skipped_curves += 1
                        continue

                    max_x_raw = max(xs)
                    max_x_h = max_x_raw * h_factor
                    if max_x_h < THRESHOLD_H:
                        continue

                    kept_curves += 1
                    rel_path = csv_file.relative_to(BASE_DIR).as_posix()

                    # Need y values. Re-read csv fully to keep alignment.
                    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        try:
                            next(reader)  # skip header
                        except StopIteration:
                            continue
                        ys: list[float] = []
                        for r in reader:
                            if len(r) < 2:
                                ys.append("")
                                continue
                            try:
                                ys.append(float(r[1]))
                            except ValueError:
                                ys.append(r[1])

                    for x, y in zip(xs, ys):
                        rows_buffer.append({
                            "top_dir": top,
                            "doi": doi,
                            "figure_folder": fig_dir.name,
                            "figure_label": fig_label,
                            "figure_short": fig_short,
                            "panel_letter": panel_letter,
                            "series_id": series_id,
                            "unit_label": unit_label or "?",
                            "max_x_raw": round(max_x_raw, 6),
                            "max_x_hours": round(max_x_h, 6),
                            "n_points": len(xs),
                            "csv_file": rel_path,
                            "x": x,
                            "y": y,
                        })
                        points_written += 1

    # Stream the rows to disk in one go
    fieldnames = [
        "top_dir", "doi", "figure_folder", "figure_label", "figure_short",
        "panel_letter", "series_id", "unit_label", "max_x_raw", "max_x_hours",
        "n_points", "csv_file", "x", "y",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_buffer)

    print(f"Curves scanned : {total_curves}")
    print(f"Curves skipped (no x data) : {skipped_curves}")
    print(f"Curves kept (>= {THRESHOLD_H} h) : {kept_curves}")
    print(f"Data rows written : {points_written}")
    print(f"Unique DOIs : {len({r['doi'] for r in rows_buffer})}")
    print(f"Unique (DOI, Figure) : {len({(r['doi'], r['figure_short']) for r in rows_buffer})}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
