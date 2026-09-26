#!/usr/bin/env python3
"""Build a manifest for every dataset whose max time is >= 200 hours.

Walks the four time-axes directories (x_time_day, x_time_h, x_time_min,
x_time_week_month_year), identifies which DOI folders contain at least one
curve whose max-x-converted-to-hours is >= 200h, and records the qualifying
figure folders without copying their files. Consumers resolve the manifest
entries against the canonical source directories.

Layout produced:
  <BASE>/curves_over_200h_full/
      manifest.csv        # every kept (top_dir, doi, figure_folder, csv) row
      manifest.json       # structured summary (counts, units, etc.)

Run:
  python3 analysis_report/merge_over_200h.py [--dest <custom_dir>]
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from find_under_200h import (  # type: ignore
    get_replot_or_validation_unit,
    read_x_values_from_csv,
    doi_from_dirname,
)


BASE_DIR = Path("/Users/shunhao/Desktop/ML/lab/05_accepted_all copy")
DEFAULT_DEST = BASE_DIR / "curves_over_200h_full"
THRESHOLD_H = 200.0
TOP_DIRS = ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]


def scan_qualifying(base: Path) -> dict:
    """For each top_dir, return
        { top_dir: {
              'qualifying_dois': {doi_name: {figure_folder: [csv_file, ...]}},
              'unit_summary': {doi_name: set([unit_label, ...])},
          } }
    """
    summary: dict = {}
    for top in TOP_DIRS:
        top_path = base / top
        if not top_path.exists():
            continue
        qualifying: dict[str, dict[str, list]] = {}
        units: dict[str, set] = defaultdict(set)
        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi_name = doi_dir.name
            doi_qualifies = False
            kept_figures: dict[str, list] = {}
            for fig_dir in doi_dir.iterdir():
                if not fig_dir.is_dir() or fig_dir.name == "image":
                    continue
                h_factor, unit_label = get_replot_or_validation_unit(fig_dir)
                accepted_dir = fig_dir / "accepted"
                csv_files = sorted(accepted_dir.glob("*.csv")) if accepted_dir.is_dir() else []
                if not csv_files:
                    csv_files = sorted(fig_dir.glob("*.csv"))
                kept_csvs: list[str] = []
                for csv_file in csv_files:
                    xs = read_x_values_from_csv(csv_file)
                    if not xs:
                        continue
                    max_h = max(xs) * h_factor
                    if max_h >= THRESHOLD_H:
                        kept_csvs.append(str(csv_file.relative_to(base).as_posix()))
                        doi_qualifies = True
                        units[doi_name].add(unit_label or "?")
                if kept_csvs:
                    kept_figures[fig_dir.name] = kept_csvs
            if doi_qualifies:
                qualifying[doi_name] = kept_figures
        summary[top] = {
            "qualifying_dois": qualifying,
            "units": {d: sorted(s) for d, s in units.items()},
        }
    return summary


def merge(base: Path, dest: Path, summary: dict) -> dict:
    """Write a compact selection manifest without duplicating source files."""
    counts = {
        "dois_selected": 0,
        "figures_selected": 0,
        "source_files_referenced": 0,
        "by_top": {},
    }
    manifest_rows: list[dict] = []

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for top, info in summary.items():
        n_dois = 0
        n_figs = 0
        n_files = 0

        for doi_name, figures in info["qualifying_dois"].items():
            doi_src = base / top / doi_name
            n_dois += 1
            n_figs += len(figures)

            for fig_name in sorted(figures):
                fig_src = doi_src / fig_name
                n_files += sum(1 for path in fig_src.rglob("*") if path.is_file())
                for csv_rel in figures[fig_name]:
                    manifest_rows.append({
                        "top_dir": top,
                        "doi_dir": doi_name,
                        "doi": doi_from_dirname(doi_name),
                        "figure_folder": fig_name,
                        "csv_file": csv_rel,
                    })

            n_files += sum(1 for _ in doi_src.glob("*.md"))

        counts["by_top"][top] = {
            "dois_selected": n_dois,
            "figures_selected": n_figs,
            "source_files_referenced": n_files,
        }
        counts["dois_selected"] += n_dois
        counts["figures_selected"] += n_figs
        counts["source_files_referenced"] += n_files

    # Manifest CSV
    with open(dest / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["top_dir", "doi_dir", "doi", "figure_folder", "csv_file"])
        w.writeheader()
        w.writerows(manifest_rows)

    # Summary JSON
    with open(dest / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold_hours": THRESHOLD_H,
            "storage_mode": "manifest_only",
            "source_root": str(base),
            "totals": {
                "dois": counts["dois_selected"],
                "figures": counts["figures_selected"],
                "source_files": counts["source_files_referenced"],
                "qualifying_csv_curves": len(manifest_rows),
            },
            "by_top": counts["by_top"],
            "units_by_doi": {top: info["units"] for top, info in summary.items()},
            "doi_dirs_by_top": {top: sorted(info["qualifying_dois"].keys()) for top, info in summary.items()},
        }, f, indent=2, ensure_ascii=False)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Destination directory (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    base = BASE_DIR
    dest = args.dest

    print(f"Scanning: {base}")
    print(f"Threshold: max time >= {THRESHOLD_H} h")
    summary = scan_qualifying(base)
    for top, info in summary.items():
        print(f"  {top}: {len(info['qualifying_dois'])} DOI(s) qualify")

    print(f"\nWriting manifest into: {dest}")
    counts = merge(base, dest, summary)
    print("\nDone.")
    for top, c in counts["by_top"].items():
        print(
            f"  {top}: {c['dois_selected']} DOI(s), "
            f"{c['figures_selected']} figure(s), "
            f"{c['source_files_referenced']} source file(s)"
        )
    print(
        f"\nTotals -> {counts['dois_selected']} DOI(s), "
        f"{counts['figures_selected']} figure folder(s), "
        f"{counts['source_files_referenced']} source file(s) referenced"
    )
    print(f"Manifest: {dest / 'manifest.csv'}")
    print(f"Summary : {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
