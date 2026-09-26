#!/usr/bin/env python3
"""Regenerate raw 0--200 h plots for both selected curve cohorts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.data import discover_curves
from som200h.plotting import (
    plot_selected_raw_curves,
    plot_selected_raw_curves_by_scale,
)
from som200h.reporting import write_checksums


def main() -> None:
    config = AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )
    curves = discover_curves(config)
    curves_by_id = {curve.curve_id: curve for curve in curves}
    selection_dir = PROJECT / "01_data_selection"

    specifications = (
        (
            "main_selected_curves.csv",
            "Main selected raw curves",
            "main_selected_raw_curves_0_200h.png",
        ),
        (
            "paper_sensitivity_selected_curves.csv",
            "Paper-sensitivity selected raw curves",
            "paper_sensitivity_selected_raw_curves_0_200h.png",
        ),
    )
    for manifest_name, title, output_name in specifications:
        selected = pd.read_csv(selection_dir / manifest_name)
        plot_selected_raw_curves(
            curves_by_id,
            selected,
            selection_dir / output_name,
            title,
            config.window_hours,
        )
        print(f"Saved {output_name}: {len(selected):,} raw curves")
        scale_output_name = output_name.replace(".png", "_by_scale.png")
        plot_selected_raw_curves_by_scale(
            curves_by_id,
            selected,
            selection_dir / scale_output_name,
            title,
            config.window_hours,
        )
        print(f"Saved {scale_output_name}: {len(selected):,} raw curves")
    write_checksums(config)
    print("Updated checksums.sha256")


if __name__ == "__main__":
    main()
