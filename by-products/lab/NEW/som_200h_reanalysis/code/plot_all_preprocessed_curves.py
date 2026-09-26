#!/usr/bin/env python3
"""Plot every final preprocessed curve for both 200 h cohorts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.plotting import plot_all_preprocessed_curves
from som200h.reporting import write_checksums


def main() -> None:
    config = AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )
    preprocessed_root = PROJECT / "03_preprocessed"
    specifications = (
        ("main_high_quality_min10", "Main high-quality preprocessed curves"),
        ("paper_aligned_min4", "Paper-aligned preprocessed curves"),
    )
    for dataset_name, title in specifications:
        dataset_dir = preprocessed_root / dataset_name
        time_grid = pd.read_csv(dataset_dir / "time_grid.csv")[
            "time_h"
        ].to_numpy(dtype=float)
        curves = np.load(dataset_dir / "X_smoothed.npy")
        output_name = f"{dataset_name}_all_preprocessed_curves.png"
        plot_all_preprocessed_curves(
            time_grid,
            curves,
            preprocessed_root / output_name,
            title,
        )
        print(f"Saved {output_name}: {curves.shape[0]:,} curves")

    write_checksums(config)
    print("Updated checksums.sha256")


if __name__ == "__main__":
    main()
