#!/usr/bin/env python3
"""Regenerate reports, file index, and checksums from completed result tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.modeling import choose_cluster_count
from som200h.reporting import (
    write_checksums,
    write_file_index,
    write_main_report,
    write_si_method_report,
)


def main() -> None:
    config = AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )
    main_name = "main_high_quality_min10"
    paper_name = "paper_aligned_min4"
    som_root = PROJECT / "04_som_results"
    decision_root = PROJECT / "05_cluster_number_decision"

    main_metrics = pd.read_csv(
        som_root / main_name / "som_cluster_number_metrics.csv"
    )
    paper_metrics = pd.read_csv(
        som_root / paper_name / "som_cluster_number_metrics.csv"
    )
    main_n, main_decision = choose_cluster_count(main_metrics, config)
    paper_n, paper_decision = choose_cluster_count(paper_metrics, config)
    main_decision.to_csv(
        decision_root / main_name / "cluster_number_decision.csv",
        index=False,
    )
    paper_decision.to_csv(
        decision_root / paper_name / "cluster_number_decision.csv",
        index=False,
    )
    (PROJECT / "run_config.json").write_text(
        json.dumps(config.to_jsonable(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_flow = pd.read_csv(
        PROJECT / "01_data_selection" / "selection_flow.csv"
    )
    final_summary = pd.read_csv(
        PROJECT / "06_final_model" / "final_cluster_summary.csv"
    )
    kmeans_comparison = json.loads(
        (PROJECT / "06_final_model" / "kmeans_comparison.json").read_text(
            encoding="utf-8"
        )
    )

    write_si_method_report(
        config,
        main_metrics,
        paper_metrics,
        main_decision,
        paper_decision,
        main_n,
        paper_n,
    )
    write_main_report(
        config,
        selection_flow,
        main_n,
        paper_n,
        main_metrics,
        paper_metrics,
        final_summary,
        kmeans_comparison,
    )
    write_file_index(config)
    write_checksums(config)
    print(f"Reports finalized: {PROJECT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
