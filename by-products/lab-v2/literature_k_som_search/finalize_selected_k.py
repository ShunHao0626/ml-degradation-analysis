#!/usr/bin/env python3
"""Freeze the paper-rule K decision and save one author-style seed=None model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "06_selected_paper_rule_result_k4"


def load_prepared() -> dict:
    archive = np.load(ROOT / "02_preprocessed_150h" / "strict_paper_preprocessed_150h.npz")
    metadata = pd.read_csv(ROOT / "02_preprocessed_150h" / "curve_metadata.csv")
    return {
        "metadata": metadata,
        "time_h": archive["time_hours"],
        "raw": archive["raw"],
        "normalized": archive["normalized"],
        "smoothed": archive["smoothed"],
    }


def main() -> None:
    prepared = load_prepared()
    n = 4
    sigma = 0.5
    learning_rate = 0.1
    model, labels = pipeline.train_run(
        prepared["smoothed"], n, sigma, learning_rate, seed=None
    )
    record = {
        "parameter_pair_id": 1,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "n_clusters": n,
        "layout": list(pipeline.LAYOUTS[n]),
        "iterations": pipeline.CFG.iterations,
        "comparison_seed": None,
        "selection_role": "final_author_style_seed_none_after_paper_rule_K_selection",
    }
    metrics = pipeline.save_run(
        FINAL_DIR,
        model,
        labels,
        prepared,
        "Selected K=4 by paper rule; final MiniSom omits random_seed like author notebook",
        record,
    )
    (FINAL_DIR / "selection_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 4,
                "decision_basis": "QE elbow at 4 plus visual underfit/overlap review; IFO targets not used",
                "final_model_metrics": metrics,
                "important": "Saved assignments/weights/model define this nondeterministic seed=None run exactly.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
