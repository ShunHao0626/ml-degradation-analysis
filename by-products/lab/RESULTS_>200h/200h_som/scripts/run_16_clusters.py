#!/usr/bin/env python3
"""Generate stand-alone n=16 (= 4×4) SOM analysis.

Reads the preprocessed feature matrix saved by the main run (default
``outputs/200h_som/X_preprocessed.npy``) and trains a 4×4 SOM (16
clusters) using the same σ, lr, iterations, random seed and training
method as the main 2×2 SOM.

All artefacts are saved into ``outputs/200h_som/n16``.

Run::

    PYTHONPATH=. python3 scripts/run_16_clusters.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import config
from src import plotting
from scripts.run_7_to_10_clusters import run_n_clusters, _centroid_diagnostics


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som_n16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default=str(config.OUTPUT_DIR / "X_preprocessed.npy"))
    parser.add_argument("--included", default=str(config.OUTPUT_DIR / "included_samples.csv"))
    args = parser.parse_args()

    logger = setup_logging(config.OUTPUT_DIR)
    warnings.filterwarnings("ignore")

    logger.info("=== Loading preprocessed matrix ===")
    X = np.load(args.X)
    inc = pd.read_csv(args.included)
    sample_ids = inc["sample_id"].tolist()
    max_pce_200h = inc["max_pce_0_200h"].astype(float).tolist()
    logger.info("X.shape=%s   #samples=%d", X.shape, len(sample_ids))

    base_cfg = config.SOM_CONFIG
    n_dir = config.OUTPUT_DIR / "n16"
    n_fig = n_dir / "figures"
    s = run_n_clusters(
        n=16,
        som_shape=(4, 4),
        X=X,
        sample_ids=sample_ids,
        max_pce_200h=max_pce_200h,
        out_dir=n_dir,
        fig_dir=n_fig,
        sigma=float(base_cfg["sigma"]),
        learning_rate=float(base_cfg["learning_rate"]),
        iterations=int(base_cfg["iterations"]),
        random_seed=int(base_cfg["random_seed"]),
    )
    _centroid_diagnostics(n_dir)
    plotting.plot_cluster_size_distribution_n(
        s["cluster_sizes"],
        "n=16 cluster sizes",
        config.FIG_DIR / "n16_cluster_size_distribution.png",
    )

    # Append to the running all_n_summary.csv if it exists
    all_csv = config.OUTPUT_DIR / "all_n_summary.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        new_row = {
            "total_samples": 1812,
            "model": "n16_4x4",
            "n_nodes": 16,
            "topology": "4x4",
            "quantisation_error": s["quantisation_error"],
            "n_empty_nodes": s["n_empty_nodes"],
            "cluster_sizes": json.dumps(s["cluster_sizes"]),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = (
            df.drop_duplicates(subset=["model", "n_nodes"], keep="last")
            .sort_values("quantisation_error", ascending=False)
            .reset_index(drop=True)
        )
        df.to_csv(all_csv, index=False)

    logger.info("=== Done. n16 outputs in outputs/n16/ ===")


if __name__ == "__main__":
    main()
