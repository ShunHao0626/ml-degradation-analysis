#!/usr/bin/env python3
"""Re-run all SOM analyses (n=4, 5, 6, 7, 8, 9, 10, 16) using the
Akima-interpolated **but unsmoothed** feature matrix.

Rationale
---------
The default pipeline (``scripts/run_200h_som.py``) applies a
Savitzky–Golay filter (window=71, polyorder=2) AFTER per-curve max-PCE
normalisation. This smears away short-scale features. For a robustness
study we want to see whether the cluster structure persists when every
curve is left in its raw-interpolated form.

Outputs are saved to ``outputs/no_smooth/{n4, n5, n6, n7, n8, n9, n10, n16}/``
alongside the smoothed variants, and a combined comparison
``outputs/no_smooth/all_n_no_smooth_summary.csv`` is written for the
cross-n analysis.

Run::

    PYTHONPATH=. python3 scripts/run_no_smooth.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import config
from src import data_loading
from src import preprocessing
from src import som_analysis
from src import plotting


# Reuse the per-n training helper from the n=7..10 script so all artefacts
# (assignments, summary, shape metrics, plots, analysis.md) match.
from scripts.run_7_to_10_clusters import run_n_clusters  # noqa: E402


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som_no_smooth")


def _main_grid_jobs() -> list[tuple[int, tuple[int, int], str]]:
    """(n, som_shape, subdir) for every cluster count we have run before."""
    return [
        (4,  (2, 2),  "n4"),
        (5,  (1, 5),  "n5"),
        (6,  (2, 3),  "n6"),
        (7,  (1, 7),  "n7"),
        (8,  (2, 4),  "n8"),
        (9,  (3, 3),  "n9"),
        (10, (2, 5),  "n10"),
        (16, (4, 4),  "n16"),
    ]


def _centroid_diagnostics(out_dir: Path) -> None:
    summary = pd.read_csv(out_dir / "cluster_summary.csv")
    if "mean_curve" not in summary.columns:
        return
    centroids = []
    for raw in summary["mean_curve"].dropna().values:
        try:
            centroids.append(np.asarray(json.loads(raw), dtype=float))
        except Exception:
            continue
    if not centroids:
        return
    centroids = np.stack(centroids, axis=0)
    pdists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    df = pd.DataFrame(
        pdists,
        index=[f"c{i}" for i in range(pdists.shape[0])],
        columns=[f"c{j}" for j in range(pdists.shape[1])],
    )
    df.to_csv(out_dir / "cluster_centroid_distances.csv")


def build_no_smooth_matrix(logger: logging.Logger):
    """Rerun preprocessing but skip Savitzky–Golay. Returns
    (X, sample_ids, max_pce_200h, included_records)."""
    logger.info("=== Re-loading raw curves ===")
    manifest = data_loading.discover_csv_files()
    long_df = data_loading.load_all_curves(manifest)
    logger.info("Loaded %d observation rows from %d curves",
                len(long_df), long_df["sample_id"].nunique())

    logger.info("=== Per-curve preprocessing (NO savgol) ===")
    sids = long_df["sample_id"].unique().tolist()
    all_records = []
    for sid in sids:
        rec = preprocessing.preprocess_one_curve(long_df, sid)
        if not rec["ok"]:
            rec["exclusion_reason"] = rec.pop("reason", "unknown") or "unknown"
        all_records.append(rec)
    included = [r for r in all_records if r["ok"]]
    excluded = [r for r in all_records if not r["ok"]]
    logger.info("Included: %d   Excluded: %d", len(included), len(excluded))

    # Verify the included set is identical to the smoothed run
    main_inc_path = config.OUTPUT_DIR / "included_samples.csv"
    if main_inc_path.exists():
        main_ids = pd.read_csv(main_inc_path)["sample_id"].tolist()
        cur_ids = [r["sample_id"] for r in included]
        if main_ids != cur_ids:
            logger.warning(
                "Included set differs from smoothed run (n_main=%d, n_now=%d). "
                "Proceeding anyway — diffs are usually caused by timestamp jitter.",
                len(main_ids), len(cur_ids),
            )
        else:
            logger.info("Included set matches smoothed run exactly (n=%d).", len(main_ids))

    sample_ids = [r["sample_id"] for r in included]
    max_pce_200h = [r["max_pce_200h"] for r in included]

    logger.info("=== Building no-smooth feature matrix ===")
    X = preprocessing.build_feature_matrix_no_smooth(included)
    info = preprocessing.verify_preprocessed_matrix(X)
    logger.info("No-smooth feature matrix: %s", info)

    return X, sample_ids, max_pce_200h, included, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(config.OUTPUT_DIR / "no_smooth"),
        help="Directory in which to write all no-smooth artefacts",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(root)
    warnings.filterwarnings("ignore")
    logger.info("=== 200h SOM analysis — NO-SMOOTH variant ===")
    logger.info("Run start (UTC): %s", datetime.utcnow().isoformat(timespec="seconds"))
    logger.info("Output root: %s", root)

    X, sample_ids, max_pce_200h, included, excluded = build_no_smooth_matrix(logger)

    # Save the raw no-smooth matrix + sample table
    np.save(root / "X_no_smooth.npy", X)
    pd.DataFrame({"sample_id": sample_ids,
                  "max_pce_0_200h": max_pce_200h}).to_csv(root / "included_samples.csv", index=False)

    pd.DataFrame(
        [{"sample_id": r["sample_id"], "reason": r.get("exclusion_reason", "unknown")}
         for r in excluded]
    ).to_csv(root / "excluded_samples.csv", index=False)

    # Build the per-n directories and train every cluster count
    fig_root = root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    base_cfg = config.SOM_CONFIG
    sigma = float(base_cfg["sigma"])
    learning_rate = float(base_cfg["learning_rate"])
    iterations = int(base_cfg["iterations"])
    random_seed = int(base_cfg["random_seed"])

    summaries = []
    for n, shape, subdir in _main_grid_jobs():
        n_dir = root / subdir
        n_fig = n_dir / "figures"
        s = run_n_clusters(
            n=n,
            som_shape=shape,
            X=X,
            sample_ids=sample_ids,
            max_pce_200h=max_pce_200h,
            out_dir=n_dir,
            fig_dir=n_fig,
            sigma=sigma,
            learning_rate=learning_rate,
            iterations=iterations,
            random_seed=random_seed,
        )
        # Tag analysis.md as no-smooth variant
        md = n_dir / "analysis.md"
        if md.exists():
            md.write_text(
                "# NO-SMOOTH variant — " + md.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        # Add a short note explaining the variant
        note = (
            "\n\n## No-smooth note\n\n"
            "The feature matrix used here is the **Akima-interpolated, "
            "per-curve-normalised** curve **without** Savitzky–Golay "
            "smoothing. Every other step (QC, normalisation, SOM training, "
            "cluster labelling) is identical to the smoothed run in "
            "``outputs/n{n}/``.\n"
        ).format(n=n)
        md.write_text(md.read_text(encoding="utf-8") + note, encoding="utf-8")

        # Centroid distance diagnostics + size distribution plot
        _centroid_diagnostics(n_dir)
        plotting.plot_cluster_size_distribution_n(
            s["cluster_sizes"],
            f"n={n} cluster sizes (no smooth)",
            fig_root / f"n{n}_cluster_size_distribution.png",
        )

        summaries.append(s)

    # Combined comparison
    comp_rows = []
    for s in summaries:
        comp_rows.append({
            "variant": "no_smooth",
            "model": f"n{s['n']}_{s['som_shape'][0]}x{s['som_shape'][1]}",
            "n_nodes": s["n"],
            "topology": f"{s['som_shape'][0]}x{s['som_shape'][1]}",
            "quantisation_error": s["quantisation_error"],
            "topographic_error": s["topographic_error"],
            "n_empty_nodes": s["n_empty_nodes"],
            "total_samples": sum(s["cluster_sizes"]),
            "cluster_sizes": json.dumps(s["cluster_sizes"]),
        })
    comp = (
        pd.DataFrame(comp_rows)
        .sort_values("quantisation_error", ascending=False)
        .reset_index(drop=True)
    )
    comp.to_csv(root / "all_n_no_smooth_summary.csv", index=False)

    # Also write a side-by-side comparison vs the smoothed run
    smooth_csv = config.OUTPUT_DIR / "all_n_summary.csv"
    if smooth_csv.exists():
        sm = pd.read_csv(smooth_csv).copy()
        sm = sm.assign(variant="smooth")
        ns = comp.copy()
        # Normalise columns
        for col in ["variant", "model", "n_nodes", "topology",
                    "quantisation_error", "n_empty_nodes", "cluster_sizes"]:
            if col not in ns.columns:
                ns[col] = np.nan
        # Match the smoothed column set
        keep = ["variant", "model", "n_nodes", "topology",
                "quantisation_error", "n_empty_nodes",
                "total_samples", "cluster_sizes"]
        ns = ns.reindex(columns=keep)
        sm = sm.reindex(columns=keep)
        side_by_side = pd.concat([sm, ns], ignore_index=True)
        side_by_side = side_by_side.sort_values(
            ["variant", "quantisation_error"], ascending=[True, False]
        ).reset_index(drop=True)
        side_by_side.to_csv(root / "smooth_vs_no_smooth.csv", index=False)

    logger.info("=== Done. No-smooth outputs in %s ===", root)


if __name__ == "__main__":
    main()