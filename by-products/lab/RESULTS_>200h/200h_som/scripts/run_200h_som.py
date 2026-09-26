#!/usr/bin/env python3
"""End-to-end driver for the 200 h SOM clustering pipeline.

Usage::

    python scripts/run_200h_som.py
    python scripts/run_200h_som.py --limit 200   # quick smoke test

Outputs are written to ``outputs/200h_som/`` (see `src/config.py`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ensure src package import works when invoked from this directory
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import config
from src import data_loading
from src import preprocessing
from src import som_analysis
from src import validation
from src import plotting
from src import reporting


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som")


def _flat_cluster_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append({
            "sample_id": rec.get("sample_id"),
            "original_start_time": rec.get("min_obs_time"),
            "original_end_time": rec.get("max_obs_time"),
            "n_original_points": rec.get("n_original_points"),
            "qc_status": "excluded" if not rec.get("ok") else "included",
            "exclusion_reason": rec.get("exclusion_reason", ""),
            "max_pce_0_200h": rec.get("max_pce_0_200h"),
            "n_duplicate_points": rec.get("n_duplicate_points", 0),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def main(limit: int | None = None) -> None:
    run_start = datetime.utcnow().isoformat(timespec="seconds")
    logger = setup_logging()
    warnings.filterwarnings("ignore")

    logger.info("=== 200h SOM analysis pipeline ===")
    logger.info("Run start (UTC): %s", run_start)
    logger.info("Output dir: %s", config.OUTPUT_DIR)
    logger.info("Dataset root: %s", config.DATASET_ROOT)

    # ---- 1. Discover data --------------------------------------------------
    logger.info("Step 1: discovering CSV files …")
    manifest = data_loading.discover_csv_files()
    if limit is not None and limit > 0:
        manifest = manifest.head(limit).copy()
    n_total = len(manifest)
    logger.info("Candidate CSV files: %d", n_total)

    # ---- 2. Load all curves into long-format dataframe ---------------------
    logger.info("Step 2: loading curves …")
    long_df = data_loading.load_all_curves(manifest)
    logger.info("Loaded %d observation rows from %d curves", len(long_df), long_df["sample_id"].nunique())

    per_curve = data_loading.per_curves_summary = data_loading.per_curve_summary(long_df)
    per_curve.to_csv(config.OUTPUT_DIR / "data_overview.csv", index=False)

    # ---- 3. Quality-control + preprocessing -------------------------------
    logger.info("Step 3: preprocessing per curve …")
    all_records = []
    sids = long_df["sample_id"].unique().tolist()
    for sid in sids:
        rec = preprocessing.preprocess_one_curve(long_df, sid)
        if not rec["ok"]:
            rec["exclusion_reason"] = rec.pop("reason", "unknown") or "unknown"
        all_records.append(rec)
    included = [r for r in all_records if r["ok"]]
    excluded = [r for r in all_records if not r["ok"]]
    n_included = len(included)
    n_excluded = len(excluded)
    logger.info("Included: %d   Excluded: %d", n_included, n_excluded)

    # QC report
    qc_df = _flat_cluster_records(all_records)
    qc_df.to_csv(config.OUTPUT_DIR / "quality_control_report.csv", index=False)
    included_df = qc_df[qc_df["qc_status"] == "included"].copy()
    excluded_df = qc_df[qc_df["qc_status"] == "excluded"].copy()
    included_df.to_csv(config.OUTPUT_DIR / "included_samples.csv", index=False)
    excluded_df.to_csv(config.OUTPUT_DIR / "excluded_samples.csv", index=False)

    if n_included == 0:
        logger.error("No curves passed QC. Stopping.")
        return

    # Sample IDs in matrix order
    sample_ids = [rec["sample_id"] for rec in included]
    max_pce_200h = [rec["max_pce_200h"] for rec in included]

    # ---- 4. Save time grid and feature matrix -----------------------------
    logger.info("Step 4: building preprocessed matrix …")
    X = preprocessing.build_feature_matrix(included)
    X_norm = np.stack([rec["y_normalized"] for rec in included], axis=0)
    X_interp = np.stack([rec["y_interpolated"] for rec in included], axis=0)
    np.save(config.OUTPUT_DIR / "X_preprocessed.npy", X)
    pd.DataFrame({"time_h": config.TIME_GRID}).to_csv(config.OUTPUT_DIR / "time_grid.csv", index=False)
    preprocessing.build_preprocessed_curve_table(included).to_csv(
        config.OUTPUT_DIR / "preprocessed_curves.csv", index=False
    )

    info = preprocessing.verify_preprocessed_matrix(X)
    logger.info("Feature matrix: %s", info)
    assert info["shape"][1] == config.N_GRID_POINTS, "Grid size mismatch"
    assert info["finite"], "Non-finite values in feature matrix"
    assert X.shape[0] == n_included

    # ---- 5. Quality-control plots -----------------------------------------
    logger.info("Step 5: drawing QC plots …")
    plotting.plot_curve_duration_distribution(per_curve, config.FIG_DIR)
    plotting.plot_curve_overview(long_df, config.FIG_DIR)
    plotting.plot_normalized_overview(X, sample_ids, config.FIG_DIR)

    # ---- 6. Train main SOM ------------------------------------------------
    logger.info("Step 6: training main 2x2 SOM (sigma=0.5, lr=0.1) …")
    main_spec = som_analysis.spec_from_config(name="main")
    som = som_analysis.train_som(X, main_spec)
    coords, dist = som_analysis.assign_bmu_clusters(som, X)
    cluster_ids = som_analysis.bmu_to_cluster_id(coords, main_spec.som_shape)
    qe = som_analysis.quantization_error(som, X)
    te = som_analysis.topographic_error(som, X)
    logger.info("Main QE = %.4f  TE = %.4f", qe, te)

    # Save model and weights
    model_path = config.OUTPUT_DIR / "som_model.pkl"
    som_analysis.save_som(som, model_path)
    np.save(config.OUTPUT_DIR / "som_weights.npy", som.get_weights())

    # ---- 7. Build cluster summaries ---------------------------------------
    assignments, summary = som_analysis.summarize_clusters(
        sample_ids=sample_ids,
        coords=coords,
        distances=dist,
        cluster_ids=cluster_ids,
        X=X,
        X_interpolated=X_interp,
        y_normalized=X_norm,
        max_pce_200h=max_pce_200h,
        som_shape=main_spec.som_shape,
        som=som,
    )
    assignments.to_csv(config.OUTPUT_DIR / "cluster_assignments.csv", index=False)

    # long-format summary (mean curves as JSON-friendly lists)
    summary_for_csv = summary.copy()
    for col in ["mean_curve", "median_curve", "std_curve", "q25_curve", "q75_curve", "codebook_vector"]:
        if col in summary_for_csv.columns:
            summary_for_csv[col] = summary_for_csv[col].apply(lambda v: json.dumps(list(v)) if isinstance(v, np.ndarray) else v)
    summary_for_csv.to_csv(config.OUTPUT_DIR / "cluster_summary.csv", index=False)

    shape_df = som_analysis.shape_metrics(included, assignments)
    shape_df.to_csv(config.OUTPUT_DIR / "cluster_shape_metrics.csv", index=False)

    # ---- 8. SOM plots ------------------------------------------------------
    logger.info("Step 8: drawing SOM plots …")
    plotting.plot_som_clusters(X, cluster_ids, main_spec.som_shape, sample_ids, summary, config.FIG_DIR, name="main")
    plotting.plot_codebook_vectors(som, main_spec.som_shape, config.FIG_DIR)
    plotting.plot_u_matrix(som, config.FIG_DIR)
    plotting.plot_hit_map(cluster_ids, main_spec.som_shape, config.FIG_DIR, name="main")
    plotting.plot_cluster_size_distribution(summary, config.FIG_DIR)
    plotting.plot_bmu_distance_distribution(dist, config.FIG_DIR)

    # ---- 9. QE sweep -------------------------------------------------------
    logger.info("Step 9: quantisation-error sweep (n=2..10) …")
    qe_df = validation.qe_sweep(X)
    qe_df.to_csv(config.OUTPUT_DIR / "quantisation_error_by_n.csv", index=False)
    plotting.plot_qe_elbow(qe_df, config.FIG_DIR)

    # ---- 10. n = 2, 4, 5, 6 comparison -------------------------------------
    logger.info("Step 10: comparing n=2,4,5,6 …")
    for n in (2, 4, 5, 6):
        from_minisom = som_analysis._import_minisom()
        spec = som_analysis.SOMSpec(
            name=f"n{n}",
            som_shape=(1, n) if n != 4 else (2, 2),
            sigma=float(config.SOM_CONFIG["sigma"]),
            learning_rate=float(config.SOM_CONFIG["learning_rate"]),
            iterations=int(config.SOM_CONFIG["iterations"]),
            random_seed=int(config.SOM_CONFIG["random_seed"]),
        )
        som_n = som_analysis.train_som(X, spec)
        coords_n, _ = som_analysis.assign_bmu_clusters(som_n, X)
        cids_n = som_analysis.bmu_to_cluster_id(coords_n, spec.som_shape)
        if n == 4:
            plotting.plot_som_clusters(X, cids_n, (2, 2), sample_ids, summary, config.FIG_DIR, name=f"n{n}")
        else:
            plotting.plot_cluster_curves_n_n(X, cids_n, n, config.FIG_DIR, name_suffix=f"")
        # Save per-n centroid curves for the comparison report
        per_n = []
        for cid in range(n):
            mask = cids_n == cid
            n_c = int(mask.sum())
            if n_c == 0:
                per_n.append({"n_nodes": n, "cluster_id": cid, "n_samples": 0, "mean_curve": []})
                continue
            per_n.append({
                "n_nodes": n,
                "cluster_id": cid,
                "n_samples": n_c,
                "mean_curve": json.dumps(X[mask].mean(0).tolist()),
                "codebook": json.dumps(som_n.get_weights().reshape(-1, X.shape[1])[cid].tolist()) if n == 1 or n == 2 else json.dumps(som_n.get_weights()[cid // spec.som_shape[1], cid % spec.som_shape[1]].tolist()),
            })
        # write
        pd.DataFrame(per_n).to_csv(config.OUTPUT_DIR / f"som_n{n}_centroids.csv", index=False)

    # ---- 11. Sensitivity --------------------------------------------------
    logger.info("Step 11: parameter sensitivity …")
    sens_assign, sens_results = validation.run_sensitivity(X, sample_ids, max_pce_200h)

    # Compute ARI / NMI vs main labels
    base_labels = cluster_ids
    extra_rows = []
    for res in sens_results:
        ari, nmi = validation.ari_nmi(base_labels, res["cluster_ids"])
        extra_rows.append({
            "model": res["name"],
            "sigma": res["spec"].sigma,
            "learning_rate": res["spec"].learning_rate,
            "quantisation_error": res["quantisation_error"],
            "ARI_vs_main": ari,
            "NMI_vs_main": nmi,
        })
    sens_full = sens_assign.merge(pd.DataFrame(extra_rows), on=["model", "sigma", "learning_rate", "quantisation_error"], how="left")
    sens_full.to_csv(config.OUTPUT_DIR / "sensitivity_summary.csv", index=False)

    plotting.plot_sensitivity_comparison_v2(sens_results, X, config.FIG_DIR)
    for res in sens_results:
        plotting.plot_simple_model(
            X, res["cluster_ids"],
            n_x=res["spec"].som_shape[0],
            n_y=res["spec"].som_shape[1],
            name=res["name"],
            out_dir=config.FIG_DIR,
        )

    # ---- 12. K-means validation -------------------------------------------
    logger.info("Step 12: k-means validation …")
    kmeans_df, kmeans_results = validation.run_kmeans(X)
    kmeans_df.to_csv(config.OUTPUT_DIR / "kmeans_wcss.csv", index=False)
    plotting.plot_kmeans_elbow(kmeans_df, config.FIG_DIR)
    kmeans_k4 = next((r for r in kmeans_results if r["k"] == 4), None)
    if kmeans_k4 is not None:
        ari_km, nmi_km = validation.ari_nmi(cluster_ids, kmeans_k4["labels"])
        logger.info("SOM vs KMeans(k=4): ARI=%.4f  NMI=%.4f", ari_km, nmi_km)
        plotting.plot_kmeans_k4(X, kmeans_k4["labels"], config.FIG_DIR)
        km_summary = {
            "ari_to_kmeans": ari_km,
            "nmi_to_kmeans": nmi_km,
            "k_for_ari": 4,
        }
    else:
        km_summary = {"ari_to_kmeans": None, "nmi_to_kmeans": None, "k_for_ari": None}

    # ---- 13. Reload SOM & verify ------------------------------------------
    logger.info("Step 13: reloading SOM and verifying reproducibility …")
    reloaded = som_analysis.load_som(model_path)
    re_coords, _ = som_analysis.assign_bmu_clusters(reloaded, X)
    re_cids = som_analysis.bmu_to_cluster_id(re_coords, main_spec.som_shape)
    same = bool(np.array_equal(re_cids, cluster_ids))
    logger.info("Reloaded SOM yields identical cluster_ids? %s", same)

    # ---- 14. QC diagnostics -----------------------------------------------
    logger.info("Step 14: reporting …")
    exc_reasons = Counter(r["exclusion_reason"] for r in excluded)
    excluded_breakdown = dict(exc_reasons.most_common(10))

    # ---- 15. Build run-config and report ----------------------------------
    pkg_versions = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "minisom", "matplotlib"):
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkg_versions[pkg] = "not installed"
    pkg_versions["run_start"] = run_start

    # final main-model info for the report
    n_nodes_main = main_spec.som_shape[0] * main_spec.som_shape[1]
    main_qe_row = qe_df[qe_df["n_nodes"] == n_nodes_main]
    main_info = {
        "quantisation_error": main_qe_row["quantisation_error"].iloc[0] if not main_qe_row.empty else qe,
        "topographic_error": float(te),
    }

    # Build conclusions / caveats / next steps
    n_empty = int((summary["n_samples"] == 0).sum())
    cluster_sizes = summary["n_samples"].tolist()
    caveats = [
        "Pre-specified 2×2 (4-node) topology: this analysis uses a 4-node SOM as a fixed setting for the main model; it does NOT prove 4 is optimal.",
        "For n ≠ 4, the QE-sweep topology is **1×n fallback** — an **implementation assumption** because the author code does not document a topology rule for arbitrary n.",
        f"n=4 2×2 topology yields QE = {main_info['quantisation_error']:.4f}; the QE sweep shows strictly decreasing QE with more nodes, which is expected but does not imply physical meaning.",
        "Cluster descriptive names are post-hoc — assigned by inspecting the mean curves. They are not predetermined physical categories.",
        "Curves not reaching 200 h are excluded. No extrapolation to 200 h is performed.",
        "K-means validation uses Euclidean distance instead of the author's DTW metric. This is an implementation assumption.",
        f"Empty nodes count: {n_empty}. Sensitivity models may show different empty-node patterns.",
    ]
    conclusions = [
        "This is an exploratory 4-node SOM analysis. The 2×2 topology is a user-specified setting, not a statistically determined optimum.",
        "Cluster IDs do NOT have predetermined physical names. The numbers (0,1,2,3) reflect SOM node coordinates only.",
        "Different cluster numbers between models do not correspond one-to-one across runs; matching is performed via Hungarian assignment on codebook vectors.",
        "Findings should be cross-checked against quantile-based PCE grouping (Figure 3 in the paper) and sensitivity analyses before drawing degradation-mechanism conclusions.",
    ]
    next_steps = [
        "Compare QE between 2×2 (main) and 1×4 topologies to assess sensitivity of physical interpretation to topology choice.",
        "Incorporate device metadata (composition, architecture) as secondary variables.",
        "Run longer QE sweeps with different random seeds to assess stability.",
        "Compare with quantile regression of relative PCE change vs. max PCE bin.",
        "Examine cluster overlap by computing pairwise distance between cluster centroid curves.",
    ]
    report_path = config.OUTPUT_DIR / "analysis_report.md"
    reporting.write_analysis_report(
        out_path=report_path,
        n_total=n_total,
        n_included=n_included,
        n_excluded=n_excluded,
        excluded_breakdown=excluded_breakdown,
        som_summary={},
        qe_summary=qe_df,
        kmeans_summary=kmeans_df,
        sensitivity_summary=sens_full,
        cluster_summary_df=summary,
        cluster_shape_df=shape_df,
        package_versions=pkg_versions,
        main_model_info=main_info,
        main_cluster_metrics=km_summary,
        caveats=caveats,
        conclusions=conclusions,
        next_steps=next_steps,
    )

    config_path = config.OUTPUT_DIR / "run_config.json"
    reporting.write_run_config(
        run_config_path=config_path,
        source_csv_files=manifest["rel_csv_file"].tolist(),
        n_total=n_total,
        n_included=n_included,
        n_excluded=n_excluded,
        som_model_path=str(model_path.relative_to(config.OUTPUT_ROOT)),
        qe_sweep_path="quantisation_error_by_n.csv",
        analysis_report_path=str(report_path.relative_to(config.OUTPUT_ROOT)),
    )

    logger.info("=== Pipeline finished ===")
    logger.info("Outputs in: %s", config.OUTPUT_DIR)
    logger.info("Plots in:   %s", config.FIG_DIR)
    logger.info("Report:     %s", report_path)

    # Also copy the core reference_parameter_audit.md into outputs so the
    # outputs directory is fully self-contained.
    src_audit = config.OUTPUT_ROOT / "reference_parameter_audit.md"
    dst_audit = config.OUTPUT_DIR / "reference_parameter_audit.md"
    if src_audit.exists() and not dst_audit.exists():
        dst_audit.write_text(src_audit.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Copied reference_parameter_audit.md into outputs/")

    # Make som_n4_curves.png available with the conventional name
    src_n4 = config.FIG_DIR / "som_cluster_curves_n4.png"
    dst_n4 = config.FIG_DIR / "som_n4_curves.png"
    if src_n4.exists() and not dst_n4.exists():
        dst_n4.write_bytes(src_n4.read_bytes())
        logger.info("Copied som_n4_curves.png into figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="200h SOM clustering pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of input curves (for smoke-testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
