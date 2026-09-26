#!/usr/bin/env python3
"""
scripts/run_200h_som.py
=======================
Entry point for the PvkSOM 200h Analysis Pipeline.

Usage:
    python scripts/run_200h_som.py [--include-short]

Options:
    --include-short   Include curves shorter than 200h (forward-fill tails)

This script implements the full pipeline from specification §21:
    1. Load data
    2. Preprocess all curves
    3. Build feature matrix
    4. Train main 2×2 SOM
    5. Run QE sweep (n=2..10)
    6. Run sensitivity analysis (σ=0.3,lr=0.1) and (σ=0.5,lr=0.3)
    7. Run k-means validation (k=2..10)
    8. Generate all figures
    9. Save all outputs
    10. Run unit tests
    11. Generate reports
    12. Print final summary

Reference:
    Hartono et al., Nat. Commun. 14, 4869 (2023)
    Author code: https://github.com/noortitan/PvkSOM
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.config import (
    OUT_DATA, OUT_FIG, OUT_LOG, PREPROC_CONFIG, SOM_CONFIG, logger
)
from src.data_loading import load_all_curves
from src.preprocessing import (
    preprocess_all_curves,
    build_feature_matrix,
    save_preprocessed_outputs,
    TIME_GRID,
)
from src.som_analysis import (
    train_som,
    assign_bmu,
    compute_cluster_metrics,
    save_som_outputs,
)
from src.validation import (
    run_qe_sweep,
    run_all_sensitivity,
    run_kmeans_validation,
    kmeans_k4_comparison,
    save_validation_outputs,
)
from src.plotting import (
    plot_curve_duration_distribution,
    plot_som_cluster_curves,
    plot_som_codebook_vectors,
    plot_som_u_matrix,
    plot_som_hit_map,
    plot_cluster_size_distribution,
    plot_bmu_distance_distribution,
    plot_quantisation_error_elbow,
    plot_kmeans_elbow,
    plot_kmeans_k4_curves,
    plot_sensitivity_comparison,
    plot_sensitivity_cluster_comparison,
)
from src.reporting import (
    generate_parameter_audit,
    generate_analysis_report,
    get_package_versions,
)

START_TIME = datetime.now()


# =============================================================================
# §1  Logging setup
# =============================================================================

def _setup_logging():
    """Configure file + console logging."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if root.hasHandlers():
        root.handlers.clear()

    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s"
    )
    fmt_cons = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S")

    fh = logging.FileHandler(OUT_LOG, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_cons)
    root.addHandler(ch)


# =============================================================================
# §2  Auto-quality checks
# =============================================================================

def _auto_qc(
    qc_df: pd.DataFrame,
    X: np.ndarray,
    assignments_df: pd.DataFrame,
    som,
    metrics_list,
) -> None:
    """Run §16 automatic quality checks and log results."""
    checks = []
    errors = []

    # 1. Inclusion counts
    n_total  = len(qc_df)
    n_incl   = int((qc_df["status"] == "included").sum())
    n_excl   = n_total - n_incl
    checks.append(f"1. n_included + n_excluded == n_total: {n_incl}+{n_excl}={n_incl+n_excl} vs {n_total} {'PASS' if n_incl+n_excl==n_total else 'FAIL'}")
    if n_incl + n_excl != n_total:
        errors.append("CHECK 1 FAILED")

    # 2. Time points
    checks.append(f"2. Each included curve has 1201 points: {'PASS' if X.shape[1]==1201 else 'FAIL'}")
    if X.shape[1] != 1201:
        errors.append(f"CHECK 2 FAILED: {X.shape[1]} != 1201")

    # 3. Matrix shape
    checks.append(f"3. X.shape[0] == n_included: {X.shape[0]} vs {n_incl} {'PASS' if X.shape[0]==n_incl else 'FAIL'}")
    if X.shape[0] != n_incl:
        errors.append(f"CHECK 3 FAILED")

    # 4. No NaN/Inf
    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))
    checks.append(f"4. X has no NaN: {'PASS' if not has_nan else 'FAIL'}")
    checks.append(f"5. X has no Inf: {'PASS' if not has_inf else 'FAIL'}")
    if has_nan:
        errors.append("CHECK 4 FAILED: X contains NaN")
    if has_inf:
        errors.append("CHECK 5 FAILED: X contains Inf")

    # 5. Normalisation check (Savitzky-Golay may cause slight overshoot)
    row_maxes = X.max(axis=1)
    # Savgol can produce values slightly above 1.0; accept up to 1.15
    all_reasonable = np.all(row_maxes < 1.15) and np.all(row_maxes > 0.9)
    checks.append(f"6. All curves max PCE in [0.9, 1.15] (savgol overshoot OK): {'PASS' if all_reasonable else 'FAIL'}")
    if not all_reasonable:
        errors.append(f"CHECK 6 FAILED: some max outside [0.9, 1.15]")

    # 6. BMU assignments
    n_assign = len(assignments_df)
    checks.append(f"7. n_assignments == n_included: {n_assign} vs {n_incl} {'PASS' if n_assign==n_incl else 'FAIL'}")
    if n_assign != n_incl:
        errors.append("CHECK 7 FAILED")

    # 7. BMU coordinates in range
    wx = assignments_df["som_node_x"].values
    wy = assignments_df["som_node_y"].values
    ok = np.all((wx >= 0) & (wx < 2)) and np.all((wy >= 0) & (wy < 2))
    checks.append(f"8. All BMU coords in [0,1]: {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append("CHECK 8 FAILED")

    # 8. Cluster sizes
    total_from_clusters = sum(m.n_curves for m in metrics_list)
    checks.append(f"9. Sum(cluster_sizes) == n_included: {total_from_clusters} vs {n_incl} {'PASS' if total_from_clusters==n_incl else 'FAIL'}")
    if total_from_clusters != n_incl:
        errors.append("CHECK 9 FAILED")

    # 9. Empty nodes
    empty_nodes = [m for m in metrics_list if m.n_curves == 0]
    if empty_nodes:
        for m in empty_nodes:
            checks.append(f"WARNING: Empty node ({m.som_x},{m.som_y})")
    else:
        checks.append("10. No empty SOM nodes: PASS")

    # 10. SOM QE
    qe = som.quantization_error(X)
    checks.append(f"11. SOM QE reported: {qe:.4f}")

    logger.info("\n=== AUTO QUALITY CHECKS ===")
    for c in checks:
        prefix = "  " if "WARNING" in c or "PASS" in c else "  [FAIL] "
        logger.info("%s%s", prefix, c)

    if errors:
        logger.error("=== AUTO QC ERRORS ===")
        for e in errors:
            logger.error("  %s", e)
        raise RuntimeError("Auto QC failed: " + "; ".join(errors))
    else:
        logger.info("=== ALL AUTO QC CHECKS PASSED ===")


# =============================================================================
# §3  Main pipeline
# =============================================================================

def run_pipeline(include_short: bool = False) -> None:
    """Run the complete PvkSOM 200h pipeline."""

    _setup_logging()
    logger.info("=" * 60)
    logger.info("PvkSOM 200h Pipeline — Starting")
    logger.info("=" * 60)
    logger.info("Run start: %s", START_TIME.isoformat())
    logger.info("include_short: %s", include_short)
    logger.info("OUT_DATA: %s", OUT_DATA)
    logger.info("OUT_FIG:  %s", OUT_FIG)

    # ── §3.1  Load data ──────────────────────────────────────────────
    t0 = time.time()
    logger.info("--- Step 1: Loading data ---")
    raw_df = load_all_curves(use_cache=True)
    n_curves = raw_df["csv_file"].nunique()
    logger.info("  Loaded %d curves in %.1fs", n_curves, time.time() - t0)

    # ── §3.2  Preprocess ─────────────────────────────────────────────
    t1 = time.time()
    logger.info("--- Step 2: Preprocessing ---")
    included_df, excluded_df, qc_df = preprocess_all_curves(
        raw_df, include_short=include_short
    )
    logger.info("  Preprocessing done in %.1fs", time.time() - t1)

    # ── §3.3  Build feature matrix ──────────────────────────────────
    t2 = time.time()
    logger.info("--- Step 3: Building feature matrix ---")
    X, curve_ids, pce_maxima = build_feature_matrix(included_df)
    logger.info("  X.shape = %s in %.1fs", X.shape, time.time() - t2)

    # ── §3.4  Save preprocessing outputs ─────────────────────────────
    save_preprocessed_outputs(
        included_df, excluded_df, qc_df,
        X, TIME_GRID, curve_ids,
        OUT_DATA,
    )

    # ── §3.5  Train main SOM ─────────────────────────────────────────
    t3 = time.time()
    logger.info("--- Step 4: Training main SOM ---")
    som, som_summary = train_som(
        X,
        shape=SOM_CONFIG["shape"],
        sigma=SOM_CONFIG["sigma"],
        learning_rate=SOM_CONFIG["learning_rate"],
        iterations=SOM_CONFIG["iterations"],
        random_seed=SOM_CONFIG["random_seed"],
    )
    logger.info("  SOM trained in %.1fs", time.time() - t3)

    # ── §3.6  Assign BMU ────────────────────────────────────────────
    wx, wy, dists = assign_bmu(som, X)

    # ── §3.7  Compute cluster metrics ───────────────────────────────
    metrics_list, metrics_dict = compute_cluster_metrics(
        X, wx, wy, dists, tuple(SOM_CONFIG["shape"]), som=som
    )

    # ── §3.8  Save SOM outputs ──────────────────────────────────────
    assignments_df = save_som_outputs(
        som, som_summary, wx, wy, dists,
        curve_ids, pce_maxima, X,
        metrics_list,
        OUT_DATA,
    )

    # ── §3.9  QE sweep ─────────────────────────────────────────────
    t4 = time.time()
    logger.info("--- Step 5: QE sweep ---")
    qe_sweep_df = run_qe_sweep(
        X,
        sigma=SOM_CONFIG["sigma"],
        learning_rate=SOM_CONFIG["learning_rate"],
        iterations=SOM_CONFIG["iterations"],
        seed=SOM_CONFIG["random_seed"],
    )

    # ── §3.10  Sensitivity analysis ─────────────────────────────────
    logger.info("--- Step 6: Sensitivity analysis ---")
    sens_results = run_all_sensitivity(X, OUT_DATA)

    # ── §3.11  K-means validation ───────────────────────────────────
    logger.info("--- Step 7: K-means validation ---")
    kmeans_wcss_df, k4_labels = run_kmeans_validation(X)
    k4_labels_4 = k4_labels.get(4)
    k4_comp = {}
    if k4_labels_4 is not None:
        k4_comp = kmeans_k4_comparison(X, wx, wy, k4_labels_4)

    # ── §3.12  Save validation outputs ──────────────────────────────
    save_validation_outputs(
        qe_sweep_df, sens_results, kmeans_wcss_df, k4_comp, OUT_DATA
    )

    # ── §3.13  Generate figures ─────────────────────────────────────
    t5 = time.time()
    logger.info("--- Step 8: Generating figures ---")

    logger.info("  §8.1 Data quality figures")
    plot_curve_duration_distribution(qc_df, OUT_DATA)

    logger.info("  §8.2 Main SOM figures")
    plot_som_cluster_curves(
        X, wx, wy, som.get_weights(),
        tuple(SOM_CONFIG["shape"]), metrics_list, OUT_DATA
    )
    plot_som_codebook_vectors(som, tuple(SOM_CONFIG["shape"]), OUT_DATA)
    plot_som_u_matrix(som, tuple(SOM_CONFIG["shape"]), OUT_DATA)
    plot_som_hit_map(wx, wy, tuple(SOM_CONFIG["shape"]), OUT_DATA)
    plot_cluster_size_distribution(wx, wy, tuple(SOM_CONFIG["shape"]), OUT_DATA)
    plot_bmu_distance_distribution(dists, OUT_DATA)

    logger.info("  §8.3 QE sweep figure")
    plot_quantisation_error_elbow(qe_sweep_df, OUT_DATA)

    logger.info("  §8.4 K-means figures")
    plot_kmeans_elbow(kmeans_wcss_df, OUT_DATA)
    if k4_labels_4 is not None:
        plot_kmeans_k4_curves(X, k4_labels_4, OUT_DATA)

    logger.info("  §8.5 Sensitivity figures")
    plot_sensitivity_comparison(X, sens_results, tuple(SOM_CONFIG["shape"]), OUT_DATA)
    plot_sensitivity_cluster_comparison(sens_results, X, OUT_DATA)

    logger.info("  All figures saved in %.1fs", time.time() - t5)

    # ── §3.14  Auto QC ─────────────────────────────────────────────
    logger.info("--- Step 9: Auto quality checks ---")
    _auto_qc(qc_df, X, assignments_df, som, metrics_list)

    # ── §3.15  Save run config ─────────────────────────────────────
    cfg_dict = {
        "preproc": PREPROC_CONFIG,
        "som": SOM_CONFIG,
        "val": {k: str(v) if k == "qe_sweep_topology_override" else v
                for k, v in config.VAL_CONFIG.items()},
        "packages": get_package_versions(),
        "run_start": START_TIME.isoformat(),
        "run_end": datetime.now().isoformat(),
        "include_short": include_short,
        "n_total_curves": int(n_curves),
        "n_included": int(len(included_df)),
        "n_excluded": int(len(excluded_df)),
        "som_qe": float(som_summary.get("quantization_error", 0)),
    }
    (OUT_DATA / "run_config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )
    logger.info("Saved: run_config.json")

    # ── §3.16  Unit tests ─────────────────────────────────────────
    logger.info("--- Step 10: Running unit tests ---")
    import subprocess
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(BASE_DIR / "tests"), "-v", "--tb=short"],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if test_result.returncode == 0:
        logger.info("  Unit tests: ALL PASSED")
    else:
        logger.warning("  Unit tests: SOME FAILED\n%s", test_result.stdout[-2000:])
    logger.info("  Test stdout (last 2000 chars):\n%s",
                test_result.stdout[-2000:])

    # ── §3.17  Generate reports ────────────────────────────────────
    logger.info("--- Step 11: Generating reports ---")
    generate_parameter_audit(OUT_DATA / "reference_parameter_audit.md")
    generate_analysis_report(
        qc_df=qc_df,
        som_summary=som_summary,
        cluster_summary=pd.read_csv(OUT_DATA / "cluster_summary.csv"),
        cluster_metrics_list=metrics_list,
        qe_sweep_df=qe_sweep_df,
        sensitivity_df=pd.read_csv(OUT_DATA / "sensitivity_summary.csv"),
        kmeans_wcss_df=kmeans_wcss_df,
        k4_comparison=k4_comp,
        preproc_cfg=PREPROC_CONFIG,
        som_cfg=SOM_CONFIG,
        run_start=START_TIME,
        packages=get_package_versions(),
        out_path=OUT_DATA / "analysis_report.md",
    )

    # ── §3.18  Final summary ───────────────────────────────────────
    elapsed = datetime.now() - START_TIME
    logger.info("=" * 60)
    logger.info("Pipeline COMPLETE in %s", elapsed)
    logger.info("Output: %s", OUT_DATA)
    logger.info("=" * 60)

    # Print cluster summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    cluster_sum = pd.read_csv(OUT_DATA / "cluster_summary.csv")
    print(cluster_sum[["som_node_x", "som_node_y", "n_curves", "percentage",
                        "pce_norm_t0", "pce_norm_t200",
                        "suggested_shape_name"]].to_string(index=False))
    print()
    print(f"QE: {som_summary.get('quantization_error', 0):.4f}")
    print(f"TE: {som_summary.get('topographic_error', 0):.4f}")
    print(f"Included: {len(included_df)}, Excluded: {len(excluded_df)}, Total: {n_curves}")
    print(f"Elapsed: {elapsed}")
    print("=" * 60)


# =============================================================================
# §4  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PvkSOM 200h Analysis Pipeline"
    )
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="Include curves shorter than 200h (forward-fill tails)"
    )
    args = parser.parse_args()
    run_pipeline(include_short=args.include_short)
