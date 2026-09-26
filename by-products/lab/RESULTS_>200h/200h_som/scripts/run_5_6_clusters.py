#!/usr/bin/env python3
"""Generate stand-alone n=5 and n=6 SOM analyses.

Reads the preprocessed feature matrix saved by the main run (default
``outputs/200h_som/X_preprocessed.npy``) and trains two additional SOMs:

* n=5 → 1×5 topology   (matches the QE-sweep 1×n fallback)
* n=6 → 2×3 topology   (natural 2-D rectangular arrangement)

Both models share the same σ, lr, iterations, random seed and training
method as the main 2×2 SOM, so the runs are directly comparable.

All artefacts are saved into ``outputs/200h_som/n5`` and
``outputs/200h_som/n6``. A combined comparison ``n5_vs_n6_summary.csv``
sits in ``outputs/200h_som/``.

Run::

    PYTHONPATH=. python3 scripts/run_5_6_clusters.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import config
from src import som_analysis
from src import plotting
from src import reporting


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som_n5_n6")


# ----------------------------------------------------------------------------
# Per-n orchestration
# ----------------------------------------------------------------------------
def run_n_clusters(
    n: int,
    som_shape: tuple[int, int],
    X: np.ndarray,
    sample_ids: list[str],
    max_pce_200h: list[float],
    out_dir: Path,
    fig_dir: Path,
    sigma: float,
    learning_rate: float,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Train a SOM with `n = som_shape[0]*som_shape[1]` clusters and write all outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    spec = som_analysis.SOMSpec(
        name=f"n{n}",
        som_shape=som_shape,
        sigma=sigma,
        learning_rate=learning_rate,
        iterations=iterations,
        random_seed=random_seed,
    )

    logger = logging.getLogger("200h_som_n5_n6")
    logger.info("Training n=%d SOM (shape=%s, σ=%.2f, lr=%.2f)",
                n, som_shape, sigma, learning_rate)
    som = som_analysis.train_som(X, spec)
    coords, dist = som_analysis.assign_bmu_clusters(som, X)
    cluster_ids = som_analysis.bmu_to_cluster_id(coords, som_shape)
    qe = som_analysis.quantization_error(som, X)
    te = som_analysis.topographic_error(som, X)
    logger.info("n=%d:  QE=%.4f  TE=%.4f", n, qe, te)

    # Save model + weights
    som_analysis.save_som(som, out_dir / "som_model.pkl")
    np.save(out_dir / "som_weights.npy", som.get_weights())

    # Recover per-curve max PCE for reporting.  The saved X matrix is
    # already smoothed AND per-curve normalised (savgol was applied AFTER
    # the per-curve division by max PCE).  We therefore pass X directly
    # as y_normalized and reconstruct y_interpolated ≈ y_normalized * max.
    inc = pd.read_csv(config.OUTPUT_DIR / "included_samples.csv")
    assert list(inc["sample_id"]) == sample_ids, "Order mismatch with main run"
    max_pc = inc["max_pce_0_200h"].to_numpy(dtype=float)
    X_norm = X.astype(float).copy()
    X_interp = X_norm * max_pc[:, None]
    max_pce = max_pc.tolist()

    # Summaries (assignments + per-cluster table)
    assignments, summary = som_analysis.summarize_clusters(
        sample_ids=sample_ids,
        coords=coords,
        distances=dist,
        cluster_ids=cluster_ids,
        X=X,
        X_interpolated=X_interp,
        y_normalized=X_norm,
        max_pce_200h=max_pce_200h,
        som_shape=som_shape,
        som=som,
    )
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)

    summary_for_csv = summary.copy()
    for col in ["mean_curve", "median_curve", "std_curve", "q25_curve", "q75_curve", "codebook_vector"]:
        if col in summary_for_csv.columns:
            summary_for_csv[col] = summary_for_csv[col].apply(
                lambda v: json.dumps(list(v)) if isinstance(v, np.ndarray) else v
            )
    summary_for_csv.to_csv(out_dir / "cluster_summary.csv", index=False)

    shape_df = som_analysis.shape_metrics(
        included=[{"sample_id": s, "y_normalized": X_norm[i]} for i, s in enumerate(sample_ids)],
        assignments=assignments,
    )
    shape_df.to_csv(out_dir / "cluster_shape_metrics.csv", index=False)

    # ---- Plots ---------------------------------------------------------------
    if som_shape[0] == 1 and som_shape[1] > 1:
        plotting.plot_cluster_curves_n_n(X, cluster_ids, som_shape[1], fig_dir, name_suffix="")
    else:
        plotting.plot_som_clusters(X, cluster_ids, som_shape, sample_ids, summary, fig_dir, name=f"n{n}")

    plotting.plot_codebook_vectors(som, som_shape, fig_dir)
    plotting.plot_u_matrix(som, fig_dir)
    plotting.plot_hit_map(cluster_ids, som_shape, fig_dir, name=f"n{n}")
    plotting.plot_bmu_distance_distribution(dist, fig_dir)

    # ---- Compact per-n report ----------------------------------------------
    cluster_pct = (summary["n_samples"] / summary["n_samples"].sum() * 100.0).round(2)
    n_empty = int((summary["n_samples"] == 0).sum())
    cluster_sizes = summary["n_samples"].astype(int).tolist()

    lines = [f"# n={n} SOM analysis ({som_shape[0]}×{som_shape[1]})", ""]
    lines.append(f"**Topology:** `{som_shape[0]}×{som_shape[1]}` ({n} nodes)")
    lines.append(f"**σ:** {sigma}  •  **learning_rate:** {learning_rate}  •  "
                 f"**iterations:** {iterations}  •  **seed:** {random_seed}")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat(timespec='seconds')} UTC")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Curves (n) | {assignments.shape[0]} |")
    lines.append(f"| Quantisation error | {qe:.4f} |")
    lines.append(f"| Topographic error | {te:.4f} |")
    lines.append(f"| Empty nodes | {n_empty} |")
    lines.append(f"| Largest cluster | {max(cluster_sizes) if cluster_sizes else 0} ({cluster_pct.max() if cluster_pct.size else 0}%) |")
    lines.append(f"| Smallest cluster | {min(cluster_sizes) if cluster_sizes else 0} ({cluster_pct.min() if cluster_pct.size else 0}%) |")
    lines.append("")
    lines.append("## Per-cluster summary")
    lines.append("")
    lines.append("| cluster_id | node | n (%) | pce@0h | pce@200h | Δ 0–200h | slope 100–200 (1/h) | suggested name |")
    lines.append("|-----------:|:----:|------:|------:|------:|------:|------:|------|")
    sh_lookup = (
        shape_df.set_index("raw_cluster_id").to_dict(orient="index")
        if not shape_df.empty else {}
    )
    for _, row in summary.iterrows():
        cid = int(row["raw_cluster_id"])
        sh = sh_lookup.get(cid, {})
        sn = sh.get("suggested_shape_name", "-")
        lines.append(
            f"| {cid} | ({int(row['som_node_x'])},{int(row['som_node_y'])}) | "
            f"{int(row['n_samples'])} ({row['fraction']*100:.1f}%) | "
            f"{row['pce_norm_0h_mean']:.4f} | {row['pce_norm_200h_mean']:.4f} | "
            f"{(row['pce_norm_200h_mean']-row['pce_norm_0h_mean']):.4f} | "
            f"{row['slope_100_200h_per_h']:.5f} | "
            f"`{sn}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Cluster IDs are SOM node coordinates, **not** predetermined physical labels.")
    lines.append("- \"suggested name\" is a post-hoc descriptor derived from metrics only.")
    lines.append(f"- Empty-node check: {n_empty} empty nodes "
                 f"(no auto-retry of seed; reported to user).")
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "n": n,
        "som_shape": list(som_shape),
        "quantisation_error": float(qe),
        "topographic_error": float(te),
        "n_empty_nodes": n_empty,
        "cluster_sizes": cluster_sizes,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
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
    sigma = float(base_cfg["sigma"])
    learning_rate = float(base_cfg["learning_rate"])
    iterations = int(base_cfg["iterations"])
    random_seed = int(base_cfg["random_seed"])

    # ---- n=5 (1x5) ---------------------------------------------------------
    n5_dir = config.OUTPUT_DIR / "n5"
    n5_fig = n5_dir / "figures"
    n5_summary = run_n_clusters(
        n=5,
        som_shape=(1, 5),
        X=X,
        sample_ids=sample_ids,
        max_pce_200h=max_pce_200h,
        out_dir=n5_dir,
        fig_dir=n5_fig,
        sigma=sigma,
        learning_rate=learning_rate,
        iterations=iterations,
        random_seed=random_seed,
    )

    # ---- n=6 (2x3) ---------------------------------------------------------
    n6_dir = config.OUTPUT_DIR / "n6"
    n6_fig = n6_dir / "figures"
    n6_summary = run_n_clusters(
        n=6,
        som_shape=(2, 3),
        X=X,
        sample_ids=sample_ids,
        max_pce_200h=max_pce_200h,
        out_dir=n6_dir,
        fig_dir=n6_fig,
        sigma=sigma,
        learning_rate=learning_rate,
        iterations=iterations,
        random_seed=random_seed,
    )

    # ---- Comparison summary -------------------------------------------------
    logger.info("=== Writing n5 vs n6 comparison ===")
    # Main model QE for context
    main_qe_row = pd.read_csv(config.OUTPUT_DIR / "quantisation_error_by_n.csv")
    main_qe_row = main_qe_row[main_qe_row["topology_source"].str.contains("main")]
    main_qe = main_qe_row["quantisation_error"].iloc[0] if not main_qe_row.empty else float("nan")

    comp = pd.DataFrame([
        {"model": "main 2x2", "n_nodes": 4, "topology": "2x2",
         "quantisation_error": main_qe, "n_empty_nodes": 0,
         "cluster_sizes": json.dumps([])},
        {"model": "n5_1x5", "n_nodes": 5, "topology": "1x5",
         "quantisation_error": n5_summary["quantisation_error"],
         "n_empty_nodes": n5_summary["n_empty_nodes"],
         "cluster_sizes": json.dumps(n5_summary["cluster_sizes"])},
        {"model": "n6_2x3", "n_nodes": 6, "topology": "2x3",
         "quantisation_error": n6_summary["quantisation_error"],
         "n_empty_nodes": n6_summary["n_empty_nodes"],
         "cluster_sizes": json.dumps(n6_summary["cluster_sizes"])},
    ])
    comp.insert(0, "total_samples",
                comp["cluster_sizes"].apply(lambda s: sum(int(x) for x in json.loads(s))))
    # main 2x2 has no cluster_sizes — patch to 1812 (the master input size)
    comp.loc[comp["model"] == "main 2x2", "total_samples"] = 1812
    comp = comp[["total_samples", "model", "n_nodes", "topology",
                 "quantisation_error", "n_empty_nodes", "cluster_sizes"]]
    comp = comp.sort_values("quantisation_error", ascending=False).reset_index(drop=True)
    comp.to_csv(config.OUTPUT_DIR / "n5_vs_n6_summary.csv", index=False)

    # Plot: side-by-side mean curves
    fig5_dir = config.FIG_DIR
    fig5_dir.mkdir(parents=True, exist_ok=True)
    plotting.plot_cluster_size_distribution_n(n5_summary["cluster_sizes"], "n=5 cluster sizes", fig5_dir / "n5_cluster_size_distribution.png")
    plotting.plot_cluster_size_distribution_n(n6_summary["cluster_sizes"], "n=6 cluster sizes", fig5_dir / "n6_cluster_size_distribution.png")

    # Pairwise distance between cluster centroids (illustrative)
    _centroid_diagnostics(n5_dir, "n5")
    _centroid_diagnostics(n6_dir, "n6")

    logger.info("=== Done. Outputs: %s, %s ===", n5_dir, n6_dir)


def _centroid_diagnostics(out_dir: Path, tag: str) -> None:
    """Write a small CSV containing pairwise distances between cluster centroids."""
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


if __name__ == "__main__":
    main()
