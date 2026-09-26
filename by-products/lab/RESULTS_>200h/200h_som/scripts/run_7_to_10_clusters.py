#!/usr/bin/env python3
"""Generate stand-alone n=7, 8, 9, 10 SOM analyses.

Reads the preprocessed feature matrix saved by the main run (default
``outputs/200h_som/X_preprocessed.npy``) and trains four additional SOMs:

* n=7 → 1×7 topology
* n=8 → 2×4 topology
* n=9 → 3×3 topology
* n=10 → 2×5 topology

All models share the same σ, lr, iterations, random seed and training
method as the main 2×2 SOM, so the runs are directly comparable.

All artefacts are saved into ``outputs/200h_som/n7``, ``n8``, ``n9`` and
``n10``. A combined comparison ``n7_8_9_10_summary.csv`` sits in
``outputs/200h_som/``.

Run::

    PYTHONPATH=. python3 scripts/run_7_to_10_clusters.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
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


# ----------------------------------------------------------------------------
# Re-run of the per-n logic used in run_5_6_clusters.py
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

    logger = logging.getLogger("200h_som_n7_n10")
    logger.info("Training n=%d SOM (shape=%s, σ=%.2f, lr=%.2f)",
                n, som_shape, sigma, learning_rate)
    som = som_analysis.train_som(X, spec)
    coords, dist = som_analysis.assign_bmu_clusters(som, X)
    cluster_ids = som_analysis.bmu_to_cluster_id(coords, som_shape)
    qe = som_analysis.quantization_error(som, X)
    te = som_analysis.topographic_error(som, X)
    logger.info("n=%d:  QE=%.4f  TE=%.4f", n, qe, te)

    som_analysis.save_som(som, out_dir / "som_model.pkl")
    np.save(out_dir / "som_weights.npy", som.get_weights())

    inc = pd.read_csv(config.OUTPUT_DIR / "included_samples.csv")
    assert list(inc["sample_id"]) == sample_ids, "Order mismatch with main run"
    max_pc = inc["max_pce_0_200h"].to_numpy(dtype=float)
    X_norm = X.astype(float).copy()
    X_interp = X_norm * max_pc[:, None]
    max_pce = max_pc.tolist()

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
    if cluster_sizes:
        lines.append(f"| Largest cluster | {max(cluster_sizes)} ({cluster_pct.max()}%) |")
        lines.append(f"| Smallest cluster | {min(cluster_sizes)} ({cluster_pct.min()}%) |")
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


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("200h_som_n7_n10")


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

    # Topology choices (kept as documented in this script):
    #   n=7  → 1×7   (matches QE-sweep 1×n fallback)
    #   n=8  → 2×4   (rectangular 2-D)
    #   n=9  → 3×3   (square 2-D)
    #   n=10 → 2×5   (rectangular 2-D)
    jobs = [
        (7, (1, 7)),
        (8, (2, 4)),
        (9, (3, 3)),
        (10, (2, 5)),
    ]

    summaries: list[dict] = []
    for n, shape in jobs:
        n_dir = config.OUTPUT_DIR / f"n{n}"
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
        summaries.append(s)
        _centroid_diagnostics(n_dir)

        # Top-level size distribution plot
        plotting.plot_cluster_size_distribution_n(
            s["cluster_sizes"],
            f"n={n} cluster sizes",
            config.FIG_DIR / f"n{n}_cluster_size_distribution.png",
        )

    # Combined comparison
    logger.info("=== Writing n7_8_9_10 comparison ===")
    main_qe_row = pd.read_csv(config.OUTPUT_DIR / "quantisation_error_by_n.csv")
    main_qe_row = main_qe_row[main_qe_row["topology_source"].str.contains("main")]
    main_qe = float(main_qe_row["quantisation_error"].iloc[0]) if not main_qe_row.empty else float("nan")

    comp_rows = [
        {
            "model": "main 2x2", "n_nodes": 4, "topology": "2x2",
            "quantisation_error": main_qe, "n_empty_nodes": 0,
            "cluster_sizes": "",
        },
    ]
    for s in summaries:
        comp_rows.append({
            "model": f"n{s['n']}_{s['som_shape'][0]}x{s['som_shape'][1]}",
            "n_nodes": s["n"],
            "topology": f"{s['som_shape'][0]}x{s['som_shape'][1]}",
            "quantisation_error": s["quantisation_error"],
            "n_empty_nodes": s["n_empty_nodes"],
            "cluster_sizes": json.dumps(s["cluster_sizes"]),
        })
    comp = pd.DataFrame(comp_rows)
    comp.insert(0, "total_samples",
                comp["cluster_sizes"].apply(
                    lambda s: sum(int(x) for x in json.loads(s))
                    if s else 1812  # main 2x2 row uses 1812 directly
                ))
    comp = comp[["total_samples", "model", "n_nodes", "topology",
                 "quantisation_error", "n_empty_nodes", "cluster_sizes"]]
    comp = comp.sort_values("quantisation_error", ascending=False).reset_index(drop=True)
    comp.to_csv(config.OUTPUT_DIR / "n7_8_9_10_summary.csv", index=False)

    # Cross-model between n=5,6,7,8,9,10 — incremental companion summary
    prev = config.OUTPUT_DIR / "n5_vs_n6_summary.csv"
    if prev.exists():
        prev_df = pd.read_csv(prev)
        # If prev has no total_samples column (older format), add it now
        if "total_samples" not in prev_df.columns:
            prev_df.insert(0, "total_samples",
                           prev_df["cluster_sizes"].apply(
                               lambda s: sum(int(x) for x in json.loads(s))
                               if pd.notna(s) and s != "" else 1812
                           ))
        merged = pd.concat([prev_df, comp], ignore_index=True)
        merged = merged.drop_duplicates(subset=["model", "n_nodes"], keep="last")
        merged = merged.sort_values("quantisation_error", ascending=False).reset_index(drop=True)
        merged.to_csv(config.OUTPUT_DIR / "all_n_summary.csv", index=False)

    logger.info("=== Done. n7..n10 outputs in outputs/n7..n10/ ===")


if __name__ == "__main__":
    main()
