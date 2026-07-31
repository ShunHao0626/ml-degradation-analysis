#!/usr/bin/env python3
"""Reproduce every SOM class count explicitly shown in the article/SI."""

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
SI_OUT = PROJECT / "09_si_exact_replication"
sys.path.insert(0, str(HERE))

from som200h.config import AnalysisConfig
from som200h.modeling import (
    _davies_bouldin_score_stable,
    _pairwise_seed_stability,
    _validation_embedding,
    build_cluster_outputs,
    train_som,
)
from som200h.plotting import plot_som_clusters
from som200h.preprocessing import PreprocessedDataset

sns.set_theme(style="whitegrid", context="notebook")

SI_EXPLICIT_SOM_COUNTS = (16, 2, 4, 5, 6)
SI_QE_SCAN_COUNTS = tuple(range(2, 11))
TOPOLOGY_16 = (4, 4)


def config() -> AnalysisConfig:
    return AnalysisConfig(
        project_root=PROJECT,
        dataset_root=REPO / "lab" / "data_all",
        work_md=REPO / "thesis" / "paper" / "work.md",
        supplementary_md=REPO / "thesis" / "paper" / "Supplementary.md",
    )


def load_dataset(name: str) -> PreprocessedDataset:
    root = PROJECT / "03_preprocessed" / name
    return PreprocessedDataset(
        name=name,
        metadata=pd.read_csv(root / "curve_metadata.csv"),
        time_grid=pd.read_csv(root / "time_grid.csv")["time_h"].to_numpy(),
        x_interpolated=np.load(root / "X_interpolated.npy"),
        x_normalized=np.load(root / "X_normalized.npy"),
        x_smoothed=np.load(root / "X_smoothed.npy"),
    )


def run_n16(
    dataset: PreprocessedDataset,
    analysis_config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = PROJECT / "04_som_results" / dataset.name / "n_16"
    result.mkdir(parents=True, exist_ok=True)
    runs = {}
    labels_by_seed: dict[int, np.ndarray] = {}
    seed_rows: list[dict] = []
    for seed in analysis_config.som_stability_seeds:
        print(f"{dataset.name}: n=16 seed={seed}", flush=True)
        run = train_som(
            dataset.x_smoothed,
            n_nodes=16,
            topology=TOPOLOGY_16,
            seed=seed,
            config=analysis_config,
        )
        runs[seed] = run
        labels_by_seed[seed] = run.labels
        sizes = np.bincount(run.labels, minlength=16)
        seed_rows.append(
            {
                "dataset": dataset.name,
                "n_nodes": 16,
                "topology": "4x4",
                "seed": seed,
                "quantisation_error": run.qe,
                "topographic_error": run.te,
                "n_empty_nodes": int(np.sum(sizes == 0)),
                "min_cluster_size": int(np.min(sizes)),
                "max_cluster_size": int(np.max(sizes)),
            }
        )

    primary = runs[analysis_config.som_primary_seed]
    stability, mean_ari, min_ari = _pairwise_seed_stability(labels_by_seed)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(result / "seed_metrics.csv", index=False)
    stability.to_csv(result / "pairwise_seed_stability.csv", index=False)

    assignments, summary, centroids, pairwise = build_cluster_outputs(
        dataset,
        primary,
    )
    selected_database_root = (
        PROJECT / "02_selected_curve_database" / dataset.name
    )
    assignments["selected_database_absolute_path"] = assignments[
        "source_relative_path"
    ].map(lambda relative: str(selected_database_root / relative))
    assignments.to_csv(result / "cluster_assignments.csv", index=False)
    summary.to_csv(result / "cluster_summary.csv", index=False)
    centroids.to_csv(result / "centroid_curves.csv", index=False)
    pairwise.to_csv(result / "centroid_pairwise_similarity.csv", index=False)
    np.save(result / "som_weights.npy", primary.som.get_weights())
    with (result / "som_model.pkl").open("wb") as handle:
        pickle.dump(primary.som, handle)

    embedding = _validation_embedding(dataset.x_smoothed, analysis_config)
    validation_distances = cdist(embedding, embedding, metric="euclidean")
    sizes = np.bincount(primary.labels, minlength=16)
    offdiag = pairwise["cluster_i"] != pairwise["cluster_j"]
    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset.name,
                "n_nodes": 16,
                "topology": "4x4",
                "primary_seed": analysis_config.som_primary_seed,
                "qe_primary_seed": primary.qe,
                "qe_mean_across_seeds": seed_frame[
                    "quantisation_error"
                ].mean(),
                "qe_std_across_seeds": seed_frame[
                    "quantisation_error"
                ].std(ddof=0),
                "te_primary_seed": primary.te,
                "te_mean_across_seeds": seed_frame[
                    "topographic_error"
                ].mean(),
                "mean_pairwise_seed_ARI": mean_ari,
                "min_pairwise_seed_ARI": min_ari,
                "silhouette_pca20": float(
                    silhouette_score(
                        validation_distances,
                        primary.labels,
                        metric="precomputed",
                    )
                ),
                "davies_bouldin_pca20": _davies_bouldin_score_stable(
                    embedding,
                    primary.labels,
                ),
                "calinski_harabasz_pca20": float(
                    calinski_harabasz_score(embedding, primary.labels)
                ),
                "max_centroid_correlation": float(
                    pairwise.loc[offdiag, "pearson_correlation"].max()
                ),
                "min_centroid_rmse": float(
                    pairwise.loc[offdiag, "centroid_rmse"].min()
                ),
                "n_empty_nodes": int(np.sum(sizes == 0)),
                "min_cluster_size": int(np.min(sizes)),
                "min_cluster_fraction": float(np.min(sizes) / len(sizes)),
                "max_cluster_size": int(np.max(sizes)),
                "max_cluster_fraction": float(
                    np.max(sizes) / dataset.x_smoothed.shape[0]
                ),
            }
        ]
    )
    # Correct the denominator after constructing a single-row metrics table.
    metrics["min_cluster_fraction"] = (
        int(np.min(sizes)) / dataset.x_smoothed.shape[0]
    )
    metrics.to_csv(result / "n16_metrics.csv", index=False)
    plot_som_clusters(
        dataset,
        primary,
        result / "cluster_curves.png",
        analysis_config.plot_max_curves_per_cluster,
    )
    return metrics, assignments


def build_traceable_class_folders(
    dataset_name: str,
    n_nodes: int,
    assignments: pd.DataFrame,
) -> None:
    output = SI_OUT / dataset_name / f"n_{n_nodes:02d}" / "classes"
    source_database = (
        PROJECT / "02_selected_curve_database" / dataset_name
    )
    assignments = assignments.copy()
    assignments["selected_database_absolute_path"] = assignments[
        "source_relative_path"
    ].map(lambda relative: str(source_database / relative))
    for class_label, members in assignments.groupby("class_label"):
        class_dir = output / class_label
        class_dir.mkdir(parents=True, exist_ok=True)
        members.to_csv(class_dir / "members.csv", index=False)
        raw_root = class_dir / "raw_curves"
        for row in members.itertuples(index=False):
            source = source_database / row.source_relative_path
            destination = raw_root / row.source_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)


def load_assignments(dataset_name: str, n_nodes: int) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT
        / "04_som_results"
        / dataset_name
        / f"n_{n_nodes:02d}"
        / "cluster_assignments.csv"
    )


def write_si_mapping() -> None:
    rows = [
        {
            "article_or_si_location": "Supplementary Fig. 1",
            "method": "SOM",
            "n_or_range": "16",
            "role": "initial overview of diverse degradation shapes",
            "local_result": "04_som_results/{dataset}/n_16/",
        },
        {
            "article_or_si_location": "Supplementary Fig. 6",
            "method": "SOM quantisation error",
            "n_or_range": "2-10",
            "role": "elbow scan",
            "local_result": "04_som_results/{dataset}/n_02 through n_10/",
        },
        {
            "article_or_si_location": "Main Fig. 4",
            "method": "SOM",
            "n_or_range": "4",
            "role": "selected main classification",
            "local_result": "04_som_results/{dataset}/n_04/",
        },
        {
            "article_or_si_location": "Supplementary Fig. 15",
            "method": "SOM",
            "n_or_range": "2",
            "role": "under-clustered comparison",
            "local_result": "04_som_results/{dataset}/n_02/",
        },
        {
            "article_or_si_location": "Supplementary Fig. 16",
            "method": "SOM",
            "n_or_range": "5",
            "role": "overlap comparison",
            "local_result": "04_som_results/{dataset}/n_05/",
        },
        {
            "article_or_si_location": "Supplementary Fig. 17",
            "method": "SOM",
            "n_or_range": "6",
            "role": "more pronounced overlap comparison",
            "local_result": "04_som_results/{dataset}/n_06/",
        },
        {
            "article_or_si_location": "Supplementary Fig. 8",
            "method": "K-means reference",
            "n_or_range": "4",
            "role": "independent clustering comparison",
            "local_result": "05_cluster_number_decision/{dataset}/kmeans_validation.csv",
        },
    ]
    pd.DataFrame(rows).to_csv(
        SI_OUT / "SI_FIGURE_TO_LOCAL_RESULT_MAP.csv",
        index=False,
    )


def centroid_table(dataset_name: str, n_nodes: int) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT
        / "04_som_results"
        / dataset_name
        / f"n_{n_nodes:02d}"
        / "centroid_curves.csv"
    ).sort_values("ordered_class_id")


def plot_centroid_panel(
    ax: plt.Axes,
    dataset_name: str,
    n_nodes: int,
    time_grid: np.ndarray,
) -> None:
    centroids = centroid_table(dataset_name, n_nodes)
    time_columns = [
        column for column in centroids.columns if column.startswith("t_")
    ]
    colors = sns.color_palette("viridis", n_nodes)
    for color, row in zip(colors, centroids.itertuples(index=False)):
        curve = np.asarray(
            [getattr(row, column) for column in time_columns],
            dtype=float,
        )
        ax.plot(
            time_grid,
            curve,
            color=color,
            linewidth=1.5,
            label=f"C{int(row.ordered_class_id)} (n={int(row.n_curves)})",
        )
    ax.set_title(f"SOM n={n_nodes}")
    ax.set_xlim(0, 200)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Normalized PCE")
    if n_nodes <= 6:
        ax.legend(fontsize=7, ncol=2, loc="lower left")


def plot_si_overview(
    dataset_name: str,
    time_grid: np.ndarray,
) -> None:
    # The sixth panel is an elbow plot, so it must not share curve axes.
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for n_nodes, ax in zip(SI_EXPLICIT_SOM_COUNTS, axes.flat[:5]):
        plot_centroid_panel(ax, dataset_name, n_nodes, time_grid)

    qe = pd.read_csv(
        PROJECT
        / "04_som_results"
        / dataset_name
        / "som_cluster_number_metrics.csv"
    )
    ax = axes.flat[5]
    ax.errorbar(
        qe["n_nodes"],
        qe["qe_mean_across_seeds"],
        yerr=qe["qe_std_across_seeds"],
        color="#2563EB",
        marker="o",
        capsize=3,
    )
    ax.axvspan(3.5, 6.5, color="#FDE68A", alpha=0.35)
    ax.axvline(4, color="#DC2626", linestyle="--")
    ax.set_title("SI Fig. 6 equivalent: QE n=2–10")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Quantisation error")
    ax.set_xlim(1.7, 10.3)
    ax.set_ylim(bottom=0)
    fig.suptitle(
        f"SI class-count replication — {dataset_name}\n"
        "n=16 overview; n=2, 4, 5, 6 comparisons; QE scan n=2–10",
        y=0.99,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    fig.savefig(
        SI_OUT / dataset_name / "si_class_count_overview.png",
        dpi=220,
    )
    plt.close(fig)


def write_combined_metrics(
    dataset_name: str,
    n16_metrics: pd.DataFrame,
) -> None:
    scan = pd.read_csv(
        PROJECT
        / "04_som_results"
        / dataset_name
        / "som_cluster_number_metrics.csv"
    )
    columns = [
        "dataset",
        "n_nodes",
        "topology",
        "qe_primary_seed",
        "qe_mean_across_seeds",
        "qe_std_across_seeds",
        "te_primary_seed",
        "te_mean_across_seeds",
        "mean_pairwise_seed_ARI",
        "min_pairwise_seed_ARI",
        "silhouette_pca20",
        "davies_bouldin_pca20",
        "calinski_harabasz_pca20",
        "max_centroid_correlation",
        "min_centroid_rmse",
        "n_empty_nodes",
        "min_cluster_size",
        "min_cluster_fraction",
        "max_cluster_size",
        "max_cluster_fraction",
    ]
    combined = pd.concat(
        [scan[columns], n16_metrics[columns]],
        ignore_index=True,
    ).sort_values("n_nodes")
    combined.to_csv(
        SI_OUT / dataset_name / "som_metrics_n2_to_n10_plus_n16.csv",
        index=False,
    )


def write_guide(
    main_n16: pd.DataFrame,
    paper_n16: pd.DataFrame,
) -> None:
    main = main_n16.iloc[0]
    paper = paper_n16.iloc[0]
    lines = [
        "# SI中全部SOM类别数量的复现",
        "",
        "## SI实际使用的类别数量",
        "",
        "- Supplementary Fig. 1：n=16，作为初始曲线形状概览。",
        "- Supplementary Fig. 6：n=2–10，用于Quantisation Error elbow扫描。",
        "- Main Fig. 4：n=4，最终主分类。",
        "- Supplementary Figs. 15–17：n=2、5、6，用于展示分类不足和分类重叠。",
        "",
        "因此，类别数层面需要运行的是n=2–10以及额外的n=16。二者现已全部完成。",
        "",
        "## 新增n=16结果",
        "",
        f"- 主数据集：QE={main['qe_mean_across_seeds']:.4f}，"
        f"平均种子ARI={main['mean_pairwise_seed_ARI']:.4f}，"
        f"最小类别占比={main['min_cluster_fraction']:.2%}。",
        f"- 文献一致性数据集：QE={paper['qe_mean_across_seeds']:.4f}，"
        f"平均种子ARI={paper['mean_pairwise_seed_ARI']:.4f}，"
        f"最小类别占比={paper['min_cluster_fraction']:.2%}。",
        "",
        "n=16用于展示数据多样性，不用于替代最终n=4结论。随着类别增加，"
        "部分节点样本很少且质心相互接近，正是SI随后进行elbow分析和类别合并判断的原因。",
        "",
        "## 文件",
        "",
        "- `SI_FIGURE_TO_LOCAL_RESULT_MAP.csv`：SI图号与本地结果一一对应。",
        "- `{dataset}/si_class_count_overview.png`：n=16、2、4、5、6和QE扫描总览。",
        "- `{dataset}/som_metrics_n2_to_n10_plus_n16.csv`：2–10及16类指标。",
        "- `{dataset}/n_XX/classes/class_XX/members.csv`：每一类具体曲线。",
        "- 每个`raw_curves/`目录按原始相对路径保存可浏览曲线硬链接。",
        "",
    ]
    (SI_OUT / "SI_EXACT_REPLICATION_CN.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    SI_OUT.mkdir(parents=True, exist_ok=True)
    analysis_config = config()
    datasets = {
        name: load_dataset(name)
        for name in ("main_high_quality_min10", "paper_aligned_min4")
    }
    n16_metrics: dict[str, pd.DataFrame] = {}
    for name, dataset in datasets.items():
        metrics, assignments = run_n16(dataset, analysis_config)
        n16_metrics[name] = metrics
        build_traceable_class_folders(name, 16, assignments)
        for n_nodes in (2, 4, 5, 6):
            build_traceable_class_folders(
                name,
                n_nodes,
                load_assignments(name, n_nodes),
            )
        plot_si_overview(name, dataset.time_grid)
        write_combined_metrics(name, metrics)

    write_si_mapping()
    write_guide(
        n16_metrics["main_high_quality_min10"],
        n16_metrics["paper_aligned_min4"],
    )
    (SI_OUT / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "si_explicit_som_counts": list(SI_EXPLICIT_SOM_COUNTS),
                "si_qe_scan_counts": list(SI_QE_SCAN_COUNTS),
                "n16_topology": list(TOPOLOGY_16),
                "seeds": list(analysis_config.som_stability_seeds),
                "iterations_per_seed": analysis_config.som_iterations,
                "datasets": {
                    name: int(dataset.x_smoothed.shape[0])
                    for name, dataset in datasets.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SI exact replication complete: {SI_OUT}")


if __name__ == "__main__":
    main()
