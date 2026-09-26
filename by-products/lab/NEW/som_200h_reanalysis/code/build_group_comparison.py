#!/usr/bin/env python3
"""Build a slide-ready comparison package for every SOM size n=2..10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "08_group_presentation_comparison"
DATASET = "main_high_quality_min10"
sys.path.insert(0, str(HERE))

sns.set_theme(style="whitegrid", context="notebook")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    metrics = pd.read_csv(
        PROJECT
        / "04_som_results"
        / DATASET
        / "som_cluster_number_metrics.csv"
    )
    kmeans = pd.read_csv(
        PROJECT
        / "05_cluster_number_decision"
        / DATASET
        / "kmeans_validation.csv"
    ).rename(columns={"k": "n_nodes"})
    time_grid = pd.read_csv(
        PROJECT
        / "03_preprocessed"
        / DATASET
        / "time_grid.csv"
    )["time_h"].to_numpy()
    return metrics, kmeans, time_grid


def build_tables(
    metrics: pd.DataFrame,
    kmeans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision = pd.read_csv(
        PROJECT
        / "05_cluster_number_decision"
        / DATASET
        / "cluster_number_decision.csv"
    )
    combined = metrics.merge(
        kmeans[
            [
                "n_nodes",
                "WCSS_full_1201d",
                "relative_WCSS_improvement_from_previous",
            ]
        ],
        on="n_nodes",
        how="left",
    ).merge(
        decision[
            [
                "n_nodes",
                "in_SI_candidate_range_4_to_6",
                "passes_all_SI_style_rules",
                "selected_n",
                "selection_reason",
            ]
        ],
        on="n_nodes",
        how="left",
    )
    combined.to_csv(OUT / "n2_to_n10_metrics_comparison.csv", index=False)

    size_rows: list[dict] = []
    index_rows: list[dict] = []
    for n_nodes in range(2, 11):
        result = PROJECT / "04_som_results" / DATASET / f"n_{n_nodes:02d}"
        summary = pd.read_csv(result / "cluster_summary.csv")
        for row in summary.itertuples(index=False):
            size_rows.append(
                {
                    "n_nodes": n_nodes,
                    "topology": combined.loc[
                        combined["n_nodes"] == n_nodes, "topology"
                    ].iloc[0],
                    "raw_cluster_id": int(row.raw_cluster_id),
                    "ordered_class_id": int(row.ordered_class_id),
                    "class_label": row.class_label,
                    "n_curves": int(row.n_curves),
                    "fraction": float(row.fraction),
                    "pce_norm_200h_mean": float(row.pce_norm_200h_mean),
                    "suggested_shape_name": row.suggested_shape_name,
                }
            )
        index_rows.append(
            {
                "n_nodes": n_nodes,
                "topology": combined.loc[
                    combined["n_nodes"] == n_nodes, "topology"
                ].iloc[0],
                "cluster_curve_figure": str(
                    (result / "cluster_curves.png").relative_to(PROJECT)
                ),
                "cluster_assignments": str(
                    (result / "cluster_assignments.csv").relative_to(PROJECT)
                ),
                "cluster_summary": str(
                    (result / "cluster_summary.csv").relative_to(PROJECT)
                ),
                "centroid_curves": str(
                    (result / "centroid_curves.csv").relative_to(PROJECT)
                ),
                "seed_metrics": str(
                    (result / "seed_metrics.csv").relative_to(PROJECT)
                ),
                "som_model": str(
                    (result / "som_model.pkl").relative_to(PROJECT)
                ),
            }
        )

    sizes = pd.DataFrame(size_rows)
    result_index = pd.DataFrame(index_rows)
    sizes.to_csv(OUT / "cluster_sizes_and_endpoints_all_n.csv", index=False)
    result_index.to_csv(OUT / "per_n_result_index.csv", index=False)
    return combined, sizes, result_index


def centroid_frame(n_nodes: int) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT
        / "04_som_results"
        / DATASET
        / f"n_{n_nodes:02d}"
        / "centroid_curves.csv"
    ).sort_values("ordered_class_id")


def plot_centroids(
    ax: plt.Axes,
    n_nodes: int,
    time_grid: np.ndarray,
    metrics: pd.DataFrame,
    show_legend: bool = True,
) -> None:
    centroids = centroid_frame(n_nodes)
    time_columns = [column for column in centroids if column.startswith("t_")]
    colors = sns.color_palette("viridis", n_nodes)
    for color, row in zip(colors, centroids.itertuples(index=False)):
        curve = np.asarray(
            [getattr(row, column) for column in time_columns],
            dtype=float,
        )
        ax.plot(
            time_grid,
            curve,
            linewidth=1.8,
            color=color,
            label=f"C{int(row.ordered_class_id)} (n={int(row.n_curves)})",
        )
    metric = metrics.loc[metrics["n_nodes"] == n_nodes].iloc[0]
    ax.set_title(
        f"n={n_nodes} ({metric['topology']})  "
        f"QE={metric['qe_mean_across_seeds']:.3f}  "
        f"Sil={metric['silhouette_pca20']:.3f}  "
        f"min RMSE={metric['min_centroid_rmse']:.3f}"
    )
    ax.set_xlim(0, 200)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Relative ageing time (h)")
    ax.set_ylabel("Normalized PCE")
    if show_legend:
        ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    if n_nodes == 4:
        for spine in ax.spines.values():
            spine.set_color("#DC2626")
            spine.set_linewidth(2.2)


def plot_all_centroid_overview(
    metrics: pd.DataFrame,
    time_grid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), sharex=True, sharey=True)
    for n_nodes, ax in zip(range(2, 11), axes.flat):
        plot_centroids(ax, n_nodes, time_grid, metrics)
    fig.suptitle(
        "SOM n=2–10 centroid comparison — main dataset (1,442 curves)\n"
        "Red border marks the selected n=4 model",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "som_n2_to_n10_centroid_overview.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_candidate_comparison(
    metrics: pd.DataFrame,
    time_grid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True)
    for n_nodes, ax in zip((4, 5, 6), axes):
        plot_centroids(ax, n_nodes, time_grid, metrics)
    fig.suptitle(
        "SI candidate range comparison: n=4, 5 and 6\n"
        "At n=5–6 the closest-centroid RMSE falls below 0.10",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "som_n4_n5_n6_candidate_comparison.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cluster_size_heatmap(sizes: pd.DataFrame) -> None:
    pivot = (
        sizes.assign(fraction_percent=sizes["fraction"] * 100.0)
        .pivot(
            index="n_nodes",
            columns="ordered_class_id",
            values="fraction_percent",
        )
        .reindex(index=range(2, 11), columns=range(1, 11))
    )
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(
        pivot,
        mask=pivot.isna(),
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Percentage of curves"},
        ax=ax,
    )
    ax.set_title("Cluster-size evolution as SOM node count increases")
    ax.set_xlabel("Ordered class (higher → lower PCE at 200 h)")
    ax.set_ylabel("Number of SOM nodes")
    fig.tight_layout()
    fig.savefig(OUT / "som_n2_to_n10_cluster_size_heatmap.png", dpi=220)
    plt.close(fig)


def write_guide(combined: pd.DataFrame) -> None:
    table = combined[
        [
            "n_nodes",
            "topology",
            "qe_mean_across_seeds",
            "qe_relative_improvement_from_previous",
            "mean_pairwise_seed_ARI",
            "silhouette_pca20",
            "davies_bouldin_pca20",
            "min_centroid_rmse",
            "min_cluster_fraction",
            "selected_n",
        ]
    ].copy()
    table["min_cluster_fraction"] *= 100.0
    table = table.round(4)

    def markdown(frame: pd.DataFrame) -> str:
        header = "| " + " | ".join(frame.columns) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *rows])

    lines = [
        "# SOM 2–10类课题组汇报对比包",
        "",
        "主分析数据为1,442条曲线。n=2–10均已分别训练，不是只运行n=4。",
        "每个n均使用5个随机种子和50,000次训练。",
        "",
        "## 推荐汇报顺序",
        "",
        "1. `som_n2_to_n10_centroid_overview.png`：先展示2–10类形状如何逐步细分。",
        "2. `som_n4_n5_n6_candidate_comparison.png`：重点比较SI候选范围4–6类。",
        "3. `som_n2_to_n10_cluster_size_heatmap.png`：展示新增类别的样本占比。",
        "4. `../05_cluster_number_decision/main_high_quality_min10/cluster_number_decision.png`：展示QE、稳定性、质心分离和K-means WCSS。",
        "",
        "## 结论",
        "",
        "- n=4、5、6均具有很高的随机种子稳定性。",
        "- n=4的最近质心RMSE为0.121，仍高于0.10分离阈值。",
        "- n=5和n=6分别降至约0.074和0.068，开始拆分已有主要形状。",
        "- 因此n=4作为主分类；n=5和n=6作为更细粒度的对照结果保留。",
        "",
        "## 全部数值",
        "",
        markdown(table),
        "",
        "## 每个分类数的完整结果",
        "",
    ]
    for n_nodes in range(2, 11):
        lines.extend(
            [
                f"- n={n_nodes}: `../04_som_results/{DATASET}/n_{n_nodes:02d}/`",
            ]
        )
    (OUT / "PRESENTATION_GUIDE_CN.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, kmeans, time_grid = load_inputs()
    combined, sizes, result_index = build_tables(metrics, kmeans)
    plot_all_centroid_overview(combined, time_grid)
    plot_candidate_comparison(combined, time_grid)
    plot_cluster_size_heatmap(sizes)
    write_guide(combined)
    (OUT / "build_summary.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "n_curves": 1442,
                "models_compared": list(range(2, 11)),
                "selected_n": 4,
                "files_indexed": int(len(result_index)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Group comparison package: {OUT}")


if __name__ == "__main__":
    main()
