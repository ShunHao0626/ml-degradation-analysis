#!/usr/bin/env python3
"""Restore the 1000-curve dataset to its values before per-curve max normalization.

The generator recorded the divisor used for each curve as
``normalization_reference_raw``.  Multiplying every normalized observation by
that curve-specific divisor reconstructs the pre-normalization synthetic PCE
values (up to the CSV's 12-significant-digit serialization precision).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ("bridge", "hill", "slope", "valley")
REGIMES = (
    "subhour_to_2h",
    "short_8_to_80h",
    "medium_120_to_1000h",
    "long_1200_to_4500h",
)
COLORS = {
    "bridge": "#2f6db0",
    "hill": "#d95f02",
    "slope": "#2a9d58",
    "valley": "#8e5bb7",
}
VALUE_COLUMN = "pce_unnormalized"


def load_scales(path: Path) -> tuple[dict[str, float], list[dict[str, object]]]:
    scales: dict[str, float] = {}
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            curve_id = str(record["curve_id"])
            scale = float(record["parameters"]["normalization_reference_raw"])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid restoration scale for {curve_id}: {scale}")
            if curve_id in scales:
                raise ValueError(f"duplicate curve_id in generation parameters: {curve_id}")
            scales[curve_id] = scale
            record["restored_output"] = {
                "value_column": VALUE_COLUMN,
                "operation": "normalized_pce * normalization_reference_raw",
                "normalization_removed": True,
                "unit": "synthetic PCE scale; not an absolute measured percentage",
            }
            records.append(record)
    return scales, records


def restore(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scales, parameter_records = load_scales(source / "generation_parameters.jsonl")

    long_table = pd.read_csv(source / "curve_data_long.csv")
    required = {
        "curve_id",
        "target_class",
        "observation_index",
        "time_h",
        "normalized_pce",
        "is_injected_outlier",
    }
    if set(long_table.columns) != required:
        raise ValueError(f"unexpected long-table columns: {list(long_table.columns)}")
    if set(long_table["curve_id"].unique()) != set(scales):
        raise ValueError("curve IDs differ between long table and generation parameters")

    long_table["restoration_scale"] = long_table["curve_id"].map(scales)
    long_table[VALUE_COLUMN] = (
        long_table["normalized_pce"] * long_table["restoration_scale"]
    )
    restored_long = long_table[
        [
            "curve_id",
            "target_class",
            "observation_index",
            "time_h",
            VALUE_COLUMN,
            "is_injected_outlier",
        ]
    ]
    restored_long.to_csv(output / "curve_data_long.csv", index=False, float_format="%.12g")

    metadata = pd.read_csv(source / "curve_metadata.csv")
    metadata["restoration_scale"] = metadata["curve_id"].map(scales)
    metadata["min_pce_unnormalized"] = (
        metadata["min_normalized_pce"] * metadata["restoration_scale"]
    )
    metadata["max_pce_unnormalized"] = (
        metadata["max_normalized_pce"] * metadata["restoration_scale"]
    )
    metadata["final_pce_unnormalized"] = (
        metadata["final_normalized_pce"] * metadata["restoration_scale"]
    )
    metadata["normalization_removed"] = 1
    metadata["pce_unit_note"] = "synthetic PCE scale; not absolute measured percent"
    metadata = metadata.drop(
        columns=["min_normalized_pce", "max_normalized_pce", "final_normalized_pce"]
    )
    metadata.to_csv(output / "curve_metadata.csv", index=False, float_format="%.12g")

    curves_root = output / "curves"
    for target in CLASSES:
        (curves_root / target).mkdir(parents=True, exist_ok=True)
    for curve_id, group in restored_long.groupby("curve_id", sort=False):
        target = str(group["target_class"].iloc[0])
        group[["time_h", VALUE_COLUMN, "is_injected_outlier"]].to_csv(
            curves_root / target / f"{curve_id}.csv", index=False, float_format="%.12g"
        )

    with (output / "generation_parameters.jsonl").open("w", encoding="utf-8") as handle:
        for record in parameter_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    shutil.copy2(source / "reference_catalog.csv", output / "reference_catalog.csv")
    for filename in ("class_distribution.png", "duration_distribution.png"):
        shutil.copy2(source / filename, output / filename)

    grouped = restored_long.groupby("curve_id", sort=False)
    audit_rows: list[dict[str, object]] = []
    for curve_id, group in grouped:
        restored_max = float(group[VALUE_COLUMN].max())
        scale = scales[str(curve_id)]
        audit_rows.append(
            {
                "curve_id": curve_id,
                "restoration_scale": scale,
                "restored_max": restored_max,
                "absolute_max_error": abs(restored_max - scale),
                "n_observations": len(group),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "restoration_audit.csv", index=False, float_format="%.12g")

    class_counts = metadata["target_class"].value_counts().reindex(CLASSES).to_dict()
    values = restored_long[VALUE_COLUMN].to_numpy()
    scale_values = audit["restoration_scale"].to_numpy()
    summary = {
        "source_dataset": str(source.resolve()),
        "restoration_operation": "pce_unnormalized = normalized_pce * normalization_reference_raw",
        "curves_total": int(len(metadata)),
        "observations_total": int(len(restored_long)),
        "curves_per_class": {key: int(value) for key, value in class_counts.items()},
        "value_column": VALUE_COLUMN,
        "value_interpretation": "pre-normalization synthetic PCE scale; not absolute measured PCE percent",
        "normalization_applied": False,
        "pce_value_min": float(values.min()),
        "pce_value_median": float(np.median(values)),
        "pce_value_max": float(values.max()),
        "curve_max_min": float(scale_values.min()),
        "curve_max_median": float(np.median(scale_values)),
        "curve_max_max": float(scale_values.max()),
        "all_time_units": "hour",
        "fixed_analysis_window": False,
        "clustering_performed": False,
        "quality_checks": {
            "unique_curve_ids": bool(metadata["curve_id"].is_unique),
            "class_counts_unchanged": class_counts
            == {"bridge": 90, "hill": 160, "slope": 500, "valley": 250},
            "all_values_finite": bool(np.isfinite(values).all()),
            "all_values_nonnegative": bool((values >= 0).all()),
            "time_strictly_increasing_per_curve": bool(
                grouped["time_h"].apply(lambda series: (series.diff().dropna() > 0).all()).all()
            ),
            "restored_curve_max_matches_recorded_scale": bool(
                np.allclose(audit["restored_max"], audit["restoration_scale"], atol=2e-11)
            ),
            "individual_curve_file_count": len(list(curves_root.glob("*/*.csv"))) == 1000,
        },
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_preview(restored_long, metadata, output)
    plot_scale_distribution(audit, output)
    write_dataset_card(output, summary)


def plot_preview(long_table: pd.DataFrame, metadata: pd.DataFrame, output: Path) -> None:
    regimes = list(REGIMES)
    fig, axes = plt.subplots(4, len(regimes), figsize=(18, 14), squeeze=False)
    for row, target in enumerate(CLASSES):
        for column, regime in enumerate(regimes):
            axis = axes[row, column]
            ids = metadata.loc[
                (metadata["target_class"] == target)
                & (metadata["duration_regime"] == regime),
                "curve_id",
            ].head(7)
            for curve_id in ids:
                group = long_table[long_table["curve_id"] == curve_id]
                axis.plot(
                    group["time_h"],
                    group[VALUE_COLUMN],
                    marker="o",
                    markersize=1.7,
                    linewidth=0.9,
                    alpha=0.58,
                    color=COLORS[target],
                )
            if row == 0:
                axis.set_title(regime.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{target.capitalize()}\nunnormalized synthetic PCE")
            if row == len(CLASSES) - 1:
                axis.set_xlabel("time (h)")
            axis.grid(alpha=0.18)
    fig.suptitle("Restored pre-normalization synthetic PCE: variable real-hour durations", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "dataset_preview_unnormalized.png", dpi=180)
    plt.close(fig)


def plot_scale_distribution(audit: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(audit["restoration_scale"], bins=32, color="#3976ad", alpha=0.85)
    axis.axvline(
        audit["restoration_scale"].median(),
        color="#b23b3b",
        linewidth=2,
        label=f"median = {audit['restoration_scale'].median():.3f}",
    )
    axis.set_xlabel("curve-specific pre-normalization maximum")
    axis.set_ylabel("curves")
    axis.set_title("Restoration scales recorded by the generator")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "restoration_scale_distribution.png", dpi=180)
    plt.close(fig)


def write_dataset_card(output: Path, summary: dict[str, object]) -> None:
    card = f"""# 1000 条未归一化合成 PCE 曲线

## 定位

本目录把 `eg_based_synthetic_dataset_1000` 中的每条曲线恢复到“除以自身最大值之前”的数值。恢复公式为：

```text
pce_unnormalized = normalized_pce × normalization_reference_raw
```

`normalization_reference_raw` 是生成器在最初构建每条曲线时记录的实际除数，因此本恢复不需要估计最大值。CSV 序列化使用 12 位有效数字，恢复值与生成器归一化前数值的差异只来自这一写出精度。

这里的 `pce_unnormalized` 是生成器的原始合成尺度。生成器基线约在 1 附近，并未从真实器件的绝对 PCE 百分数分布中拟合，因此不能把数值 1.2 解释为真实实验中的 1.2% PCE。

## 规模

| 类别 | 曲线数 |
|---|---:|
| Bridge | 90 |
| Hill | 160 |
| Slope | 500 |
| Valley | 250 |
| 合计 | 1000 |

- 总观测点：{summary['observations_total']:,}；
- 全部时间仍为 `time_h`，没有统一 200 h 截断；
- 数值范围：{summary['pce_value_min']:.6f}–{summary['pce_value_max']:.6f}；
- 各曲线原始最大值范围：{summary['curve_max_min']:.6f}–{summary['curve_max_max']:.6f}；
- 不规则采样、异常点标志、类别比例和时间跨度均未改变；
- 本步骤未运行聚类或无监督学习。

## 文件

- `curve_data_long.csv`：全部观测长表，纵轴字段为 `pce_unnormalized`；
- `curve_metadata.csv`：每条曲线一行，含恢复除数和未归一化极值；
- `curves/<class>/<curve_id>.csv`：1000 个独立曲线文件；
- `generation_parameters.jsonl`：原生成参数及恢复说明；
- `restoration_audit.csv`：逐曲线恢复除数、恢复最大值和误差；
- `dataset_summary.json`：结构及自动质检；
- `dataset_preview_unnormalized.png`：未归一化数据预览；
- `restoration_scale_distribution.png`：1000 条曲线原始最大值的分布。
- `class_distribution.png`：四类数量与比例；
- `duration_distribution.png`：四类内部时长分布。

## 无监督学习输入边界

聚类时可以使用 `time_h` 与 `pce_unnormalized`。`target_class`、`morphology_qc_pass`、`restoration_scale`、生成随机种子和参考图片字段只能用于追踪或事后评价，不应作为聚类输入。

## 可复现命令

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
MPLCONFIGDIR=/private/tmp/restore_pce_mpl \\
python3 restore_unnormalized_pce_dataset.py \\
  --source eg_based_synthetic_dataset_1000 \\
  --output eg_based_synthetic_dataset_1000_unnormalized_pce
```
"""
    (output / "DATASET_CARD_CN.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=base / "eg_based_synthetic_dataset_1000")
    parser.add_argument(
        "--output", type=Path, default=base / "eg_based_synthetic_dataset_1000_unnormalized_pce"
    )
    args = parser.parse_args()
    restore(args.source, args.output)
    print((args.output / "dataset_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
