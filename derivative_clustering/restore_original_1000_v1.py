#!/usr/bin/env python3
"""Rebuild the first 1000-curve release: normalization by the initial PCE.

This is the version that preceded the later "maximum PCE equals 1" change.
It uses the same curve IDs, time grids, stochastic observations and class
counts, then applies the original rule y_relative = raw_PCE / raw_PCE(t=0).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_eg_based_dataset import COLORS, REGIMES, morphology_qc_pass


CLASSES = ("bridge", "hill", "slope", "valley")


def rebuild(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source / "curve_data_long.csv")
    metadata_source = pd.read_csv(source / "curve_metadata.csv")
    expected = {
        "curve_id",
        "target_class",
        "observation_index",
        "time_h",
        "pce_unnormalized",
        "is_injected_outlier",
    }
    if set(raw.columns) != expected:
        raise ValueError(f"unexpected source columns: {list(raw.columns)}")

    initial = raw.groupby("curve_id", sort=False)["pce_unnormalized"].first()
    if (initial <= 0).any() or not np.isfinite(initial).all():
        raise ValueError("all initial PCE references must be finite and positive")
    raw["initial_pce_reference"] = raw["curve_id"].map(initial)
    raw["y_relative"] = np.clip(
        raw["pce_unnormalized"] / raw["initial_pce_reference"], 0.02, 1.65
    )
    long_table = raw[
        [
            "curve_id",
            "target_class",
            "observation_index",
            "time_h",
            "y_relative",
            "is_injected_outlier",
        ]
    ]
    long_table.to_csv(output / "curve_data_long.csv", index=False, float_format="%.12g")

    records: list[dict[str, object]] = []
    qc_counts = {target: 0 for target in CLASSES}
    curves_root = output / "curves"
    for target in CLASSES:
        (curves_root / target).mkdir(parents=True, exist_ok=True)
    metadata_lookup = metadata_source.set_index("curve_id")
    for curve_id, group in long_table.groupby("curve_id", sort=False):
        source_row = metadata_lookup.loc[curve_id]
        target = str(source_row["target_class"])
        times = group["time_h"].to_numpy(dtype=float)
        values = group["y_relative"].to_numpy(dtype=float)
        minimum = int(np.argmin(values))
        maximum = int(np.argmax(values))
        qc_pass = int(morphology_qc_pass(target, times, values))
        qc_counts[target] += qc_pass
        records.append(
            {
                "curve_id": curve_id,
                "target_class": target,
                "core_definition_version": source_row["core_definition_version"],
                "duration_regime": source_row["duration_regime"],
                "duration_h": float(times[-1]),
                "n_observations": int(len(times)),
                "sampling_pattern": source_row["sampling_pattern"],
                "reference_examples": source_row["reference_examples"],
                "random_seed": int(source_row["random_seed"]),
                "min_y": float(values[minimum]),
                "max_y": float(values[maximum]),
                "final_y": float(values[-1]),
                "time_of_min_h": float(times[minimum]),
                "time_of_max_h": float(times[maximum]),
                "morphology_qc_pass": qc_pass,
            }
        )
        group[["time_h", "y_relative", "is_injected_outlier"]].to_csv(
            curves_root / target / f"{curve_id}.csv", index=False, float_format="%.12g"
        )

    metadata = pd.DataFrame(records)
    metadata.to_csv(output / "curve_metadata.csv", index=False, float_format="%.12g")
    audit = pd.DataFrame(
        {
            "curve_id": initial.index,
            "initial_pce_reference": initial.to_numpy(),
            "reconstructed_initial_y_relative": 1.0,
        }
    )
    audit.to_csv(output / "initial_normalization_audit.csv", index=False, float_format="%.12g")

    with (source / "generation_parameters.jsonl").open(encoding="utf-8") as input_handle, (
        output / "generation_parameters.jsonl"
    ).open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            item = json.loads(line)
            curve_id = str(item["curve_id"])
            item["restored_v1_output"] = {
                "normalization": "initial_observation_equals_1",
                "initial_pce_reference": float(initial[curve_id]),
                "value_column": "y_relative",
                "clip_range": [0.02, 1.65],
            }
            output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    for filename in (
        "reference_catalog.csv",
        "class_distribution.png",
        "duration_distribution.png",
    ):
        shutil.copy2(source / filename, output / filename)

    counts = metadata["target_class"].value_counts().reindex(CLASSES).to_dict()
    regime_counts = {
        target: {
            regime: int(
                ((metadata["target_class"] == target) & (metadata["duration_regime"] == regime)).sum()
            )
            for regime in REGIMES
        }
        for target in CLASSES
    }
    summary = {
        "release_identity": "reconstructed original 1000-curve v1",
        "source_pre_normalization_dataset": str(source.resolve()),
        "normalization": "each curve divided by its first observed synthetic PCE; t=0 equals 1",
        "value_column": "y_relative",
        "clip_range": [0.02, 1.65],
        "curves_total": int(len(metadata)),
        "observations_total": int(len(long_table)),
        "curves_per_class": {key: int(value) for key, value in counts.items()},
        "duration_regime_counts_per_class": regime_counts,
        "duration_h_min": float(metadata["duration_h"].min()),
        "duration_h_median": float(metadata["duration_h"].median()),
        "duration_h_max": float(metadata["duration_h"].max()),
        "observations_min": int(metadata["n_observations"].min()),
        "observations_median": float(metadata["n_observations"].median()),
        "observations_max": int(metadata["n_observations"].max()),
        "all_time_units": "hour",
        "fixed_analysis_window": False,
        "clustering_performed": False,
        "morphology_qc": {
            "type": "fixed rule-based post-generation check; no clustering or learned model",
            "pass_counts": qc_counts,
            "pass_rates": {target: qc_counts[target] / counts[target] for target in CLASSES},
        },
        "quality_checks": {
            "unique_curve_ids": bool(metadata["curve_id"].is_unique),
            "class_counts_match_original_v1": counts
            == {"bridge": 90, "hill": 160, "slope": 500, "valley": 250},
            "observations_match_original_v1": len(long_table) == 39887,
            "each_curve_initial_y_equals_1": bool(
                np.allclose(long_table.groupby("curve_id", sort=False)["y_relative"].first(), 1.0)
            ),
            "time_strictly_increasing_per_curve": bool(
                long_table.groupby("curve_id")["time_h"]
                .apply(lambda series: (series.diff().dropna() > 0).all())
                .all()
            ),
            "all_values_within_original_clip_range": bool(
                long_table["y_relative"].between(0.02, 1.65).all()
            ),
            "individual_curve_file_count": len(list(curves_root.glob("*/*.csv"))) == 1000,
        },
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_preview(long_table, metadata, output)
    write_card(output, summary)


def plot_preview(long_table: pd.DataFrame, metadata: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), squeeze=False)
    for row, target in enumerate(CLASSES):
        for column, regime in enumerate(REGIMES):
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
                    group["y_relative"],
                    marker="o",
                    markersize=1.8,
                    linewidth=0.9,
                    alpha=0.58,
                    color=COLORS[target],
                )
            if row == 0:
                axis.set_title(regime.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{target.capitalize()}\ny relative (initial = 1)")
            if row == len(CLASSES) - 1:
                axis.set_xlabel("time (h)")
            axis.grid(alpha=0.18)
    fig.suptitle("Original 1000-curve v1: initial-value-normalized synthetic PCE", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "dataset_preview.png", dpi=180)
    plt.close(fig)


def write_card(output: Path, summary: dict[str, object]) -> None:
    qc = summary["morphology_qc"]
    card = f"""# 原始 1000 条不均衡人工数据集（v1 重建版）

## 这是不是最先生成的 1000 条版本

是。本目录重建的是后来改成“每条曲线最大值为 1”之前的 1000 条版本。曲线编号、随机种子、时间点、噪声、异常点和类别数量均与当时相同。

该版本仍然做过归一化，规则为：

```text
y_relative = 原始合成 PCE / 该曲线第一个观测值
```

因此每条曲线在 `time_h=0` 时的 `y_relative` 严格为 1。数值按原规则限制在 `[0.02, 1.65]`。它不同于未归一化原始尺度，也不同于后来“最大值等于 1”的版本。

## 数据规模

| 类别 | 数量 | 比例 |
|---|---:|---:|
| Bridge | 90 | 9% |
| Hill | 160 | 16% |
| Slope | 500 | 50% |
| Valley | 250 | 25% |
| 合计 | 1000 | 100% |

- 总观测点：{summary['observations_total']:,}；
- 横坐标：`time_h`；
- 持续时间：{summary['duration_h_min']:.4f}–{summary['duration_h_max']:.2f} h；
- 每条曲线：{summary['observations_min']}–{summary['observations_max']} 个不规则观测点；
- 没有固定 0–200 h 截断；
- 没有运行无监督学习。

## 文件

- `curve_data_long.csv`：全部观测长表，纵轴字段为 `y_relative`；
- `curve_metadata.csv`：逐曲线元数据，字段与最初版本一致；
- `curves/<class>/<curve_id>.csv`：1000 个独立曲线文件；
- `generation_parameters.jsonl`：完整生成参数与重建基准；
- `initial_normalization_audit.csv`：逐曲线初值除数；
- `dataset_summary.json`：规模与自动质检；
- `dataset_preview.png`：四类 × 四档时长预览。

## 规则形态质检

固定规则通过数为 Bridge {qc['pass_counts']['bridge']}/90、Hill {qc['pass_counts']['hill']}/160、Slope {qc['pass_counts']['slope']}/500、Valley {qc['pass_counts']['valley']}/250。该标志只用于生成后核查，不能作为无监督学习输入。

## 可复现命令

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
MPLCONFIGDIR=/private/tmp/original_v1_mpl \\
python3 restore_original_1000_v1.py
```
"""
    (output / "DATASET_CARD_CN.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=Path,
        default=base / "eg_based_synthetic_dataset_1000_unnormalized_pce",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "eg_based_synthetic_dataset_1000_original_v1_initial_normalized",
    )
    args = parser.parse_args()
    rebuild(args.source, args.output)
    print((args.output / "dataset_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
