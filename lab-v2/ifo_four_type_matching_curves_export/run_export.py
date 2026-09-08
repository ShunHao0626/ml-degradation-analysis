#!/usr/bin/env python3
"""Export strict four-topology matches into paired CSV and PNG class folders."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE_DATASET = REPO / "lab-v2" / "som_references" / "accepted" / "samples_test"
STRICT_RESULTS = (
    REPO
    / "lab-v2"
    / "ifo_four_shape_discovery"
    / "06_final_four_classes"
    / "strict_four_class_matches.csv"
)
CLASSES = ["IFO-Bridge", "IFO-Hill", "IFO-Slope", "IFO-Valley"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    matches = pd.read_csv(STRICT_RESULTS)
    matches = matches[matches["strict_topology_match"]].copy()
    matches = matches[matches["strict_class"].isin(CLASSES)].copy()
    matches = matches.sort_values(["strict_class", "record", "series_name"]).reset_index(drop=True)

    for kind in ("csv", "png"):
        for class_name in CLASSES:
            (ROOT / kind / class_name).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    seen_destinations: set[Path] = set()
    for row in matches.itertuples(index=False):
        source_csv = REPO / row.source_file
        source_png = source_csv.with_suffix(".png")
        if not source_csv.is_relative_to(SOURCE_DATASET):
            raise RuntimeError(f"CSV outside requested dataset: {source_csv}")
        if not source_csv.exists() or not source_png.exists():
            raise FileNotFoundError(f"Missing paired source for {row.curve_id}")
        dest_csv = ROOT / "csv" / row.strict_class / source_csv.name
        dest_png = ROOT / "png" / row.strict_class / source_png.name
        if dest_csv in seen_destinations or dest_png in seen_destinations:
            raise RuntimeError(f"Duplicate destination filename for {row.curve_id}")
        seen_destinations.update((dest_csv, dest_png))
        shutil.copy2(source_csv, dest_csv)
        shutil.copy2(source_png, dest_png)
        manifest_rows.append(
            {
                "curve_id": row.curve_id,
                "strict_class": row.strict_class,
                "strict_topology_match": bool(row.strict_topology_match),
                "window_hours": int(row.window_hours),
                "confidence_level": row.confidence_level,
                "source_csv": source_csv.relative_to(REPO).as_posix(),
                "source_png": source_png.relative_to(REPO).as_posix(),
                "exported_csv": dest_csv.relative_to(ROOT).as_posix(),
                "exported_png": dest_png.relative_to(ROOT).as_posix(),
                "csv_sha256": sha256(dest_csv),
                "png_sha256": sha256(dest_png),
                "duration_hours": row.duration_hours,
                "gain_ratio": row.gain_ratio,
                "early_peak_drop_ratio": row.early_peak_drop_ratio,
                "early_drop_ratio": row.early_drop_ratio,
                "recovery_ratio": row.recovery_ratio,
                "post_recovery_drop_ratio": row.post_recovery_drop_ratio,
                "early_slope": row.early_slope,
                "late_slope": row.late_slope,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(ROOT / "export_manifest.csv", index=False, float_format="%.10g")
    manifest[manifest["confidence_level"].eq("high")].to_csv(
        ROOT / "stable_rule_matches.csv", index=False, float_format="%.10g"
    )
    manifest[manifest["confidence_level"].eq("low_review")].to_csv(
        ROOT / "low_stability_review_queue.csv", index=False, float_format="%.10g"
    )
    summary = (
        manifest.groupby("strict_class", sort=False)
        .agg(curve_count=("curve_id", "size"), csv_count=("exported_csv", "size"), png_count=("exported_png", "size"))
        .reindex(CLASSES, fill_value=0)
        .reset_index()
    )
    confidence_counts = (
        manifest.groupby(["strict_class", "confidence_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASSES, fill_value=0)
    )
    summary["high_count"] = confidence_counts.get("high", 0).to_numpy()
    summary["low_review_count"] = confidence_counts.get("low_review", 0).to_numpy()
    summary.to_csv(ROOT / "class_summary.csv", index=False)

    selection = {
        "source_dataset": str(SOURCE_DATASET),
        "source_strict_result": str(STRICT_RESULTS),
        "selection_scope": "strict_topology_match == True only",
        "analysis_window_hours": sorted(manifest["window_hours"].unique().tolist()),
        "class_counts": dict(zip(summary["strict_class"], summary["curve_count"].astype(int))),
        "confidence_counts": manifest["confidence_level"].value_counts().to_dict(),
        "total_curves": int(len(manifest)),
        "note": (
            "This is strict topology screening, not a pure-unsupervised discovery result. "
            "Topology direction can match even when the visual amplitude is weak."
        ),
    }
    (ROOT / "selection_basis.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# 四类严格匹配曲线导出

本目录从 `{SOURCE_DATASET}` 的218条曲线中，导出了符合四种目标拓扑严格条件的曲线。

## 数量

- IFO-Bridge：{selection['class_counts']['IFO-Bridge']} 条
- IFO-Hill：{selection['class_counts']['IFO-Hill']} 条
- IFO-Slope：{selection['class_counts']['IFO-Slope']} 条
- IFO-Valley：{selection['class_counts']['IFO-Valley']} 条
- 合计：{selection['total_curves']} 条
- 其中置信度 high 58 条，low_review 3 条；61 条均通过严格拓扑条件。

## 目录

- `csv/IFO-*/`：原始曲线 CSV 副本。
- `png/IFO-*/`：与 CSV 同名、同类别的原始 PNG 副本。
- `export_manifest.csv`：逐条来源、导出路径、形态证据和 SHA-256。
- `stable_rule_matches.csv`：规则扰动下仍稳定的 58 条，不代表形态振幅一定很强。
- `low_stability_review_queue.csv`：规则扰动稳定性较低、建议复核的 3 条。
- `class_summary.csv`：分类数量汇总。
- `selection_basis.json`：筛选来源和边界。

## 方法边界

仅导出 `strict_topology_match=True` 的曲线，统一来自750 h严格形态筛选。`low_review` 表示该曲线在规则扰动下稳定性较低，虽然通过严格条件，仍建议人工复核图片。这里的“匹配”首先指上升、下降、恢复等方向顺序符合规则；部分曲线变化振幅较弱，不能理解为与概念示意图等强度复现。它是依据目标拓扑定义进行的严格筛选，不应表述为纯无监督 SOM 自然发现。原始数据没有被移动或修改。
"""
    (ROOT / "README_CN.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
