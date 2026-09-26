#!/usr/bin/env python3
"""Narrow the manual queue to source figures lacking 0–200 h observations."""
from __future__ import annotations

import csv
from pathlib import Path

from early_data import HERE, prepare, rows, write_rows
from diagram_stage_classifier import classify, unique_observations


OUTPUT = HERE / "reviews/outputs/unresolved_only_20260925"
SOURCE = HERE.parents[1] / "original_curves_2250"
PAPER_CONFIRMED_PCE = {"F01223", "F01224"}


def initial_support(row):
    path = SOURCE / row["source_csv"]
    times = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        field = "x" if row["folder"] == "data_all" else "time_h"
        factor = float(row["time_factor"])
        for point in reader:
            try:
                h = float(point[field]) * factor
                if 0 <= h <= 200:
                    times.add(h)
            except (TypeError, ValueError):
                continue
    return len(times), min(times) if times else ""


def main():
    unresolved = rows(HERE / "delivery/unresolved.csv")
    frozen = {r["file_id"]: r for r in rows(HERE / "manifests/frozen_inputs.csv")}
    if len(unresolved) != 41:
        raise ValueError("unresolved source changed")
    audit, manual = [], []
    for r in unresolved:
        fid = r["curve_id"]
        source = frozen[fid]
        base = dict(curve_id=fid, source_group=r["source_group"], source_csv=r["source_csv"],
                    source_image=r["source_image"], original_issue=r["review_reason"],
                    target_series=Path(r["source_csv"]).stem.split("__")[-2]
                    if "__" in Path(r["source_csv"]).stem else "",
                    strict_unique_times="", first_strict_time_h="", disposition="",
                    automatic_class="", evidence="")
        if r["review_reason"] == "fewer_than_two_distinct_observations_0_200h":
            count, first = initial_support(source)
            base.update(strict_unique_times=count, first_strict_time_h=first,
                        disposition="manual_source_image_check",
                        evidence="0–200 h 原始 CSV 只有 0 或 1 个不同时间；须看原图是否另有可读的初期轨迹。")
            manual.append(base)
        elif fid in PAPER_CONFIRMED_PCE:
            revised = dict(source, y_kind="pce")
            item, error = prepare(revised, SOURCE,
                                  dict(window_h=200, boundary_limit_h=250, boundary_max_points=3))
            if error:
                raise ValueError(fid + ": " + error)
            t, y = unique_observations(item["points"], item["divisor"])
            result = classify(t, y)
            base.update(strict_unique_times=item["unique_window_times"],
                        first_strict_time_h=item["window_start_h"],
                        disposition="auto_resolved_from_paper_and_csv",
                        automatic_class=result["candidate_class"],
                        evidence="本地论文正文明确称 Fig. 6c 为 normalized PCE；用指定原始 CSV 计算初期阶段。")
        else:
            base.update(disposition="not_a_target_elapsed_hour_pce_curve",
                        evidence="原图/轴核验为其他物理量、PCE 差值或等效时间；不属于当前实际小时 PCE 四类输入。")
        audit.append(base)
    if len(manual) != 15 or sum(r["disposition"] == "auto_resolved_from_paper_and_csv" for r in audit) != 2:
        raise ValueError("manual triage counts changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = list(audit[0])
    write_rows(OUTPUT / "triage_audit.csv", audit, fields)
    write_rows(OUTPUT / "manual_queue.csv", manual, fields)
    print(dict(unresolved=len(audit), manual=len(manual), auto_resolved=2, out_of_scope=24))


if __name__ == "__main__":
    main()
