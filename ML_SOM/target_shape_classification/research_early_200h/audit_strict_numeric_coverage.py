#!/usr/bin/env python3
"""Separate 0–200 h numeric candidates from image-only impressions.

The integrated archive remains immutable. This view makes the exact-hour
evidence limit explicit without deleting the user's source-image opinions.
"""
from __future__ import annotations

import json
from collections import Counter

from early_data import HERE, digest, read_observations, rows, write_rows


OUT = HERE / "evaluation" / "strict_numeric_coverage_20260925"
SOURCE = HERE.parents[1] / "original_curves_2250"
AXIS_NOTES = {
    "F00104": "source figure: Time (days), 0–200 days; 200 h = 8.33 days",
    "F00383": "source figure: Time (Days), 0–300 days; 200 h = 8.33 days",
    "F00384": "source figure: Time (Days), 0–300 days; 200 h = 8.33 days",
    "F00412": "source figure: Aging period [day], 0–200 days; 200 h = 8.33 days",
    "F00619": "source figure: Time (days), 0–300 days; 200 h = 8.33 days",
    "F00620": "source figure: Time (days), 0–300 days; 200 h = 8.33 days",
    "F00621": "source figure: Time (days), 0–300 days; 200 h = 8.33 days",
    "F00901": "source figure: Time (h), 0–4000 h; early markers are compressed",
    "F00903": "source figure: Time (hours), displayed series starts beyond 200 h",
}


def main():
    canonical = rows(HERE / "final_integrated_20260925" / "canonical_index.csv")
    original_class_index = rows(HERE / "final_integrated_20260925" / "class_index.csv")
    frozen = {r["file_id"]: r for r in rows(HERE / "manifests" / "frozen_inputs.csv")}
    if len(canonical) != 2246 or len(original_class_index) != 2216:
        raise ValueError("integrated archive count changed")
    image_only = [r for r in canonical if r["classification_origin"] == "single_user_source_image"
                  and r["integrated_class"]]
    if len(image_only) != 9 or set(AXIS_NOTES) != {r["curve_id"] for r in image_only}:
        raise ValueError("image-only cohort changed; inspect sources")
    held = []
    for row in image_only:
        fid = row["curve_id"]
        source = frozen[fid]
        if digest(SOURCE / source["source_csv"]) != source["source_sha256"]:
            raise ValueError("source hash changed: " + fid)
        points = read_observations(source, SOURCE)
        strict = sorted({hour for hour, _, _ in points if 0 <= hour <= 200})
        if len(strict) != int(row["numeric_strict_points"]) or len(strict) > 1:
            raise ValueError("strict observation support changed: " + fid)
        nonnegative = sorted({hour for hour, _, _ in points if hour >= 0})
        held.append(dict(curve_id=fid, original_integrated_class=row["integrated_class"],
                         human_image_label=row["human_image_label"],
                         strict_200h_class="", strict_200h_numeric_times=len(strict),
                         strict_200h_times=";".join(f"{hour:.6g}" for hour in strict),
                         first_nonnegative_source_time_h=(nonnegative[0] if nonnegative else ""),
                         first_source_time_after_200h=next((h for h in nonnegative if h > 200), ""),
                         figure_axis_note=AXIS_NOTES[fid], source_csv=row["source_csv"],
                         source_sha256=row["source_sha256"], source_image=row["source_image"],
                         interpretation="human source-image impression retained; strict numeric shape unresolved"))
    held_ids = {r["curve_id"] for r in held}
    strict_index = [dict(r, strict_200h_candidate_class=r["integrated_class"],
                         strict_200h_coverage_status="numeric_rule_candidate_unvalidated")
                    for r in original_class_index if r["curve_id"] not in held_ids]
    if len(strict_index) != 2207:
        raise ValueError("numeric candidate count changed")
    coverage = []
    for row in canonical:
        fid = row["curve_id"]
        if fid in held_ids:
            status = "image_opinion_only_strict_unresolved"
            strict_class = ""
        elif row["integrated_class"]:
            status = "numeric_rule_candidate_unvalidated"
            strict_class = row["integrated_class"]
        elif row["classification_origin"] == "single_user_source_image":
            status = "human_nonconforming_image_only"
            strict_class = ""
        else:
            status = "out_of_scope"
            strict_class = ""
        coverage.append(dict(curve_id=fid, strict_200h_candidate_class=strict_class,
                             strict_200h_coverage_status=status,
                             original_integrated_class=row["integrated_class"],
                             classification_origin=row["classification_origin"],
                             evidence_status=row["evidence_status"],
                             source_group=row["source_group"],
                             source_csv=row["source_csv"], source_image=row["source_image"]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "strict_200h_class_index.csv", strict_index, list(strict_index[0]))
    write_rows(OUT / "held_image_opinions.csv", held, list(held[0]))
    write_rows(OUT / "strict_200h_coverage.csv", coverage, list(coverage[0]))
    report = dict(integrated_class_count=len(original_class_index),
                  strict_numeric_candidate_count=len(strict_index),
                  held_image_opinion_count=len(held),
                  strict_numeric_candidate_classes=dict(sorted(Counter(
                      r["integrated_class"] for r in strict_index).items())),
                  coverage_statuses=dict(sorted(Counter(
                      r["strict_200h_coverage_status"] for r in coverage).items())),
                  note="Image-only opinions are preserved separately; no independent accuracy claim. "
                       "Some figures do not show any 0–200 h trajectory to digitize.")
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
