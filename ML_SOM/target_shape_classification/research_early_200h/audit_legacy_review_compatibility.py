#!/usr/bin/env python3
"""Trace historical image opinions against the strict 0–200 h rule protocol.

This is a protocol-compatibility audit, not a correctness adjudication.
"""
from __future__ import annotations

import json
from collections import Counter

from early_data import HERE, digest, rows, write_rows


OUT = HERE / "evaluation" / "legacy_review_compatibility_20260925"
LEGACY = HERE.parent / "review_inputs" / "source_reviews_200h.csv"


def main():
    historical = rows(LEGACY)
    stage = {r["curve_id"]: r for r in rows(HERE / "diagram_stage_20260925" / "stage_features.csv")}
    frozen = {r["file_id"]: r for r in rows(HERE / "manifests" / "frozen_inputs.csv")}
    if len(historical) != 46 or len(stage) != 2205:
        raise ValueError("historical review or current rule cohort changed")
    records = []
    for row in historical:
        fid = row["file_id"]
        source, current = frozen[fid], stage[fid]
        if row["source_image_sha256"] != source["image_sha256"]:
            raise ValueError("historical figure hash changed: " + fid)
        verified = row["review_status"] == "verified"
        old_class = row["reviewed_class"] if verified else ""
        new_class = current["candidate_class"]
        aligned = bool(verified and old_class == new_class)
        if not verified:
            detail = "historical_ambiguous"
        elif aligned:
            detail = "same_name_different_protocol"
        elif old_class == "valley" and new_class == "slope":
            detail = ("valley_vs_slope_with_post_200_recovery_hint"
                      if current["boundary_valley_hint"] == "True" else
                      "valley_vs_slope_without_window_recovery")
        elif old_class == "bridge" and new_class == "hill":
            detail = "bridge_vs_hill_peak_width_or_decay_threshold"
        else:
            detail = "other_legacy_vs_strict_rule_difference"
        records.append(dict(curve_id=fid, source_group=source["source_group"],
                            split=source["split"], historical_review_status=row["review_status"],
                            historical_class=old_class, historical_note=row["notes"],
                            current_rule_class=new_class,
                            current_rule_status=current["evidence_status"],
                            same_name=int(aligned) if verified else "",
                            discrepancy_type=detail,
                            primary_turn_h=current["primary_turn_h"],
                            secondary_turn_h=current["secondary_turn_h"],
                            strict_end_h=current["strict_end_h"],
                            strict_points=current["strict_points"],
                            boundary_valley_hint=current["boundary_valley_hint"],
                            source_csv=current["source_csv"],
                            source_image=current["source_image"],
                            source_image_sha256=row["source_image_sha256"],
                            interpretation="historical opinion; not a locked strict-200h reference"))
    verified = [r for r in records if r["historical_review_status"] == "verified"]
    differences = [r for r in verified if not r["same_name"]]
    if len(verified) != 36 or len(differences) != 16:
        raise ValueError("historical compatibility count changed")
    report = dict(protocol="legacy_image_opinion_vs_new_strict_rule_v1",
                  historical_file_sha256=digest(LEGACY),
                  stage_features_sha256=digest(HERE / "diagram_stage_20260925" /
                                               "stage_features.csv"),
                  historical_total=len(records), verified=len(verified),
                  ambiguous=len(records) - len(verified),
                  same_name=sum(r["same_name"] for r in verified), different_name=len(differences),
                  difference_types=dict(sorted(Counter(r["discrepancy_type"] for r in differences).items())),
                  old_valley_new_slope=sum(r["historical_class"] == "valley" and
                                           r["current_rule_class"] == "slope" for r in differences),
                  old_valley_new_slope_with_post_200_hint=sum(
                      r["discrepancy_type"] == "valley_vs_slope_with_post_200_recovery_hint"
                      for r in differences),
                  note="Agreement/disagreement between an older image-review protocol and a new automatic "
                       "strict-window rule. Neither side is an independent reference under the other protocol.")
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "all_historical_reviews.csv", records, list(records[0]))
    write_rows(OUT / "verified_differences.csv", differences, list(differences[0]))
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
