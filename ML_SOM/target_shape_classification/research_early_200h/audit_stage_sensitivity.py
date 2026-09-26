#!/usr/bin/env python3
"""Audit rule-candidate changes under observation-window and sampling changes.

This is a paired robustness audit of the 2205 numeric source curves, not an
accuracy evaluation. The frozen 200 h normalization is used in every scenario
so that changing the observation cutoff does not also change the value scale.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter

import numpy as np

import diagram_stage_classifier as stage
from early_data import HERE, digest, read_observations, rows, write_rows


OUT = HERE / "evaluation" / "stage_sensitivity_20260925"
SOURCE = HERE.parents[1] / "original_curves_2250"
SCENARIOS = ("window_150h", "window_250h", "thin_even", "thin_odd",
             "remove_primary_turn_neighborhood")
NOISE_AMPLITUDES = (0.005, 0.01, 0.02)
NOISE_REPLICATES = 10


def scenario_arrays(name, t, y, primary_turn):
    if name.startswith("window_"):
        return t, y, True, 150.0 if name == "window_150h" else 250.0
    strict = np.flatnonzero(t <= 200.0)
    keep = np.ones(len(t), dtype=bool)
    if name in ("thin_even", "thin_odd"):
        parity = 0 if name == "thin_even" else 1
        for i in range(1, len(strict) - 1):
            if i % 2 != parity:
                keep[strict[i]] = False
    else:
        if primary_turn == "":
            return t, y, False, 200.0
        turns = np.flatnonzero(np.isclose(t[strict], float(primary_turn), rtol=0, atol=1e-7))
        if len(turns) != 1 or turns[0] == 0 or turns[0] == len(strict) - 1:
            return t, y, False, 200.0
        k = int(turns[0])
        for i in (k - 1, k, k + 1):
            if 0 < i < len(strict) - 1:
                keep[strict[i]] = False
    return t[keep], y[keep], bool(np.any(~keep)), 200.0


def summarize(records, noise_records, baseline):
    summaries = []
    for name in SCENARIOS:
        group = [r for r in records if r["scenario"] == name]
        eligible = [r for r in group if r["eligible"]]
        changed = [r for r in eligible if r["class_changed"]]
        status = [r for r in eligible if r["status_changed"]]
        worsened = [r for r in eligible if r["baseline_status"] == "stage_candidate"
                    and r["scenario_status"] != "stage_candidate"]
        summary = dict(scenario=name, cohort=len(group), eligible=len(eligible),
                       class_changed=len(changed), class_changed_fraction=(len(changed) / len(eligible)
                                                                        if eligible else None),
                       status_changed=len(status), stage_candidate_to_lower_support=len(worsened),
                       class_transitions=dict(sorted(Counter(
                           r["baseline_class"] + "->" + r["scenario_class"] for r in changed).items())),
                       by_baseline_class={cls: dict(eligible=sum(r["baseline_class"] == cls for r in eligible),
                                                    class_changed=sum(r["baseline_class"] == cls for r in changed),
                                                    status_changed=sum(r["baseline_class"] == cls for r in status))
                                          for cls in ("bridge", "hill", "slope", "valley")})
        summaries.append(summary)
    noise_summaries = []
    for amplitude in NOISE_AMPLITUDES:
        group = [r for r in noise_records if r["amplitude"] == amplitude]
        changed_ids = {r["curve_id"] for r in group if r["class_changed"]}
        status_ids = {r["curve_id"] for r in group if r["status_changed"]}
        noise_summaries.append(dict(amplitude=amplitude, replicates=NOISE_REPLICATES,
                                    perturbed_evaluations=len(group),
                                    class_changed_evaluations=sum(r["class_changed"] for r in group),
                                    status_changed_evaluations=sum(r["status_changed"] for r in group),
                                    curves_with_any_class_change=len(changed_ids),
                                    curves_with_any_status_change=len(status_ids),
                                    by_baseline_class={cls: dict(
                                        curves=sum(r["candidate_class"] == cls for r in baseline),
                                        curves_with_any_class_change=sum(
                                            r["candidate_class"] == cls and r["curve_id"] in changed_ids
                                            for r in baseline))
                                        for cls in ("bridge", "hill", "slope", "valley")}))
    return dict(protocol="fixed_200h_scale_paired_stage_rule_sensitivity_v1",
                baseline_curve_count=len(baseline), scenario_summaries=summaries,
                noise_summaries=noise_summaries,
                input_hashes=dict(stage_features_csv=digest(HERE / "diagram_stage_20260925" /
                                                           "stage_features.csv"),
                                  integrated_canonical_index_csv=digest(HERE / "final_integrated_20260925" /
                                                                         "canonical_index.csv")),
                interpretation="Candidate and evidence-status changes under hypothetical observation edits; "
                               "not accuracy, measurement uncertainty, or a reason to retune the 200 h rule.",
                limitations=["Only the 2205 numeric source curves in the existing stage baseline are evaluated; "
                             "9 image-only human classes and 2 paper-axis recovered curves are excluded.",
                             "150/250 h uses the frozen 200 h normalization divisor and the same 0.01 event trigger.",
                             "Thinning preserves first/last strict observations and all post-200 h observations; "
                             "removed points exist only in experiment arrays.",
                             "Primary-turn removal is a targeted worst-case diagnostic, conditional on an interior "
                             "baseline turn; it is not random missingness.",
                             "Noise scenarios add independent uniform ±0.005/0.01/0.02 normalized-PCE jitter "
                             "at every observed time for 10 deterministic replicates. This is a hypothetical "
                             "stress test, not an estimate of digitization or measurement error."])


def per_curve_flags(records, noise_records, baseline):
    scenarios = {(r["curve_id"], r["scenario"]): r for r in records}
    noise = Counter((r["curve_id"], r["amplitude"]) for r in noise_records if r["class_changed"])
    output = []
    for row in baseline:
        fid = row["curve_id"]
        record = dict(curve_id=fid, source_group=row["source_group"],
                      baseline_class=row["candidate_class"], baseline_status=row["evidence_status"])
        for name in SCENARIOS:
            variant = scenarios[(fid, name)]
            record[name + "_eligible"] = variant["eligible"]
            record[name + "_class_changed"] = variant["class_changed"]
            record[name + "_status_changed"] = variant["status_changed"]
        for amplitude in NOISE_AMPLITUDES:
            key = f"noise_{amplitude:g}_class_changes_of_{NOISE_REPLICATES}"
            record[key] = noise[(fid, amplitude)]
        output.append(record)
    return output


def main():
    baseline = rows(HERE / "diagram_stage_20260925" / "stage_features.csv")
    frozen = {r["file_id"]: r for r in rows(HERE / "manifests" / "frozen_inputs.csv")}
    index = {r["curve_id"]: r for r in rows(HERE / "final_integrated_20260925" / "canonical_index.csv")}
    if len(baseline) != 2205 or len({r["curve_id"] for r in baseline}) != len(baseline):
        raise ValueError("the frozen 2205-curve stage baseline changed")
    details = []
    noise_records = []
    for idx, row in enumerate(baseline, 1):
        fid = row["curve_id"]
        source = frozen[fid]
        source_path = SOURCE / source["source_csv"]
        if digest(source_path) != source["source_sha256"]:
            raise ValueError("source hash changed: " + fid)
        points = read_observations(source, SOURCE)
        t, y = stage.unique_observations(points, float(index[fid]["normalization_divisor"]))
        stage.WINDOW = 200.0
        check = stage.classify(t, y)
        if (check["candidate_class"], check["evidence_status"]) != (
                row["candidate_class"], row["evidence_status"]):
            raise ValueError("200 h baseline did not reproduce: " + fid)
        for name in SCENARIOS:
            ts, ys, eligible, window = scenario_arrays(name, t, y, row["primary_turn_h"])
            stage.WINDOW = window
            variant = stage.classify(ts, ys) if eligible else check
            stage.WINDOW = 200.0
            details.append(dict(curve_id=fid, source_group=row["source_group"], split=row["split"],
                                scenario=name, window_h=window, eligible=int(eligible),
                                baseline_class=row["candidate_class"],
                                scenario_class=variant["candidate_class"],
                                class_changed=int(eligible and variant["candidate_class"] != row["candidate_class"]),
                                baseline_status=row["evidence_status"],
                                scenario_status=variant["evidence_status"],
                                status_changed=int(eligible and variant["evidence_status"] != row["evidence_status"]),
                                baseline_strict_points=row["strict_points"],
                                scenario_strict_points=variant.get("strict_points", ""),
                                baseline_primary_turn_h=row["primary_turn_h"],
                                scenario_primary_turn_h=variant.get("primary_turn_h", ""),
                                baseline_secondary_turn_h=row["secondary_turn_h"],
                                scenario_secondary_turn_h=variant.get("secondary_turn_h", "")))
        for amplitude in NOISE_AMPLITUDES:
            for replicate in range(NOISE_REPLICATES):
                seed_bytes = hashlib.sha256(f"{fid}|{amplitude}|{replicate}".encode()).digest()[:8]
                rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
                perturbed = y + rng.uniform(-amplitude, amplitude, size=len(y))
                variant = stage.classify(t, perturbed)
                noise_records.append(dict(curve_id=fid, source_group=row["source_group"],
                                          split=row["split"], amplitude=amplitude,
                                          replicate=replicate, baseline_class=row["candidate_class"],
                                          scenario_class=variant["candidate_class"],
                                          class_changed=int(variant["candidate_class"] != row["candidate_class"]),
                                          baseline_status=row["evidence_status"],
                                          scenario_status=variant["evidence_status"],
                                          status_changed=int(variant["evidence_status"] != row["evidence_status"])))
        if idx % 500 == 0:
            print(f"audited {idx}/{len(baseline)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "per_curve.csv", details, list(details[0]))
    write_rows(OUT / "noise_replicates.csv", noise_records, list(noise_records[0]))
    flags = per_curve_flags(details, noise_records, baseline)
    write_rows(OUT / "per_curve_flags.csv", flags, list(flags[0]))
    report = summarize(details, noise_records, baseline)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
