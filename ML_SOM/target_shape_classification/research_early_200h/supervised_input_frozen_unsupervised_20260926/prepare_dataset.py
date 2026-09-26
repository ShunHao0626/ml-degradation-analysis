"""Create a target-guided input subset; keep all labels out of model inputs.

This is the deliberately supervised boundary of the experiment. It uses the
previous anonymous fit and old *unvalidated* automatic class candidates only
to decide which curves enter the next run. It never edits PCE measurements.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREVIOUS = PARENT / "shape_only_input_curation_20260925"
UPSTREAM = PARENT / "shape_only_unsupervised_20260925/results"
CANDIDATES = PARENT / "evaluation/strict_numeric_coverage_20260925/strict_200h_class_index.csv"
PREVIOUS_ASSIGNMENTS = PREVIOUS / "results/anonymous_assignments.csv"
PREVIOUS_EXCLUSIONS = PREVIOUS / "results/excluded_input_curves.csv"
CLUSTER_CANDIDATE = {0: "slope", 1: "hill", 2: "valley", 3: "bridge"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original = read_rows(UPSTREAM / "input_index.csv")
    prior = {r["curve_id"]: r for r in read_rows(PREVIOUS_ASSIGNMENTS)}
    prior_excluded = {r["curve_id"]: r for r in read_rows(PREVIOUS_EXCLUSIONS)}
    old = {r["curve_id"]: r for r in read_rows(CANDIDATES)}
    assert len(original) == 2152 and len(prior) == 1897 and len(prior_excluded) == 255
    assert not set(prior) & set(prior_excluded)
    assert set(r["curve_id"] for r in original) == set(prior) | set(prior_excluded)
    all_decisions, removed = [], []
    retained_ids = []
    for row in original:
        cid = row["curve_id"]
        automatic = old[cid]
        if cid in prior_excluded:
            previous_cluster = ""
            decision = "excluded_prior_quality"
            reason = prior_excluded[cid]["reason"]
        else:
            previous_cluster = int(prior[cid]["anonymous_cluster"])
            candidate = automatic["strict_200h_candidate_class"]
            # Preserve the large decline cluster. Curate only the three small
            # groups, where prior anonymous shape and old candidate disagree.
            small_range = float(prior[cid]["observed_range"]) <= .01
            if (previous_cluster != 0 and not small_range
                    and candidate != CLUSTER_CANDIDATE[previous_cluster]):
                decision = "excluded_supervised_input"
                reason = "small_cluster_old_candidate_disagreement"
                removed.append(dict(curve_id=cid, previous_anonymous_cluster=previous_cluster,
                                    old_automatic_candidate=candidate,
                                    old_evidence_status=automatic["evidence_status"],
                                    distinct_observations=row["distinct_observations"],
                                    reason=reason))
            else:
                decision = "included"
                reason = ("preserved_observed_range_at_most_0.01"
                          if previous_cluster != 0 and small_range
                          and candidate != CLUSTER_CANDIDATE[previous_cluster] else "")
                retained_ids.append(cid)
        all_decisions.append(dict(curve_id=cid, decision=decision, reason=reason,
                                  previous_anonymous_cluster=previous_cluster,
                                  old_automatic_candidate=automatic["strict_200h_candidate_class"],
                                  old_evidence_status=automatic["evidence_status"],
                                  distinct_observations=row["distinct_observations"]))
    assert len(retained_ids) == 1842 and len(removed) == 55
    assert sum(r["decision"] == "excluded_prior_quality" for r in all_decisions) == 255
    keep = set(retained_ids)
    selected_rows = [r for r in original if r["curve_id"] in keep]
    assert [r["curve_id"] for r in selected_rows] == retained_ids
    write_rows(HERE / "input_index.csv", selected_rows, list(original[0]))
    write_rows(HERE / "supervised_input_exclusions.csv", removed, list(removed[0]))
    write_rows(HERE / "full_input_decisions.csv", all_decisions, list(all_decisions[0]))
    with np.load(UPSTREAM / "shape_inputs.npz") as arrays:
        source_ids = arrays["curve_id"].tolist()
        assert source_ids == [r["curve_id"] for r in original]
        positions = np.array([i for i, cid in enumerate(source_ids) if cid in keep])
        output = {key: arrays[key][positions] for key in arrays.files}
    assert output["curve_id"].tolist() == retained_ids
    np.savez_compressed(HERE / "shape_inputs.npz", **output)
    summary = dict(protocol="old-candidate-guided-input-only-v1",
                   source_curves=len(original), prior_quality_excluded=len(prior_excluded),
                   supervised_input_excluded=len(removed), model_input=len(retained_ids),
                   small_range_disagreement_preserved=sum(
                       r["reason"] == "preserved_observed_range_at_most_0.01"
                       for r in all_decisions),
                   prior_small_cluster_majority=CLUSTER_CANDIDATE,
                   removed_by_prior_cluster=dict(Counter(r["previous_anonymous_cluster"] for r in removed)),
                   old_candidates_are_independent_truth=False,
                   previous_assignments_sha256=sha(PREVIOUS_ASSIGNMENTS),
                   old_candidate_index_sha256=sha(CANDIDATES),
                   source_index_sha256=sha(UPSTREAM / "input_index.csv"))
    (HERE / "input_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
