"""Construction audit against old automatic candidates, not validation.

The old candidates were used upstream in prepare_dataset.py. They remain
absent from run.py and its numeric model inputs, but agreement here is partly
built into the selected dataset and cannot be treated as independent evidence.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

from run import HERE, OUT, read_rows, write_rows


def main() -> None:
    path = HERE.parent / "evaluation/strict_numeric_coverage_20260925/strict_200h_class_index.csv"
    previous = {r["curve_id"]: r for r in read_rows(path)}
    assigned = read_rows(OUT / "anonymous_assignments.csv")
    labels, candidates = [], []
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in assigned:
        prior = previous.get(row["curve_id"])
        if not prior or not prior["strict_200h_candidate_class"]:
            continue
        cluster = row["anonymous_cluster"]
        candidate = prior["strict_200h_candidate_class"]
        labels.append(cluster)
        candidates.append(candidate)
        table[(cluster, candidate)][prior["evidence_status"] or "unspecified"] += 1
    output = []
    for (cluster, candidate), evidence in sorted(table.items()):
        output.append(dict(anonymous_cluster=cluster, old_automatic_candidate=candidate,
                           total=sum(evidence.values()),
                           stage_candidate=evidence["stage_candidate"],
                           insufficient_evidence=evidence["insufficient_evidence"],
                           pattern_conflict=evidence["pattern_conflict"],
                           other_evidence=sum(n for key, n in evidence.items()
                                              if key not in {"stage_candidate", "insufficient_evidence",
                                                             "pattern_conflict"})))
    write_rows(OUT / "posthoc_old_candidate_contingency.csv", output, list(output[0]))
    report = dict(assigned_curves=len(assigned), compared_to_old_automatic_candidates=len(labels),
                  adjusted_rand_index=adjusted_rand_score(candidates, labels),
                  comparison_is_independent_truth=False,
                  current_input_selection_used_old_candidates=True,
                  inherited_feature_model_was_target_informed=True,
                  candidate_counts=dict(Counter(candidates)))
    (OUT / "posthoc_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
