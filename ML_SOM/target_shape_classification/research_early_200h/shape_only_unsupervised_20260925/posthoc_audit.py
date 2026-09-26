"""Compare frozen anonymous assignments with old unvalidated candidates.

This is never imported by run.py or make_viewer.py. Do not use these numbers
to select or refit the clustering parameters on the same collection.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
STRICT = HERE.parent / "evaluation/strict_numeric_coverage_20260925/strict_200h_class_index.csv"
SHAPES = ("bridge", "hill", "slope", "valley")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Refuse a post-hoc audit before the anonymous result is frozen on disk.
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["shape_names_used_in_fit"] is False
    assignment = read(OUT / "anonymous_assignments.csv")
    old = {r["curve_id"]: r for r in read(STRICT)}
    assert len(assignment) == 2152
    matching = [r for r in assignment if r["curve_id"] in old]
    assert len(matching) == 2152
    report = {}
    for k in (16, 32, 64):
        counts: dict[int, Counter] = defaultdict(Counter)
        for row in matching:
            counts[int(row[f"anonymous_k{k}"])][old[row["curve_id"]]["strict_200h_candidate_class"]] += 1
        records = []
        for cluster in sorted(counts):
            count = counts[cluster]
            total = sum(count.values())
            record = dict(cluster=cluster, total=total,
                          **{shape: count[shape] for shape in SHAPES},
                          dominant_old_candidate=count.most_common(1)[0][0],
                          dominant_fraction=f"{count.most_common(1)[0][1] / total:.6f}")
            records.append(record)
        write(OUT / f"posthoc_old_candidate_contingency_k{k}.csv", records,
              ["cluster", "total", *SHAPES, "dominant_old_candidate", "dominant_fraction"])
        report[str(k)] = dict(ari_vs_unvalidated_old_candidates=adjusted_rand_score(
            [r[f"anonymous_k{k}"] for r in matching],
            [old[r["curve_id"]]["strict_200h_candidate_class"] for r in matching]),
            old_candidate_counts=dict(Counter(old[r["curve_id"]]["strict_200h_candidate_class"]
                                      for r in matching)))
    (OUT / "posthoc_old_candidate_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
