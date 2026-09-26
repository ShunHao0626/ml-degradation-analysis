#!/usr/bin/env python3
"""Show accepted re-digitizations separately from the original-only analysis."""
from __future__ import annotations

import json
from pathlib import Path

from early_data import HERE, read_observations, rows, write_rows


def main():
    config = json.loads((HERE / "config.json").read_text())
    frozen = rows(HERE / "manifests/frozen_inputs.csv")
    original_root = Path(config["source_root"])
    revised_root = Path(config["legacy_results_root"])
    records = []
    for row in frozen:
        if row["historical_analysis_version"] == row["source_csv"]:
            continue
        raw = read_observations(row, original_root)
        revised_row = dict(row, source_csv=row["historical_analysis_version"])
        revised = read_observations(revised_row, revised_root)
        raw_window = [p for p in raw if 0 <= p[0] <= 200]
        revised_window = [p for p in revised if 0 <= p[0] <= 200]
        records.append(dict(file_id=row["file_id"], canonical_id=row["canonical_id"],
                            source_group=row["source_group"], source_csv=row["source_csv"],
                            source_sha256=row["source_sha256"],
                            revised_csv=row["historical_analysis_version"],
                            revised_sha256=row["historical_analysis_sha256"],
                            raw_window_points=len(raw_window), revised_window_points=len(revised_window),
                            raw_window_max=max((p[1] for p in raw_window), default=""),
                            revised_window_max=max((p[1] for p in revised_window), default=""),
                            note="Primary model uses original only; revised version needs separate source-image audit."))
    assert len(records) == 12
    write_rows(HERE / "manifests/accepted_redigitization_comparison.csv", records, list(records[0]))
    print(json.dumps({"revised_versions": len(records),
                      "raw_vs_revised_window_count_changed": sum(r["raw_window_points"] != r["revised_window_points"] for r in records)}))


if __name__ == "__main__":
    main()
