#!/usr/bin/env python3
"""Fail-fast consistency checks for the complete project handoff."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "logs/verification_report.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks = []
    config = json.loads((ROOT / "config/experiment_manifest.json").read_text(encoding="utf-8"))
    pairs = {(item["sigma"], item["learning_rate"]) for item in config["som_hyperparameter_pairs"]}
    check(pairs == {(0.5, 0.1), (0.3, 0.1), (0.5, 0.3)}, "parameter whitelist changed")
    check(config["model_input"] == "normalized_smoothed_pce_trajectory_only", "model input changed")
    checks.append("parameter whitelist and single-variable input")

    expected_counts = {150: 103, 300: 94, 500: 86}
    for window, expected in expected_counts.items():
        base = ROOT / f"03_baseline_reproduction/preprocessed/{window}h"
        summary = json.loads((base / "preprocessing_summary.json").read_text(encoding="utf-8"))
        check(summary["included_curves"] == expected, f"unexpected {window} h curve count")
        bundle = np.load(base / "preprocessed_curves.npz")
        check(bundle["data"].shape[0] == expected, f"bundle count mismatch at {window} h")
        check(np.isfinite(bundle["data"]).all(), f"non-finite data at {window} h")
    checks.append("preprocessing counts and finite model matrices")

    decisions = read_csv(ROOT / "05_cluster_selection/cluster_count_decisions.csv")
    check({int(row["selected_k"]) for row in decisions} == {5}, "selected k is not documented k=5")
    for row in decisions:
        window = int(row["window_hours"])
        summary = json.loads(
            (ROOT / f"05_cluster_selection/baseline/{window}h/k05/summary.json").read_text(
                encoding="utf-8"
            )
        )
        check(summary["occupied_clusters"] == 5, f"unoccupied selected node at {window} h")
        check(sum(summary["cluster_sizes"]) == expected_counts[window], f"cluster size mismatch at {window} h")
    checks.append("selected cluster models and membership totals")

    representatives = read_csv(
        ROOT
        / "06_curve_type_results/provisional_individual_representatives/representative_provenance.csv"
    )
    check(len(representatives) == 4, "expected four provisional representatives")
    check(len({row["morphology"] for row in representatives}) == 4, "representative labels duplicate")
    check(all(Path(row["csv_path"]).exists() for row in representatives), "representative source missing")
    checks.append("four post-clustering representatives and source paths")

    retrieval = json.loads(
        (ROOT / "08_literature_review/metadata/retrieval_summary.json").read_text(encoding="utf-8")
    )
    check(retrieval["raw_records"] == 1150, "bulk raw literature count changed")
    check(retrieval["deduplicated_records"] == 1068, "bulk deduplicated count changed")
    check(len(read_csv(ROOT / "08_literature_review/screening/priority_shortlist.csv")) == 20, "shortlist count")
    checks.append("bulk literature corpus and priority shortlist")

    archive = read_csv(ROOT / "08_literature_review/metadata/oa_archive_manifest.csv")
    validated = [row for row in archive if row["status"] == "validated_pdf"]
    check(len(validated) == 11, "validated OA PDF count changed")
    for row in validated:
        path = Path(row["path"])
        check(path.exists() and path.read_bytes()[:4] == b"%PDF", f"invalid PDF: {path}")
        check(digest(path) == row["sha256"], f"PDF checksum mismatch: {path}")
    checks.append("11 validated open-access PDFs and SHA-256 checksums")

    required = [
        ROOT / "reports/final_report.md",
        ROOT / "05_cluster_selection/baseline_candidate_contact_sheet.png",
        ROOT / "04_literature_parameter_sweeps/published_parameter_contact_sheet.png",
        ROOT / "06_curve_type_results/provisional_individual_representatives/four_ifo_like_individuals.png",
        ROOT / "08_literature_review/evidence_pages/literature_evidence_contact_sheet.png",
    ]
    check(all(path.exists() and path.stat().st_size > 0 for path in required), "required handoff missing")
    checks.append("reports and key visual artifacts")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"status": "PASS", "checks": checks, "check_count": len(checks)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
