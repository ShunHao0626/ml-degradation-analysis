#!/usr/bin/env python3
"""Verify paired export counts, paths, and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CLASSES = ["IFO-Bridge", "IFO-Hill", "IFO-Slope", "IFO-Valley"]
EXPECTED = {"IFO-Bridge": 3, "IFO-Hill": 3, "IFO-Slope": 51, "IFO-Valley": 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_checks() -> dict:
    manifest = pd.read_csv(ROOT / "export_manifest.csv")
    stable = pd.read_csv(ROOT / "stable_rule_matches.csv")
    review = pd.read_csv(ROOT / "low_stability_review_queue.csv")
    summary = pd.read_csv(ROOT / "class_summary.csv")
    counts = manifest["strict_class"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
    csv_files = list((ROOT / "csv").rglob("*.csv"))
    png_files = list((ROOT / "png").rglob("*.png"))
    path_pairs_ok = True
    hashes_ok = True
    source_hashes_ok = True
    for row in manifest.itertuples(index=False):
        exported_csv = ROOT / row.exported_csv
        exported_png = ROOT / row.exported_png
        source_csv = ROOT.parents[1] / row.source_csv
        source_png = ROOT.parents[1] / row.source_png
        path_pairs_ok = path_pairs_ok and exported_csv.exists() and exported_png.exists()
        hashes_ok = hashes_ok and sha256(exported_csv) == row.csv_sha256 and sha256(exported_png) == row.png_sha256
        source_hashes_ok = source_hashes_ok and sha256(source_csv) == row.csv_sha256 and sha256(source_png) == row.png_sha256
    checks = {
        "manifest_has_61_curves": len(manifest) == 61,
        "class_counts_match": counts == EXPECTED,
        "exactly_61_exported_csv": len(csv_files) == 61,
        "exactly_61_exported_png": len(png_files) == 61,
        "curve_ids_unique": manifest["curve_id"].nunique() == 61,
        "csv_export_paths_unique": manifest["exported_csv"].nunique() == 61,
        "png_export_paths_unique": manifest["exported_png"].nunique() == 61,
        "all_csv_png_pairs_exist": path_pairs_ok,
        "all_export_hashes_match_manifest": hashes_ok,
        "all_copies_identical_to_sources": source_hashes_ok,
        "all_window_hours_are_750": manifest["window_hours"].eq(750).all(),
        "all_strict_topology_match": manifest["strict_topology_match"].eq(True).all(),
        "confidence_distribution_preserved": manifest["confidence_level"].value_counts().to_dict()
        == {"high": 58, "low_review": 3},
        "stable_manifest_has_58_high_rows": len(stable) == 58 and stable["confidence_level"].eq("high").all(),
        "review_queue_has_3_low_rows": len(review) == 3 and review["confidence_level"].eq("low_review").all(),
        "stable_and_review_partition_manifest": set(stable["curve_id"]).isdisjoint(set(review["curve_id"]))
        and set(stable["curve_id"]) | set(review["curve_id"]) == set(manifest["curve_id"]),
        "summary_total_is_61": int(summary["curve_count"].sum()) == 61,
        "summary_confidence_totals_match": int(summary["high_count"].sum()) == 58
        and int(summary["low_review_count"].sum()) == 3,
        "readme_exists": (ROOT / "README_CN.md").exists(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"all_passed": all(checks.values()), "passed": sum(checks.values()), "total": len(checks), "checks": checks}


def write_verification(result: dict) -> None:
    (ROOT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    with (ROOT / "checksums.sha256").open("w", encoding="utf-8") as f:
        for path in files:
            f.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")


def verify_checksums() -> bool:
    checksum_path = ROOT / "checksums.sha256"
    if not checksum_path.exists():
        return False
    ok = True
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        ok = ok and path.exists() and sha256(path) == expected
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run_checks()
    if args.write:
        write_verification(result)
    checksum_ok = verify_checksums() if (ROOT / "checksums.sha256").exists() else None
    print(json.dumps({**result, "checksums_passed": checksum_ok}, ensure_ascii=False, indent=2))
    if not result["all_passed"] or checksum_ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
