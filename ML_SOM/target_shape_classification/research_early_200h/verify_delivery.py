#!/usr/bin/env python3
"""Check complete canonical and source-file coverage plus materialized charts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from early_data import CLASSES, HERE, digest, rows


def verify():
    delivery = HERE / "delivery"
    files = rows(delivery / "all_files.csv")
    index = rows(delivery / "class_index.csv")
    unresolved = rows(delivery / "unresolved.csv")
    canonical = rows(HERE / "manifests/frozen_inputs.csv")
    canonical_ids = {r["canonical_id"] for r in canonical}
    source_ids = {r["file_id"] for r in canonical}
    source_by_id = {r["file_id"]: r for r in canonical}
    assert len(files) == len(source_ids) == 2250
    assert {r["file_id"] for r in files} == source_ids
    for row in files:
        frozen = source_by_id[row["file_id"]]
        assert row["source_csv"] == frozen["source_csv"]
        assert row["source_sha256"] == frozen["source_sha256"]
        assert row["canonical_id"] == frozen["canonical_id"]
    ids = [r["curve_id"] for r in index + unresolved]
    assert len(ids) == len(set(ids)) == len(canonical_ids) == 2246
    assert set(ids) == canonical_ids
    directory_ids = []
    for label in CLASSES:
        for p in (delivery / label).glob("F*.csv"):
            directory_ids.append(p.stem)
    assert len(directory_ids) == len(set(directory_ids)) == len(index)
    assert set(directory_ids) == {r["curve_id"] for r in index}
    for r in index:
        assert r["candidate_class"] in CLASSES
        assert Path(r["curve_csv"]).parts[0] == r["candidate_class"]
        assert r["final_class"] == ""
        for key in ("curve_csv", "curve_png"):
            p = delivery / r[key]
            assert p.is_file() and p.stat().st_size > 100, (r["curve_id"], key)
        if r["boundary_png"]:
            assert (delivery / r["boundary_png"]).is_file()
        with (delivery / r["curve_png"]).open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
    counts = Counter(r["candidate_class"] for r in index)
    report = json.loads((HERE / "runs/summary.json").read_text(encoding="utf-8"))
    assert dict(counts) == report["classes"]
    assert len(unresolved) == report["unresolved"]
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert report["config"]["config_sha256"] == digest(HERE / "config.json")
    assert report["snapshot"]["protocol"] == config["protocol_version"]
    assert report["selected_model_sha256"] == digest(HERE / "runs" / f"{report['selected_run']}.npz")
    groups = rows(HERE / "manifests/source_split.csv")
    assert len(groups) == len({r["source_group"] for r in groups})
    assert all(r["split"] == "train" for r in groups if r["historical_review_group"] == "1")
    strict = rows(HERE / "reviews/blind_strict_form.csv")
    boundary = rows(HERE / "reviews/blind_boundary_form.csv")
    assert len(strict) == len(boundary) == 520
    assert {r["curve_id"] for r in strict} == {r["curve_id"] for r in boundary}
    assert all(not r["class_or_nonconforming"] and not r["expert_id"] and
               (HERE / "reviews" / r["strict_chart"]).is_file() for r in strict)
    result = dict(source_files=len(files), canonical_curves=len(ids),
                  class_counts=dict(counts), unresolved=len(unresolved),
                  checked_csv=len(index), checked_png=len(index),
                  boundary_png=sum(bool(r["boundary_png"]) for r in index),
                  blind_forms=len(strict), selected_model=report["selected_run"])
    (delivery / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
