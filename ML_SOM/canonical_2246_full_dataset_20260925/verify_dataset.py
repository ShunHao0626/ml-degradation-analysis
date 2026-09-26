"""Verify copied source bytes and every derived coordinate in the dataset."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from build_dataset import HERE, SOURCE, digest, read_rows, safe_source_path


def number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 2e-10 * max(1, abs(a), abs(b))


def main() -> None:
    manifest = read_rows(HERE / "dataset_manifest.csv")
    aliases = read_rows(HERE / "provenance/alias_mapping_2250_to_2246.csv")
    assert len(manifest) == 2246 and len(aliases) == 2250
    ids = {r["curve_id"] for r in manifest}
    assert len(ids) == 2246 and {r["canonical_id"] for r in aliases} == ids
    assert len(list((HERE / "derived").glob("*.csv"))) == 2246
    assert len(list((HERE / "dataset").glob("*.csv"))) == 2246
    assert {r["dataset_id"] for r in manifest} == {f"curve_{i:04d}" for i in range(1, 2247)}
    total_rows = 0
    for record in manifest:
        raw_path = HERE / record["curve_csv"]
        standardized_path = HERE / record["derived_csv"]
        source_path = safe_source_path(record["source_csv"])
        assert source_path.is_relative_to(SOURCE.resolve())
        assert digest(source_path) == digest(raw_path) == record["source_sha256"]
        source_rows = read_rows(raw_path)
        converted = read_rows(standardized_path)
        assert len(source_rows) == len(converted) == int(record["source_rows"])
        total_rows += len(converted)
        factor = number(record["time_factor"])
        full_max = number(record["full_pce_max"])
        window_max = number(record["window_0_200h_pce_max"])
        for i, (src, dst) in enumerate(zip(source_rows, converted), 2):
            assert int(dst["source_row"]) == i
            assert dst["source_x"] == src[record["source_x_column"]]
            assert dst["source_y"] == src[record["source_y_column"]]
            x = number(dst["source_x"])
            y = number(dst["source_y"])
            hour = number(dst["hour"])
            if x is not None and factor is not None and factor > 0:
                assert hour is not None and close(hour, x * factor)
            else:
                assert dst["hour"] == ""
            for column, divisor in (("normalized_pce_full", full_max),
                                    ("normalized_pce_0_200h", window_max)):
                value = number(dst[column])
                if y is not None and divisor is not None:
                    assert value is not None and close(value, y / divisor)
                else:
                    assert dst[column] == ""
        if record["pce_status"] != "confirmed_pce":
            assert full_max is None and window_max is None
        if record["hour_status"] != "converted":
            assert all(r["hour"] == "" for r in converted)
    report = dict(checks="passed", canonical_files=len(manifest),
                  original_aliases=len(aliases), all_source_rows=total_rows,
                  copied_source_hashes="match live original CSVs and canonical manifest",
                  coordinate_rows="all checked")
    (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
