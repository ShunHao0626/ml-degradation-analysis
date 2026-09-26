"""Check the self-contained 1,842-curve inputs and, optionally, a new fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-results", action="store_true", help="also check a newly generated fit")
    args = parser.parse_args()

    index = rows(DATA / "input_index.csv")
    sources = rows(DATA / "curve_index.csv")
    ids = [row["curve_id"] for row in index]
    check(len(ids) == len(set(ids)) == 1842, "Expected 1842 unique input IDs")
    check(ids == [row["curve_id"] for row in sources], "Raw CSV index order differs")
    check(all(a["source_csv"] == b["source_csv"] for a, b in zip(index, sources)),
          "Source paths differ between indexes")

    for row in sources:
        path = (DATA / row["raw_csv"]).resolve()
        check(path.is_relative_to((DATA / "raw_csv").resolve()), "Raw CSV path escapes data directory")
        check(path.is_file(), f"Missing raw CSV: {path}")
        check(hashlib.sha256(path.read_bytes()).hexdigest() == row["csv_sha256"],
              f"Raw CSV checksum mismatch: {path}")
    for row in index:
        path = (DATA / row["normalized_csv"]).resolve()
        check(path.is_relative_to((DATA / "points").resolve()), "Point path escapes data directory")
        check(path.is_file(), f"Missing normalized points: {path}")

    with np.load(DATA / "shape_inputs.npz") as shapes:
        check(shapes["curve_id"].tolist() == ids, "Shape input IDs/order differ")
        check(shapes["raw_rank"].shape == (1842, 64), "Expected 1842 x 64 shape input")
        check(np.isfinite(shapes["raw_rank"]).all(), "Nonfinite shape input")

    report = {"inputs_valid": True, "curves": len(ids), "raw_csv_files": len(sources)}
    if args.check_results:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        output = (HERE / config["output_dir"]).resolve()
        assignments = rows(output / "anonymous_assignments.csv")
        check([row["curve_id"] for row in assignments] == ids, "Result IDs/order differ")
        counts = Counter(int(row["anonymous_cluster"]) for row in assignments)
        check(counts == {0: 1710, 1: 44, 2: 21, 3: 67}, "Unexpected cluster sizes")
        with np.load(output / "model.npz") as model, np.load(DATA / "shape_inputs.npz") as shapes:
            check(model["curve_id"].tolist() == ids, "Model IDs/order differ")
            check(np.array_equal(model["raw_rank"], shapes["raw_rank"]),
                  "Model did not use supplied shape inputs")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        check(summary["included"] == 1842 and summary["excluded_from_input"] == 0,
              "Unexpected included/excluded counts")
        for name in ("Supplementary_Fig_S1_zoom.png", "Supplementary_Fig_S1_zoom.svg"):
            check((output / "figures" / name).is_file(), f"Missing zoom figure: {name}")
        report["results_valid"] = True
        report["cluster_sizes"] = dict(sorted(counts.items()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
