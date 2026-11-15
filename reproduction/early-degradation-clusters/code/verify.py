"""Check the self-contained 1,842-curve inputs and, optionally, a new fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np


HERE = Path(__file__).resolve().parent.parent
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
        config = json.loads((HERE / "code" / "config.json").read_text(encoding="utf-8"))
        output = (HERE / config["output_dir"]).resolve()
        assignments = rows(output / "cluster_assignments.csv")
        check([row["curve_id"] for row in assignments] == ids, "Result IDs/order differ")
        counts = Counter(int(row["cluster"]) for row in assignments)
        check(counts == {1: 1710, 2: 44, 3: 21, 4: 67}, "Unexpected cluster sizes")
        source_index = rows(DATA / "raw_csv" / "curve_index.csv")
        check([(row["curve_id"], row["cluster"]) for row in source_index] ==
              [(row["curve_id"], row["cluster"]) for row in assignments],
              "Raw source index cluster IDs differ from assignments")
        with np.load(output / "model.npz") as model, np.load(DATA / "shape_inputs.npz") as shapes:
            check(model["curve_id"].tolist() == ids, "Model IDs/order differ")
            check(np.array_equal(model["cluster"],
                                 np.array([int(row["cluster"]) for row in assignments])),
                  "Model cluster IDs differ from assignments")
            check(np.array_equal(model["center_cluster"], np.arange(1, 5)),
                  "Cluster center numbering differs")
            check(np.array_equal(model["raw_rank"], shapes["raw_rank"]),
                  "Model did not use supplied shape inputs")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        check(summary["included"] == 1842 and summary["excluded_from_input"] == 0,
              "Unexpected included/excluded counts")
        for name in ("Supplementary_Fig_S1_zoom.png", "Supplementary_Fig_S1_zoom.svg"):
            check((output / "figures" / name).is_file(), f"Missing zoom figure: {name}")
        check((output / "cluster_four_groups.png").is_file(), "Missing cluster overview figure")
        raw_root = (DATA / "raw_csv").resolve()
        source_by_id = {row["curve_id"]: row for row in sources}
        for cluster in range(1, 5):
            folder = output / f"cluster_{cluster}"
            members = rows(folder / "members.csv")
            expected = [row for row in assignments if int(row["cluster"]) == cluster]
            check([row["curve_id"] for row in members] ==
                  [row["curve_id"] for row in expected],
                  f"Cluster {cluster} membership list differs")
            doi_folders = set()
            for member in members:
                source = source_by_id[member["curve_id"]]
                source_csv = (DATA / source["raw_csv"]).resolve()
                check(source_csv.is_relative_to(raw_root), "Raw CSV path escapes source directory")
                relative = source_csv.relative_to(raw_root)
                copied = folder / relative
                check(copied.is_file(), f"Missing copied CSV: {copied}")
                check(member["copied_csv"] == relative.as_posix(),
                      f"Copied CSV path differs: {copied}")
                check(hashlib.sha256(copied.read_bytes()).hexdigest() == source["csv_sha256"],
                      f"Copied CSV checksum mismatch: {copied}")
                doi_folders.add(relative.parent)
            if config["cluster_export_mode"] == "full_doi":
                for doi in doi_folders:
                    originals = {p.name for p in (raw_root / doi).iterdir() if p.is_file()}
                    copies = {p.name for p in (folder / doi).iterdir() if p.is_file()}
                    check(copies == originals, f"Incomplete DOI folder copy: {folder / doi}")
        report["results_valid"] = True
        report["cluster_sizes"] = dict(sorted(counts.items()))
        report["cluster_folders_valid"] = True
        dashboard_path = output / "dashboard.html"
        dashboard_data = json.loads((output / "dashboard_data.json").read_text(encoding="utf-8"))
        check(dashboard_path.is_file(), "Missing dashboard HTML")
        html = dashboard_path.read_text(encoding="utf-8")
        marker = '<script type="application/json" id="dashboard-data">'
        check(marker in html and "__DASHBOARD_DATA__" not in html,
              "Dashboard data was not embedded")
        embedded = html.split(marker, 1)[1].split("</script>", 1)[0]
        check(json.loads(embedded) == dashboard_data, "Dashboard embedded data differs")
        cards = dashboard_data["records"]
        check([row["curve_id"] for row in cards] == ids, "Dashboard curve IDs/order differ")
        check([int(row["cluster"]) for row in cards] ==
              [int(row["cluster"]) for row in assignments],
              "Dashboard cluster IDs differ")
        expected_names = {1: "Slope", 2: "Bridge", 3: "Valley", 4: "Hill"}
        for card in cards:
            check(card["cluster_name"] == expected_names[int(card["cluster"])],
                  f"Dashboard cluster name differs: {card['curve_id']}")
            check(all(card[field] for field in ("doi", "figure", "legend")),
                  f"Dashboard metadata missing: {card['curve_id']}")
            for field in ("csv", "original", "replot"):
                check((output / unquote(card[field])).is_file(),
                      f"Dashboard {field} missing: {card['curve_id']}")
        report["dashboard_valid"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
