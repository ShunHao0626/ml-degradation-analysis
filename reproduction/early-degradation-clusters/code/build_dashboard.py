"""Build a self-contained, offline browser for the four fitted clusters."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent.parent
TEMPLATE = HERE / "code" / "dashboard_template.html"
CLUSTER_NAMES = {1: "Slope", 2: "Bridge", 3: "Valley", 4: "Hill"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def figure_label(image_path: str) -> str:
    stem = Path(image_path).stem
    match = re.match(r"^(FigS?\d+)(?:_([A-Za-z]+))?", stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify figure number: {image_path}")
    figure, panel = match.groups()
    if not panel:
        return figure
    return f"{figure} ({panel})" if panel.lower() == "whole" else f"{figure} {panel}"


def curve_legend(source: dict[str, str]) -> str:
    stem = Path(source["dataset_csv"]).stem
    parts = stem.split("__")
    if len(parts) >= 6:
        return parts[-2]
    image_stem = re.sub(r"__variant_\d+$", "", Path(source["dataset_image"]).stem)
    doi_folder = Path(source["dataset_csv"]).parent.name
    prefix = f"{doi_folder}_{image_stem}_"
    if stem.startswith(prefix):
        return stem[len(prefix):].split("__Series_", 1)[0]
    raise ValueError(f"Cannot identify curve legend: {source['dataset_csv']}")


def build_dashboard(results_dir: Path) -> None:
    assignments = read_rows(results_dir / "cluster_assignments.csv")
    sources = read_rows(HERE / "data" / "raw_csv" / "curve_index.csv")
    by_id = {row["curve_id"]: row for row in sources}
    if len(by_id) != len(sources) or set(by_id) != {row["curve_id"] for row in assignments}:
        raise ValueError("Dashboard source index and assignments differ")

    records = []
    for assignment in assignments:
        source = by_id[assignment["curve_id"]]
        cluster = int(assignment["cluster"])
        if cluster not in CLUSTER_NAMES or cluster != int(source["cluster"]):
            raise ValueError(f"Cluster mismatch: {assignment['curve_id']}")
        prefix = f"cluster_{cluster}/"
        paths = {
            "csv": prefix + source["dataset_csv"],
            "original": prefix + source["dataset_image"],
            "replot": prefix + source["dataset_replot"],
        }
        for label, path in paths.items():
            if not (results_dir / path).is_file():
                raise ValueError(f"Missing {label} for {assignment['curve_id']}: {path}")
        records.append({
            "curve_id": assignment["curve_id"],
            "cluster": cluster,
            "cluster_name": CLUSTER_NAMES[cluster],
            "doi": source["source_group"],
            "figure": figure_label(source["dataset_image"]),
            "legend": curve_legend(source),
            "csv": quote(paths["csv"], safe="/"),
            "original": quote(paths["original"], safe="/"),
            "replot": quote(paths["replot"], safe="/"),
        })
    if len(records) != 1842:
        raise ValueError("Expected 1842 dashboard records")

    data = {"records": records, "cluster_names": CLUSTER_NAMES}
    (results_dir / "dashboard_data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DASHBOARD_DATA__", embedded)
    (results_dir / "dashboard.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    build_dashboard(args.results_dir)
