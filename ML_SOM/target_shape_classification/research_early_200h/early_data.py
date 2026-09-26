"""Frozen source audit and non-extrapolating 0–200 h observations."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEGACY = PROJECT / "target_shape_classification"
CLASSES = ("bridge", "hill", "slope", "valley")


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path escapes configured root: " + relative)
    return path


def read_observations(row, source_root):
    """Return every finite source row, including duplicate/negative times."""
    path = inside(source_root, row["source_csv"])
    columns = ("x", "y") if row["folder"] == "data_all" else ("time_h", "normalized_pce")
    factor = float(row["time_factor"])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("unresolved_hour_axis")
    if row["folder"] == "final_data" and factor != 1:
        raise ValueError("final_data_time_h_must_remain_hours")
    points = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError("missing_coordinates")
        for source_row, item in enumerate(reader, 2):
            try:
                hour = float(item[columns[0]]) * factor
                value = float(item[columns[1]])
            except (TypeError, ValueError):
                continue
            if math.isfinite(hour) and math.isfinite(value):
                points.append((hour, value, source_row))
    return sorted(points, key=lambda p: p[0])


def freeze(config, root):
    source = Path(config["source_root"])
    images = Path(config["image_root"])
    legacy = Path(config["legacy_results_root"])
    manifest = rows(legacy / "input_manifest.csv")
    canonical = rows(legacy / "canonical_curves.csv")
    reviews = rows(LEGACY / "review_inputs/source_reviews_200h.csv")
    digit = rows(legacy / "digitization_manifest.csv")
    if len(manifest) != 2250 or len(canonical) != 2246:
        raise ValueError("input counts changed; review protocol before continuing")
    if len({r["file_id"] for r in manifest}) != len(manifest):
        raise ValueError("duplicate file ID")
    if len({r["file_id"] for r in canonical}) != len(canonical):
        raise ValueError("duplicate canonical ID")
    by_file = {r["file_id"]: r for r in manifest}
    by_canonical = {r["file_id"]: r for r in canonical}
    if any(r["canonical_id"] not in by_canonical for r in manifest):
        raise ValueError("missing canonical reference")
    hashes = {}
    for r in manifest:
        path = inside(source, r["source_csv"])
        actual = digest(path)
        if actual != r["source_sha256"]:
            raise ValueError("source CSV hash changed: " + r["file_id"])
        mirror = inside(images, r["source_csv"])
        if digest(mirror) != actual:
            raise ValueError("source mirror changed: " + r["file_id"])
        image = r["source_image"]
        if image and image not in hashes:
            hashes[image] = digest(inside(images, image))
        if image and hashes[image] != r["image_sha256"]:
            raise ValueError("source image changed: " + r["file_id"])
        if r["metadata_file"]:
            inside(images, r["metadata_file"]).stat()
    for r in digit:
        if r["status"] != "accepted":
            continue
        original = next((x for x in manifest if x["source_csv"] == r["source_csv"]), None)
        if not original or original["source_sha256"] != r["source_sha256"]:
            raise ValueError("accepted digitization source changed")
        if digest(inside(legacy, r["digitized_csv"])) != r["digitized_sha256"]:
            raise ValueError("accepted digitization changed")
        if r["source_image_sha256"] != hashes[r["source_image"]]:
            raise ValueError("accepted digitization image changed")
        inside(legacy, r["calibration_json"]).stat()
        inside(legacy, r["overlay_png"]).stat()
    review_map = {}
    for r in reviews:
        source_row = by_file[r["file_id"]]
        if r["source_image_sha256"] != hashes[source_row["source_image"]]:
            raise ValueError("old review image changed: " + r["file_id"])
        review_map[r["file_id"]] = r
    reviewed_groups = {by_file[r["file_id"]]["source_group"] for r in reviews}
    groups = sorted({r["source_group"] for r in manifest})
    other = [group for group in groups if group not in reviewed_groups]
    random.Random(config["split_seed"]).shuffle(other)
    # Reserve validation/test groups before training; prior manually reviewed groups stay train.
    n_test = round(len(groups) * (1 - config["train_fraction"] - config["validation_fraction"]))
    n_val = round(len(groups) * config["validation_fraction"])
    if len(other) <= n_test + n_val:
        raise ValueError("too many reviewed groups for requested split")
    split = {group: "train" for group in groups}
    split.update({group: "test" for group in other[:n_test]})
    split.update({group: "validation" for group in other[n_test:n_test + n_val]})
    split_rows = [dict(source_group=group, split=split[group],
                       historical_review_group=int(group in reviewed_groups)) for group in groups]
    frozen = []
    for r in manifest:
        frozen.append(dict(file_id=r["file_id"], canonical_id=r["canonical_id"],
                           source_csv=r["source_csv"], source_sha256=r["source_sha256"],
                           source_image=r["source_image"], image_sha256=r["image_sha256"],
                           metadata_file=r["metadata_file"], source_group=r["source_group"],
                           split=split[r["source_group"]], folder=r["folder"],
                           y_kind=r["y_kind"], time_factor=r["time_factor"],
                           time_unit=r["time_unit"], conversion=r["conversion"],
                           historical_review_status=review_map.get(r["file_id"], {}).get("review_status", ""),
                           historical_review_class=review_map.get(r["file_id"], {}).get("reviewed_class", ""),
                           historical_analysis_version=r["analysis_version"],
                           historical_analysis_sha256=r["analysis_sha256"]))
    out = Path(root) / "manifests"
    write_rows(out / "frozen_inputs.csv", frozen, list(frozen[0]))
    write_rows(out / "source_split.csv", split_rows, list(split_rows[0]))
    snapshot = dict(protocol=config["protocol_version"], input_manifest_sha256=digest(legacy / "input_manifest.csv"),
                    canonical_manifest_sha256=digest(legacy / "canonical_curves.csv"),
                    historical_reviews_sha256=digest(LEGACY / "review_inputs/source_reviews_200h.csv"),
                    axis_overrides_sha256=digest(LEGACY / "review_inputs/source_axis_overrides.csv"),
                    digitization_manifest_sha256=digest(legacy / "digitization_manifest.csv"),
                    source_count=len(manifest), canonical_count=len(canonical), image_count=len(hashes),
                    accepted_digitization_count=sum(r["status"] == "accepted" for r in digit),
                    split_groups={s: sum(v == s for v in split.values()) for s in ("train", "validation", "test")},
                    note="Original source CSV is the sole primary input; accepted digitizations are audited only.")
    (out / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return frozen, by_canonical, split, snapshot


def prepare(row, source_root, config):
    if row["y_kind"] != "pce":
        return None, "response_identity_" + row["y_kind"]
    try:
        points = read_observations(row, source_root)
    except (OSError, ValueError) as exc:
        return None, str(exc)
    limit = config["window_h"]
    inside_points = [p for p in points if 0 <= p[0] <= limit]
    if len({p[0] for p in inside_points}) < 2:
        return None, "fewer_than_two_distinct_observations_0_200h"
    divisor = max(p[1] for p in inside_points)
    if divisor <= 0 or not math.isfinite(divisor):
        return None, "nonpositive_window_maximum"
    normalized = [(h, v / divisor, source_row) for h, v, source_row in points]
    by_time = defaultdict(list)
    for h, y, source_row in normalized:
        if 0 <= h <= limit:
            by_time[h].append(y)
    t = np.array(sorted(by_time), dtype=float)
    y = np.array([np.median(by_time[h]) for h in t])
    boundary = [(h, v, sr) for h, v, sr in normalized if limit < h <= config["boundary_limit_h"]]
    boundary = boundary[:config["boundary_max_points"]]
    return dict(points=normalized, t=t, y=y, boundary=boundary,
                divisor=float(divisor), duplicates=len(inside_points) - len(t),
                window_start_h=float(t[0]), window_end_h=float(t[-1]),
                window_points=len(inside_points), unique_window_times=len(t),
                max_gap_h=float(np.max(np.diff(t)))), ""


def features(item, config):
    """Three groups; NaN means no actual point support, not zero PCE."""
    grid = np.arange(0, config["window_h"] + 1e-9, config["grid_step_h"])
    t, y = item["t"], item["y"]
    trajectory = np.full(len(grid), np.nan)
    for k, h in enumerate(grid):
        index = np.searchsorted(t, h)
        if index < len(t) and np.isclose(t[index], h, atol=1e-9):
            trajectory[k] = y[index]
        elif 0 < index < len(t) and t[index] - t[index - 1] <= config["maximum_interpolation_gap_h"]:
            trajectory[k] = np.interp(h, t[index - 1:index + 1], y[index - 1:index + 1])
    velocities = []
    for step_h in (10, 20):
        shift = round(step_h / config["grid_step_h"])
        v = np.full(len(grid), np.nan)
        v[shift:] = (trajectory[shift:] - trajectory[:-shift]) / step_h
        velocities.append(v * 10)  # fixed units: change per 10 h
    detail = np.full((20, 4), np.nan)
    for k in range(20):
        mask = (t >= k * 10) & (t < (k + 1) * 10 if k < 19 else t <= 200)
        values = y[mask]
        if len(values):
            detail[k, :2] = [min(values), max(values)]
        if len(values) >= 2:
            diffs = np.diff(values)
            detail[k, 2:] = [sum(diffs[diffs > 0]), -sum(diffs[diffs < 0])]
    return (trajectory, np.concatenate(velocities), detail.ravel())
