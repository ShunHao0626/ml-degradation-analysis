"""Build 2246 byte-identical deduplicated source CSVs plus optional derived views."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "original_curves_2250"
CANONICAL = PROJECT / "target_shape_classification/results/canonical_curves.csv"
INPUT_MANIFEST = PROJECT / "target_shape_classification/results/input_manifest.csv"
PROVENANCE = HERE / "provenance"
RAW = HERE / "dataset"
CURVES = HERE / "derived"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.15g}"


def safe_source_path(relative: str) -> Path:
    root = SOURCE.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path escapes source root")
    return path


def main() -> None:
    canonical = read_rows(CANONICAL)
    originals = read_rows(INPUT_MANIFEST)
    if len(canonical) != 2246 or len(originals) != 2250:
        raise ValueError("manifest counts changed")
    ids = {r["file_id"] for r in canonical}
    if len(ids) != 2246 or {r["canonical_id"] for r in originals} != ids:
        raise ValueError("canonical mapping is inconsistent")
    aliases: dict[str, list[str]] = defaultdict(list)
    for r in originals:
        aliases[r["canonical_id"]].append(r["file_id"])
    RAW.mkdir(parents=True, exist_ok=True)
    CURVES.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL, HERE / "canonical_curves.csv")
    shutil.copyfile(INPUT_MANIFEST, PROVENANCE / "input_manifest_2250.csv")
    manifest: list[dict] = []
    total_source_rows = 0
    for sequence, row in enumerate(canonical, 1):
        curve_id = row["file_id"]
        dataset_id = f"curve_{sequence:04d}"
        original_path = safe_source_path(row["source_csv"])
        if digest(original_path) != row["source_sha256"]:
            raise ValueError("source hash mismatch: " + curve_id)
        raw_path = RAW / f"{dataset_id}.csv"
        shutil.copyfile(original_path, raw_path)
        if digest(raw_path) != row["source_sha256"]:
            raise ValueError("copied source hash mismatch: " + curve_id)
        x_col, y_col = (("x", "y") if row["folder"] == "data_all"
                        else ("time_h", "normalized_pce"))
        with raw_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {x_col, y_col}.issubset(reader.fieldnames or []):
                raise ValueError("missing source axis column: " + curve_id)
            source_rows = list(reader)
        factor = finite_number(row["time_factor"])
        if row["folder"] == "final_data" and factor is not None and factor != 1:
            raise ValueError("final_data time_h unexpectedly needs conversion: " + curve_id)
        hour_status = "converted" if factor is not None and factor > 0 else "unresolved_hour_axis"
        pce_status = "confirmed_pce" if row["y_kind"] == "pce" else row["y_kind"]
        parsed = []
        for source_row, point in enumerate(source_rows, 2):
            raw_x, raw_y = point[x_col], point[y_col]
            x, y = finite_number(raw_x), finite_number(raw_y)
            hour = x * factor if x is not None and hour_status == "converted" else None
            parsed.append((source_row, raw_x, raw_y, hour, y))
        all_y = [p[4] for p in parsed if p[4] is not None]
        full_max = max(all_y) if all_y else None
        if full_max is not None and full_max <= 0:
            full_max = None
        window_y = [p[4] for p in parsed if p[3] is not None and
                    0 <= p[3] <= 200 and p[4] is not None]
        window_max = max(window_y) if window_y else None
        if window_max is not None and window_max <= 0:
            window_max = None
        confirmed = pce_status == "confirmed_pce"
        full_divisor = full_max if confirmed else None
        window_divisor = window_max if confirmed else None
        converted = []
        for source_row, raw_x, raw_y, hour, y in parsed:
            converted.append(dict(source_row=source_row, source_x=raw_x, source_y=raw_y,
                                  hour=fmt(hour),
                                  normalized_pce_full=fmt(y / full_divisor)
                                  if y is not None and full_divisor is not None else "",
                                  normalized_pce_0_200h=fmt(y / window_divisor)
                                  if y is not None and window_divisor is not None else ""))
        curve_path = CURVES / f"{dataset_id}.csv"
        write_rows(curve_path, converted,
                   ["source_row", "source_x", "source_y", "hour",
                    "normalized_pce_full", "normalized_pce_0_200h"])
        total_source_rows += len(parsed)
        finite_hours = [p[3] for p in parsed if p[3] is not None]
        manifest.append(dict(dataset_id=dataset_id, curve_id=curve_id,
                             file_ids=";".join(aliases[curve_id]),
                             source_csv=row["source_csv"], source_sha256=row["source_sha256"],
                             curve_csv=f"dataset/{dataset_id}.csv",
                             derived_csv=f"derived/{dataset_id}.csv",
                             source_image=row["source_image"], source_group=row["source_group"],
                             source_x_column=x_col, source_y_column=y_col,
                             source_x_unit=row["x_unit"], source_y_name=row["y_name"],
                             source_y_unit=row["y_unit"], time_unit=row["time_unit"],
                             time_factor=row["time_factor"], hour_status=hour_status,
                             pce_status=pce_status, source_rows=len(parsed),
                             finite_hours=len(finite_hours),
                             first_hour=fmt(min(finite_hours)) if finite_hours else "",
                             last_hour=fmt(max(finite_hours)) if finite_hours else "",
                             observed_rows_0_200h=len(window_y),
                             full_pce_max=fmt(full_divisor),
                             window_0_200h_pce_max=fmt(window_divisor)))
    write_rows(HERE / "dataset_manifest.csv", manifest, list(manifest[0]))
    write_rows(PROVENANCE / "alias_mapping_2250_to_2246.csv",
               [dict(file_id=r["file_id"], canonical_id=r["canonical_id"],
                     source_csv=r["source_csv"], source_sha256=r["source_sha256"])
                for r in originals],
               ["file_id", "canonical_id", "source_csv", "source_sha256"])
    summary = dict(canonical_curves=len(manifest), original_file_aliases=len(originals),
                   duplicate_alias_rows=len(originals) - len(manifest),
                   source_rows=total_source_rows,
                   hour_status=dict(Counter(r["hour_status"] for r in manifest)),
                   pce_status=dict(Counter(r["pce_status"] for r in manifest)),
                   full_pce_normalized_curves=sum(bool(r["full_pce_max"]) for r in manifest),
                   window_0_200h_normalized_curves=sum(bool(r["window_0_200h_pce_max"])
                                                       for r in manifest),
                   source_manifest_sha256=digest(CANONICAL),
                   alias_manifest_sha256=digest(INPUT_MANIFEST),
                   build_code_sha256=digest(HERE / "build_dataset.py"))
    (HERE / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
