"""Small utilities for the UCR/UEA ``.ts`` files used by the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_ts_file(path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a univariate, labelled UCR/UEA ``.ts`` file.

    This deliberately supports the subset needed by ECG5000, Trace, and Plane
    and raises a clear error for multivariate or timestamped data.
    """

    source = Path(path)
    series: list[np.ndarray] = []
    labels: list[str] = []
    in_data = False
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            fields = line.split(":")
            if len(fields) != 2:
                raise ValueError(
                    f"{source}:{line_number}: expected univariate values followed by one label"
                )
            values = np.fromstring(fields[0], sep=",", dtype=float)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{source}:{line_number}: invalid or missing numeric data")
            series.append(values)
            labels.append(fields[1].strip())

    if not in_data:
        raise ValueError(f"{source}: missing @data marker")
    if not series:
        raise ValueError(f"{source}: no series found")
    return series, np.asarray(labels, dtype=str)


def load_ucr_dataset(
    data_dir: str | Path,
    name: str,
    *,
    split: str = "both",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load train/test/both splits and return series, labels, source indices."""

    split = split.lower()
    if split not in {"train", "test", "both"}:
        raise ValueError("split must be 'train', 'test', or 'both'")
    root = Path(data_dir) / name
    requested = [split.upper()] if split != "both" else ["TRAIN", "TEST"]
    all_series: list[np.ndarray] = []
    all_labels: list[str] = []
    all_indices: list[str] = []
    for part in requested:
        path = root / f"{name}_{part}.ts"
        values, labels = load_ts_file(path)
        all_series.extend(values)
        all_labels.extend(labels.tolist())
        all_indices.extend(f"{part.lower()}:{i}" for i in range(len(values)))
    return all_series, np.asarray(all_labels), np.asarray(all_indices)


def stratified_sample(
    series: list[np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    classes: int,
    per_class: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Deterministically sample the requested number of series per class."""

    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    if classes > unique.size:
        raise ValueError(f"requested {classes} classes, but only {unique.size} are available")
    chosen_labels = rng.choice(unique, size=classes, replace=False)
    selected: list[int] = []
    for label in chosen_labels:
        candidates = np.flatnonzero(labels == label)
        if per_class > candidates.size:
            raise ValueError(f"class {label} contains fewer than {per_class} samples")
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    chosen = np.asarray(selected, dtype=int)
    return [series[i] for i in chosen], labels[chosen], indices[chosen]
