#!/usr/bin/env python3
"""Download the three public UCR/UEA datasets used in the paper."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path


DATASETS = ("ECG5000", "Trace", "Plane")
URL = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"


def download(name: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} ...")
    with urllib.request.urlopen(URL.format(name=name), timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        wanted = {f"{name}_TRAIN.ts", f"{name}_TEST.ts"}
        missing = wanted.difference(archive.namelist())
        if missing:
            raise RuntimeError(f"archive for {name} is missing {sorted(missing)}")
        for member in sorted(wanted):
            target = destination / member
            target.write_bytes(archive.read(member))
            print(f"  wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    for name in DATASETS:
        download(name, args.data_dir / name)


if __name__ == "__main__":
    main()
