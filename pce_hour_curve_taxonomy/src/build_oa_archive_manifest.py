#!/usr/bin/env python3
"""Validate and inventory the local open-access literature archive."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "08_literature_review/screening/priority_shortlist.csv"
PAPERS = ROOT / "08_literature_review/open_access_papers"
OUTPUT = ROOT / "08_literature_review/metadata/oa_archive_manifest.csv"


def safe_name(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with SHORTLIST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = []
    for row in rows:
        path = PAPERS / safe_name(row["doi"])
        if path.exists() and path.read_bytes()[:4] == b"%PDF":
            document = fitz.open(path)
            manifest.append(
                {
                    "doi": row["doi"],
                    "title": row["title"],
                    "status": "validated_pdf",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "pages": len(document),
                    "sha256": sha256(path),
                }
            )
        else:
            manifest.append(
                {
                    "doi": row["doi"],
                    "title": row["title"],
                    "status": "metadata_only_or_download_blocked",
                    "path": "",
                    "bytes": 0,
                    "pages": 0,
                    "sha256": "",
                }
            )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["doi", "title", "status", "path", "bytes", "pages", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
