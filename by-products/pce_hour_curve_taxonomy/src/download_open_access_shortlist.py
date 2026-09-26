#!/usr/bin/env python3
"""Lawfully archive verified open-access PDFs from the priority shortlist."""

from __future__ import annotations

import csv
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST = ROOT / "08_literature_review/screening/priority_shortlist.csv"
OUT = ROOT / "08_literature_review/open_access_papers"
LOG = ROOT / "08_literature_review/metadata/oa_download_log.csv"


def safe_name(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".pdf"


def candidate_urls(row: dict[str, str]) -> list[str]:
    urls = []
    for field in ("primary_pdf_url", "oa_url"):
        url = (row.get(field) or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with SHORTLIST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    log_rows = []
    context = ssl.create_default_context()
    for row in rows:
        doi = row["doi"]
        destination = OUT / safe_name(doi)
        if destination.exists() and destination.read_bytes()[:4] == b"%PDF":
            log_rows.append(
                {"doi": doi, "status": "already_present", "url": "", "path": str(destination), "bytes": destination.stat().st_size, "error": ""}
            )
            continue
        if str(row.get("is_open_access", "")).lower() != "true":
            log_rows.append(
                {"doi": doi, "status": "not_open_access", "url": "", "path": "", "bytes": 0, "error": ""}
            )
            continue
        success = False
        last_error = "no candidate URL"
        last_url = ""
        for url in candidate_urls(row):
            last_url = url
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 pce-hour-curve-review/1.0",
                        "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5",
                    },
                )
                with urllib.request.urlopen(request, timeout=90, context=context) as response:
                    payload = response.read()
                if not payload.startswith(b"%PDF"):
                    last_error = f"response was not PDF ({len(payload)} bytes)"
                    continue
                destination.write_bytes(payload)
                log_rows.append(
                    {"doi": doi, "status": "downloaded", "url": url, "path": str(destination), "bytes": len(payload), "error": ""}
                )
                success = True
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if not success:
            log_rows.append(
                {"doi": doi, "status": "download_failed", "url": last_url, "path": "", "bytes": 0, "error": last_error}
            )
    with LOG.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doi", "status", "url", "path", "bytes", "error"])
        writer.writeheader()
        writer.writerows(log_rows)


if __name__ == "__main__":
    main()
