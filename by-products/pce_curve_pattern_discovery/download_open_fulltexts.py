#!/usr/bin/env python3
"""Download only explicitly open PDF URLs from the screened queue."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LIT = ROOT / "07_literature"
QUEUE = LIT / "open_fulltext_download_queue.csv"
OUT = LIT / "fulltext_open"


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return name[:120] or "no_doi"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE).fillna("")
    rows = []
    for row in queue.to_dict("records"):
        download_id = str(row["download_id"])
        url = str(row["open_pdf_url"])
        target = OUT / f"{download_id}__{safe_name(str(row['doi']))}.pdf"
        status = ""
        error = ""
        size = 0
        digest = ""
        content_type = ""
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PCE-curve-literature-audit/1.0 (metadata and open-access research retrieval)",
                    "Accept": "application/pdf",
                },
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content_type = str(response.headers.get("Content-Type", ""))
                body = response.read(60 * 1024 * 1024 + 1)
            if len(body) > 60 * 1024 * 1024:
                status = "skipped_over_60MB"
            elif not body.startswith(b"%PDF"):
                status = "not_pdf_response"
                error = f"content_type={content_type}"
            else:
                target.write_bytes(body)
                status = "downloaded_open_pdf"
                size = len(body)
                digest = hashlib.sha256(body).hexdigest()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = "download_failed"
            error = f"{type(exc).__name__}: {exc}"[:500]
        rows.append(
            {
                **row,
                "download_status": status,
                "local_file": target.relative_to(ROOT).as_posix() if target.exists() else "",
                "bytes": size,
                "sha256": digest,
                "content_type": content_type,
                "error": error,
            }
        )
        print(download_id, status, size)
        time.sleep(0.25)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(LIT / "initial_python_dns_attempt_manifest.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_size": len(manifest),
        "status_counts": manifest["download_status"].value_counts().to_dict(),
        "downloaded_files": int((manifest["download_status"] == "downloaded_open_pdf").sum()),
        "downloaded_bytes": int(manifest["bytes"].sum()),
        "policy": "Only URLs explicitly exposed as open PDF locations were attempted; no access-control bypass.",
    }
    (LIT / "initial_python_dns_attempt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
