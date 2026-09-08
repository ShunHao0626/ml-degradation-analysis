#!/usr/bin/env python3
"""Validate, extract, and render relevant pages from downloaded open PDFs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parent
LIT = ROOT / "07_literature"
PDF_DIR = LIT / "fulltext_open"
TEXT_DIR = LIT / "fulltext_text"
RENDER_DIR = LIT / "rendered_relevant_pages"
QUEUE = LIT / "open_fulltext_download_queue.csv"

SIGNAL = re.compile(
    r"power conversion efficien|\bPCE\b|maximum power point|\bMPP\b|\bMPPT\b|"
    r"operational stabil|light[- ]soak|burn[- ]in|recover|reversib|self[- ]heal|"
    r"\bT80\b|hours?|ageing|aging|degrad",
    re.I,
)
NONMONOTONIC = re.compile(
    r"recover|reversib|self[- ]heal|light[- ]soak|burn[- ]in|initial gain|"
    r"increase.*efficien|efficien.*increase|restor|regener|diurnal|day.?night",
    re.I | re.S,
)
FIGURE = re.compile(r"(?:Fig(?:ure)?\.?\s*\d+|Extended Data Fig|Supplementary Fig)", re.I)


def id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE).fillna("").set_index("download_id")
    rows = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        did = id_from_path(pdf_path)
        metadata = queue.loc[did].to_dict() if did in queue.index else {}
        raw = pdf_path.read_bytes()
        page_texts: list[str] = []
        error = ""
        try:
            document = fitz.open(pdf_path)
            for page in document:
                page_texts.append(page.get_text("text"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "download_id": did,
                    **metadata,
                    "local_file": pdf_path.relative_to(ROOT).as_posix(),
                    "pdf_valid": False,
                    "page_count": 0,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "signal_pages": "",
                    "nonmonotonic_signal_pages": "",
                    "rendered_pages": "",
                    "error": error,
                }
            )
            continue

        full_text = "\n\n".join(f"===== PAGE {i + 1} =====\n{text}" for i, text in enumerate(page_texts))
        (TEXT_DIR / f"{did}.txt").write_text(full_text, encoding="utf-8")
        signal_pages = [i + 1 for i, value in enumerate(page_texts) if SIGNAL.search(value)]
        nonmono_pages = [i + 1 for i, value in enumerate(page_texts) if NONMONOTONIC.search(value)]
        figure_signal_pages = [
            i + 1
            for i, value in enumerate(page_texts)
            if SIGNAL.search(value) and FIGURE.search(value)
        ]
        render_pages = []
        for page_number in figure_signal_pages[:3]:
            page = document[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            out_dir = RENDER_DIR / did
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"page_{page_number:03d}.png"
            pix.save(out)
            render_pages.append(out.relative_to(ROOT).as_posix())
        document.close()
        rows.append(
            {
                "download_id": did,
                **metadata,
                "local_file": pdf_path.relative_to(ROOT).as_posix(),
                "pdf_valid": raw.startswith(b"%PDF"),
                "page_count": len(page_texts),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "signal_pages": ";".join(map(str, signal_pages)),
                "nonmonotonic_signal_pages": ";".join(map(str, nonmono_pages)),
                "rendered_pages": ";".join(render_pages),
                "error": error,
            }
        )

    manifest = pd.DataFrame(rows).sort_values("download_id")
    manifest.to_csv(LIT / "open_fulltext_audit_manifest.csv", index=False)
    final_download = queue.reset_index()[
        ["download_id", "doi", "title", "year", "open_pdf_url", "landing_url", "relevance_score"]
    ].copy()
    actual = manifest.set_index("download_id")
    final_download["download_status"] = final_download["download_id"].map(
        lambda value: "downloaded_open_pdf" if value in actual.index else "unavailable_after_curl_fallback"
    )
    final_download["local_file"] = final_download["download_id"].map(
        lambda value: actual.loc[value, "local_file"] if value in actual.index else ""
    )
    final_download["bytes"] = final_download["download_id"].map(
        lambda value: int(actual.loc[value, "bytes"]) if value in actual.index else 0
    )
    final_download["sha256"] = final_download["download_id"].map(
        lambda value: actual.loc[value, "sha256"] if value in actual.index else ""
    )
    final_download.to_csv(LIT / "open_fulltext_download_manifest.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid_open_pdfs": int(manifest["pdf_valid"].sum()),
        "total_pages": int(manifest["page_count"].sum()),
        "total_bytes": int(manifest["bytes"].sum()),
        "with_nonmonotonic_text_signal": int(manifest["nonmonotonic_signal_pages"].astype(bool).sum()),
        "rendered_page_count": sum(len(text.split(";")) for text in manifest["rendered_pages"] if text),
        "note": "Text signals are triage only; rendered pages require visual review before curve-shape claims.",
    }
    (LIT / "open_fulltext_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    download_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_size": len(final_download),
        "status_counts": final_download["download_status"].value_counts().to_dict(),
        "downloaded_files": int((final_download["download_status"] == "downloaded_open_pdf").sum()),
        "downloaded_bytes": int(final_download["bytes"].sum()),
        "note": "The first Python attempt was DNS-blocked; a permitted curl fallback was then attempted for all 50 URLs.",
        "policy": "Only explicitly open PDF URLs were attempted; no access-control bypass.",
    }
    (LIT / "open_fulltext_download_summary.json").write_text(
        json.dumps(download_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
