#!/usr/bin/env python3
"""Download an auditable OA subset and inspect PDF text/figure captions.

The output explicitly labels this as machine-assisted full-text inspection. Visual
inspection is a separate downstream step. Only OpenAlex-provided OA URLs are used.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "high_priority_for_fulltext.csv"
PDF_DIR = ROOT / "oa_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = ROOT / "fulltext_machine_audit.csv"
MANIFEST = ROOT / "fulltext_audit_manifest.json"

TARGET_SUCCESS = 45
MAX_ATTEMPTS = 100


def slug_doi(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", doi)[:180]


def excerpt(text: str, match: re.Match, radius: int = 260) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    value = re.sub(r"\s+", " ", text[start:end]).strip()
    return value[:700]


def page_evidence(page_text: str) -> tuple[bool, list[str], list[str]]:
    normalized = re.sub(r"\s+", " ", page_text)
    graph_patterns = [
        re.compile(
            r"(?:Fig(?:ure)?\.?\s*[A-Za-z0-9S\.\-]+).{0,700}?"
            r"(?:PCE|power conversion efficiency|normalized efficiency).{0,500}?"
            r"(?:time|hours?|\bh\b|ageing|aging|operation|MPPT|maximum power point)",
            re.I,
        ),
        re.compile(
            r"(?:PCE|power conversion efficiency|normalized efficiency).{0,500}?"
            r"(?:time|hours?|\bh\b|ageing|aging|operation|MPPT|maximum power point).{0,700}?"
            r"(?:Fig(?:ure)?\.?\s*[A-Za-z0-9S\.\-]+)",
            re.I,
        ),
        re.compile(
            r"(?:stabilized power output|MPP tracking|MPPT stability|operational stability)"
            r".{0,600}?(?:hours?|\bh\b|time|duration)",
            re.I,
        ),
    ]
    shape_patterns = [
        re.compile(
            r"(?:initial|first|early).{0,80}?"
            r"(?:increase|improvement|rise|gain|enhancement)",
            re.I,
        ),
        re.compile(r"non[- ]?monotonic|burn[- ]?in|metastab", re.I),
        re.compile(
            r"(?:PCE|efficiency|performance).{0,120}?"
            r"(?:recover|regenerat|reversib|self[- ]heal)",
            re.I,
        ),
        re.compile(r"(?:dark rest|light/dark cycl|light soaking)", re.I),
        re.compile(
            r"(?:rapid|fast).{0,50}?decay.{0,180}?(?:slow|linear).{0,50}?decay",
            re.I,
        ),
    ]
    graph = []
    shape = []
    for pattern in graph_patterns:
        match = pattern.search(normalized)
        if match:
            graph.append(excerpt(normalized, match))
    for pattern in shape_patterns:
        match = pattern.search(normalized)
        if match:
            shape.append(excerpt(normalized, match))
    return bool(graph), graph, shape


def download(url: str, destination: Path) -> tuple[str, str]:
    if destination.exists() and destination.stat().st_size > 1000:
        if destination.read_bytes()[:5] == b"%PDF-":
            return "cached_pdf", ""
    temporary = destination.with_suffix(".part")
    if temporary.exists():
        temporary.unlink()
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "120",
        "-A",
        "Mozilla/5.0 academic-literature-audit",
        "-o",
        str(temporary),
        url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        if temporary.exists():
            temporary.unlink()
        return "download_failed", completed.stderr.strip()[:300]
    header = temporary.read_bytes()[:5]
    if header != b"%PDF-":
        size = temporary.stat().st_size
        temporary.unlink()
        return "not_a_pdf", f"downloaded_bytes={size}"
    temporary.replace(destination)
    return "downloaded_pdf", ""


def inspect_pdf(path: Path) -> dict:
    try:
        document = fitz.open(path)
    except Exception as exc:
        return {
            "parse_status": "parse_failed",
            "parse_error": str(exc)[:300],
            "pdf_pages": "",
            "extracted_characters": 0,
            "graph_evidence_pages": "",
            "shape_evidence_pages": "",
            "graph_evidence_excerpt": "",
            "shape_evidence_excerpt": "",
            "machine_curve_figure_evidence": False,
            "machine_nonmonotonic_shape_evidence": False,
        }
    graph_pages = []
    shape_pages = []
    graph_excerpts = []
    shape_excerpts = []
    total_characters = 0
    for index, page in enumerate(document):
        text = page.get_text("text") or ""
        total_characters += len(text)
        graph, graph_found, shape_found = page_evidence(text)
        if graph:
            graph_pages.append(index + 1)
            graph_excerpts.extend(graph_found[:1])
        if shape_found:
            shape_pages.append(index + 1)
            shape_excerpts.extend(shape_found[:1])
    pages = document.page_count
    document.close()
    return {
        "parse_status": "full_text_machine_reviewed",
        "parse_error": "",
        "pdf_pages": pages,
        "extracted_characters": total_characters,
        "graph_evidence_pages": ";".join(map(str, graph_pages)),
        "shape_evidence_pages": ";".join(map(str, shape_pages)),
        "graph_evidence_excerpt": " || ".join(graph_excerpts[:3]),
        "shape_evidence_excerpt": " || ".join(shape_excerpts[:3]),
        "machine_curve_figure_evidence": bool(graph_pages),
        "machine_nonmonotonic_shape_evidence": bool(shape_pages),
    }


def write_csv(rows: list[dict]) -> None:
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    candidates = list(csv.DictReader(INPUT.open(encoding="utf-8")))
    candidates = [
        row
        for row in candidates
        if row.get("pdf_url") and row.get("is_oa") == "True"
    ]
    candidates.sort(
        key=lambda row: (
            row.get("nonmonotonic_recovery_flag") != "True",
            -int(row.get("screen_score") or 0),
            -int(row.get("publication_year") or 0),
            -int(row.get("cited_by_count") or 0),
        )
    )

    audit = []
    successes = 0
    for candidate in candidates[:MAX_ATTEMPTS]:
        doi = candidate["doi"]
        destination = PDF_DIR / f"{slug_doi(doi)}.pdf"
        download_status, download_error = download(candidate["pdf_url"], destination)
        row = {
            "doi": doi,
            "title": candidate["title"],
            "publication_year": candidate["publication_year"],
            "journal": candidate["journal"],
            "landing_page_url": candidate["landing_page_url"],
            "pdf_source_url": candidate["pdf_url"],
            "local_pdf": str(destination) if destination.exists() else "",
            "download_status": download_status,
            "download_error": download_error,
            "title_abstract_nonmonotonic_flag": candidate[
                "nonmonotonic_recovery_flag"
            ],
            "title_abstract_nonmonotonic_hits": candidate[
                "nonmonotonic_recovery_hits"
            ],
        }
        if destination.exists():
            row.update(inspect_pdf(destination))
            if row["parse_status"] == "full_text_machine_reviewed":
                successes += 1
        else:
            row.update(
                {
                    "parse_status": "not_reviewed_no_pdf",
                    "parse_error": "",
                    "pdf_pages": "",
                    "extracted_characters": 0,
                    "graph_evidence_pages": "",
                    "shape_evidence_pages": "",
                    "graph_evidence_excerpt": "",
                    "shape_evidence_excerpt": "",
                    "machine_curve_figure_evidence": False,
                    "machine_nonmonotonic_shape_evidence": False,
                }
            )
        audit.append(row)
        write_csv(audit)
        if successes >= TARGET_SUCCESS:
            break

    counts = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "OpenAlex-marked OA PDFs within A_curve_likely_original; title/abstract "
            "recovery candidates first, then screen score/year/citations."
        ),
        "target_successful_full_text_machine_reviews": TARGET_SUCCESS,
        "download_attempts": len(audit),
        "valid_pdfs_parsed": sum(
            row["parse_status"] == "full_text_machine_reviewed" for row in audit
        ),
        "failed_or_non_pdf": sum(
            row["parse_status"] != "full_text_machine_reviewed" for row in audit
        ),
        "machine_curve_figure_evidence": sum(
            str(row["machine_curve_figure_evidence"]) == "True" for row in audit
        ),
        "machine_nonmonotonic_shape_evidence": sum(
            str(row["machine_nonmonotonic_shape_evidence"]) == "True" for row in audit
        ),
        "important_scope_note": (
            "PDF text/caption inspection is not visual verification of plotted curve shape."
        ),
    }
    MANIFEST.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
