#!/usr/bin/env python3
"""Index evidence-bearing terms in downloaded open-access PDFs."""

from __future__ import annotations

import csv
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "08_literature_review/open_access_papers"
OUTPUT = ROOT / "08_literature_review/evidence_tables/fulltext_term_index.csv"
TERMS = [
    "PCE",
    "degradation curve",
    "clustering",
    "self-organizing map",
    "quantization error",
    "light-soaking",
    "light soaking",
    "recovery",
    "burn-in",
    "maximum power point",
    "MPPT",
    "dark",
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(PAPERS.glob("*.pdf")):
        document = fitz.open(path)
        texts = [page.get_text().lower() for page in document]
        for term in TERMS:
            pages = [str(index + 1) for index, text in enumerate(texts) if term.lower() in text]
            if pages:
                rows.append(
                    {
                        "pdf_file": path.name,
                        "page_count": len(document),
                        "term": term,
                        "pages_one_based": ";".join(pages),
                    }
                )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pdf_file", "page_count", "term", "pages_one_based"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
