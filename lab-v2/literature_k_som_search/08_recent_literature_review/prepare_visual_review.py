#!/usr/bin/env python3
"""Select and render one high-signal stability-figure page per parsed PDF."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "fulltext_machine_audit.csv"
OUT_DIR = ROOT / "figure_review" / "candidate_pages"
QUEUE = ROOT / "figure_review" / "visual_review_queue.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180]


def page_score(text: str) -> int:
    score = 0
    rules = [
        (r"Time\s*\(\s*h\s*\)", 15),
        (r"(?:maximum power point|MPP|MPPT)\s*(?:tracking)?", 8),
        (r"operational stability|stabilized power output", 7),
        (r"(?:\d[\d,\.]*\s*h\b|hours?)", 4),
        (r"(?:PCE|power conversion efficiency|normalized efficiency)", 3),
        (r"(?:Fig(?:ure)?\.?\s*[A-Za-z0-9S\.\-]+)", 2),
        (r"continuous (?:illumination|operation)|ISOS[- ]L", 5),
        (r"recover|initial increase|burn[- ]?in|non[- ]?monotonic", 4),
    ]
    for pattern, weight in rules:
        score += weight * min(len(re.findall(pattern, text, re.I)), 3)
    return score


def best_caption(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", text)
    options = []
    for match in re.finditer(r"Fig(?:ure)?\.?\s*([A-Za-z0-9S\.\-]+)\s*[|:]?", normalized, re.I):
        piece = normalized[match.start() : match.start() + 1300]
        score = page_score(piece)
        options.append((score, match.group(1), piece[:1200]))
    if not options:
        return "", ""
    options.sort(reverse=True)
    _, number, piece = options[0]
    return number, piece


def main() -> None:
    records = [
        row
        for row in csv.DictReader(AUDIT.open(encoding="utf-8"))
        if row["parse_status"] == "full_text_machine_reviewed"
    ]
    queue = []
    for record in records:
        pdf_path = Path(record["local_pdf"])
        document = fitz.open(pdf_path)
        scored = []
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            scored.append((page_score(text), index, text))
        score, index, text = max(scored)
        figure_number, caption = best_caption(text)
        output = OUT_DIR / f"{record['publication_year']}_{slug(record['doi'])}_p{index+1}.png"
        if not output.exists():
            page = document[index]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
            pixmap.save(output)
        document.close()
        queue.append(
            {
                "doi": record["doi"],
                "year": record["publication_year"],
                "title": record["title"],
                "selected_pdf_page": index + 1,
                "selection_score": score,
                "machine_figure_number_guess": figure_number,
                "machine_caption_excerpt": caption,
                "rendered_page": str(output),
                "human_visual_review_status": "pending",
                "human_figure_number": "",
                "human_time_axis_is_hours": "",
                "human_protocol": "",
                "human_shape": "",
                "human_notes": "",
            }
        )
    fields = list(queue[0])
    with QUEUE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(queue)
    print(f"rendered={len(queue)} queue={QUEUE}")


if __name__ == "__main__":
    main()
