#!/usr/bin/env python3
"""Render high-scoring PCE-time pages for a human visual review.

The scoring only routes pages for inspection.  It does not assign curve shapes.
Those labels must be entered after looking at the rendered plots.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "fulltext_machine_audit.csv"
OUT = ROOT / "figure_review" / "routed_pages"
SHEETS = ROOT / "figure_review" / "contact_sheets"


def page_score(text: str) -> int:
    compact = re.sub(r"\s+", " ", text)
    score = 0
    weighted = [
        (r"normalized\s+PCE", 10),
        (r"PCE\s*\(%\)", 9),
        (r"Time\s*\((?:h|hour|hours)\)", 12),
        (r"operational stability", 9),
        (r"maximum[- ]power[- ]point|MPP tracking|MPPT", 7),
        (r"continuous (?:operation|illumination|light)", 5),
        (r"light[- ]dark|light cycling|dark recovery", 6),
        (r"retained|retention|decreased|decay|degradation", 3),
        (r"Fig(?:ure)?\.?\s*\d+", 2),
    ]
    for pattern, weight in weighted:
        score += min(len(re.findall(pattern, compact, re.I)), 4) * weight
    return score


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def render_page(document: fitz.Document, index: int, destination: Path) -> None:
    pixmap = document[index].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    pixmap.save(destination)


def make_sheets(records: list[dict]) -> None:
    SHEETS.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 760, 980
    cell_w, cell_h = 800, 1060
    per_sheet = 6
    for sheet_index in range(math.ceil(len(records) / per_sheet)):
        subset = records[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        canvas = Image.new("RGB", (cell_w * 3, cell_h * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for position, record in enumerate(subset):
            row, col = divmod(position, 3)
            image = Image.open(record["rendered_page"]).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = col * cell_w + (cell_w - image.width) // 2
            y = row * cell_h + 60
            canvas.paste(image, (x, y))
            label = f"{record['doi']}  PDF p.{record['pdf_page']}  score={record['routing_score']}"
            draw.text((col * cell_w + 12, row * cell_h + 18), label, fill="black")
        canvas.save(SHEETS / f"contact_sheet_{sheet_index + 1:02d}.jpg", quality=88)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(AUDIT.open(encoding="utf-8")))
    routed = []
    for row in rows:
        if row.get("parse_status") != "full_text_machine_reviewed":
            continue
        pdf_path = Path(row["local_pdf"])
        if not pdf_path.exists():
            continue
        document = fitz.open(pdf_path)
        scores = []
        for index, page in enumerate(document):
            score = page_score(page.get_text("text") or "")
            scores.append((score, index))
        top = sorted(scores, reverse=True)[:2]
        for score, index in top:
            destination = OUT / f"{slug(row['doi'])}__page_{index + 1:02d}.png"
            render_page(document, index, destination)
            routed.append(
                {
                    "doi": row["doi"],
                    "publication_year": row["publication_year"],
                    "title": row["title"],
                    "pdf_page": index + 1,
                    "routing_score": score,
                    "rendered_page": str(destination),
                }
            )
        document.close()

    routed.sort(key=lambda item: (-int(item["publication_year"]), item["doi"], -item["routing_score"]))
    with (ROOT / "figure_review" / "routed_pages_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(routed[0]))
        writer.writeheader()
        writer.writerows(routed)
    make_sheets(routed)
    print(f"rendered={len(routed)} sheets={math.ceil(len(routed) / 6)}")


if __name__ == "__main__":
    main()
