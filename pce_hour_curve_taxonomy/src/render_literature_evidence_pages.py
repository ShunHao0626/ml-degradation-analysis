#!/usr/bin/env python3
"""Render selected literature pages for visual evidence checking."""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "08_literature_review/open_access_papers"
OUT = ROOT / "08_literature_review/evidence_pages"

# Page numbers are one-based and were chosen after text-term localization.
SELECTIONS = [
    ("10.1002_aenm.202304452.pdf", 2, "2024 outdoor light-soaking recovery"),
    ("10.1002_aenm.202501906.pdf", 2, "2025 four-year outdoor seasonality"),
    ("10.1038_s41586-024-08161-x.pdf", 3, "2024 continuous versus cycling decay"),
    ("10.1021_acsaem.1c00588.pdf", 4, "2021 bias-dependent degradation/recovery"),
    ("10.1002_adma.202110239.pdf", 4, "2022 self-healing/light-soaking"),
    ("10.1038_s41467-023-40585-3.pdf", 4, "2023 SOM curve-shape clustering"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for filename, page_number, label in SELECTIONS:
        document = fitz.open(PAPERS / filename)
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        output = OUT / f"{Path(filename).stem}_page_{page_number:02d}.png"
        pixmap.save(output)
        image = Image.open(output).convert("RGB")
        target_width = 640
        target_height = round(image.height * target_width / image.width)
        image = image.resize((target_width, target_height))
        card = Image.new("RGB", (target_width, target_height + 48), "white")
        card.paste(image, (0, 48))
        draw = ImageDraw.Draw(card)
        draw.text((12, 14), f"{label} | page {page_number}", fill="black")
        cards.append(card)
    cols = 2
    rows = (len(cards) + cols - 1) // cols
    width = max(card.width for card in cards) * cols
    row_heights = []
    for row in range(rows):
        row_heights.append(max(card.height for card in cards[row * cols : (row + 1) * cols]))
    sheet = Image.new("RGB", (width, sum(row_heights)), "#dddddd")
    y = 0
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if index < len(cards):
                sheet.paste(cards[index], (col * cards[index].width, y))
        y += row_heights[row]
    sheet.save(OUT / "literature_evidence_contact_sheet.png", quality=92)


if __name__ == "__main__":
    main()
