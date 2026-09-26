#!/usr/bin/env python3
"""Assemble baseline candidate cluster plots for visual-overlap selection audit."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_cluster_selection/baseline"
OUT = ROOT / "05_cluster_selection/baseline_candidate_contact_sheet.png"


def main() -> None:
    windows = (150, 300, 500)
    ks = (3, 4, 5, 6)
    cards = []
    for k in ks:
        row = []
        for window in windows:
            source = BASE / f"{window}h/k{k:02d}/clusters.png"
            image = Image.open(source).convert("RGB")
            width = 600
            height = round(image.height * width / image.width)
            image = image.resize((width, height))
            card = Image.new("RGB", (width, height + 44), "white")
            card.paste(image, (0, 44))
            ImageDraw.Draw(card).text(
                (12, 13), f"{window} h | k={k} | baseline sigma=0.5, lr=0.1", fill="black"
            )
            row.append(card)
        cards.append(row)
    col_width = max(card.width for row in cards for card in row)
    row_heights = [max(card.height for card in row) for row in cards]
    sheet = Image.new("RGB", (col_width * len(windows), sum(row_heights)), "#d8d8d8")
    y = 0
    for row, row_height in zip(cards, row_heights):
        for col, card in enumerate(row):
            sheet.paste(card, (col * col_width, y))
        y += row_height
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=93)


if __name__ == "__main__":
    main()
