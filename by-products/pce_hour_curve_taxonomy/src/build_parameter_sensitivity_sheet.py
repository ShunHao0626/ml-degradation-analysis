#!/usr/bin/env python3
"""Compare paper-documented SOM hyperparameters at independently selected k."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04_literature_parameter_sweeps/published_parameter_contact_sheet.png"
WINDOW_K = {150: 5, 300: 5, 500: 5}
CONFIGS = [
    ("baseline", ROOT / "05_cluster_selection/baseline", "sigma=0.5, lr=0.1"),
    (
        "sigma0p3_lr0p1",
        ROOT / "04_literature_parameter_sweeps/sigma0p3_lr0p1",
        "sigma=0.3, lr=0.1",
    ),
    (
        "sigma0p5_lr0p3",
        ROOT / "04_literature_parameter_sweeps/sigma0p5_lr0p3",
        "sigma=0.5, lr=0.3",
    ),
]


def main() -> None:
    rows = []
    for _, base, label in CONFIGS:
        cards = []
        for window, k in WINDOW_K.items():
            source = base / f"{window}h/k{k:02d}/clusters.png"
            image = Image.open(source).convert("RGB")
            width = 620
            height = round(image.height * width / image.width)
            image = image.resize((width, height))
            card = Image.new("RGB", (width, height + 44), "white")
            card.paste(image, (0, 44))
            ImageDraw.Draw(card).text(
                (12, 13), f"{window} h | selected k={k} | {label}", fill="black"
            )
            cards.append(card)
        rows.append(cards)
    col_width = max(card.width for row in rows for card in row)
    row_heights = [max(card.height for card in row) for row in rows]
    sheet = Image.new("RGB", (col_width * 3, sum(row_heights)), "#d8d8d8")
    y = 0
    for row, row_height in zip(rows, row_heights):
        for col, card in enumerate(row):
            sheet.paste(card, (col * col_width, y))
        y += row_height
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=93)


if __name__ == "__main__":
    main()
