#!/usr/bin/env python3
"""Convert two-column PDF to readable Markdown using PyMuPDF.

Strategy:
- For each page, get text blocks with bounding boxes.
- Detect 2-column layout by checking x0 bimodality.
- Sort columns top-down, then y-then-x within each column.
- Read each column fully, then merge columns by interleaving rows
  according to their physical y position.
- Dehyphenate line breaks and collapse multi-line paragraphs.
"""
import fitz
import re
from pathlib import Path

pdf_path = Path("/Users/shunhao/Desktop/ML/thesis/paper/work.pdf")
out_path = Path("/Users/shunhao/Desktop/ML/thesis/paper/work.md")

doc = fitz.open(pdf_path)


def get_blocks(page):
    blocks = page.get_text("blocks")
    out = []
    for b in blocks:
        x0, y0, x1, y1, text, bno, btype = b
        if btype != 0:
            continue
        text = text.strip()
        if not text:
            continue
        out.append((x0, y0, x1, y1, text))
    return out


def detect_column_divider(blocks, page_width):
    xs = [b[0] for b in blocks]
    if len(xs) < 4:
        return None
    xs_sorted = sorted(set(round(x, 0) for x in xs))
    if len(xs_sorted) < 4:
        return None
    # Hypothesis: divider sits at the largest gap between distinct x0 values
    gaps = []
    for i in range(len(xs_sorted) - 1):
        gap = xs_sorted[i+1] - xs_sorted[i]
        if gap > 50:
            gaps.append((gap, (xs_sorted[i] + xs_sorted[i+1]) / 2))
    if not gaps:
        return None
    gaps.sort(reverse=True)
    return gaps[0][1]


def split_columns(blocks, divider):
    left = [b for b in blocks if b[0] < divider]
    right = [b for b in blocks if b[0] >= divider]
    return left, right


def block_paragraphs(blocks):
    """Take column blocks and produce list of paragraphs from their text."""
    paras = []
    for b in blocks:
        text = b[4]
        # Replace newlines with spaces; collapse
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        # De-hyphenate: "stabil-\nity" -> "stability"
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        paras.append(text)
    return paras


def interleave_columns(left_paras, right_paras, left_blocks, right_blocks):
    """Merge column paragraphs based on y position of their first block."""
    # Pair each paragraph with the y0 of its first block
    left_pairs = []
    for p, blocks in zip(left_paras, [left_blocks[i:i+1] for i in range(len(left_blocks))]):
        pass
    # Simpler: just pair blocks->paragraphs in order
    out = []
    li = ri = 0
    while li < len(left_blocks) or ri < len(right_blocks):
        ly = left_blocks[li][1] if li < len(left_blocks) else 1e9
        ry = right_blocks[ri][1] if ri < len(right_blocks) else 1e9
        if ly <= ry:
            out.append(left_paras[li])
            li += 1
        else:
            out.append(right_paras[ri])
            ri += 1
    return out


# --- patterns ---------------------------------------------------------------

JOURNAL_NOISE = re.compile(
    r"^Nature Communications\|"
    r"|^Article\s*$"
    r"|^1234567890\(\)\)?$"
    r"|^https://doi\.org/10\.1038/s41467-023-40585-3\s*$"
    r"|^Article\s+https://doi"
)
FIG_RE = re.compile(r"^Fig(?:\.|ure)?\.?\s*\d", re.IGNORECASE)
TABLE_RE = re.compile(r"^Table\s*\d", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^(Results and discussion|Discussion|Methods|Conclusion"
    r"|Dataset description|Cell grouping and relative change in PCE"
    r"|Degradation curve shape clustering|References|Acknowledgements"
    r"|Author contributions|Funding|Competing interests"
    r"|Additional information|Correspondence|Peer review"
    r"|Reprints|Publisher|Open Access|Supplementary"
    r"|Data availability|Code availability|Ageing of solar cells"
    r"|Data analysis)$"
)
EQ_RE = re.compile(r"Relative change in PCE|xMaxAbsScaler")
AFFIL_RE = re.compile(r"^\dHelmholtz|^\dDepartment|^\dThese authors")
RECEIVED_ACCEPTED = re.compile(r"^(Received|Accepted):", re.IGNORECASE)
EMAIL_LINE = re.compile(r"^e-mail:", re.IGNORECASE)
PURE_NUM_RE = re.compile(r"^\d{1,3}$")
CHECK_FOR_UPDATES = re.compile(r"^Check for updates\s*$", re.IGNORECASE)
# Figure axis labels / inline chart text — single symbol or short fragments
AXIS_FRAG_RE = re.compile(
    r"^[\)>\(\.,;:%°\d\s]+$"
    r"|^[a-zA-Z]{1,2}\s*$"
    r"|^Degradation time"
    r"|^Maximum PCE$"
    r"|^PCE @ 150 hours$"
    r"|^Relative frequency$"
    r"|^Time \(hours\)$"
    r"|^Normalised PCE$"
    r"|^ΔPCE, rel \(%\)$"
    r"|^Maximum PCE group \(%\)$"
    r"|^Maximum PCE group mean \(%\)$"
    r"|^ΔPCE, rel mean \(%\)$"
    r"|^Solar energy$"
    r"|^Ein\s*=\s*Eout$"
    r"|^Excess energy$"
    r"|^potentially$"
    r"|^triggering$"
    r"|^degradation$"
    r"|^\(e\.g\. activation$"
    r"|^of degradation$"
    r"|^reactions\) is$"
    r"|^large$"
    r"|^small$"
    r"|^Actual$"
    r"|^energy$"
    r"|^extracted is$"
    r"|^extracted$"
    r"|^is large$"
    r"|^is small$"
    r"|^Other energy$"
    r"|^losses \(e\.g\.$"
    r"|^transmission\)$"
    r"|^Low-efficiency perovskite solar cells$"
    r"|^High-efficiency perovskite solar cells$"
    r"|^Solar energy$"
    r"|^Solar energy$"
    r"|^Cluster$"
    r"|^Cluster 1:.+"
    r"|^Cluster 2:.+"
    r"|^Cluster 3:.+"
    r"|^Cluster 4:.+"
    r"|^Relative frequency$"
    r"|^150$"
    r"|^>\s*19\.2$"
    r"|^<\s*10\.4$"
    r"|^16\.8-19\.2$"
    r"|^10\.4-14\.2$"
    r"|^14\.2-16\.8$"
    r"|^>\s*19\.2$"
    r"|^0\s*$"
    r"|^50\s*$"
    r"|^100\s*$"
    r"|^11\s*$"
    r"|^10\s*$"
    r"|^14\s*$"
    r"|^16\s*$"
    r"|^18\s*$"
    r"|^20\s*$"
    r"|^12\s*$"
    r"|^13\s*$"
    r"|^5\s*$"
    r"|^15\s*$"
    r"|^20\s*$"
    r"|^40\s*$"
    r"|^30\s*$"
    r"|^Ein\s*$"
    r"|^Eout\s*$"
    r"|^Ein = Eout\s*$"
    r"|^a\s*$"
    r"|^b\s*$"
    r"|^c\s*$"
    r"|^d\s*$"
    r"|^a \||^b \|"
    r"|^> 19\.2$"
)
# Authors line pattern
AUTHORS_RE = re.compile(
    r"^Noor Titan Putri Hartono.*Antonio Abate"
)


def render_paragraph(p):
    s = p.strip()
    if not s:
        return []
    out = []
    # Detect compound section headers: "Results and discussion Dataset description We..."
    if re.match(r"^(Results and discussion|Methods|Dataset description|Cell grouping|Conclusions|Acknowledgements|References|Introduction|Degradation curve shape clustering|Data availability|Code availability|Author contributions|Funding|Competing interests|Additional information|Peer review information|Reprints and permissions information|Publisher|Open Access)", s) and not s.startswith("###"):
        # Try to identify section headers at the start
        section_splits = [
            "Results and discussion",
            "Dataset description",
            "Cell grouping and relative change in PCE",
            "Degradation curve shape clustering",
            "Methods",
            "Ageing of solar cells",
            "Data analysis",
            "Data availability",
            "Code availability",
            "Acknowledgements",
            "Author contributions",
            "Funding",
            "Competing interests",
            "Additional information",
            "Supplementary information",
            "References",
            "Peer review information",
            "Publisher’s note",
            "Reprints and permissions information is available",
            "Reprints and permissions information",
            "Correspondence",
        ]
        # Sort splits by length (longest first) to avoid greedy shorter matches
        section_splits_local = sorted(section_splits, key=len, reverse=True)
        cursor = 0
        emitted = []
        body = s
        found_any = False
        # Greedy search: find a section header at the start of the remaining body
        while True:
            original = body
            for sh in section_splits_local:
                if body.startswith(sh):
                    emitted.append(("section", sh))
                    body = body[len(sh):].lstrip()
                    found_any = True
                    break
            if body == original:
                break
        if found_any:
            for kind, text in emitted:
                out.append("### " + text)
            if body:
                out.append(body)
            return out
        out.append("### " + s)
        return out
    if SECTION_RE.match(s):
        out.append("### " + s)
        return out
    if FIG_RE.match(s) or TABLE_RE.match(s):
        if "|" in s:
            title, body = s.split("|", 1)
            out.append("> **" + title.strip() + "**")
            out.append(">")
            for bl in body.strip().split(" | "):
                out.append("> " + bl.strip())
        else:
            out.append("> " + s)
        return out
    if EQ_RE.search(s) and ("=" in s or "max" in s):
        out.append("```")
        out.append(s)
        out.append("```")
        return out
    if AFFIL_RE.match(s) or "Helmholtz-Zentrum" in s or "Department of Business Informatics" in s or "These authors contributed equally" in s:
        out.append("> " + s)
        return out
    if RECEIVED_ACCEPTED.match(s):
        out.append("*" + s + "*")
        return out
    if PURE_NUM_RE.match(s):
        return []
    if AXIS_FRAG_RE.match(s):
        return []

    # Detect references paragraphs that combine many numbered references
    if re.match(r"^References?\s+\d", s) or re.match(r"^\d+\.\s+[A-Z]", s) and "." in s[:8]:
        # Split by numbered pattern
        parts = re.split(r"(?=\b\d{1,2}\.\s+[A-Z])", s)
        out = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("References"):
                out.append("### References")
                # strip the leading "References N." pattern
                m = re.match(r"^References?\s+(\d+\..*)", part)
                if m:
                    out.append(m.group(1))
                continue
            out.append(part)
        return out

    out.append(s)
    return out


# --- assemble document ------------------------------------------------------

lines = []
lines.append("# Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset")
lines.append("")
lines.append("> Extracted from `work.pdf` (7 pages) — *Nature Communications* **14**:4869 (2023). DOI: 10.1038/s41467-023-40585-3")
lines.append("")
lines.append("**Authors:** Noor Titan Putri Hartono, Hans Köbler, Paolo Graniero, Mark Khenkin, Rutger Schlatmann, Carolin Ulbrich & Antonio Abate")
lines.append("")
lines.append("**Affiliations:**")
lines.append("")
lines.append("1. Helmholtz-Zentrum Berlin für Materialien und Energie, 14109 Berlin, Germany")
lines.append("2. Department of Business Informatics, Freie Universität Berlin, 14195 Berlin, Germany")
lines.append("")
lines.append("**Contact:** titan.hartono@helmholtz-berlin.de; antonio.abate@helmholtz-berlin.de")
lines.append("")
lines.append("---")
lines.append("")

title_consumed = False
for i, page in enumerate(doc, start=1):
    lines.append(f"## Page {i}")
    lines.append("")
    blocks = get_blocks(page)
    divider = detect_column_divider(blocks, page.rect.width)
    if divider:
        left, right = split_columns(blocks, divider)
        # Sort each column top-down
        left.sort(key=lambda b: (b[1], b[0]))
        right.sort(key=lambda b: (b[1], b[0]))
        left_paras = block_paragraphs(left)
        right_paras = block_paragraphs(right)
        page_paras = interleave_columns(left_paras, right_paras, left, right)
    else:
        b_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
        page_paras = block_paragraphs(b_sorted)

    for p in page_paras:
        if not p:
            continue
        # Drop journal running headers / footers
        if "Nature Communications|" in p and "4869" in p:
            continue
        if p.strip().startswith("Article https://doi.org"):
            continue
        if p.strip().startswith("https://doi.org/10.1038/s41467-023-40585-3"):
            continue
        if AUTHORS_RE.match(p):
            continue
        if i == 1 and not title_consumed and "Stability follows" in p:
            title_consumed = True
            continue
        if i == 1 and not title_consumed and "ageing dataset" in p:
            title_consumed = True
            continue
        if p.strip().startswith("e-mail:"):
            continue
        if "Helmholtz-Zentrum Berlin" in p and "Department of Business Informatics" in p:
            continue
        if CHECK_FOR_UPDATES.match(p.strip()):
            continue
        lines.extend(render_paragraph(p))
        lines.append("")
    lines.append("---")
    lines.append("")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes, {doc.page_count} pages)")
