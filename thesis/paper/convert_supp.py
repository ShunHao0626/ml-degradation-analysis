#!/usr/bin/env python3
"""Convert Supplementary.pdf to readable Markdown with two-column detection."""
import fitz
import re
from pathlib import Path

pdf_path = Path("/Users/shunhao/Desktop/ML/thesis/paper/Supplementary.pdf")
out_path = Path("/Users/shunhao/Desktop/ML/thesis/paper/Supplementary.md")

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
    paras = []
    for b in blocks:
        text = b[4]
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        paras.append(text)
    return paras


def interleave_columns(left_paras, right_paras, left_blocks, right_blocks):
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

# Match noisy page-number artifacts at top/bottom of each page
PAGE_NOISE_RE = re.compile(r"^\d{1,3}\s*$")

FIG_CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Fig(\.|ure|\s)\s*\d",
    re.IGNORECASE,
)
TABLE_CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Table\s*\d",
    re.IGNORECASE,
)
SUPP_NOTE_RE = re.compile(
    r"^(Supplementary\s+(Note|Information|Data)\b)",
    re.IGNORECASE,
)


# --- front-matter header parsing --------------------------------------------

def extract_front_matter(doc):
    """Read the title page and return header fields as a dict."""
    page = doc[0]
    text = page.get_text("text")
    info = {
        "title": "",
        "authors": "",
        "affiliations": [],
        "corresponding": "",
        "equal": "",
    }
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Title
    for i, line in enumerate(lines):
        if line == "Supplementary Information":
            continue
        if line.startswith(("1Department", "1Department")):
            break
        if line.startswith("Supplementary Information"):
            continue
        # Skip page numbers
        if PAGE_NOISE_RE.match(line):
            continue
        if not info["title"] and "Dataset" in line:
            # Title may span multiple lines; keep collecting
            info["title"] = ""
        info["title"] += (" " + line) if info["title"] else line
    # Better: re-extract title by hand
    info["title"] = ""
    capture = False
    for line in lines:
        if "Supplementary Information" in line:
            capture = True
            continue
        if line.startswith("Noor Titan"):
            capture = False
            break
        if capture and not PAGE_NOISE_RE.match(line):
            info["title"] = info["title"] + " " + line if info["title"] else line
    info["title"] = info["title"].strip()
    # Authors
    for line in lines:
        if line.startswith("Noor Titan"):
            info["authors"] = line
            break
    # Affiliations: lines starting with a single digit
    aff = []
    for line in lines:
        if re.match(r"^[123]([A-Z][a-zA-Z]+|Department|Helmholtz)", line):
            aff.append(line)
        elif "Authors contributed equally" in line or "Authors contributed" in line:
            info["equal"] = line
        elif "Corresponding authors" in line:
            info["corresponding"] = line
    info["affiliations"] = aff
    return info


# --- paragraph rendering ----------------------------------------------------

SECTION_SPLITS = [
    "Supplementary Note 1: Data Quality",
    "Supplementary Note 2",
    "Supplementary Note:",
    "Supplementary Notes",
    "Detailed Dataset Description",
    "Degradation Time Length",
    "MPPT Behaviour",
    "SOM Quantisation Error",
    "Influence of band gaps",
    "Device Architecture Impact on Clustering",
    "Correlation between Clusters and Testing Conditions",
    "Data Quality",
    "K-means Clustering",
    "Supplementary References",
    "Methods",
    "Ageing of solar cells",
    "Data analysis",
    "Acknowledgements",
    "References",
    "Author contributions",
    "Funding",
    "Competing interests",
    "Additional information",
    "Supplementary information",
    "Correspondence",
]


def render_paragraph(p):
    s = p.strip()
    if not s:
        return []
    out = []
    if PAGE_NOISE_RE.match(s):
        return []

    # Detect section header runs at start
    if re.match(r"^(Supplementary\s+Note|Detailed|Dataset Description|MPPT|Influence|Acknowledgements|Author contributions|Methods|Ageing|Data analysis|Funding|Competing interests|Additional information|Correspondence|Supplementary information|Data Quality|K-means Clustering|Supplementary References|Correlation between Clusters|Degradation Time Length|Device Architecture Impact|SOM Quantisation)", s):
        section_splits_local = sorted(SECTION_SPLITS, key=len, reverse=True)
        emitted = []
        body = s
        found_any = False
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

    # Figure / Table caption
    if FIG_CAPTION_RE.match(s) or TABLE_CAPTION_RE.match(s):
        if "." in s:
            # Split at first big full stop followed by capital word
            m = re.match(r"^((?:Supplementary\s+)?(?:Fig\.?|Figure|Table)\.?\s*\d+\.?(?:\s*[A-Z][^.]*?\.))\s*(.*)$", s, re.DOTALL)
            if m:
                title = m.group(1).strip()
                rest = m.group(2).strip()
                out.append("> **" + title + "**")
                if rest:
                    out.append(">")
                    # Split body into sentences by '. ' boundary
                    for chunk in re.split(r"(?<=\.)\s+", rest):
                        out.append("> " + chunk.strip())
            else:
                out.append("> " + s)
        else:
            out.append("> " + s)
        return out

    # Equation-like content: only short standalone lines (not paragraphs)
    if "Supplementary Eq." in s and len(s) < 100:
        out.append("```")
        out.append(s)
        out.append("```")
        return out
    # Pure equation-line: text dominantly math symbols, short, isolated
    math_only = re.sub(r"[^\W\d_a-zA-Z\u0370-\u03FF\u1D400-\u1D7FF.,()=+\-*/∑√°]", "", s)
    non_ws_chars = re.sub(r"\s", "", s)
    if 0 < len(non_ws_chars) <= 40 and (non_ws_chars in {"𝑃out", "𝑃in", "QE", "PCE"} or
        re.match(r"^[0-9]+\s*𝑛√∑", s) or
        re.match(r"^[A-Z]{1,3}\s*=\s*$", s)):
        out.append("```")
        out.append(s)
        out.append("```")
        return out

    out.append(s)
    return out


# --- assemble document ------------------------------------------------------

info = extract_front_matter(doc)

lines = []
lines.append("# Supplementary Information")
lines.append("")
if info["title"]:
    lines.append(f"**Title:** {info['title']}")
    lines.append("")
lines.append("> Extracted from `Supplementary.pdf` ({} pages)".format(doc.page_count))
lines.append("")
if info["authors"]:
    lines.append(f"**Authors:** {info['authors']}")
    lines.append("")
if info["affiliations"]:
    lines.append("**Affiliations:**")
    lines.append("")
    for a in info["affiliations"]:
        lines.append(a)
    lines.append("")
if info["equal"]:
    lines.append(f"_{info['equal']}_")
    lines.append("")
if info["corresponding"]:
    lines.append(f"_{info['corresponding']}_")
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
        left.sort(key=lambda b: (b[1], b[0]))
        right.sort(key=lambda b: (b[1], b[0]))
        left_paras = block_paragraphs(left)
        right_paras = block_paragraphs(right)
        page_paras = interleave_columns(left_paras, right_paras, left, right)
    else:
        b_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
        page_paras = block_paragraphs(b_sorted)

    for p in page_paras:
        s = p.strip()
        if not s:
            continue
        # Drop running page numbers like "2  " at start
        if PAGE_NOISE_RE.match(s):
            continue
        # Skip repeating title block on page 1 (already in front matter)
        if i == 1 and not title_consumed and "Dataset" in s and "Ageing" in s:
            title_consumed = True
            continue
        if i == 1 and "Supplementary Information" in s:
            continue
        if i == 1 and "Department Novel Materials" in s:
            continue
        if i == 1 and "PVcomB" in s:
            continue
        if i == 1 and "Department of Business Informatics" in s and "authors contributed" not in s:
            continue
        if i == 1 and "Corresponding authors" in s:
            continue
        if i == 1 and "Authors contributed equally" in s:
            continue
        if i == 1 and re.match(r"^[123]Department|^[123]Helmholtz", s):
            continue
        if i == 1 and s.startswith("Noor Titan"):
            continue
        lines.extend(render_paragraph(p))
        lines.append("")
    lines.append("---")
    lines.append("")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes, {doc.page_count} pages)")
