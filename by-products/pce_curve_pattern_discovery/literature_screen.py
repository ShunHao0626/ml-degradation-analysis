#!/usr/bin/env python3
"""Merge, deduplicate, and screen the bulk Crossref/OpenAlex retrieval."""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
LIT = ROOT / "07_literature"
CROSSREF = LIT / "raw_crossref"
OPENALEX = LIT / "raw_openalex"


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def strip_markup(value: Any) -> str:
    raw = html.unescape(text(value))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()


def normalize_doi(value: Any) -> str:
    doi = text(value).lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().rstrip(".,;)")


def normalize_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", text(value).lower())


def crossref_date(item: dict[str, Any]) -> str:
    for key in ("published-online", "published-print", "published", "issued", "created"):
        part = item.get(key) or {}
        values = part.get("date-parts") or []
        if values and values[0]:
            bits = [int(v) for v in values[0][:3]]
            return "-".join([f"{bits[0]:04d}"] + [f"{v:02d}" for v in bits[1:]])
        if part.get("date-time"):
            return text(part["date-time"])[:10]
    return ""


def abstract_from_inverted(index: Any) -> str:
    if not isinstance(index, dict) or not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, places in index.items():
        if isinstance(places, list):
            positions.extend((int(pos), str(word)) for pos in places)
    return " ".join(word for _, word in sorted(positions))


def openalex_authors(item: dict[str, Any]) -> str:
    names = []
    for authorship in item.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            names.append(text(author["display_name"]))
    return "; ".join(names)


def parse_crossref() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for path in sorted(CROSSREF.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        message = payload.get("message") or {}
        items = message.get("items") or []
        queries.append(
            {
                "source": "Crossref",
                "query_id": path.stem,
                "service_total_results": message.get("total-results"),
                "retrieved": len(items),
                "raw_file": path.relative_to(ROOT).as_posix(),
            }
        )
        for rank, item in enumerate(items, start=1):
            titles = item.get("title") or []
            containers = item.get("container-title") or []
            authors = []
            for author in item.get("author") or []:
                name = " ".join(v for v in [text(author.get("given")), text(author.get("family"))] if v)
                if name:
                    authors.append(name)
            records.append(
                {
                    "metadata_source": "Crossref",
                    "query_ids": f"cr_{path.stem}",
                    "best_query_rank": rank,
                    "openalex_id": "",
                    "doi": normalize_doi(item.get("DOI")),
                    "title": text(titles[0] if titles else ""),
                    "abstract": strip_markup(item.get("abstract")),
                    "publication_date": crossref_date(item),
                    "year": int(crossref_date(item)[:4]) if crossref_date(item)[:4].isdigit() else None,
                    "type": text(item.get("type")),
                    "venue": text(containers[0] if containers else ""),
                    "publisher": text(item.get("publisher")),
                    "authors": "; ".join(authors),
                    "cited_by_count": item.get("is-referenced-by-count") or 0,
                    "landing_url": text(item.get("URL")),
                    "open_pdf_url": "",
                    "is_oa": False,
                }
            )
    return records, queries


def parse_openalex() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for path in sorted(OPENALEX.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("results") or []
        meta = payload.get("meta") or {}
        queries.append(
            {
                "source": "OpenAlex",
                "query_id": path.stem,
                "service_total_results": meta.get("count"),
                "retrieved": len(items),
                "raw_file": path.relative_to(ROOT).as_posix(),
            }
        )
        for rank, item in enumerate(items, start=1):
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            oa = item.get("open_access") or {}
            records.append(
                {
                    "metadata_source": "OpenAlex",
                    "query_ids": f"oa_{path.stem}",
                    "best_query_rank": rank,
                    "openalex_id": text(item.get("id")),
                    "doi": normalize_doi(item.get("doi")),
                    "title": text(item.get("title")),
                    "abstract": abstract_from_inverted(item.get("abstract_inverted_index")),
                    "publication_date": text(item.get("publication_date")),
                    "year": item.get("publication_year"),
                    "type": text(item.get("type")),
                    "venue": text(source.get("display_name")),
                    "publisher": "",
                    "authors": openalex_authors(item),
                    "cited_by_count": item.get("cited_by_count") or 0,
                    "landing_url": text(location.get("landing_page_url")) or text(item.get("doi")),
                    "open_pdf_url": text(location.get("pdf_url")),
                    "is_oa": bool(oa.get("is_oa")),
                }
            )
    return records, queries


def merge_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = f"doi:{row['doi']}" if row["doi"] else f"title:{normalize_title(row['title'])}"
        if key not in {"doi:", "title:"}:
            groups[key].append(row)

    merged = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: (bool(r["abstract"]), bool(r["open_pdf_url"]), r["cited_by_count"]), reverse=True)
        base = dict(rows[0])
        for field in ("doi", "title", "abstract", "publication_date", "year", "type", "venue", "publisher", "authors", "landing_url", "open_pdf_url", "openalex_id"):
            if not base.get(field):
                base[field] = next((r[field] for r in rows if r.get(field)), base.get(field))
        base["metadata_source"] = ";".join(sorted({r["metadata_source"] for r in rows}))
        base["query_ids"] = ";".join(sorted({qid for r in rows for qid in r["query_ids"].split(";")}))
        base["best_query_rank"] = min(int(r["best_query_rank"]) for r in rows)
        base["cited_by_count"] = max(int(r["cited_by_count"] or 0) for r in rows)
        base["is_oa"] = any(bool(r["is_oa"]) for r in rows)
        base["dedup_key"] = key
        merged.append(base)
    return pd.DataFrame(merged)


def screen(frame: pd.DataFrame) -> pd.DataFrame:
    perovskite = re.compile(r"\bperovskit", re.I)
    pv = re.compile(r"solar cell|photovoltaic|power conversion efficien|\bpce\b|\bpsc(?:s)?\b", re.I)
    stability = re.compile(r"stabil|degrad|ageing|aging|lifetime|operational|durab|reliab", re.I)
    curve = re.compile(r"maximum power point|\bmppt\b|time[- ]series|trajectory|curve|temporal|over time|hours?|\bt80\b|\bt90\b", re.I)
    nonmonotonic = re.compile(r"recover|reversib|light[- ]soak|burn[- ]in|initial gain|heal|restoration|diurnal|day.?night|self[- ]repair|regener", re.I)
    clustering = re.compile(r"cluster|self[- ]organi[sz]ing map|\bsom\b|unsupervised|shape", re.I)
    review = re.compile(r"\breview\b|perspective|roadmap|overview|progress in|recent advances", re.I)

    rows = []
    for row in frame.to_dict("records"):
        title = text(row.get("title"))
        abstract = text(row.get("abstract"))
        blob = f"{title} {abstract}"
        core = bool(perovskite.search(blob) and pv.search(blob))
        has_stability = bool(stability.search(blob))
        has_curve = bool(curve.search(blob))
        has_nonmonotonic = bool(nonmonotonic.search(blob))
        has_clustering = bool(clustering.search(blob))
        is_review = bool(review.search(title))
        score = 0
        score += 12 if core else 0
        score += 5 if has_stability else 0
        score += 5 if has_curve else 0
        score += 6 if has_nonmonotonic else 0
        score += 6 if has_clustering else 0
        score += 2 if re.search(r"\b\d[\d,.]*\s*h(?:our)?s?\b", blob, re.I) else 0
        score += min(4, int(math.log10(1 + int(row.get("cited_by_count") or 0)) * 2))
        if not core:
            status = "D_exclude_not_core_perovskite_PV"
        elif is_review:
            status = "C_review_or_perspective"
        elif has_stability and has_curve:
            status = "A_curve_likely_original"
        else:
            status = "B_perovskite_stability_related"
        row.update(
            {
                "screen_status": status,
                "relevance_score": score,
                "core_perovskite_pv": core,
                "has_stability_terms": has_stability,
                "has_curve_time_terms": has_curve,
                "has_nonmonotonic_terms": has_nonmonotonic,
                "has_clustering_terms": has_clustering,
                "is_review_title": is_review,
                "doi_url": f"https://doi.org/{row['doi']}" if row.get("doi") else "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["relevance_score", "cited_by_count", "year"], ascending=[False, False, False]
    ).reset_index(drop=True)


def write_report(frame: pd.DataFrame, queries: list[dict[str, Any]]) -> None:
    raw_count = sum(q["retrieved"] for q in queries)
    statuses = Counter(frame["screen_status"])
    priority = frame[frame["screen_status"] == "A_curve_likely_original"]
    nonmono = frame[
        frame["core_perovskite_pv"]
        & frame["has_stability_terms"]
        & frame["has_nonmonotonic_terms"]
    ]
    open_priority = priority[priority["open_pdf_url"].astype(bool)]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_filter": {"from": "2010-01-01", "to": "2026-09-03"},
        "method": {
            "retrieval": "Top 200 relevance-ranked records for each of 8 queries from both Crossref and OpenAlex.",
            "deduplication": "Normalized DOI primary key; normalized title fallback.",
            "screening": "Deterministic title/abstract keyword triage; not a substitute for figure-level full-text review.",
        },
        "queries": queries,
        "counts": {
            "raw_records_with_duplicates": raw_count,
            "deduplicated_records": len(frame),
            "with_doi": int(frame["doi"].astype(bool).sum()),
            "with_abstract": int(frame["abstract"].astype(bool).sum()),
            "high_priority_curve_likely": len(priority),
            "nonmonotonic_recovery_priority": len(nonmono),
            "high_priority_with_open_pdf_url": len(open_priority),
            "screen_status": dict(statuses),
        },
    }
    (LIT / "retrieval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    query_table = "\n".join(
        f"- {q['source']} / `{q['query_id']}`：返回 {q['retrieved']} 条，服务报告总量 {q['service_total_results']}。"
        for q in queries
    )
    report = f"""# 批量文献检索报告

检索覆盖 2010-01-01 至 2026-09-03。Crossref 与 OpenAlex 各运行 8 组检索式，每组抓取
相关度排序前 200 条。原始带重复记录 {raw_count} 条，按 DOI（无 DOI 时按规范化题名）去重后
{len(frame)} 条；其中高优先级 PCE/效率—时间或稳定性曲线候选 {len(priority)} 条，包含恢复、
可逆、light-soaking、burn-in 或昼夜行为关键词的重点候选 {len(nonmono)} 条。

## 重要边界

此阶段只进行元数据、摘要和开放全文链接抓取。关键词命中不等于图中真的存在所需拓扑；
必须继续检查全文图与补充材料。付费墙内容不绕过访问控制，仅保存 DOI/出版社页面。

## 每个查询的抓取状态

{query_table}

## 输出

- `all_records_deduplicated.csv`：完整去重目录；
- `high_priority_curve_papers.csv`：高优先级曲线文献；
- `nonmonotonic_recovery_priority.csv`：Bridge/Hill/Valley 更相关的非单调候选；
- `open_fulltext_download_queue.csv`：可公开下载的全文队列；
- `retrieval_manifest.json`：检索与计数审计。
"""
    (LIT / "LITERATURE_SEARCH_REPORT_CN.md").write_text(report, encoding="utf-8")


def main() -> None:
    crossref_records, crossref_queries = parse_crossref()
    openalex_records, openalex_queries = parse_openalex()
    frame = screen(merge_records(crossref_records + openalex_records))
    frame.to_csv(LIT / "all_records_deduplicated.csv", index=False)
    core = frame[frame["core_perovskite_pv"]]
    core.to_csv(LIT / "screened_core_perovskite_pv.csv", index=False)
    priority = frame[frame["screen_status"] == "A_curve_likely_original"]
    priority.to_csv(LIT / "high_priority_curve_papers.csv", index=False)
    nonmono = frame[
        frame["core_perovskite_pv"]
        & frame["has_stability_terms"]
        & frame["has_nonmonotonic_terms"]
    ]
    nonmono.to_csv(LIT / "nonmonotonic_recovery_priority.csv", index=False)
    queue = priority[priority["open_pdf_url"].astype(bool)].head(50).copy()
    queue.insert(0, "download_id", [f"oa_{i:03d}" for i in range(1, len(queue) + 1)])
    queue[["download_id", "doi", "title", "year", "open_pdf_url", "landing_url", "relevance_score"]].to_csv(
        LIT / "open_fulltext_download_queue.csv", index=False
    )
    write_report(frame, crossref_queries + openalex_queries)
    print(json.dumps(json.loads((LIT / "retrieval_manifest.json").read_text(encoding="utf-8"))["counts"], indent=2))


if __name__ == "__main__":
    main()
