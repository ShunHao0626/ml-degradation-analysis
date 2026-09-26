#!/usr/bin/env python3
"""Bulk metadata retrieval and reproducible title/abstract screening.

This script deliberately separates metadata retrieval from full-text/figure review.
It does not alter, train, or evaluate the local SOM model.
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

DATE_FROM = "2023-01-01"
DATE_TO = "2026-12-31"
# OpenAlex returned HTTP 429 after the first two complete 200-record responses.
# Preserve and reuse those complete files; do not repeatedly hit the service.
RETRIEVE_MISSING_OPENALEX = False
USER_AGENT = (
    "perovskite-stability-literature-audit/1.0 "
    "(non-commercial academic metadata review)"
)

OPENALEX_QUERIES = {
    "oa_operational_stability": "perovskite solar cell operational stability",
    "oa_mppt": "perovskite solar cell maximum power point tracking stability",
    "oa_pce_time": "perovskite solar cell PCE time aging hours",
    "oa_long_term": "perovskite photovoltaics long-term stability T80 hours",
    "oa_light_soaking": "perovskite solar cell light soaking recovery stability",
    "oa_reversible": "perovskite solar cell reversible degradation recovery",
    "oa_burn_in": "perovskite photovoltaic burn-in nonmonotonic degradation",
    "oa_isos": "perovskite solar cell ISOS-L operational stability MPPT",
    "oa_outdoor": "perovskite solar cell outdoor stability power conversion efficiency time",
    "oa_metastability": "perovskite solar cell metastability recovery efficiency",
}

CROSSREF_QUERIES = {
    "cr_operational_stability": "perovskite solar cell operational stability",
    "cr_mppt": "perovskite solar cell maximum power point tracking",
    "cr_pce_hours": "perovskite solar cell PCE hours stability",
    "cr_light_soaking": "perovskite solar cell light soaking recovery",
    "cr_reversible": "perovskite solar cell reversible degradation recovery",
    "cr_nonmonotonic": "perovskite solar cell nonmonotonic stability burn-in",
}


def fetch_json(url: str, attempts: int = 8) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 8.0 + attempt * 6.0
                time.sleep(min(wait, 60.0))
                continue
            time.sleep(2.0 + attempt * 3.0)
        except Exception as exc:  # network services can transiently fail
            last_error = exc
            time.sleep(2.0 + attempt * 3.0)
    raise RuntimeError(f"Failed after {attempts} attempts: {url}\n{last_error}")


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(str(value)).strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.strip().rstrip(".,;)")


def clean_markup(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def abstract_from_inverted(index: dict | None) -> str:
    if not index:
        return ""
    positions = []
    for token, offsets in index.items():
        for offset in offsets:
            positions.append((int(offset), token))
    positions.sort()
    return " ".join(token for _, token in positions)


def first(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def date_from_crossref(item: dict) -> str:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = item.get(key, {}).get("date-parts", [])
        if not parts or not parts[0]:
            continue
        values = [int(x) for x in parts[0]]
        return "-".join(
            [f"{values[0]:04d}"]
            + ([f"{values[1]:02d}"] if len(values) > 1 else [])
            + ([f"{values[2]:02d}"] if len(values) > 2 else [])
        )
    return ""


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def openalex_rows() -> tuple[list[dict], list[dict]]:
    rows = []
    manifest = []
    fields = (
        "id,doi,title,publication_year,publication_date,type,cited_by_count,"
        "primary_location,open_access,abstract_inverted_index,authorships"
    )
    for name, query in OPENALEX_QUERIES.items():
        params = {
            "search": query,
            "filter": (
                f"from_publication_date:{DATE_FROM},"
                f"to_publication_date:{DATE_TO}"
            ),
            "per-page": 200,
            "select": fields,
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        raw_path = RAW / f"{name}.json"
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            retrieval_status = "complete_cached_response"
        elif not RETRIEVE_MISSING_OPENALEX:
            manifest.append(
                {
                    "source": "OpenAlex",
                    "query_id": name,
                    "query": query,
                    "service_total_results": None,
                    "retrieved_top_ranked": 0,
                    "request_url": url,
                    "retrieval_status": "not_retrieved_after_API_429",
                }
            )
            continue
        else:
            payload = fetch_json(url)
            raw_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            retrieval_status = "complete_network_response"
        items = payload.get("results", [])
        manifest.append(
            {
                "source": "OpenAlex",
                "query_id": name,
                "query": query,
                "service_total_results": payload.get("meta", {}).get("count"),
                "retrieved_top_ranked": len(items),
                "request_url": url,
                "retrieval_status": retrieval_status,
            }
        )
        for rank, item in enumerate(items, 1):
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            oa = item.get("open_access") or {}
            authors = []
            for authorship in item.get("authorships") or []:
                author = authorship.get("author") or {}
                if author.get("display_name"):
                    authors.append(author["display_name"])
            rows.append(
                {
                    "source": "OpenAlex",
                    "source_query": name,
                    "source_rank": rank,
                    "openalex_id": item.get("id", ""),
                    "doi": normalize_doi(item.get("doi")),
                    "title": clean_markup(item.get("title")),
                    "abstract": clean_markup(
                        abstract_from_inverted(item.get("abstract_inverted_index"))
                    ),
                    "publication_year": item.get("publication_year", ""),
                    "publication_date": item.get("publication_date", ""),
                    "document_type": item.get("type", ""),
                    "journal": source.get("display_name", ""),
                    "publisher": source.get("host_organization_name", ""),
                    "authors": "; ".join(authors),
                    "cited_by_count": item.get("cited_by_count", 0),
                    "landing_page_url": location.get("landing_page_url", ""),
                    "pdf_url": location.get("pdf_url", "") or oa.get("oa_url", ""),
                    "is_oa": bool(oa.get("is_oa")),
                }
            )
        time.sleep(1.25)
    return rows, manifest


def crossref_rows() -> tuple[list[dict], list[dict]]:
    rows = []
    manifest = []
    select = (
        "DOI,title,author,published,published-print,published-online,issued,created,"
        "container-title,URL,abstract,type,is-referenced-by-count,publisher"
    )
    for name, query in CROSSREF_QUERIES.items():
        params = {
            "query.bibliographic": query,
            "filter": (
                f"from-pub-date:{DATE_FROM},until-pub-date:{DATE_TO},"
                "type:journal-article"
            ),
            "rows": 200,
            "select": select,
        }
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        raw_path = RAW / f"{name}.json"
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            retrieval_status = "complete_cached_response"
        else:
            payload = fetch_json(url)
            raw_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            retrieval_status = "complete_network_response"
        message = payload.get("message", {})
        items = message.get("items", [])
        manifest.append(
            {
                "source": "Crossref",
                "query_id": name,
                "query": query,
                "service_total_results": message.get("total-results"),
                "retrieved_top_ranked": len(items),
                "request_url": url,
                "retrieval_status": retrieval_status,
            }
        )
        for rank, item in enumerate(items, 1):
            date = date_from_crossref(item)
            authors = []
            for author in item.get("author") or []:
                full = " ".join(
                    x for x in (author.get("given", ""), author.get("family", "")) if x
                )
                if full:
                    authors.append(full)
            rows.append(
                {
                    "source": "Crossref",
                    "source_query": name,
                    "source_rank": rank,
                    "openalex_id": "",
                    "doi": normalize_doi(item.get("DOI")),
                    "title": clean_markup(first(item.get("title"))),
                    "abstract": clean_markup(item.get("abstract")),
                    "publication_year": date[:4],
                    "publication_date": date,
                    "document_type": item.get("type", ""),
                    "journal": clean_markup(first(item.get("container-title"))),
                    "publisher": clean_markup(item.get("publisher")),
                    "authors": "; ".join(authors),
                    "cited_by_count": item.get("is-referenced-by-count", 0),
                    "landing_page_url": item.get("URL", ""),
                    "pdf_url": "",
                    "is_oa": "",
                }
            )
        time.sleep(1.25)
    return rows, manifest


def merge_rows(rows: list[dict]) -> list[dict]:
    merged = {}
    title_to_key = {}
    for row in rows:
        doi = row["doi"]
        tkey = title_key(row["title"])
        key = f"doi:{doi}" if doi else f"title:{tkey}"
        if doi and tkey in title_to_key and title_to_key[tkey] in merged:
            old_key = title_to_key[tkey]
            if old_key.startswith("title:"):
                existing = merged.pop(old_key)
                merged[key] = existing
        title_to_key[tkey] = key
        if key not in merged:
            base = dict(row)
            base["metadata_sources"] = row["source"]
            base["query_hits"] = row["source_query"]
            base["best_source_rank"] = int(row["source_rank"])
            merged[key] = base
            continue
        current = merged[key]
        current["metadata_sources"] = ";".join(
            sorted(set(current["metadata_sources"].split(";")) | {row["source"]})
        )
        current["query_hits"] = ";".join(
            sorted(set(current["query_hits"].split(";")) | {row["source_query"]})
        )
        current["best_source_rank"] = min(
            int(current["best_source_rank"]), int(row["source_rank"])
        )
        for field in (
            "openalex_id",
            "doi",
            "title",
            "publication_year",
            "publication_date",
            "document_type",
            "journal",
            "publisher",
            "authors",
            "landing_page_url",
            "pdf_url",
            "is_oa",
        ):
            if not current.get(field) and row.get(field):
                current[field] = row[field]
        if len(row.get("abstract", "")) > len(current.get("abstract", "")):
            current["abstract"] = row["abstract"]
        current["cited_by_count"] = max(
            int(current.get("cited_by_count") or 0),
            int(row.get("cited_by_count") or 0),
        )
    return list(merged.values())


def regex_hit(patterns: list[str], text: str) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, re.I)]


def screen(row: dict) -> dict:
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    text = f"{title} {abstract}".lower()

    perovskite = regex_hit(
        [r"\bperovskit", r"\bpsc(?:s)?\b", r"metal[- ]halide"], text
    )
    pv = regex_hit(
        [
            r"solar cell",
            r"photovoltaic",
            r"power conversion efficien",
            r"\bpce\b",
            r"device efficien",
        ],
        text,
    )
    stability = regex_hit(
        [
            r"stabilit",
            r"degrad",
            r"lifetime",
            r"durab",
            r"ageing|aging",
            r"operational",
            r"retention",
        ],
        text,
    )
    curve = regex_hit(
        [
            r"maximum power point|\bmpp(?:t)?\b",
            r"tracking",
            r"continuous (?:illumination|operation)",
            r"\bT(?:80|90|95)\b",
            r"\bhours?\b|\bhrs?\b",
            r"\b\d[\d,\.]*\s*h\b",
            r"time[- ]dependent|temporal evolution|over time",
            r"light soaking",
            r"operational stabilit",
            r"stabilized power output",
            r"ISOS[- ]L",
            r"outdoor test|outdoor operation",
        ],
        text,
    )
    nonmonotonic = regex_hit(
        [
            r"recover|reversib|regenerat|self[- ]heal",
            r"non[- ]?monotonic|metastab",
            r"initial (?:increase|improvement|gain|rise)",
            r"light[- ]soak|light soaking",
            r"burn[- ]in",
            r"partial(?:ly)? reversible",
            r"activation period|performance enhancement",
            r"dark rest|light/dark cycl",
        ],
        text,
    )
    review = bool(
        re.search(
            r"\breview\b|perspective|roadmap|progress and prospects|advances in",
            title,
            re.I,
        )
    )
    core = bool(perovskite and pv)
    score = (
        4 * bool(perovskite)
        + 4 * bool(pv)
        + 3 * bool(stability)
        + 2 * min(len(curve), 4)
        + 3 * min(len(nonmonotonic), 3)
        + 1 * bool(abstract)
        - 2 * review
    )
    if core and stability and curve and not review:
        status = "A_curve_likely_original"
    elif core and stability and not review:
        status = "B_stability_original"
    elif core and review:
        status = "C_review_or_perspective"
    elif core:
        status = "C_perovskite_PV_peripheral"
    else:
        status = "D_exclude_not_core_perovskite_PV"

    year = int(row.get("publication_year") or 0)
    screened = dict(row)
    screened.update(
        {
            "screen_status": status,
            "screen_score": score,
            "abstract_available": bool(abstract),
            "frontier_2024_2026": year >= 2024,
            "review_title_flag": review,
            "perovskite_hits": ";".join(perovskite),
            "pv_hits": ";".join(pv),
            "stability_hits": ";".join(stability),
            "curve_evidence_hits": ";".join(curve),
            "nonmonotonic_recovery_hits": ";".join(nonmonotonic),
            "nonmonotonic_recovery_flag": bool(nonmonotonic),
        }
    )
    return screened


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    oa_rows, oa_manifest = openalex_rows()
    cr_rows, cr_manifest = crossref_rows()
    all_raw = oa_rows + cr_rows
    merged = merge_rows(all_raw)
    screened = [screen(row) for row in merged]
    screened.sort(
        key=lambda row: (
            -int(row["screen_score"]),
            -int(row.get("publication_year") or 0),
            -int(row.get("cited_by_count") or 0),
            int(row["best_source_rank"]),
        )
    )
    core = [
        row
        for row in screened
        if row["screen_status"] != "D_exclude_not_core_perovskite_PV"
    ]
    priority = [
        row
        for row in screened
        if row["screen_status"] == "A_curve_likely_original"
    ]
    recovery = [
        row
        for row in priority
        if row["nonmonotonic_recovery_flag"]
    ]

    write_csv(ROOT / "all_records_deduplicated.csv", screened)
    write_csv(ROOT / "title_abstract_screened_core.csv", core)
    write_csv(ROOT / "high_priority_for_fulltext.csv", priority)
    write_csv(ROOT / "nonmonotonic_recovery_priority.csv", recovery)

    status_counts = Counter(row["screen_status"] for row in screened)
    year_counts = Counter(str(row.get("publication_year") or "unknown") for row in screened)
    source_counts = Counter(row["metadata_sources"] for row in screened)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_filter": {"from": DATE_FROM, "to": DATE_TO},
        "method": {
            "retrieval": (
                "Top 200 relevance-ranked records per query from OpenAlex and Crossref; "
                "the APIs' total result counts are reported but were not all downloaded."
            ),
            "deduplication": (
                "Normalized DOI primary key; normalized title fallback for records without DOI."
            ),
            "screening": (
                "Deterministic keyword screening of title plus available abstract. This is not "
                "a claim of full-text or figure review."
            ),
        },
        "queries": oa_manifest + cr_manifest,
        "counts": {
            "openalex_query_records_with_duplicates": len(oa_rows),
            "crossref_query_records_with_duplicates": len(cr_rows),
            "combined_query_records_with_duplicates": len(all_raw),
            "deduplicated_records": len(screened),
            "deduplicated_with_doi": sum(bool(row["doi"]) for row in screened),
            "deduplicated_with_abstract": sum(
                bool(row["abstract_available"]) for row in screened
            ),
            "core_perovskite_pv_title_abstract": len(core),
            "curve_likely_original_priority": len(priority),
            "nonmonotonic_recovery_priority": len(recovery),
            "screen_status": dict(status_counts),
            "publication_year": dict(sorted(year_counts.items())),
            "metadata_source_combinations": dict(source_counts),
        },
    }
    (ROOT / "retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
