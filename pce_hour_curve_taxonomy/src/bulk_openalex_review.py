#!/usr/bin/env python3
"""Build a reproducible recent-literature corpus from the official OpenAlex API."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_literature_review/metadata"
RAW = OUT / "openalex_raw"
FROM_DATE = "2021-01-01"
TO_DATE = "2026-09-02"

QUERIES = {
    "psc_stability_mppt": "perovskite solar cell stability degradation MPPT",
    "psc_operational_stability": "perovskite solar cell operational stability maximum power point tracking",
    "psc_light_soaking_recovery": "perovskite solar cell light soaking recovery degradation",
    "pv_degradation_clustering": "photovoltaic degradation trajectory time series clustering",
    "solar_ageing_unsupervised": "solar cell ageing curve unsupervised machine learning",
    "psc_stability_ml": "perovskite stability data machine learning",
}

POSITIVE_TERMS = {
    "perovskite": 3.0,
    "solar cell": 3.0,
    "stability": 3.0,
    "degradation": 3.0,
    "aging": 2.0,
    "ageing": 2.0,
    "operational": 2.0,
    "maximum power point": 3.0,
    "mppt": 3.0,
    "light soaking": 3.0,
    "recovery": 2.0,
    "trajectory": 3.0,
    "time series": 2.0,
    "cluster": 2.0,
    "unsupervised": 3.0,
    "machine learning": 2.0,
}
NEGATIVE_TERMS = {
    "battery": -3.0,
    "wind turbine": -3.0,
    "building load": -2.0,
    "forecasting irradiance": -2.0,
}


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positioned = [(position, word) for word, positions in inverted.items() for position in positions]
    return " ".join(word for _, word in sorted(positioned))


def fetch_query(slug: str, query: str) -> dict:
    params = {
        "search": query,
        "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE}",
        "per-page": "200",
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "publication_date",
                "type",
                "cited_by_count",
                "open_access",
                "primary_location",
                "best_oa_location",
                "authorships",
                "abstract_inverted_index",
            ]
        ),
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pce-hour-curve-review/1.0 (mailto:anonymous@example.org)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    payload["retrieval"] = {"slug": slug, "query": query, "url": url}
    with (RAW / f"{slug}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def location_fields(location: dict | None) -> tuple[str, str, str]:
    location = location or {}
    source = location.get("source") or {}
    return (
        location.get("landing_page_url") or "",
        location.get("pdf_url") or "",
        source.get("display_name") or "",
    )


def relevance_score(row: dict) -> float:
    text = (row["title"] + " " + row["abstract"]).lower()
    score = sum(weight for term, weight in POSITIVE_TERMS.items() if term in text)
    score += sum(weight for term, weight in NEGATIVE_TERMS.items() if term in text)
    if "perovskite" in text and ("stability" in text or "degradation" in text):
        score += 4.0
    if ("trajectory" in text or "time series" in text) and (
        "cluster" in text or "unsupervised" in text
    ):
        score += 4.0
    year = int(row["publication_year"] or 0)
    if year >= 2025:
        score += 2.0
    elif year >= 2023:
        score += 1.0
    citations = int(row["cited_by_count"] or 0)
    score += min(3.0, citations / 100.0)
    return round(score, 3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    query_log: list[dict] = []
    for index, (slug, query) in enumerate(QUERIES.items()):
        payload = fetch_query(slug, query)
        query_log.append(
            {
                "slug": slug,
                "query": query,
                "reported_total": payload["meta"]["count"],
                "retrieved": len(payload["results"]),
            }
        )
        for work in payload["results"]:
            work_id = work["id"]
            if work_id not in records:
                primary_landing, primary_pdf, journal = location_fields(work.get("primary_location"))
                oa_landing, oa_pdf, _ = location_fields(work.get("best_oa_location"))
                authors = [
                    (authorship.get("author") or {}).get("display_name", "")
                    for authorship in work.get("authorships") or []
                ]
                oa = work.get("open_access") or {}
                records[work_id] = {
                    "openalex_id": work_id,
                    "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                    "title": work.get("display_name") or "",
                    "publication_year": work.get("publication_year") or "",
                    "publication_date": work.get("publication_date") or "",
                    "type": work.get("type") or "",
                    "journal": journal,
                    "authors": "; ".join(authors),
                    "cited_by_count": work.get("cited_by_count") or 0,
                    "is_open_access": bool(oa.get("is_oa")),
                    "oa_status": oa.get("oa_status") or "",
                    "oa_url": oa.get("oa_url") or oa_pdf or oa_landing,
                    "primary_landing_page": primary_landing,
                    "primary_pdf_url": primary_pdf,
                    "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                    "matched_queries": [],
                }
            records[work_id]["matched_queries"].append(slug)
        if index < len(QUERIES) - 1:
            time.sleep(0.2)

    rows = list(records.values())
    for row in rows:
        row["matched_queries"] = ";".join(sorted(set(row["matched_queries"])))
        row["relevance_score"] = relevance_score(row)
        row["screening_status"] = "unscreened"
    rows.sort(
        key=lambda row: (
            -float(row["relevance_score"]),
            -int(row["publication_year"] or 0),
            -int(row["cited_by_count"] or 0),
            row["title"].lower(),
        )
    )

    fields = [
        "openalex_id",
        "doi",
        "title",
        "publication_year",
        "publication_date",
        "type",
        "journal",
        "authors",
        "cited_by_count",
        "is_open_access",
        "oa_status",
        "oa_url",
        "primary_landing_page",
        "primary_pdf_url",
        "matched_queries",
        "relevance_score",
        "screening_status",
        "abstract",
    ]
    with (OUT / "openalex_combined_deduplicated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "openalex_top100_screening.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[:100])
    with (OUT / "retrieval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "OpenAlex official API",
                "retrieved_at_local_date": TO_DATE,
                "date_filter": {"from": FROM_DATE, "to": TO_DATE},
                "per_query_cap": 200,
                "query_log": query_log,
                "raw_records": sum(item["retrieved"] for item in query_log),
                "deduplicated_records": len(rows),
                "open_access_records": sum(bool(row["is_open_access"]) for row in rows),
                "note": "Ranking is for screening only and never enters the SOM model.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
