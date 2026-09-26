#!/usr/bin/env python3
"""Create a transparent, manually reasoned shortlist from the bulk corpus."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "08_literature_review/metadata/openalex_combined_deduplicated.csv"
OUTPUT = ROOT / "08_literature_review/screening/priority_shortlist.csv"

# Relevance decisions are based on title/abstract screening.  The list mixes
# recent frontier work with two clearly marked foundational reporting papers.
SELECTION = {
    "10.1038/s41467-023-40585-3": (
        "core_method",
        "Direct source workflow: full normalized MPPT trajectories, SOM, QE elbow and visual overlap.",
    ),
    "10.3389/fenrg.2023.1118654": (
        "data_quality",
        "Shows that stability FOM choice changes conclusions and argues for sharing complete ageing curves.",
    ),
    "10.1038/s41467-022-35400-4": (
        "stability_metrics",
        "Large-database analysis of burn-in, TS80/TS80m and heterogeneous reporting constraints.",
    ),
    "10.1039/d3ta05966a": (
        "recent_ml_database",
        "2024 large-dataset ML analysis; useful for limitations of heterogeneous literature-derived stability data.",
    ),
    "10.1016/j.egyai.2025.100503": (
        "frontier_ml_database",
        "2025 open analysis reporting weak stability prediction under non-standardized data.",
    ),
    "10.1021/acsenergylett.4c00328": (
        "frontier_forecasting",
        "2024 wavelet/ML prediction of long-term outdoor performance from accelerated indoor tests.",
    ),
    "10.1021/acsenergylett.4c01943": (
        "frontier_outdoor_dynamics",
        "Two-year outdoor module monitoring with diurnal degradation/recovery and ML analysis.",
    ),
    "10.1002/aenm.202304452": (
        "recovery_morphology",
        "Long-term outdoor PCE with nighttime loss and light-soaking recovery; directly relevant to valley/rise shapes.",
    ),
    "10.1002/aenm.202501906": (
        "frontier_outdoor_dynamics",
        "Four-year outdoor dataset; seasonality and >24 h light-soaking saturation complicate curve interpretation.",
    ),
    "10.1038/s41586-024-08161-x": (
        "frontier_recovery_mechanism",
        "Continuous versus day/night cycling shows dark recovery and changing decay rates.",
    ),
    "10.1021/acsaem.1c00588": (
        "recovery_mechanism",
        "Bias-dependent degradation and recovery dynamics support non-monotonic PCE trajectories.",
    ),
    "10.1002/adma.202110239": (
        "recovery_mechanism",
        "Self-healing and light-soaking study; relevant physical context for gain/recovery curves.",
    ),
    "10.1002/ente.202200234": (
        "measurement_system",
        "High-throughput parallel MPPT setup underlying homogeneous ageing-curve acquisition.",
    ),
    "10.1149/2162-8777/ac6f1d": (
        "measurement_quality",
        "Round-robin maximum-power measurement study for metastable devices.",
    ),
    "10.1038/s41560-023-01421-6": (
        "frontier_long_duration",
        "4,500 h operational test; candidate source of long PCE-hour traces.",
    ),
    "10.1038/s41560-024-01487-w": (
        "frontier_degradation_mechanism",
        "Ion-induced field screening offers mechanistic context for evolving operational PCE.",
    ),
    "10.1016/j.apenergy.2025.126132": (
        "frontier_open_data",
        "2025 overview of open datasets for PV reliability and degradation analysis.",
    ),
    "10.1038/s41560-019-0529-5": (
        "foundational_protocol",
        "ISOS consensus: non-monotonic PCE, burn-in, recovery, reporting, and light/dark cycling.",
    ),
    "10.1016/j.solmat.2019.110284": (
        "foundational_curve_database",
        "Earlier literature-derived database explicitly containing 404 PCE-versus-time profiles.",
    ),
    "10.3390/asi9060116": (
        "frontier_forecasting",
        "2026 trajectory forecasting benchmark that reuses SOM-derived degradation modes.",
    ),
}

MANUAL = {
    "10.1038/s41467-022-35400-4": {
        "title": "Big data driven perovskite solar cell stability analysis",
        "publication_year": "2022",
        "journal": "Nature Communications",
        "is_open_access": "True",
        "oa_url": "https://www.nature.com/articles/s41467-022-35400-4.pdf",
    },
    "10.1021/acsenergylett.4c01943": {
        "title": "Diurnal Changes and Machine Learning Analysis of Perovskite Modules Based on Two Years of Outdoor Monitoring",
        "publication_year": "2024",
        "journal": "ACS Energy Letters",
        "is_open_access": "True",
        "oa_url": "https://pubs.acs.org/doi/pdf/10.1021/acsenergylett.4c01943",
    },
    "10.1038/s41560-019-0529-5": {
        "title": "Consensus statement for stability assessment and reporting for perovskite photovoltaics based on ISOS procedures",
        "publication_year": "2020",
        "journal": "Nature Energy",
        "is_open_access": "True",
        "oa_url": "https://www.nrel.gov/docs/fy20osti/74047.pdf",
    },
    "10.1016/j.solmat.2019.110284": {
        "title": "Machine learning analysis on stability of perovskite solar cells",
        "publication_year": "2020",
        "journal": "Solar Energy Materials and Solar Cells",
        "is_open_access": "False",
        "oa_url": "",
    },
    "10.3390/asi9060116": {
        "title": "On the Sufficiency of Direct Regression for Perovskite Solar Cell Degradation Forecasting",
        "publication_year": "2026",
        "journal": "Applied System Innovation",
        "is_open_access": "True",
        "oa_url": "https://www.mdpi.com/2571-5577/9/6/116/pdf",
    },
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with MASTER.open("r", encoding="utf-8", newline="") as handle:
        master_rows = list(csv.DictReader(handle))
    by_doi = {row["doi"].lower(): row for row in master_rows if row["doi"]}
    output_rows = []
    for doi, (category, rationale) in SELECTION.items():
        row = dict(by_doi.get(doi, {}))
        if doi in MANUAL:
            row.update(MANUAL[doi])
        row.update(
            {
                "doi": doi,
                "review_category": category,
                "screening_decision": "include_priority",
                "screening_rationale": rationale,
            }
        )
        output_rows.append(row)
    output_rows.sort(
        key=lambda row: (-int(row.get("publication_year") or 0), row.get("title", "").lower())
    )
    fields = [
        "doi",
        "title",
        "publication_year",
        "publication_date",
        "journal",
        "authors",
        "review_category",
        "screening_decision",
        "screening_rationale",
        "is_open_access",
        "oa_status",
        "oa_url",
        "primary_landing_page",
        "primary_pdf_url",
        "openalex_id",
        "cited_by_count",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
