#!/usr/bin/env python3
"""Build an all-file SOM delivery with an explicit 15% exclusion check.

Four-way nearest-SOM assignments remain visibly provisional when no source
review or complete stage agreement supports them. Source-audit disputes stay
in an extended review group; non-PCE and non-elapsed-time files are excluded
from the actual-hour, normalized-PCE deliverable but remain mapped.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import run


TARGETS = set(run.CLASSES)
MAX_EXCLUDED_FRACTION = .15


def delivery_group(row: dict) -> tuple[str, str, str]:
    if row['decision_group'] == 'excluded_non_comparable_pce':
        return 'excluded', 'response_not_comparable_pce', 'excluded'
    if not row['time_start_h'] or not row['time_end_h']:
        return 'excluded', 'actual_hour_axis_unresolved', 'excluded'
    if row['decision_group'] in TARGETS:
        return row['decision_group'], row['decision_basis'], 'assigned'
    if row['decision_basis'] in ('source_audit_or_shape_ambiguous',
                                 'source_audit_pending'):
        return 'extended_review_needed', row['decision_basis'], 'extended'
    if row['som_candidate_class'] in TARGETS:
        return row['som_candidate_class'], 'nearest_som_only_stage_unresolved', 'assigned'
    raise ValueError(f"No traceable delivery group for {row['file_id']}")


def main():
    results = Path(__file__).resolve().parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=results)
    parser.add_argument('--som-output', type=Path)
    args = parser.parse_args()
    results = args.results
    som_dir = args.som_output or results / 'som_primary'
    manifest_rows = run.rows(results / 'input_manifest.csv')
    decisions = run.rows(som_dir / 'classification_decision_view.csv')
    manifest = {r['file_id']: r for r in manifest_rows}
    if len(manifest) != len(manifest_rows) or len(decisions) != len(manifest):
        raise ValueError('All-file manifest and decision table must have unique, equal coverage')
    if {r['file_id'] for r in decisions} != set(manifest):
        raise ValueError('Decision table has stale or missing file IDs')
    output = []
    for row in decisions:
        source = manifest[row['file_id']]
        if row['analysis_version'] != source['analysis_version']:
            raise ValueError(f"Stale analysis version for {row['file_id']}")
        group, basis, status = delivery_group(row)
        output.append(dict(
            file_id=row['file_id'], canonical_id=row['canonical_id'],
            source_csv=row['source_csv'], source_image=row['source_image'],
            analysis_version=row['analysis_version'], analysis_sha256=source['analysis_sha256'],
            source_group=row['source_group'],
            delivery_group=group, delivery_status=status, assignment_basis=basis,
            som_candidate_class=row['som_candidate_class'],
            som_direct_group=row['som_direct_group'],
            decision_group=row['decision_group'],
            source_reviewed_class=row['source_reviewed_class'],
            corrected_series=row.get('corrected_series', ''),
            source_verified=row['source_verified'], stages_verified=row['stages_verified'],
            one_percent_source_extrema=row['one_percent_source_extrema'],
            time_start_h=row['time_start_h'], time_end_h=row['time_end_h'],
            time_conversion=source['conversion'],
            normalization_max=source['normalization_max'],
            exclusion_reason=(row['exclusion_reason'] or basis) if status == 'excluded' else '',
            source_audit_issue=row['source_audit_issue'],
            source_audit_resolution=row['source_audit_resolution']))
    counts = Counter(r['delivery_status'] for r in output)
    groups = Counter(r['delivery_group'] for r in output)
    excluded_fraction = counts['excluded'] / len(output)
    canonical_count = len({r['canonical_id'] for r in output})
    excluded_canonical = len({r['canonical_id'] for r in output
                              if r['delivery_status'] == 'excluded'})
    excluded_canonical_fraction = excluded_canonical / canonical_count
    if max(excluded_fraction, excluded_canonical_fraction) > MAX_EXCLUDED_FRACTION:
        raise ValueError('Exclusion cap exceeded for original files or unique curves')
    if any(r['delivery_status'] != 'excluded' and
           (not r['time_start_h'] or not r['time_end_h']) for r in output):
        raise ValueError('Nonexcluded file without actual hour coordinates')
    run.write_rows(som_dir / 'som_delivery_all_files.csv', output, list(output[0]))
    summary = dict(input_files=len(output), canonical_curves=canonical_count,
                   counts_by_status=dict(counts),
                   counts_by_group=dict(groups), excluded_files=counts['excluded'],
                   excluded_fraction=excluded_fraction,
                   excluded_canonical_curves=excluded_canonical,
                   excluded_canonical_fraction=excluded_canonical_fraction,
                   maximum_excluded_fraction=MAX_EXCLUDED_FRACTION,
                   exclusion_cap_met=True,
                   assigned_four_way=sum(groups[k] for k in TARGETS),
                   source_verified_four_way=sum(r['assignment_basis'] == 'source_verified'
                                                for r in output),
                   caveat='Nearest-SOM-only four-way assignments are retrieval labels, not verified target membership.')
    (som_dir / 'som_delivery_coverage.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
