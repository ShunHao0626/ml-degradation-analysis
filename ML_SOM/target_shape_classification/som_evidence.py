#!/usr/bin/env python3
"""Join SOM candidates with source evidence without treating candidates as truth."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import run


def indexed(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f'duplicate {key}: {value}')
        result[value] = row
    return result


def evidence_status(candidate, morphology_status, review, finding):
    verified = review.get('source_verified', '').lower() == 'yes'
    stages = review.get('stages_verified', '').lower() == 'yes'
    if not candidate:
        return 'not_comparable_pce'
    if verified and stages and review['reviewed_class'] != candidate:
        return 'verified_disagreement'
    if verified and stages and morphology_status == 'supported':
        return 'verified_match'
    if verified and stages:
        return 'reviewed_match_limited'
    if verified:
        return 'source_shape_ambiguous'
    if finding.get('rejected_candidate_class') == candidate:
        return 'source_audit_rejects_candidate'
    return 'unreviewed_candidate'


def build(results: Path, som_output: Path):
    manifest = run.rows(results / 'input_manifest.csv')
    som_files = run.rows(som_output / 'som_all_files.csv')
    assignments = indexed(run.rows(som_output / 'som_assignments.csv'), 'file_id')
    canonical = indexed(run.rows(results / 'canonical_curves.csv'), 'file_id')
    shape = indexed(run.rows(results / 'classification_all_files.csv'), 'file_id')
    reviews = indexed(run.rows(results / 'source_reviews.csv'), 'file_id')
    audit = indexed(run.rows(results / 'source_audit_findings.csv'), 'file_id')
    source = indexed(manifest, 'file_id')
    som = indexed(som_files, 'file_id')
    if set(source) != set(som) or set(source) != set(shape):
        raise ValueError('full-file manifests have different file IDs')
    for file_id, assignment in assignments.items():
        if assignment['analysis_sha256'] != canonical[file_id]['analysis_sha256']:
            raise ValueError(f'SOM input version stale for {file_id}')
    output = []
    for file_id in source:
        original = source[file_id]
        row = som[file_id]
        morphology = shape[file_id]
        if (row['canonical_id'] != original['canonical_id']
                or morphology['canonical_id'] != original['canonical_id']
                or row['source_sha256'] != original['source_sha256']
                or morphology['source_sha256'] != original['source_sha256']):
            raise ValueError(f'file-level provenance mismatch for {file_id}')
        canonical_id = row['canonical_id']
        version = canonical[canonical_id]['analysis_version']
        if morphology['analysis_version'] != original['analysis_version']:
            raise ValueError(f'shape analysis version stale for {file_id}')
        review = reviews.get(canonical_id, {})
        finding = audit.get(canonical_id, {})
        candidate = row['som_candidate_class']
        status = evidence_status(candidate, morphology['evidence_status'], review, finding)
        output.append(dict(
            file_id=file_id, canonical_id=canonical_id,
            source_csv=original['source_csv'], source_group=original['source_group'],
            source_image=original['source_image'], source_sha256=original['source_sha256'],
            file_analysis_version=original['analysis_version'],
            som_analysis_version=version, y_kind=original['y_kind'],
            time_start_h=morphology['time_start_h'], time_end_h=morphology['time_end_h'],
            som_candidate_class=candidate,
            som_direct_group=row.get('som_direct_group', ''),
            som_stability=row['som_stability'],
            ensemble_agreement=row['ensemble_agreement'],
            one_percent_source_extrema=row['one_percent_source_extrema'],
            som_evidence_status=status, reviewed_class=review.get('reviewed_class', ''),
            corrected_series=(review.get('corrected_series', '')
                              or finding.get('corrected_series', '')),
            source_verified=review.get('source_verified', ''),
            stages_verified=review.get('stages_verified', ''),
            shape_candidate_class=morphology['candidate_class'],
            shape_evidence_status=morphology['evidence_status'],
            shape_stages_complete=morphology['stages_complete'],
            source_audit_issue=finding.get('issue', ''),
            source_audit_resolution=finding.get('resolution_status', ''),
            source_audit_rejected_class=finding.get('rejected_candidate_class', ''),
            exclusion_reason=row['exclusion_reason']))
    path = som_output / 'som_evidence_all_files.csv'
    run.write_rows(path, output, list(output[0]))
    return dict(files=len(output), status_counts=dict(Counter(r['som_evidence_status'] for r in output)))


def refresh_motif_audit_flags(results: Path, som_output: Path):
    findings = indexed(run.rows(results / 'source_audit_findings.csv'), 'file_id')
    for name in ('som_motif_retrieval.csv', 'som_motif_shortlist.csv',
                 'input_version_sensitivity.csv', 'input_version_followup_queue.csv'):
        path = som_output / name
        if not path.exists():
            continue
        rows = run.rows(path)
        for row in rows:
            finding = findings.get(row['file_id'], {})
            if 'source_audit_issue' in row:
                row['source_audit_issue'] = finding.get('issue', '')
            if 'source_audit_resolution' in row:
                row['source_audit_resolution'] = finding.get('resolution_status', '')
            if name.startswith('som_motif_'):
                row['source_audit_rejected_class'] = finding.get('rejected_candidate_class', '')
        run.write_rows(path, rows, list(rows[0]))


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument('--results', type=Path, default=base / 'results')
    parser.add_argument('--som-output', type=Path, default=base / 'results' / 'som_primary')
    args = parser.parse_args()
    result = build(args.results, args.som_output)
    refresh_motif_audit_flags(args.results, args.som_output)
    summary_path = args.som_output / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['som_evidence_status_counts'] = result['status_counts']
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
