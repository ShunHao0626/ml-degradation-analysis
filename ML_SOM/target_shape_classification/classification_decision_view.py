#!/usr/bin/env python3
"""Combine unsupervised SOM discovery, stage evidence, and source review.

The SOM candidate and direct group remain unchanged. This is a reviewable
five-group decision view with an explicit unresolved class; source reviews do
not enter SOM fitting, seed selection, or stage retrieval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import run


def decide(evidence: dict, stage: dict, stage_votes: int = 0) -> tuple[str, str]:
    if not evidence['som_candidate_class']:
        return 'excluded_non_comparable_pce', 'not_comparable_pce'
    if not evidence.get('time_start_h') or not evidence.get('time_end_h'):
        return 'unresolved', 'actual_hour_axis_unresolved'
    if evidence['source_verified'] == 'yes' and evidence['stages_verified'] == 'yes':
        return evidence['reviewed_class'], 'source_verified'
    if evidence['som_evidence_status'] in ('source_audit_rejects_candidate',
                                           'source_shape_ambiguous'):
        return 'unresolved', 'source_audit_or_shape_ambiguous'
    if evidence.get('source_audit_resolution') in (
            'needs_pixel_level_check', 'needs_stage_review',
            'needs_shape_review', 'source_shape_review_needed'):
        return 'unresolved', 'source_audit_pending'
    if (stage and stage['stage_complete'] == 'True'
            and stage['stage_candidate_class'] == evidence['som_candidate_class']):
        return stage['stage_candidate_class'], 'provisional_som_stage_agreement'
    if stage and stage['stage_complete'] == 'True' and stage_votes >= 2:
        return stage['stage_candidate_class'], 'provisional_ensemble_stage_agreement'
    return 'unresolved', 'som_stage_unresolved_or_disagree'


def main():
    base = Path(__file__).resolve().parent / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=base)
    parser.add_argument('--som-output', type=Path)
    args = parser.parse_args()
    som_dir = args.som_output or args.results / 'som_primary'
    evidence = run.rows(som_dir / 'som_evidence_all_files.csv')
    stage_rows = run.rows(som_dir / 'som_stage_overlay.csv')
    stages = {r['file_id']: r for r in stage_rows}
    if len(stages) != len(stage_rows):
        raise ValueError('duplicate stage-overlay file ID')
    expected = {r['canonical_id'] for r in evidence if r['som_candidate_class']}
    if set(stages) != expected:
        raise ValueError('stage overlay does not cover the current comparable SOM curves')
    votes = {}
    for vote in run.rows(som_dir / 'som_seed_votes.csv'):
        counts = votes.setdefault(vote['file_id'], Counter())
        counts[vote['som_candidate_class']] += 1
    if set(votes) != expected or len({sum(c.values()) for c in votes.values()}) != 1:
        raise ValueError('seed votes do not cover the current comparable SOM curves')
    output = []
    for row in evidence:
        stage = stages.get(row['canonical_id'], {})
        if stage and stage['analysis_version'] != row['som_analysis_version']:
            raise ValueError(f'stage overlay input version stale for {row["file_id"]}')
        stage_votes = votes.get(row['canonical_id'], {}).get(
            stage.get('stage_candidate_class', ''), 0)
        group, basis = decide(row, stage, stage_votes)
        output.append(dict(
            file_id=row['file_id'], canonical_id=row['canonical_id'],
            source_csv=row['source_csv'], source_group=row['source_group'],
            source_image=row['source_image'], analysis_version=row['file_analysis_version'],
            y_kind=row['y_kind'], time_start_h=row['time_start_h'],
            time_end_h=row['time_end_h'],
            decision_group=group, decision_basis=basis,
            som_candidate_class=row['som_candidate_class'],
            som_direct_group=row['som_direct_group'],
            som_stability=row['som_stability'],
            ensemble_agreement=row['ensemble_agreement'],
            one_percent_source_extrema=row['one_percent_source_extrema'],
            stage_candidate_class=stage.get('stage_candidate_class', ''),
            stage_complete=stage.get('stage_complete', ''),
            stage_score_margin=stage.get('stage_score_margin', ''),
            stage_class_seed_votes=stage_votes,
            source_reviewed_class=row['reviewed_class'],
            corrected_series=row.get('corrected_series', ''),
            source_verified=row['source_verified'],
            stages_verified=row['stages_verified'],
            som_evidence_status=row['som_evidence_status'],
            source_audit_issue=row['source_audit_issue'],
            source_audit_resolution=row['source_audit_resolution'],
            source_audit_rejected_class=row['source_audit_rejected_class'],
            exclusion_reason=row['exclusion_reason']))
    run.write_rows(som_dir / 'classification_decision_view.csv', output, list(output[0]))
    counts = Counter((r['decision_basis'], r['decision_group']) for r in output)
    for key, count in sorted(counts.items()):
        print(*key, count)


if __name__ == '__main__':
    main()
