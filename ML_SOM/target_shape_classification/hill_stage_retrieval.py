#!/usr/bin/env python3
"""Retrieve complete Hill-shaped curves missed by the unsupervised SOM.

The shape interpretation uses no source-review labels or templates. It is a
post-training diagnostic; SOM training, seed choice, and candidates are read
unchanged from the primary run. All source points remain in the input files and
SOM features, including normalized fluctuations around one percent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run
import unsupervised_som as som


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base.parent / 'data_final')
    parser.add_argument('--results', type=Path, default=base / 'results')
    parser.add_argument('--som-output', type=Path)
    args = parser.parse_args()
    out = args.som_output or args.results / 'som_primary'
    canonical = run.rows(args.results / 'canonical_curves.csv')
    values, raw, metadata = som.input_curves(args.root, canonical)
    assignments = {r['file_id']: r for r in run.rows(out / 'som_assignments.csv')}
    evidence = {r['file_id']: r for r in run.rows(out / 'som_evidence_all_files.csv')}
    votes = defaultdict(list)
    for vote in run.rows(out / 'som_seed_votes.csv'):
        votes[vote['file_id']].append(vote['som_candidate_class'])
    rows = []
    phase = np.linspace(0, 1, som.N_GRID)
    for value, source in zip(values, metadata):
        m = run.morphology({'t': phase, 't_hour': None, 'y': value}, run.DEFAULT)
        assignment = assignments[source['file_id']]
        proof = evidence[source['file_id']]
        vote_counts = Counter(votes[source['file_id']])
        if len(votes[source['file_id']]) != 5:
            raise ValueError(f"expected five SOM seed votes for {source['file_id']}")
        rows.append(dict(
            file_id=source['file_id'], source_group=source['source_group'],
            source_csv=source['source_csv'], source_image=source['source_image'],
            analysis_version=source['analysis_version'],
            stage_candidate_class=m['candidate_class'],
            stage_complete=bool(m['stages_complete']),
            stage_score_margin=float(m['score_margin']),
            stage_class_seed_votes=vote_counts[m['candidate_class']],
            hill_seed_votes=vote_counts['hill'],
            hill_shape_score=float(m['shape_scores']['hill']),
            peak_phase=float(np.argmax(value) / (som.N_GRID - 1)),
            som_candidate_class=assignment['som_candidate_class'],
            som_direct_group=assignment['som_direct_group'],
            som_stability=assignment['som_stability'],
            som_candidate_distance=assignment['distance_to_candidate'],
            som_quantization_distance=assignment['quantization_distance'],
            one_percent_source_extrema=assignment['one_percent_source_extrema'],
            som_evidence_status=proof['som_evidence_status'],
            reviewed_class=proof['reviewed_class'],
            corrected_series=proof.get('corrected_series', ''),
            source_verified=proof['source_verified'],
            stages_verified=proof['stages_verified'],
            source_audit_issue=proof['source_audit_issue'],
            source_audit_rejected_class=proof['source_audit_rejected_class']))
    run.write_rows(out / 'som_stage_overlay.csv', rows, list(rows[0]))
    hill = [r for r in rows if r['stage_candidate_class'] == 'hill' and r['stage_complete']]
    hill.sort(key=lambda r: (-r['hill_seed_votes'], -r['stage_score_margin'], r['file_id']))
    run.write_rows(out / 'hill_stage_retrieval.csv', hill, list(rows[0]))
    corroborated = [r for r in hill if r['hill_seed_votes'] >= 2]
    run.write_rows(out / 'hill_som_corroborated.csv', corroborated, list(rows[0]))
    print(f'{len(rows)} comparable curves; {len(hill)} complete Hill stage candidates; '
          f'{sum(r["som_candidate_class"] != "hill" for r in hill)} missed by primary SOM; '
          f'{len(corroborated)} supported by at least two SOM seeds')


if __name__ == '__main__':
    main()
