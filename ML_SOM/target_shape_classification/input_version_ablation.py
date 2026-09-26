#!/usr/bin/env python3
"""Counterfactual input-version sensitivity for one corrected source curve.

The old series is known to be wrong and is used only to measure how much a
single input correction changes the unsupervised SOM map. No review labels are
read or used for training or seed selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import run
import unsupervised_som as som


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument('--root', type=Path, default=base.parent / 'data_final')
    parser.add_argument('--results', type=Path, default=base / 'results')
    parser.add_argument('--file-id', default='F01669')
    parser.add_argument('--seeds', default='41,42,43,44,45')
    parser.add_argument('--iterations', type=int, default=30000)
    args = parser.parse_args()
    canonical = run.rows(args.results / 'canonical_curves.csv')
    corrected_hashes = {r['file_id']:r['analysis_sha256'] for r in canonical}
    target = next(r for r in canonical if r['file_id'] == args.file_id)
    if target['analysis_version'] == target['source_csv']:
        raise ValueError('target has no accepted correction to compare')
    old_version = target['source_csv']
    corrected_version = target['analysis_version']
    target['analysis_version'] = old_version
    target['analysis_sha256'] = target['source_sha256']
    current_summary = json.loads((args.results / 'som_primary' / 'summary.json').read_text())
    values, raw, metadata = som.input_curves(args.root, canonical)
    x, _, branch, events = som.features(
        values, raw, current_summary['derivative_weight'],
        current_summary['variation_weight'], current_summary['drawup_weight'])
    phase = np.linspace(0, 1, som.N_GRID)
    stages = {}
    for value, row in zip(values, metadata):
        morphology = run.morphology({'t': phase, 't_hour': None, 'y': value}, run.DEFAULT)
        stages[row['file_id']] = (morphology['candidate_class']
                                  if morphology['stages_complete'] else '')
    stage_labels = [stages[row['file_id']] for row in metadata]
    seeds = [int(v) for v in args.seeds.split(',')]
    models = {}
    qualities = {}
    assignments = {}
    for seed in seeds:
        model, _ = som.train(x, branch, np.ones(len(x), bool), seed,
                             current_summary['side'], args.iterations,
                             current_summary['density_power'], current_summary['naming'],
                             current_summary['density_clip_quantile'], stage_labels,
                             current_summary.get('early_stage_balance_power', 0),
                             current_summary.get('sampling_coupling', 'cdf'))
        models[seed] = model
        qualities[seed] = som.map_quality(x, branch, model,
                                         np.ones(len(x), bool), current_summary['side'])['fit']
        assignments[seed] = som.assign(x, branch, metadata, events, model)
        print(f"old version seed={seed} topology={qualities[seed]['topographic_error']:.5f}",
              flush=True)
    stage_scores = {seed: som.balanced_stage_agreement(assignments[seed], stages)[0]
                    for seed in seeds}
    selected = som.select_seed(seeds, qualities, stage_scores,
                               current_summary['topology_tolerance'])
    corrected_seed = current_summary['seed']
    if corrected_seed not in assignments:
        raise ValueError('corrected selected seed absent from counterfactual seeds')
    # Hold seed fixed to isolate the input correction from the seed-selection
    # policy. Report policy-selected changes separately below.
    old = {r['file_id']:r for r in assignments[corrected_seed]}
    old_policy = {r['file_id']:r for r in assignments[selected]}
    corrected = {r['file_id']:r for r in run.rows(args.results / 'som_primary' / 'som_assignments.csv')}
    for file_id, row in corrected.items():
        if row['analysis_sha256'] != corrected_hashes[file_id]:
            raise ValueError(f'corrected SOM input version stale for {file_id}')
    canonical_by_id = {r['file_id']:r for r in canonical}
    audited = {r['file_id']:r for r in run.rows(args.results / 'source_audit_findings.csv')}
    if set(old) != set(corrected):
        raise ValueError('counterfactual and corrected SOM use different curves')
    rows = []
    for file_id in sorted(old):
        original = old[file_id]['som_candidate_class']
        current = corrected[file_id]['som_candidate_class']
        source = canonical_by_id[file_id]
        rows.append(dict(file_id=file_id, old_version_candidate=original,
                         corrected_candidate=current, changed=original != current,
                         corrected_som_stability=corrected[file_id]['som_stability'],
                         corrected_ensemble_agreement=corrected[file_id]['ensemble_agreement'],
                         one_percent_source_extrema=corrected[file_id]['one_percent_source_extrema'],
                         source_group=source['source_group'], source_csv=source['source_csv'],
                         source_image=source['source_image'],
                         source_audit_issue=audited.get(file_id, {}).get('issue', '')))
    output = args.results / 'som_primary'
    run.write_rows(output / 'input_version_sensitivity.csv', rows, list(rows[0]))
    queue = [r for r in rows if r['changed']]
    queue.sort(key=lambda r: (r['corrected_som_stability'] != 'stable',
                              -float(r['corrected_ensemble_agreement']), r['file_id']))
    run.write_rows(output / 'input_version_followup_queue.csv', queue, list(rows[0]))
    summary = dict(corrected_file_id=args.file_id, old_version=old_version,
                   corrected_version=corrected_version,
                   old_selected_seed=selected,
                   corrected_selected_seed=corrected_seed,
                   old_stage_agreement_by_seed={str(k): stage_scores[k] for k in seeds},
                   old_seed_quality={str(k):v for k,v in qualities.items()},
                   old_class_counts=dict(Counter(r['old_version_candidate'] for r in rows)),
                   corrected_class_counts=dict(Counter(r['corrected_candidate'] for r in rows)),
                   changed_curves=sum(r['changed'] for r in rows),
                   changed_policy_selected_curves=sum(
                       old_policy[file_id]['som_candidate_class'] != corrected[file_id]['som_candidate_class']
                       for file_id in old_policy),
                   changed_source_groups=len({r['source_group'] for r in queue}),
                   changed_stable_candidates=sum(r['corrected_som_stability']=='stable' for r in queue),
                   changed_one_percent_event_curves=sum(int(r['one_percent_source_extrema'])>0 for r in queue),
                   transitions={f'{a}->{b}':sum(r['old_version_candidate']==a and
                                                r['corrected_candidate']==b for r in rows)
                                for a in som.CLASSES for b in som.CLASSES if a != b},
                   note='Known wrong source series is used only for sensitivity. changed_curves holds the corrected representative seed fixed; changed_policy_selected_curves also includes any seed-selection change. Corrected model remains primary.')
    (output / 'input_version_sensitivity.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
