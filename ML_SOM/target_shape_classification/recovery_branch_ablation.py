#!/usr/bin/env python3
"""Exploratory SOM on no-early-gain curves with a later recovery.

The 2% routing threshold defines a broad unlabeled recovery subset. All source
points, including ~1% wiggles, remain in the feature matrix. Source reviews are
used only for post-training diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from minisom import MiniSom

import run
import unsupervised_som as som


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument('--root', type=Path, default=base.parent / 'data_final')
    parser.add_argument('--results', type=Path, default=base / 'results')
    parser.add_argument('--threshold', type=float, default=.02)
    parser.add_argument('--sides', default='6,8,10')
    parser.add_argument('--seeds', default='41,42,43')
    parser.add_argument('--iterations', type=int, default=30000)
    args = parser.parse_args()
    values, raw, metadata = som.input_curves(
        args.root, run.rows(args.results / 'canonical_curves.csv'))
    x, _, branch, _ = som.features(values, raw, .3)
    recovery = np.max(values - np.minimum.accumulate(values, axis=1), axis=1)
    subset = np.flatnonzero((branch == 0) & (recovery >= args.threshold))
    xx = x[subset]
    reviews = {r['file_id']:r['reviewed_class']
               for r in run.rows(args.results / 'source_reviews.csv')
               if r['source_verified'] == 'yes' and r['stages_verified'] == 'yes'}
    all_distances = cdist(xx, xx)
    neighbor = np.partition(all_distances, 10, axis=1)[:, 10]
    weights = np.minimum(neighbor, np.quantile(neighbor, .95)) ** 2
    weights = weights / weights.sum()
    output = []
    votes = defaultdict(dict)
    valley_ratios = defaultdict(list)
    for side in [int(v) for v in args.sides.split(',')]:
        for seed in [int(v) for v in args.seeds.split(',')]:
            rng = np.random.default_rng(seed)
            sampled = xx[rng.choice(len(xx), 5000, p=weights)]
            model = MiniSom(side, side, x.shape[1], sigma=side/4,
                            learning_rate=.1, random_seed=seed)
            model.random_weights_init(sampled)
            model.train_random(sampled, args.iterations)
            codebook = model.get_weights().reshape(side*side, -1)
            named = som.named_prototypes(codebook)
            eligible = np.array([p['neuron'] for p in named if p['label']], int)
            distances = cdist(xx, codebook)
            nearest = eligible[np.argmin(distances[:, eligible], axis=1)]
            labels = [named[j]['label'] for j in nearest]
            valley_neurons = [p['neuron'] for p in named if p['label'] == 'valley']
            closest_named = np.min(distances[:, eligible], axis=1)
            valley_distance = np.min(distances[:, valley_neurons], axis=1)
            for k, i in enumerate(subset):
                file_id = metadata[i]['file_id']
                votes[file_id][f'side_{side}_seed_{seed}'] = labels[k]
                valley_ratios[file_id].append(float(valley_distance[k] / max(closest_named[k], 1e-9)))
            counts = Counter(labels)
            matched = [(reviews.get(metadata[i]['file_id']), labels[k])
                       for k, i in enumerate(subset)]
            quality = som.map_quality(
                xx, np.zeros(len(xx), int),
                {0:(codebook, named, subset), 1:(codebook, named, subset)},
                np.ones(len(xx), bool), side)['fit']
            row = dict(threshold=args.threshold, side=side, seed=seed,
                       subset_curves=len(subset),
                       quantization_error=quality['quantization_error'],
                       topographic_error=quality['topographic_error'])
            row.update({f'{label}_candidates':counts[label] for label in som.CLASSES})
            row.update({f'{label}_review_correct':sum(a == b == label for a, b in matched)
                        for label in som.CLASSES})
            output.append(row)
            print(f"side={side} seed={seed} valley={counts['valley']} "
                  f"reviewed_valley={row['valley_review_correct']}/"
                  f"{sum(label == 'valley' for label in reviews.values())}", flush=True)
    path = args.results / 'som_primary' / 'recovery_branch_ablation.csv'
    run.write_rows(path, output, list(output[0]))
    audit = {r['file_id']:r for r in run.rows(args.results / 'source_audit_findings.csv')}
    main_som = {r['file_id']:r for r in run.rows(args.results / 'som_primary' / 'som_assignments.csv')}
    shape = {r['file_id']:r for r in run.rows(args.results / 'classification_unique_curves.csv')}
    vote_rows = []
    for i in subset:
        source = metadata[i]
        file_id = source['file_id']
        counts = Counter(votes[file_id].values())
        maximum = max(counts.values())
        winners = [label for label in som.CLASSES if counts[label] == maximum]
        finding = audit.get(file_id, {})
        row = dict(file_id=file_id, source_group=source['source_group'],
                   source_csv=source['source_csv'], source_image=source['source_image'],
                   analysis_version=source['analysis_version'],
                   recovery_amplitude=float(recovery[i]),
                   valley_votes=counts['valley'], total_votes=len(votes[file_id]),
                   median_valley_relative_distance=float(np.median(valley_ratios[file_id])),
                   majority_class=winners[0],
                   main_som_class=main_som[file_id]['som_candidate_class'],
                   main_som_stability=main_som[file_id]['som_stability'],
                   shape_candidate_class=shape[file_id]['candidate_class'],
                   shape_evidence_status=shape[file_id]['evidence_status'],
                   reviewed_class=reviews.get(file_id, ''),
                   source_audit_issue=finding.get('issue', ''),
                   source_audit_rejected_class=finding.get('rejected_candidate_class', ''),
                   **votes[file_id])
        vote_rows.append(row)
    vote_rows.sort(key=lambda r: (-r['valley_votes'], r['median_valley_relative_distance'],
                                  r['file_id']))
    out = args.results / 'som_primary'
    run.write_rows(out / 'recovery_branch_votes.csv', vote_rows, list(vote_rows[0]))
    shortlist, groups = [], set()
    for row in vote_rows:
        if row['source_group'] not in groups and len(shortlist) < 50:
            shortlist.append(row)
            groups.add(row['source_group'])
    run.write_rows(out / 'recovery_branch_shortlist.csv', shortlist, list(vote_rows[0]))
    print(path)
    print(out / 'recovery_branch_votes.csv')


if __name__ == '__main__':
    main()
