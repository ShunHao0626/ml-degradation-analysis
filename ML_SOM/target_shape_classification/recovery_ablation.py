#!/usr/bin/env python3
"""Unlabeled drawup-feature ablation for rare recovery trajectories.

Reviews are read only after SOM training and seed quality measurement. Results
are exploratory diagnostics and must not be interpreted as test accuracy.
"""
from __future__ import annotations

import argparse
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
    parser.add_argument('--weights', default='0.5,1.0')
    parser.add_argument('--seeds', default='41,42,43,44,45')
    parser.add_argument('--iterations', type=int, default=30000)
    args = parser.parse_args()
    canonical = run.rows(args.results / 'canonical_curves.csv')
    reviews = run.rows(args.results / 'source_reviews.csv')
    values, raw, metadata = som.input_curves(args.root, canonical)
    output = []
    for weight in [float(v) for v in args.weights.split(',')]:
        x, _, branch, events = som.features(values, raw, .3, 0, weight)
        for seed in [int(v) for v in args.seeds.split(',')]:
            models, _ = som.train(x, branch, np.ones(len(x), bool), seed,
                                  10, args.iterations, 2, 'prototype', .95)
            assignments = som.assign(x, branch, metadata, events, models)
            quality = som.map_quality(x, branch, models,
                                      np.ones(len(x), bool), 10)['fit']
            checked = som.review_check(assignments, reviews)
            counts = Counter(row['som_candidate_class'] for row in assignments)
            row = dict(drawup_weight=weight, seed=seed,
                       fitted_curves=len(metadata),
                       quantization_error=quality['quantization_error'],
                       topographic_error=quality['topographic_error'],
                       reviewed_total=len(checked),
                       reviewed_correct=sum(r['correct'] for r in checked))
            row.update({f'{label}_candidates': counts[label] for label in som.CLASSES})
            row.update({f'{label}_review_correct': sum(r['correct'] for r in checked
                                                       if r['reviewed_class'] == label)
                        for label in som.CLASSES})
            output.append(row)
            print(f"weight={weight} seed={seed} topology={quality['topographic_error']:.4f} "
                  f"reviewed={row['reviewed_correct']}/{row['reviewed_total']}", flush=True)
    path = args.results / 'som_primary' / 'recovery_ablation.csv'
    run.write_rows(path, output, list(output[0]))
    print(path)


if __name__ == '__main__':
    main()
