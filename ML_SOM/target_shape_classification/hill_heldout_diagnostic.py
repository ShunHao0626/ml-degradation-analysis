#!/usr/bin/env python3
"""Plot reviewed Hill shapes beside the selected source-held-out SOM prototypes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import run
import unsupervised_som as som


def main():
    base = Path(__file__).resolve().parent
    results = base / 'results'
    out = results / 'som_primary'
    summary = json.loads((out / 'summary.json').read_text())
    seed = summary['seed']
    canonical = run.rows(results / 'canonical_curves.csv')
    values, raw, metadata = som.input_curves(base.parent / 'data_final', canonical)
    features, z, branch, _ = som.features(values, raw,
                                         summary['derivative_weight'],
                                         summary['variation_weight'],
                                         summary['drawup_weight'])
    reviewed = {r['file_id'] for r in run.rows(results / 'source_reviews.csv')
                if r['source_verified'] == 'yes' and r['stages_verified'] == 'yes'
                and r['reviewed_class'] == 'hill'}
    idx = [i for i, r in enumerate(metadata) if r['file_id'] in reviewed]
    if len(idx) != len(reviewed):
        raise ValueError('A reviewed Hill curve is missing from the SOM input')
    if len(set(branch[idx])) != 1:
        raise ValueError('Reviewed Hill curves span SOM branches; plot each separately')
    branch_name = 'early_gain' if branch[idx[0]] else 'no_early_gain'
    with np.load(out / 'som_model.npz') as model:
        codebook = model[f'heldout_seed_{seed}_{branch_name}_codebook']
    named = som.named_prototypes(codebook)
    distances = cdist(features[idx], codebook).mean(axis=0)
    phase = np.linspace(0, 1, som.N_GRID)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for i in idx:
        axes[0].plot(phase, z[i], lw=1.15, alpha=.8, label=metadata[i]['file_id'])
    axes[0].legend(fontsize=7, ncol=3, loc='lower left')
    axes[0].set_title(f'{len(idx)} source reviewed Hill curves')
    colors = {'bridge': '#ed6a13', 'hill': '#1766a1', 'slope': '#c3292e'}
    for label in colors:
        candidates = [p['neuron'] for p in named if p['label'] == label]
        if not candidates:
            continue
        neuron = min(candidates, key=lambda n: distances[n])
        axes[1].plot(phase, codebook[neuron, :som.N_GRID] / .4,
                     color=colors[label], lw=2.3,
                     label=f'{label} neuron {neuron} (mean distance {distances[neuron]:.2f})')
    axes[1].legend(fontsize=8)
    axes[1].set_title(f'Source held out SOM seed {seed}: nearby named neurons')
    for ax in axes:
        ax.axhline(0, color='#888888', lw=.7)
        ax.set(xlabel='Relative observed duration', ylabel='Change / full excursion')
        ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(out / 'hill_heldout_shape_diagnostic.png', dpi=170)
    plt.close(fig)
    print(out / 'hill_heldout_shape_diagnostic.png')


if __name__ == '__main__':
    main()
