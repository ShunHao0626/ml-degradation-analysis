#!/usr/bin/env python3
"""Audit the first 200 actual hours before any full-trajectory class decision.

One-percent events are retained as candidates. The table is evidence, not a
source-verified class label: sparse markers and figure error bars still need
the original image.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

import run


EARLY_H = 200.
TRANSITION_H = 250.
MICRO = .01


def assess(t_hour: np.ndarray, normalized_pce: np.ndarray) -> dict:
    """Describe source-point movements through 200 h and its near boundary."""
    t = np.asarray(t_hour, float)
    y = np.asarray(normalized_pce, float)
    if len(t) != len(y) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('expected at least two strictly increasing hour points')
    first = max(0., float(t[0]))
    early_end = min(EARLY_H, float(t[-1]))
    transition_end = min(TRANSITION_H, float(t[-1]))
    observed = (t >= first) & (t <= early_end)
    result = dict(early_start_h=first, early_end_h=early_end,
                  early_observed_points=int(observed.sum()),
                  early_coverage_h=max(0., early_end-first),
                  first_1pct_direction='', first_1pct_hour='',
                  early_1pct_source_extrema=0,
                  early_valley_trough_h='', early_valley_drop=0.,
                  early_valley_recovery_by_250h=0.,
                  early_valley_strength=0.,
                  early_peak_h='', early_peak_gain=0.,
                  early_postpeak_drop_by_250h=0.,
                  early_end_change='', early_pattern_hint='insufficient_early_points')
    if early_end <= first or observed.sum() < 2:
        return result
    f = PchipInterpolator(t, y, extrapolate=False)
    positions = np.unique(np.r_[t[(t >= first) & (t <= transition_end)],
                                first, early_end, transition_end])
    values = f(positions)
    early = positions <= early_end
    early_positions, early_values = positions[early], values[early]
    result['early_end_change'] = float(early_values[-1]-early_values[0])
    source = (t >= first) & (t <= transition_end)
    peaks, _ = find_peaks(y[source], prominence=MICRO)
    troughs, _ = find_peaks(-y[source], prominence=MICRO)
    result['early_1pct_source_extrema'] = int(len(peaks)+len(troughs))
    crossings = np.flatnonzero(np.abs(early_values-early_values[0]) >= MICRO)
    if len(crossings):
        j = int(crossings[0])
        direction = 'rise' if early_values[j] > early_values[0] else 'decline'
        result.update(first_1pct_direction=direction,
                      first_1pct_hour=float(early_positions[j]))
    peak = int(np.argmax(early_values))
    if peak > 0:
        later = values[positions > early_positions[peak]]
        result.update(early_peak_h=float(early_positions[peak]),
                      early_peak_gain=float(early_values[peak]-early_values[0]),
                      early_postpeak_drop_by_250h=(
                          float(early_values[peak]-np.min(later)) if len(later) else 0.))
    for j in range(1, len(positions)-1):
        if positions[j] > early_end:
            break
        drop = float(np.max(values[:j])-values[j])
        recovery = float(np.max(values[j+1:])-values[j])
        strength = min(drop, recovery)
        if strength > result['early_valley_strength']:
            result.update(early_valley_trough_h=float(positions[j]),
                          early_valley_drop=drop,
                          early_valley_recovery_by_250h=recovery,
                          early_valley_strength=strength)
    if observed.sum() < 3:
        return result
    if result['first_1pct_direction'] == 'decline':
        result['early_pattern_hint'] = (
            'valley' if result['early_valley_strength'] >= MICRO else 'slope')
    elif result['first_1pct_direction'] == 'rise':
        result['early_pattern_hint'] = (
            'bridge_or_hill' if result['early_postpeak_drop_by_250h'] >= MICRO
             else 'bridge')
    else:
        result['early_pattern_hint'] = 'no_1pct_change_observed'
    return result


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base.parent / 'data_final')
    parser.add_argument('--results', type=Path, default=base / 'results')
    args = parser.parse_args()
    output = []
    for row in run.rows(args.results / 'canonical_curves.csv'):
        if row['y_kind'] != 'pce':
            continue
        item = dict(file_id=row['file_id'], source_csv=row['source_csv'],
                    analysis_version=row['analysis_version'],
                    source_image=row['source_image'],
                    source_image_sha256=row['image_sha256'])
        if not row['time_factor']:
            output.append(item | dict(early_pattern_hint='actual_hour_axis_unresolved'))
            continue
        path = args.root / row['analysis_version']
        if not path.exists() and row['analysis_version'] != row['source_csv']:
            path = args.results / row['analysis_version']
        t, y, _, _ = run.read_curve(path, row['folder'])
        if np.max(y) <= 0:
            output.append(item | dict(early_pattern_hint='nonpositive_pce'))
            continue
        output.append(item | assess(t * float(row['time_factor']), y / np.max(y)))
    fields = list(dict.fromkeys(key for item in output for key in item))
    run.write_rows(args.results / 'early_200h_audit.csv', output, fields)
    print(len(output), 'PCE curves:', Counter(r['early_pattern_hint'] for r in output))


if __name__ == '__main__':
    main()
