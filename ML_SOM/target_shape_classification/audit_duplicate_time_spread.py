#!/usr/bin/env python3
"""Audit normalized PCE spread among observations sharing one time stamp."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import run


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base.parent / 'data_final')
    parser.add_argument('--results', type=Path, default=base / 'results')
    args = parser.parse_args()
    rows = []
    for source in run.rows(args.results / 'canonical_curves.csv'):
        if source['y_kind'] != 'pce' or int(source['duplicate_times']) == 0:
            continue
        version = args.root / source['analysis_version']
        if not version.exists() and source['analysis_version'] != source['source_csv']:
            version = args.results / source['analysis_version']
        xcol, ycol = (('x', 'y') if source['folder'] == 'data_all'
                      else ('time_h', 'normalized_pce'))
        groups = defaultdict(list)
        with version.open(newline='', encoding='utf-8-sig') as stream:
            for line, point in enumerate(csv.DictReader(stream), 1):
                try:
                    x, y = float(point[xcol]), float(point[ycol])
                except (ValueError, TypeError, KeyError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    groups[x].append((line, y))
        median_max = max(np.median([y for _, y in points]) for points in groups.values())
        if median_max <= 0:
            continue
        spreads = []
        for x, points in groups.items():
            if len(points) > 1:
                values = [y for _, y in points]
                spreads.append((float((max(values) - min(values)) / median_max),
                                x, ','.join(str(line) for line, _ in points)))
        if not spreads:
            raise ValueError(f"duplicate count disagrees with source: {source['file_id']}")
        maximum, max_time, source_rows = max(spreads)
        rows.append(dict(file_id=source['file_id'], source_csv=source['source_csv'],
                         analysis_version=source['analysis_version'],
                         duplicate_observations=int(source['duplicate_times']),
                         duplicate_time_groups=len(spreads),
                         groups_with_spread_at_least_0p01=sum(v >= .01 for v, _, _ in spreads),
                         groups_with_spread_0p01_to_0p03=sum(.01 <= v <= .03 for v, _, _ in spreads),
                         groups_with_spread_above_0p03=sum(v > .03 for v, _, _ in spreads),
                         max_normalized_spread=maximum, time_at_max_spread=max_time,
                         source_rows_at_max_spread=source_rows,
                         source_image_sha256=source['image_sha256']))
    run.write_rows(args.results / 'duplicate_time_spread_audit.csv', rows, list(rows[0]))
    print(f"{len(rows)} PCE curves with duplicate times; "
          f"{sum(r['groups_with_spread_at_least_0p01'] > 0 for r in rows)} "
          "have a within-time normalized spread of at least 1%")


if __name__ == '__main__':
    main()
