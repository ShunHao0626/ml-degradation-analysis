#!/usr/bin/env python3
"""Count source panels and curves still awaiting first visual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    canonical = {r['file_id']: r for r in run.rows(args.results / 'canonical_curves.csv')}
    reviews = run.rows(args.results / 'source_reviews.csv')
    reviewed = {r['file_id'] for r in reviews if r['source_verified'].lower() == 'yes'}
    reviewed_images = {canonical[i]['source_image'] for i in reviewed
                       if i in canonical and canonical[i]['source_image']}
    flagged = {r['file_id'] for r in run.rows(args.results / 'source_review_tasks.csv')}
    priority = {r['file_id'] for r in run.rows(args.results / 'candidate_review_queue.csv')}
    pce = {i for i, r in canonical.items() if r['y_kind'] == 'pce'}

    def count(ids):
        pending = ids - reviewed
        images = {canonical[i]['source_image'] for i in pending
                  if canonical[i]['source_image']}
        return dict(unreviewed_curves=len(pending), source_images=len(images),
                    source_images_never_reviewed=len(images - reviewed_images),
                    missing_source_image_curves=sum(not canonical[i]['source_image']
                                                    for i in pending))

    summary = dict(comparable_pce_curves=len(pce), reviewed_curves=len(reviewed),
                   reviewed_source_images=len(reviewed_images),
                   reviewed_complete_four_stage_curves=sum(
                       r['source_verified'].lower() == 'yes'
                       and r['stages_verified'].lower() == 'yes' for r in reviews),
                   reviewed_stage_unconfirmed_curves=sum(
                       r['source_verified'].lower() == 'yes'
                       and r['stages_verified'].lower() != 'yes' for r in reviews),
                   generated_review_task_curves=len(flagged),
                   flagged_review_workload=count(flagged),
                   priority_candidate_workload=count(priority),
                   entire_pce_workload=count(pce),
                   note=('Review tasks are generated from incomplete or close morphology '
                         'scores; they are not a claim that every curve needs a fresh '
                         'source image. A panel may contain multiple curves.'))
    output = args.results / 'source_review_workload.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
