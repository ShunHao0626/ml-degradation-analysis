"""Explicitly weak post-hoc morphology suggestion for frozen anonymous nodes."""
from __future__ import annotations

import numpy as np


def rule_suggestion(t, y):
    """A provisional event-order heuristic; never a reference annotation."""
    if len(t) < 3:
        return "slope", "too_few_points_for_stages"
    # Approximate stage comparisons use real time bins, not interpolated point counts.
    cuts = np.linspace(t[0], min(t[-1], 200), 6)
    stage = np.array([np.median(y[(t >= cuts[k]) & (t <= cuts[k + 1])])
                      if np.any((t >= cuts[k]) & (t <= cuts[k + 1])) else np.nan for k in range(5)])
    if np.sum(np.isfinite(stage)) < 3:
        stage = np.interp(np.linspace(t[0], t[-1], 5), t, y)
    direction = stage[1] - stage[0]
    if direction < -.008:
        trough = int(np.argmin(stage[:4]))
        recovery = float(np.max(stage[trough + 1:]) - stage[trough]) if trough < 4 else 0.
        if recovery >= .01 and trough >= 1:
            return "valley", "provisional_fall_then_recovery"
        return "slope", "provisional_early_decline"
    peak = int(np.argmax(stage))
    if peak < 4 and stage[peak] - stage[0] >= .008:
        after = np.diff(stage[peak:])
        if len(after) >= 2 and after[0] < -.015 and abs(after[0]) > 1.6 * abs(after[-1]):
            return "hill", "provisional_fast_postpeak_decline"
        return "bridge", "provisional_rise_and_high_region"
    # Forced four-way candidate is flagged for review by caller.
    return ("slope" if stage[-1] < stage[0] else "bridge"), "weak_four_way_fallback"
