"""Check coverage and numerical integrity of the frozen four-group output."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from run import CONFIG, HERE, OUT, read_rows


def main() -> None:
    source = read_rows((HERE / CONFIG["input_index"]).resolve())
    assigned = read_rows(OUT / "anonymous_assignments.csv")
    excluded = read_rows(OUT / "excluded_rough_curves.csv")
    summary = json.loads((OUT / "summary.json").read_text())
    original_ids = [r["curve_id"] for r in source]
    kept_ids = [r["curve_id"] for r in assigned]
    removed_ids = [r["curve_id"] for r in excluded]
    assert len(original_ids) == len(set(original_ids))
    assert set(original_ids) == set(kept_ids) | set(removed_ids)
    assert not set(kept_ids) & set(removed_ids)
    assert len(assigned) == summary["included"] == 2135
    assert len(excluded) == summary["excluded_from_input"] == 17
    assert set(int(r["anonymous_cluster"]) for r in assigned) == set(range(4))
    assert all(float(r["total_variation_to_range"]) <= 3 for r in assigned)
    assert all(float(r["total_variation_to_range"]) > 3 for r in excluded)
    with np.load(OUT / "model.npz") as model:
        assert model["curve_id"].tolist() == kept_ids
        for key in ("raw_rank", "embedding", "normalized_shape", "coefficients",
                    "centers", "weights", "amplitudes"):
            assert np.isfinite(model[key]).all(), key
    report = dict(passed=True, input_eligible=len(source), assigned=len(assigned),
                  excluded=len(excluded),
                  cluster_sizes=dict(Counter(r["anonymous_cluster"] for r in assigned)),
                  assigned_with_observed_range_at_most_one_percent=sum(
                      float(r["observed_range"]) <= .01 for r in assigned))
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
