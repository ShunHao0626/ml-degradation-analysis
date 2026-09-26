"""Verify the supervised input boundary and frozen unsupervised fitting code."""
from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "shape_only_input_curation_20260925"
OUT = HERE / "results"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(function, include_attributes=False)


def main() -> None:
    current = json.loads((HERE / "config.json").read_text())
    earlier = json.loads((PREVIOUS / "config.json").read_text())
    for key in earlier:
        if key not in {"protocol", "input_index", "input_shapes"}:
            assert earlier[key] == current[key], key
    for name in ("original_observations", "encode", "fit"):
        assert function_ast(PREVIOUS / "run.py", name) == function_ast(HERE / "run.py", name), name
    source = rows(HERE.parent / "shape_only_unsupervised_20260925/results/input_index.csv")
    selected = rows(HERE / "input_index.csv")
    decisions = rows(HERE / "full_input_decisions.csv")
    assignments = rows(OUT / "anonymous_assignments.csv")
    run_exclusions = rows(OUT / "excluded_input_curves.csv")
    assert len(source) == len(decisions) == 2152
    assert len(selected) == len(assignments) == 1842
    assert not run_exclusions
    assert [r["curve_id"] for r in selected] == [r["curve_id"] for r in assignments]
    assert set(selected[0]) == set(source[0])
    assert not set(selected[0]) & {"integrated_class", "strict_200h_candidate_class",
                                   "evidence_status", "previous_anonymous_cluster"}
    counts = Counter(r["decision"] for r in decisions)
    assert counts == {"included": 1842, "excluded_prior_quality": 255,
                      "excluded_supervised_input": 55}
    with np.load(HERE / "shape_inputs.npz") as inputs, np.load(OUT / "model.npz") as model:
        assert inputs["curve_id"].tolist() == model["curve_id"].tolist()
        assert np.array_equal(inputs["raw_rank"], model["raw_rank"])
        for name in ("raw_rank", "embedding", "normalized_shape", "coefficients",
                     "centers", "weights", "amplitudes"):
            assert np.isfinite(model[name]).all(), name
    assert set(int(r["anonymous_cluster"]) for r in assignments) == set(range(4))
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["class_names_used_in_fit"] is False
    assert summary["current_input_selection_used_old_candidates"] is True
    assert summary["included"] == 1842 and summary["excluded_from_input"] == 0
    report = dict(passed=True, source_curves=2152, prior_quality_excluded=255,
                  supervised_input_excluded=55, fit_input=1842,
                  numeric_fit_functions_unchanged=True, hyperparameters_unchanged=True,
                  class_columns_absent_from_model_input=True,
                  class_labels_used_by_input_selector=True,
                  cluster_sizes=dict(Counter(r["anonymous_cluster"] for r in assignments)))
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
