#!/usr/bin/env python3
"""Retrain only the selected 6x6 SOM and compare it with the archived model.

This is the shorter reproducibility path. It uses the exact fixed input and
selected hyperparameters, writes into a new directory, and never modifies the
archived official result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper200h_tuned_rise_fall_som import curve_diagnostics, train_one


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
DEFAULT_INPUT = PROJECT / "03_preprocessed" / "main_high_quality_min10"
DEFAULT_REFERENCE = BUNDLE / "03_selected_model"
DEFAULT_OUTPUT = BUNDLE / "reproduced_selected_model"

EXPECTED_INPUT_SHA256 = (
    "8c9b473c1b3a56ffcda2d0b1464bcaf3f969a3cbead871f6cb9103cad8dd0096"
)
N_NODES = 36
SIGMA = 0.3
LEARNING_RATE = 0.1
SEED = 42
ITERATIONS = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    array_path = input_dir / "X_smoothed.npy"
    actual_hash = sha256(array_path)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise AssertionError(
            "X_smoothed.npy does not match the archived input: "
            f"expected {EXPECTED_INPUT_SHA256}, got {actual_hash}"
        )

    curves = np.load(array_path)
    time_h = pd.read_csv(input_dir / "time_grid.csv")["time_h"].to_numpy(dtype=float)
    metadata = pd.read_csv(input_dir / "curve_metadata.csv")
    if curves.shape != (1442, 1201):
        raise AssertionError(f"Unexpected input shape: {curves.shape}")
    if len(time_h) != curves.shape[1] or len(metadata) != curves.shape[0]:
        raise AssertionError("Input array, time grid, and metadata dimensions differ")

    target = np.asarray(
        [
            bool(curve_diagnostics(curve, time_h)["rise_then_fall_core"])
            for curve in curves
        ],
        dtype=bool,
    )
    if int(target.sum()) != 56:
        raise AssertionError(f"RTF-core target drift: expected 56, got {target.sum()}")
    return curves, time_h, metadata, target


def compare_reference(
    reference_dir: Path,
    labels: np.ndarray,
    weights: np.ndarray,
    curve_ids: pd.Series,
) -> dict[str, object]:
    archived_assignments = pd.read_csv(reference_dir / "final_curve_assignments.csv")
    archived_weights = np.load(reference_dir / "final_som_weights.npy")
    archived_model = json.loads(
        (reference_dir / "selected_model.json").read_text(encoding="utf-8")
    )
    same_curve_order = np.array_equal(
        archived_assignments["curve_id"].astype(str).to_numpy(),
        curve_ids.astype(str).to_numpy(),
    )
    labels_exact = bool(
        same_curve_order
        and np.array_equal(
            archived_assignments["cluster_id"].to_numpy(dtype=int), labels
        )
    )
    weights_max_abs_diff = float(np.max(np.abs(archived_weights - weights)))
    weights_allclose = bool(
        np.allclose(archived_weights, weights, rtol=0.0, atol=1e-12)
    )
    return {
        "same_curve_order": bool(same_curve_order),
        "labels_exact_match": labels_exact,
        "weights_allclose_atol_1e-12": weights_allclose,
        "weights_max_abs_diff": weights_max_abs_diff,
        "archived_n_nodes": int(archived_model["n_nodes"]),
        "archived_rise_then_fall_node": int(
            archived_model["rise_then_fall_node"]
        ),
    }


def main(input_dir: Path, output_dir: Path, reference_dir: Path, strict: bool) -> None:
    if output_dir in (BUNDLE.resolve(), reference_dir.resolve()):
        raise ValueError("Refusing to overwrite the archived official result")
    output_dir.mkdir(parents=True, exist_ok=True)

    curves, time_h, metadata, target = load_input(input_dir)
    run = train_one(
        curves=curves,
        time_h=time_h,
        target=target,
        n_nodes=N_NODES,
        sigma=SIGMA,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        iterations=ITERATIONS,
    )

    assignments = pd.DataFrame(
        {
            "array_row": np.arange(len(curves)),
            "curve_id": metadata["curve_id"],
            "cluster_id": run.labels,
            "som_node_x": run.coords[:, 0],
            "som_node_y": run.coords[:, 1],
            "rise_then_fall_core": target,
        }
    )
    assignments.to_csv(output_dir / "reproduced_assignments.csv", index=False)
    np.save(output_dir / "reproduced_som_weights.npy", run.som.get_weights())

    comparison = compare_reference(
        reference_dir=reference_dir,
        labels=run.labels,
        weights=run.som.get_weights(),
        curve_ids=metadata["curve_id"],
    )
    member = run.labels == run.best_node
    metrics = {
        "parameters": {
            "n_nodes": N_NODES,
            "topology": [6, 6],
            "sigma": SIGMA,
            "learning_rate": LEARNING_RATE,
            "iterations": ITERATIONS,
            "seed": SEED,
            "input_len": int(curves.shape[1]),
        },
        "input": {
            "directory": str(input_dir),
            "shape": list(curves.shape),
            "X_smoothed_sha256": sha256(input_dir / "X_smoothed.npy"),
            "rtf_core_count": int(target.sum()),
        },
        "reproduced_metrics": {
            "qe": float(run.qe),
            "te": float(run.te),
            "rise_then_fall_node": int(run.best_node),
            "node_size": int(member.sum()),
            "node_rtf_core_count": int(np.sum(member & target)),
            "node_precision": float(run.best_metrics["precision"]),
            "node_recall": float(run.best_metrics["recall"]),
            "node_f1": float(run.best_metrics["f1"]),
        },
        "comparison_with_archived_result": comparison,
    }
    passed = bool(
        run.best_node == 22
        and int(member.sum()) == 9
        and int(np.sum(member & target)) == 8
        and comparison["labels_exact_match"]
        and comparison["weights_allclose_atol_1e-12"]
    )
    metrics["status"] = "PASS" if passed else "DIFFERENT_FROM_ARCHIVE"
    (output_dir / "reproduction_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain and compare only the selected 6x6 SOM."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code unless labels and weights match the archive.",
    )
    args = parser.parse_args()
    main(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        reference_dir=args.reference_dir.expanduser().resolve(),
        strict=args.strict,
    )
