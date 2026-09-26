"""Check frozen inputs, complete pairwise distances, and native DTW math."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from run import CONFIG, HERE, OUT, INPUT_INDEX, digest, read_rows


def python_dtw(level: np.ndarray, short: np.ndarray, long: np.ndarray,
               weights: np.ndarray, band: int) -> float:
    n = len(level[0])
    cost = np.full((n + 1, n + 1), np.inf)
    length = np.zeros((n + 1, n + 1), dtype=int)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            local = (weights[0] * abs(level[0, i-1] - level[1, j-1])
                     + weights[1] * abs(short[0, i-1] - short[1, j-1])
                     + weights[2] * abs(long[0, i-1] - long[1, j-1]))
            options = [(cost[i-1, j-1], length[i-1, j-1]),
                       (cost[i-1, j], length[i-1, j]),
                       (cost[i, j-1], length[i, j-1])]
            best, steps = min(options, key=lambda item: item[0])
            cost[i, j] = best + local
            length[i, j] = steps + 1
    return cost[n, n] / length[n, n]


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_index_sha256"] == digest(INPUT_INDEX)
    assert summary["config_sha256"] == digest(HERE / "config.json")
    assert summary["code_sha256"] == digest(HERE / "run.py")
    assert summary["cpp_sha256"] == digest(HERE / "shape_dtw.cpp")
    index = read_rows(OUT / "input_index.csv")
    assignments = read_rows(OUT / "anonymous_assignments.csv")
    assert len(index) == len(assignments) == summary["eligible"] == 2152
    assert [r["curve_id"] for r in index] == [r["curve_id"] for r in assignments]
    assert len({r["curve_id"] for r in index}) == len(index)
    with np.load(OUT / "shape_inputs.npz") as model:
        level, short, long, raw = (model[x] for x in ("level", "short_change", "long_change", "raw_rank"))
        assert level.shape == short.shape == long.shape == raw.shape == (2152, 64)
        assert np.isfinite(level).all() and np.isfinite(short).all() and np.isfinite(long).all()
        assert np.allclose(level, raw - raw[:, :1])
    with np.load(OUT / "all_pair_dtw_distances.npz") as saved:
        pairs, distances = saved["pairs"], saved["distance"]
        assert len(pairs) == len(distances) == len(index) * (len(index) - 1) // 2
        assert np.all(pairs[:, 0] < pairs[:, 1])
        assert np.isfinite(distances).all() and (distances >= 0).all()
    with np.load(OUT / "shape_dtw_edges.npz") as saved:
        edges, edge_distances = saved["pairs"], saved["distance"]
        assert len(edges) == len(edge_distances) == summary["neighbor_graph_edges"]
    # Independent implementation check on a small pair, including warping.
    a, b = 10, 57
    grid = 12
    channels = [np.ascontiguousarray(x[[a, b], :grid]) for x in (level, short, long)]
    weights = np.ascontiguousarray(CONFIG["dtw_channel_weights"], dtype=np.float64)
    pair = np.array([[0, 1]], dtype=np.int32)
    output = np.empty(1, dtype=np.float64)
    library = ctypes.CDLL(str(HERE / "shape_dtw.dylib"))
    double = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    int32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    library.shape_pair_dtw.argtypes = [double, double, double, ctypes.c_int, int32,
                                       ctypes.c_int, ctypes.c_int, double, double]
    library.shape_pair_dtw(*channels, grid, pair, 1, 6, weights, output)
    reference = python_dtw(*channels, weights, 6)
    assert abs(output[0] - reference) < 1e-12, (output[0], reference)
    report = dict(valid=True, eligible=len(index), exact_pairs=len(pairs),
                  graph_edges=len(edges), native_vs_python_dtw_absolute_error=abs(output[0]-reference),
                  note="Checks data and computation, not four-shape classification accuracy.")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
