"""Independent output consistency checks for the 0-200 h DTW run."""
from __future__ import annotations

import csv
import ctypes
import json
from collections import Counter

import numpy as np

import run


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_native_dtw() -> None:
    libpath = run.HERE / ("masked_dtw.dylib" if run.platform.system() == "Darwin" else "masked_dtw.so")
    lib = ctypes.CDLL(str(libpath))
    fun = lib.masked_pair_dtw
    fun.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    y = np.array([[1., .9, .8, .7, .6], [1., .9, .8, .7, .6],
                  [1., .8, .6, .8, 1.]], dtype=np.float64)
    d = np.zeros_like(y)
    mask = np.ones_like(y, dtype=np.uint8)
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
    out = np.empty((2, 3), dtype=np.float64)
    fun(y.ctypes.data, d.ctypes.data, mask.ctypes.data, 5,
        pairs.ctypes.data, 2, 1, 3, 2, out.ctypes.data)
    assert np.allclose(out[0], 0), out[0]
    assert out[1, 0] > 0 and out[1, 2] > 0, out[1]
    assert out[1, 1] == 0, out[1]
    mask[2, :] = 0
    fun(y.ctypes.data, d.ctypes.data, mask.ctypes.data, 5,
        pairs.ctypes.data, 2, 1, 3, 2, out.ctypes.data)
    assert np.isinf(out[1]).all(), out[1]


def main() -> None:
    out = run.OUTPUT
    aliases = rows(out / "all_file_index.csv")
    index = rows(out / "curve_index.csv")
    assigned = rows(out / "anonymous_assignments.csv")
    queue = rows(out / "event_review_queue.csv")
    assert len(aliases) == 2250 and len(index) == len(assigned) == 2246
    assert len({r["file_id"] for r in aliases}) == 2250
    ids = {r["curve_id"] for r in index}
    assert len(ids) == 2246 and {r["curve_id"] for r in assigned} == ids
    assert {r["canonical_id"] for r in aliases} == ids
    arrays = np.load(out / "model_inputs.npz")
    model_ids = set(map(str, arrays["curve_id"]))
    eligible = {r["curve_id"] for r in index if r["model_status"] == "model_eligible"}
    assert model_ids == eligible and len(eligible) == 2065
    assert arrays["level"].shape == arrays["derivative"].shape == arrays["mask"].shape == (2065, 101)
    assert np.isfinite(arrays["level"]).all() and np.isfinite(arrays["derivative"]).all()
    assert np.array_equal(arrays["grid_hour"], np.arange(0, 201, 2))
    assert len(queue) == 200
    for r in assigned:
        if r["curve_id"] in eligible:
            assert all(r[f"cluster_fused_k{k}"] != "" for k in (4, 8, 12))
        else:
            assert r["cluster_fused_k8"] == ""
    point_file_count = 0
    for r in index:
        if not r["normalized_csv"]:
            continue
        point_file_count += 1
        values = rows(out / r["normalized_csv"])
        assert len(values) == int(r["original_points_0_200"])
        hours = [float(x["hour"]) for x in values]
        normalized = [float(x["normalized_pce"]) for x in values]
        assert hours == sorted(hours) and all(0 <= h <= 200 for h in hours)
        assert abs(max(normalized) - 1.0) < 1e-8
        divisor = float(r["normalization_divisor"])
        assert max(abs(float(x["source_y"]) / divisor - float(x["normalized_pce"]))
                   for x in values) < 2e-8
    distances = np.load(out / "dtw_distances.npz")
    p = distances["pairs"]
    assert len(p) == 126701 and np.all(p[:, 0] < p[:, 1])
    assert p.min() >= 0 and p.max() < 2065
    for key in ("level_dtw", "derivative_dtw", "fused_dtw"):
        a = distances[key]
        assert len(a) == len(p) and np.isfinite(a).all() and (a >= 0).all()
    check_native_dtw()
    summary = dict(checks="passed", original_files=len(aliases), canonical_curves=len(index),
                   normalized_point_files=point_file_count, model_eligible=len(eligible),
                   compared_pairs=len(p), event_review_rows=len(queue),
                   source_statuses=dict(Counter(r["model_status"] for r in index)))
    (out / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
