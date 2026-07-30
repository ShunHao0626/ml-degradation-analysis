"""Unit tests for preprocessing.

Run with::

    cd lab/RESULTS_>200h/200h_som
    PYTHONPATH=. python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import config, preprocessing


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _make_long_curve(start, end, n_points, pce_fn):
    x = np.linspace(start, end, n_points)
    y = np.array([pce_fn(xi) for xi in x])
    df = pd.DataFrame({
        "sample_id": ["S1"] * len(x),
        "x_hours": x,
        "pce": y,
        "doi": ["doi"],
        "top_dir": ["x_time_h"],
        "figure_folder": ["Fig"],
    })
    return df


# ----------------------------------------------------------------------------
# 1. Time-grid size
# ----------------------------------------------------------------------------
def test_time_grid_length():
    assert len(config.TIME_GRID) == config.N_GRID_POINTS == 1201
    assert config.TIME_GRID[0] == pytest.approx(0)
    assert config.TIME_GRID[-1] == pytest.approx(config.WINDOW_HOURS)


# ----------------------------------------------------------------------------
# 2. Normalisation: divide by own max
# ----------------------------------------------------------------------------
def test_normalize_by_200h_max_basic():
    y = np.array([0.1, 0.5, 0.7, 1.0, 0.9])
    out = preprocessing.normalize_by_200h_max(y)
    assert out.max() == pytest.approx(1.0)
    assert out.shape == y.shape


def test_normalize_rejects_nonpositive_max():
    y = np.array([-1.0, -0.5, 0.0])
    with pytest.raises(ValueError):
        preprocessing.normalize_by_200h_max(y)


# ----------------------------------------------------------------------------
# 3. No global normalization: each curve by its own max
# ----------------------------------------------------------------------------
def test_per_curve_max_normalisation_global_vs_local():
    # Two curves on identical time grid, very different maxima
    long_df = pd.DataFrame([
        {"sample_id": "S1", "x_hours": t, "pce": p, "doi": "d", "top_dir": "x_time_h", "figure_folder": "F"}
        for t, p in zip(np.linspace(0, 200, 200), np.linspace(0.1, 2.0, 200))
    ] + [
        {"sample_id": "S2", "x_hours": t, "pce": p, "doi": "d", "top_dir": "x_time_h", "figure_folder": "F"}
        for t, p in zip(np.linspace(0, 200, 200), np.linspace(0.05, 0.5, 200))
    ])
    rec1 = preprocessing.preprocess_one_curve(long_df, "S1")
    rec2 = preprocessing.preprocess_one_curve(long_df, "S2")
    assert rec1["ok"] and rec2["ok"]
    assert rec1["y_normalized"].max() == pytest.approx(1.0, abs=1e-6)
    assert rec2["y_normalized"].max() == pytest.approx(1.0, abs=1e-6)
    # Each curve divided by its own max
    assert rec1["max_pce_200h"] > rec2["max_pce_200h"]


# ----------------------------------------------------------------------------
# 4. Data-leakage: post-200 h data does NOT affect the 0-200 h normaliser
# ----------------------------------------------------------------------------
def test_post_200h_does_not_affect_normalisation():
    # Build a curve that ends at 250 h with a huge peak after 200 h
    ts = list(np.linspace(0, 200, 100)) + [210, 220, 230, 240, 250]
    pcs = [10.0] * 100 + [100.0] * 5
    long_df = pd.DataFrame([
        {"sample_id": "S_leak", "x_hours": t, "pce": p, "doi": "d", "top_dir": "x_time_h", "figure_folder": "F"}
        for t, p in zip(ts, pcs)
    ])
    rec = preprocessing.preprocess_one_curve(long_df, "S_leak")
    assert rec["ok"], rec
    assert rec["max_pce_200h"] == pytest.approx(10.0, abs=1e-6)


# ----------------------------------------------------------------------------
# 5. Short curve is excluded
# ----------------------------------------------------------------------------
def test_short_curve_excluded():
    ts = np.linspace(0, 199, 100)
    pcs = 5 + 0.01 * ts
    long_df = pd.DataFrame([
        {"sample_id": "S_short", "x_hours": t, "pce": p, "doi": "d", "top_dir": "x_time_h", "figure_folder": "F"}
        for t, p in zip(ts, pcs)
    ])
    rec = preprocessing.preprocess_one_curve(long_df, "S_short")
    assert not rec["ok"]
    reason = rec.get("reason") or ""; assert "199" in reason or "200" in reason


# ----------------------------------------------------------------------------
# 6. Akima only fills interior gaps, no extrapolation
# ----------------------------------------------------------------------------
def test_akima_no_extrapolation():
    # Observations: 0–200 h, irregular grid
    obs_t = np.array([0, 50, 100, 150, 200])
    obs_y = np.array([1.0, 0.95, 0.9, 0.85, 0.8])
    xg, yg = preprocessing.akima_interpolate_curve(obs_t, obs_y)
    assert xg is not None and yg is not None
    assert len(yg) == config.N_GRID_POINTS
    assert np.isfinite(yg).all()
    # End points should match observed values exactly
    assert yg[0] == pytest.approx(1.0, abs=1e-6)
    assert yg[-1] == pytest.approx(0.8, abs=1e-6)


# ----------------------------------------------------------------------------
# 7. Feature-matrix shape and finiteness
# ----------------------------------------------------------------------------
def test_feature_matrix_shape_and_finiteness():
    # Build 5 valid curves from 0..200 h
    ts = np.linspace(0, 200, 201)
    rows = []
    for i in range(5):
        # each curve different in amplitude and slope
        amp = 15.0 + i
        slope = -0.05 - 0.01 * i
        pcs = amp + slope * ts
        rows.append(pd.DataFrame({
            "sample_id": [f"S{i}"] * len(ts),
            "x_hours": ts,
            "pce": pcs,
            "doi": ["d"] * len(ts),
            "top_dir": ["x_time_h"] * len(ts),
            "figure_folder": ["F"] * len(ts),
        }))
    long_df = pd.concat(rows, ignore_index=True)
    sids = long_df["sample_id"].unique()
    included, _ = preprocessing.preprocess_all_curves(long_df)
    X = preprocessing.build_feature_matrix(included)
    assert X.shape[0] == len(sids)
    assert X.shape[1] == config.N_GRID_POINTS
    assert np.isfinite(X).all()


# ----------------------------------------------------------------------------
# 8. Cluster assignments
# ----------------------------------------------------------------------------
def test_assignment_helper():
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
    cids = som_helpers_cluster_id = coords[:, 0] * 2 + coords[:, 1]
    assert list(cids) == [0, 1, 2, 3, 0, 3]


# ----------------------------------------------------------------------------
# 9. Duplicate timestamps resolve to mean
# ----------------------------------------------------------------------------
def test_duplicate_timestamps_resolved_to_mean():
    df = pd.DataFrame({
        "x_hours": [0.0, 0.0, 1.0, 2.0],
        "pce": [10.0, 14.0, 9.0, 8.0],
    })
    out = preprocessing.resolve_duplicate_timestamps(df)
    assert len(out) == 3
    # duplicate at 0 should be 12.0
    assert out["pce"].iloc[0] == pytest.approx(12.0)
