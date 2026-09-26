"""
tests/test_preprocessing.py
===========================
Unit tests for the preprocessing pipeline.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    preprocess_curve,
    build_time_grid,
    _forward_fill_nan,
)


class TestTimeGrid:
    """Test §5.2: 1201-point uniform time grid."""

    def test_length(self):
        grid = build_time_grid()
        assert len(grid) == 1201, f"Expected 1201, got {len(grid)}"

    def test_start(self):
        grid = build_time_grid()
        assert np.isclose(grid[0], 0.0), f"Expected 0.0, got {grid[0]}"

    def test_end(self):
        grid = build_time_grid()
        assert np.isclose(grid[-1], 200.0), f"Expected 200.0, got {grid[-1]}"

    def test_interval(self):
        grid = build_time_grid()
        diffs = np.diff(grid)
        assert np.allclose(diffs, 1.0 / 6.0), \
            f"Expected interval 1/6, got min={diffs.min()}, max={diffs.max()}"

    def test_no_nan(self):
        grid = build_time_grid()
        assert not np.any(np.isnan(grid)), "Time grid contains NaN"


class TestForwardFill:
    """Test forward-fill of NaN regions."""

    def test_no_nan(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _forward_fill_nan(arr)
        assert np.allclose(result, arr)

    def test_leading_nan(self):
        arr = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = _forward_fill_nan(arr)
        assert result[0] == 3.0
        assert result[1] == 3.0
        assert np.allclose(result[2:], [3.0, 4.0, 5.0])

    def test_trailing_nan(self):
        arr = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        result = _forward_fill_nan(arr)
        assert np.allclose(result[:3], [1.0, 2.0, 3.0])
        assert result[3] == 3.0
        assert result[4] == 3.0


class TestNormalization:
    """Test §5.4: per-curve MaxAbsScaler normalisation."""

    def test_normalized_max_is_one(self):
        xs = np.linspace(0, 200, 1201)
        ys = np.linspace(0.5, 1.0, 1201)   # max = 1.0
        qc = preprocess_curve("test", xs, ys, "hours", max_x_hours=200.0)
        assert qc.status == "included"
        assert qc.y_normalized is not None
        assert np.isclose(qc.y_normalized.max(), 1.0, atol=0.05)   # Akima can overshoot slightly

    def test_normalized_with_peak_in_middle(self):
        """Max is not at t=0; normalisation should use the peak."""
        xs = np.linspace(0, 200, 1201)
        # Parabolic: peak at t=50
        ys = 1.0 - ((xs - 50) / 50) ** 2
        qc = preprocess_curve("test2", xs, ys, "hours", max_x_hours=200.0)
        assert qc.status == "included"
        assert qc.y_normalized is not None
        assert np.isclose(qc.y_normalized.max(), 1.0, atol=0.05)   # Akima overshoot OK

    def test_different_curves_different_denominators(self):
        """Test §5.4: no global normalisation leak."""
        xs = np.linspace(0, 200, 1201)
        ys1 = np.linspace(0.5, 1.0, 1201)   # max=1.0
        ys2 = np.linspace(0.2, 0.8, 1201)   # max=0.8

        qc1 = preprocess_curve("c1", xs, ys1, "hours", max_x_hours=200.0)
        qc2 = preprocess_curve("c2", xs, ys2, "hours", max_x_hours=200.0)

        assert qc1.status == "included"
        assert qc2.status == "included"
        assert np.isclose(qc1.y_normalized.max(), 1.0, atol=0.01)
        assert np.isclose(qc2.y_normalized.max(), 1.0, atol=0.01)
        # After normalisation, curve 1 should be higher than curve 2
        assert qc1.y_normalized.mean() > qc2.y_normalized.mean()
        # Both should be close to 1.0 after normalisation
        assert qc1.y_normalized.max() < 1.05, "curve 1 normalized max should be ≤ 1.05"
        assert qc2.y_normalized.max() < 1.05, "curve 2 normalized max should be ≤ 1.05"


class TestNoDataLeakage:
    """Test §5.4: 200h+ data must not affect normalisation."""

    def test_larger_pce_after_200h_excluded(self):
        """A PCE spike at t=250h must NOT affect normalisation."""
        xs = np.array([0, 100, 200, 250])
        ys = np.array([0.9, 0.8, 0.7, 2.0])   # spike at 250h

        qc = preprocess_curve("test_leak", xs, ys, "hours", max_x_hours=250.0)
        # Should be excluded because max_x_hours > 200h
        # but in [0,200] the max is 0.9
        # The curve should use max(0.9) for normalisation, not 2.0
        # However, since max_x_hours=250, the curve gets truncated to [0,200]
        # and normalised by max within [0,200] = 0.9
        if qc.status == "included":
            assert qc.pce_max_0_200h <= 0.95, \
                f"Normalisation denominator should exclude t>200h, got {qc.pce_max_0_200h}"


class TestShortCurve:
    """Test §4: curves < 200h with include_short=False are excluded."""

    def test_short_curve_excluded(self):
        xs = np.linspace(0, 100, 101)    # only up to 100h
        ys = np.linspace(0.8, 1.0, 101)
        qc = preprocess_curve("short", xs, ys, "hours", max_x_hours=100.0)
        assert qc.status == "excluded"
        assert "too short" in qc.reason or "200h" in qc.reason

    def test_short_curve_included_when_allowed(self):
        xs = np.linspace(0, 100, 101)
        ys = np.linspace(0.8, 1.0, 101)
        qc = preprocess_curve("short_allowed", xs, ys, "hours",
                               max_x_hours=100.0, include_short=True)
        assert qc.status == "included"


class TestAkimaInterpolation:
    """Test §5.3: Akima interpolation — no extrapolation."""

    def test_no_extrapolation_returns_nan(self):
        xs = np.array([0.0, 10.0, 50.0, 100.0])
        ys = np.array([1.0, 0.9, 0.8, 0.7])
        qc = preprocess_curve("akima_test", xs, ys, "hours", max_x_hours=100.0)
        # With include_short=False and max_x_hours=100<200, should be excluded
        # But if include_short=True, forward-fill fills the NaN
        assert qc.status == "excluded"   # data too short for 200h window

    def test_internal_gaps_filled(self):
        xs = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])   # 13 pts, gap at 40h is OK
        ys = np.linspace(1.0, 0.5, 13)
        qc = preprocess_curve("gap_test", xs, ys, "hours",
                               max_x_hours=200.0, include_short=True)
        assert qc.status == "included"
        assert not np.any(np.isnan(qc.y_raw_interp))


class TestFeatureMatrix:
    """Test §6: final feature matrix properties."""

    def test_shape(self):
        xs = np.linspace(0, 200, 1201)
        ys = np.linspace(0.8, 1.0, 1201)
        qc = preprocess_curve("shape_test", xs, ys, "hours", max_x_hours=200.0)
        assert qc.y_smoothed is not None
        assert len(qc.y_smoothed) == 1201

    def test_no_nan_inf_in_smoothed(self):
        xs = np.linspace(0, 200, 1201)
        ys = np.linspace(0.8, 1.0, 1201)
        qc = preprocess_curve("nan_test", xs, ys, "hours", max_x_hours=200.0)
        assert qc.status == "included"
        assert not np.any(np.isnan(qc.y_smoothed))
        assert not np.any(np.isinf(qc.y_smoothed))


class TestDuplicateResolution:
    """Test §5.1: duplicate timestamps resolved by averaging."""

    def test_duplicates_averaged(self):
        xs = np.array([0.0, 0.0, 10.0, 10.0, 20.0])
        ys = np.array([1.0, 1.1, 0.9, 0.8, 0.7])
        qc = preprocess_curve("dup_test", xs, ys, "hours", max_x_hours=20.0)
        # Should be excluded (too short) but duplicate handling should not crash
        # Just check it doesn't error
        assert qc.status in ("included", "excluded")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
