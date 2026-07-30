"""
tests/test_som_outputs.py
========================
Tests for SOM training, BMU assignment, and output consistency.
"""

import numpy as np
import pytest
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.som_analysis import train_som, assign_bmu


class TestSOMOutputs:
    """Test SOM model outputs and reproducibility."""

    @pytest.fixture
    def sample_X(self):
        """Small synthetic feature matrix for testing."""
        np.random.seed(42)
        # 50 samples, 1201 features
        base = np.linspace(0, 200, 1201) / 200.0   # 0..1
        X1 = 1.0 - 0.1 * (base ** 1.5)   # stable-ish
        X2 = 1.0 - 0.5 * (base ** 0.7)   # fast decay
        X3 = 1.0 - 0.2 * base            # slow decay
        X4 = 1.0 - 0.01 * base           # very stable
        X = np.stack([X1, X2, X3, X4] * 12 + [X1, X2])
        # Add small noise
        X = X + np.random.randn(*X.shape) * 0.005
        X = np.clip(X, 0, 1.1)
        return X

    def test_som_shapes(self, sample_X):
        som, summary = train_som(sample_X, shape=[2, 2], iterations=1000)
        weights = som.get_weights()
        assert weights.shape == (2, 2, 1201)
        assert summary["som_x"] == 2
        assert summary["som_y"] == 2

    def test_all_bmu_assigned(self, sample_X):
        som, _ = train_som(sample_X, shape=[2, 2], iterations=1000)
        wx, wy, dists = assign_bmu(som, sample_X)
        assert len(wx) == len(sample_X)
        assert len(wy) == len(sample_X)
        assert len(dists) == len(sample_X)

    def test_bmu_in_range(self, sample_X):
        som, _ = train_som(sample_X, shape=[2, 2], iterations=1000)
        wx, wy, _ = assign_bmu(som, sample_X)
        assert np.all(wx >= 0) and np.all(wx < 2)
        assert np.all(wy >= 0) and np.all(wy < 2)

    def test_quantization_error_positive(self, sample_X):
        som, summary = train_som(sample_X, shape=[2, 2], iterations=1000)
        assert summary["quantization_error"] >= 0

    def test_reproducibility(self, sample_X):
        som1, s1 = train_som(sample_X, shape=[2, 2], iterations=1000, random_seed=99)
        som2, s2 = train_som(sample_X, shape=[2, 2], iterations=1000, random_seed=99)
        assert np.allclose(som1.get_weights(), som2.get_weights())
        assert s1["quantization_error"] == s2["quantization_error"]

    def test_different_seeds_different_weights(self, sample_X):
        som1, _ = train_som(sample_X, shape=[2, 2], iterations=1000, random_seed=1)
        som2, _ = train_som(sample_X, shape=[2, 2], iterations=1000, random_seed=2)
        assert not np.allclose(som1.get_weights(), som2.get_weights())

    def test_pickle_roundtrip(self, sample_X, tmp_path):
        som, _ = train_som(sample_X, shape=[2, 2], iterations=500)
        wx, wy, _ = assign_bmu(som, sample_X)

        # Save and reload
        path = tmp_path / "som_test.pkl"
        with open(path, "wb") as f:
            pickle.dump(som, f)
        with open(path, "rb") as f:
            som2 = pickle.load(f)

        wx2, wy2, _ = assign_bmu(som2, sample_X)
        assert np.array_equal(wx, wx2)
        assert np.array_equal(wy, wy2)


class TestClusterAssignment:
    """Test that every included sample gets exactly one BMU."""

    @pytest.fixture
    def sample_X(self):
        np.random.seed(7)
        X = np.random.rand(30, 1201)
        # Make sure each row normalises to ~1
        for i in range(30):
            X[i] = X[i] / X[i].max()
        return X

    def test_one_bmu_per_sample(self, sample_X):
        som, _ = train_som(sample_X, shape=[2, 2], iterations=1000)
        wx, wy, dists = assign_bmu(som, sample_X)
        assert len(wx) == len(sample_X)
        assert len(wy) == len(sample_X)
        assert len(dists) == len(sample_X)

    def test_no_unassigned(self, sample_X):
        som, _ = train_som(sample_X, shape=[2, 2], iterations=1000)
        wx, wy, _ = assign_bmu(som, sample_X)
        assert not np.any((wx < 0) | (wx >= 2))
        assert not np.any((wy < 0) | (wy >= 2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
