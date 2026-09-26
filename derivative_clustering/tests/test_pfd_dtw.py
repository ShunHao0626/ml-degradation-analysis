import numpy as np

from pfd_dtw import (
    hp_filter,
    pairwise_distance_matrix,
    pfd_dtw_distance,
    pfd_dtw_hierarchical_clustering,
    pfd_transform,
)


def test_hp_filter_reduces_second_difference_energy():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 80)
    noisy = x + rng.normal(0, 0.2, x.size)
    smooth = hp_filter(noisy, smoothing=1000)
    assert np.linalg.norm(np.diff(smooth, n=2)) < np.linalg.norm(np.diff(noisy, n=2))


def test_linear_polynomial_has_expected_derivatives():
    x = np.arange(1, 31, dtype=float)
    transformed = pfd_transform(2.5 * x - 4, smoothing=0, degree=1)
    np.testing.assert_allclose(transformed.features[:, 0], 2.5, atol=1e-10)
    np.testing.assert_allclose(transformed.features[:, 1:], 0, atol=1e-10)


def test_pfd_dtw_is_zero_for_identical_features_and_symmetric():
    first = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]])
    second = np.array([[0.1, 1.2, 1.7], [0.9, 0.1, -0.8], [1.1, 0.0, -1.0]])
    assert pfd_dtw_distance(first, first) == 0
    np.testing.assert_allclose(pfd_dtw_distance(first, second), pfd_dtw_distance(second, first))


def test_pairwise_matrix_and_clustering_pipeline():
    x = np.linspace(0, 1, 50)
    series = [x, x**1.05, 1 - x, 1 - x**1.05]
    result = pfd_dtw_hierarchical_clustering(
        series, n_clusters=2, smoothing=10, degree=3, linkage_method="complete"
    )
    assert result.distance_matrix.shape == (4, 4)
    np.testing.assert_allclose(result.distance_matrix, result.distance_matrix.T)
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]


def test_pairwise_distance_rejects_too_few_items():
    try:
        pairwise_distance_matrix([np.array([1.0])], lambda a, b: 0.0)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("expected ValueError")
