import numpy as np
import pandas as pd

from eg_dtw_derivative_experiment import run_experiment as MODULE


def test_constrained_dtw_identity_and_symmetry():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 3))
    weights = np.array([0.5, 0.3, 0.2])
    assert MODULE.constrained_dtw(x, x, weights, radius=3) == 0.0
    y = rng.normal(size=(20, 3))
    np.testing.assert_allclose(
        MODULE.constrained_dtw(x, y, weights, radius=3),
        MODULE.constrained_dtw(y, x, weights, radius=3),
    )


def test_prepare_curve_uses_relative_progress_derivative():
    # y = 1 + 2u, observed at irregular physical hours.  d/du should be 2.
    time = np.array([0.0, 1.0, 4.0, 10.0, 30.0, 100.0])
    u = time / time[-1]
    frame = pd.DataFrame({"time_h": time, "y_relative": 1.0 + 2.0 * u})
    config = MODULE.Config(grid_points=48)
    _, channels, replacements, _ = MODULE.prepare_curve(frame, config)
    assert replacements == 0
    np.testing.assert_allclose(channels[:, 0], 1.0 + 2.0 * np.linspace(0, 1, 48), atol=1e-8)
    np.testing.assert_allclose(channels[4:-4, 1], 2.0, atol=1e-8)


def test_channel_scale_is_label_free_and_finite():
    values = np.arange(2 * 10 * 4, dtype=float).reshape(2, 10, 4)
    scaled, center, scale = MODULE.robust_channel_scale(values)
    assert scaled.shape == values.shape
    assert center.shape == scale.shape == (4,)
    assert np.all(np.isfinite(scaled))
    assert np.all(scale > 0)


def test_knn_affinity_is_symmetric_and_has_unit_diagonal():
    matrix = np.array(
        [[0.0, 1.0, 3.0, 4.0], [1.0, 0.0, 2.0, 5.0],
         [3.0, 2.0, 0.0, 1.0], [4.0, 5.0, 1.0, 0.0]]
    )
    affinity = MODULE.distance_to_knn_affinity(matrix, neighbors=2)
    np.testing.assert_allclose(affinity, affinity.T)
    np.testing.assert_allclose(np.diag(affinity), 1.0)
    assert np.all((affinity >= 0) & (affinity <= 1))
