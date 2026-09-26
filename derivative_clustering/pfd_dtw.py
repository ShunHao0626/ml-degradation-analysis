"""Reproduction of the PFD-DTW-HC method proposed by Kang et al. (2024).

The implementation follows Equations (1)--(20) in the paper:

1. Hodrick--Prescott filtering.
2. Polynomial curve fitting selected by R-squared.
3. First-, second-, and third-order derivative features.
4. Weighted Euclidean local costs and dynamic time warping.
5. Hierarchical clustering from the resulting precomputed distances.

The paper does not report the derivative weights, polynomial search range, or
hierarchical linkage.  They are explicit constructor arguments here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Union

import numpy as np
from numpy.polynomial import Polynomial
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import eye as sparse_eye
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import squareform


ArrayLike1D = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class PFDTransform:
    """Intermediate products for one input time series."""

    original: np.ndarray
    smoothed: np.ndarray
    fitted: np.ndarray
    features: np.ndarray
    polynomial: Polynomial
    degree: int
    r_squared: float


@dataclass(frozen=True)
class ClusteringResult:
    """Outputs of PFD-DTW hierarchical clustering."""

    labels: np.ndarray
    distance_matrix: np.ndarray
    linkage_matrix: np.ndarray
    transforms: tuple[PFDTransform, ...]


def _as_finite_1d(values: ArrayLike1D, *, name: str = "series") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def hp_filter(series: ArrayLike1D, smoothing: float = 1000.0) -> np.ndarray:
    """Return the HP trend V = (I + lambda D.T D)^(-1) Y.

    ``D`` is the second-difference matrix used in Equations (1)--(4).  A
    sparse solve avoids explicitly forming the inverse written in the paper.
    """

    y = _as_finite_1d(series)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    n = y.size
    if n < 3 or smoothing == 0:
        return y.copy()

    diagonals = (
        np.ones(n - 2),
        -2.0 * np.ones(n - 2),
        np.ones(n - 2),
    )
    difference = diags(diagonals, offsets=(0, 1, 2), shape=(n - 2, n), format="csc")
    system = sparse_eye(n, format="csc") + smoothing * (difference.T @ difference)
    return np.asarray(spsolve(system, y), dtype=float)


def paper_r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Goodness of fit as printed in Equation (13)."""

    observed = _as_finite_1d(observed, name="observed")
    fitted = _as_finite_1d(fitted, name="fitted")
    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted must have the same shape")
    centered = observed - observed.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(observed, fitted) else 0.0
    explained = fitted - observed.mean()
    return float(np.dot(explained, explained) / denominator)


def fit_polynomial(
    series: ArrayLike1D,
    *,
    degree: int | None = None,
    max_degree: int = 20,
) -> tuple[Polynomial, np.ndarray, int, float]:
    """Fit a polynomial and return model, fitted values, degree, and R-squared.

    When ``degree`` is omitted, all degrees from 1 through ``max_degree`` are
    tested and the degree with the largest Equation (13) R-squared is used.
    The bounded search is necessary because the paper does not disclose a
    stopping rule and ordinary R-squared is non-decreasing with model degree.
    """

    y = _as_finite_1d(series)
    n = y.size
    upper = min(int(max_degree), n - 1)
    if upper < 0:
        raise ValueError("max_degree must be non-negative")

    x = np.arange(1, n + 1, dtype=float)
    if degree is not None:
        candidates = [int(degree)]
    elif upper == 0:
        candidates = [0]
    else:
        candidates = list(range(1, upper + 1))

    if any(candidate < 0 or candidate >= n for candidate in candidates):
        raise ValueError(f"polynomial degree must be between 0 and {n - 1}")

    best: tuple[float, int, Polynomial, np.ndarray] | None = None
    for candidate in candidates:
        model = Polynomial.fit(x, y, deg=candidate)
        fitted = np.asarray(model(x), dtype=float)
        score = paper_r_squared(y, fitted)
        # Prefer the smaller degree only for numerical ties.
        key = (score, -candidate)
        if best is None or key > (best[0], -best[1]):
            best = (score, candidate, model, fitted)

    assert best is not None
    score, selected_degree, model, fitted = best
    return model, fitted, selected_degree, score


def derivative_features(
    polynomial: Polynomial,
    length: int,
    orders: int = 3,
) -> np.ndarray:
    """Evaluate polynomial derivatives 1..``orders`` at time points 1..n."""

    if length <= 0:
        raise ValueError("length must be positive")
    if orders <= 0:
        raise ValueError("orders must be positive")
    x = np.arange(1, length + 1, dtype=float)
    return np.column_stack([polynomial.deriv(order)(x) for order in range(1, orders + 1)])


def pfd_transform(
    series: ArrayLike1D,
    *,
    smoothing: float = 1000.0,
    degree: int | None = None,
    max_degree: int = 20,
    derivative_orders: int = 3,
) -> PFDTransform:
    """Apply HP filtering, polynomial fitting, and PFD feature extraction."""

    original = _as_finite_1d(series)
    smoothed = hp_filter(original, smoothing=smoothing)
    polynomial, fitted, selected_degree, score = fit_polynomial(
        smoothed, degree=degree, max_degree=max_degree
    )
    features = derivative_features(polynomial, original.size, orders=derivative_orders)
    return PFDTransform(
        original=original.copy(),
        smoothed=smoothed,
        fitted=fitted,
        features=features,
        polynomial=polynomial,
        degree=selected_degree,
        r_squared=score,
    )


def _validate_weights(weights: Iterable[float] | None, dimensions: int) -> np.ndarray:
    if weights is None:
        # The paper only states k1 > k2 > k3, without numerical values.
        result = np.arange(dimensions, 0, -1, dtype=float)
    else:
        result = np.asarray(tuple(weights), dtype=float)
    if result.shape != (dimensions,):
        raise ValueError(f"expected {dimensions} weights, got shape {result.shape}")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("all weights must be finite and positive")
    return result


def pfd_dtw_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    weights: Iterable[float] | None = None,
    window: int | None = None,
) -> float:
    """Compute PFD-DTW using Equations (20) and the paper's recurrence.

    The returned value is the unnormalised cumulative path cost, as defined in
    the article.  ``window=None`` gives unconstrained DTW.
    """

    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("PFD inputs must be two-dimensional [time, derivative order]")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("PFD inputs must not be empty")
    if x.shape[1] != y.shape[1]:
        raise ValueError("PFD inputs must have the same number of derivative orders")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("PFD inputs contain NaN or infinite values")

    local_weights = _validate_weights(weights, x.shape[1])
    n, m = x.shape[0], y.shape[0]
    if window is None:
        radius = max(n, m)
    else:
        if window < 0:
            raise ValueError("window must be non-negative")
        radius = max(int(window), abs(n - m))

    previous = np.full(m + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf, dtype=float)
        lower = max(1, i - radius)
        upper = min(m, i + radius)
        for j in range(lower, upper + 1):
            delta = x[i - 1] - y[j - 1]
            local_cost = float(np.sqrt(np.dot(local_weights, delta * delta)))
            current[j] = local_cost + min(previous[j - 1], previous[j], current[j - 1])
        previous = current
    return float(previous[m])


def scalar_dtw_distance(
    first: ArrayLike1D,
    second: ArrayLike1D,
    *,
    window: int | None = None,
) -> float:
    """Conventional DTW baseline with absolute Euclidean point costs."""

    x = _as_finite_1d(first, name="first")[:, None]
    y = _as_finite_1d(second, name="second")[:, None]
    return pfd_dtw_distance(x, y, weights=(1.0,), window=window)


def pairwise_distance_matrix(
    items: Sequence[np.ndarray],
    distance: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Build a symmetric pairwise distance matrix."""

    n = len(items)
    if n < 2:
        raise ValueError("at least two items are required")
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(distance(items[i], items[j]))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid distance for pair ({i}, {j}): {value}")
            matrix[i, j] = matrix[j, i] = value
    return matrix


def hierarchical_cluster(
    distance_matrix: np.ndarray,
    *,
    n_clusters: int,
    linkage_method: str = "single",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster a precomputed distance matrix and return 0-based labels."""

    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if n_clusters < 1 or n_clusters > matrix.shape[0]:
        raise ValueError("n_clusters must be between 1 and the number of samples")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
        raise ValueError("distance_matrix must be symmetric with a zero diagonal")
    if linkage_method == "ward":
        raise ValueError("Ward linkage is invalid for these precomputed non-Euclidean distances")

    condensed = squareform(matrix, checks=True)
    linkage_matrix = linkage(condensed, method=linkage_method)
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1
    return labels.astype(int), linkage_matrix


def pfd_dtw_hierarchical_clustering(
    series: Sequence[ArrayLike1D],
    *,
    n_clusters: int,
    smoothing: float = 1000.0,
    degree: int | None = None,
    max_degree: int = 20,
    derivative_orders: int = 3,
    weights: Iterable[float] | None = None,
    window: int | None = None,
    linkage_method: str = "single",
) -> ClusteringResult:
    """Run the complete PFD-DTW-HC pipeline."""

    transforms = tuple(
        pfd_transform(
            values,
            smoothing=smoothing,
            degree=degree,
            max_degree=max_degree,
            derivative_orders=derivative_orders,
        )
        for values in series
    )
    features = [item.features for item in transforms]
    matrix = pairwise_distance_matrix(
        features,
        lambda a, b: pfd_dtw_distance(a, b, weights=weights, window=window),
    )
    labels, linkage_matrix = hierarchical_cluster(
        matrix, n_clusters=n_clusters, linkage_method=linkage_method
    )
    return ClusteringResult(labels, matrix, linkage_matrix, transforms)
