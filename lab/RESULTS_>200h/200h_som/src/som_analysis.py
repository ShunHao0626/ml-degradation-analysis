"""SOM training and analysis utilities.

Wraps MiniSom with explicit parameter handling, deterministic seeding
and BMU/distance tracking.  All tunable knobs live in `config.SOM_CONFIG`.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _import_minisom():
    try:
        from minisom import MiniSom
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "minisom is required for SOM analysis. Install via `pip install minisom`."
        ) from exc
    return MiniSom


# ----------------------------------------------------------------------------
# Configuration objects
# ----------------------------------------------------------------------------
@dataclass
class SOMSpec:
    name: str
    som_shape: tuple[int, int]
    sigma: float
    learning_rate: float
    iterations: int
    random_seed: int
    topology: str = "rectangular"
    neighborhood_function: str = "gaussian"
    activation_distance: str = "euclidean"
    initialization: str = "random_weights_init"
    training_method: str = "train"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["som_shape"] = list(self.som_shape)
        return d


def spec_from_config(name: str = "main") -> SOMSpec:
    c = config.SOM_CONFIG
    return SOMSpec(
        name=name,
        som_shape=tuple(c["shape"]),
        sigma=float(c["sigma"]),
        learning_rate=float(c["learning_rate"]),
        iterations=int(c["iterations"]),
        random_seed=int(c["random_seed"]),
        topology=c.get("topology", "rectangular"),
        neighborhood_function=c.get("neighborhood_function", "gaussian"),
        activation_distance=c.get("activation_distance", "euclidean"),
        initialization=c.get("initialization", "random_weights_init"),
        training_method=c.get("training_method", "train"),
    )


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _seed_random_state(seed: int) -> np.random.RandomState:
    """MiniSom relies on numpy.random.RandomState seeded externally.

    Some MiniSom versions also honour the global numpy random state. We
    set the seed both ways for determinism.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    return rng


def train_som(X: np.ndarray, spec: SOMSpec) -> "object":
    """Train a MiniSom model on X (n_samples × n_features) and return it."""
    MiniSom = _import_minisom()
    rng = _seed_random_state(spec.random_seed)
    n_x, n_y = spec.som_shape
    som = MiniSom(
        x=n_x,
        y=n_y,
        input_len=X.shape[1],
        sigma=spec.sigma,
        learning_rate=spec.learning_rate,
        topology=spec.topology,
        neighborhood_function=spec.neighborhood_function,
        activation_distance=spec.activation_distance,
        random_seed=spec.random_seed,
    )
    if spec.initialization == "random_weights_init":
        # MiniSom's random_weights_init uses the internal RNG seeded at
        # construction time via `random_seed`. We additionally seed the
        # global numpy RNG for determinism.
        som.random_weights_init(X)
    elif spec.initialization == "pca_weights_init":
        som.pca_weights_init(X)
    else:
        som.random_weights_init(X)
    if spec.training_method == "train_random":
        som.train_random(X, spec.iterations, verbose=False)
    else:
        som.train(X, spec.iterations, verbose=False)
    return som


# ----------------------------------------------------------------------------
# Cluster assignments
# ----------------------------------------------------------------------------
def assign_bmu_clusters(som: "object", X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bmu_xy, bmu_distance)."""
    coords = np.zeros((X.shape[0], 2), dtype=int)
    distances = np.zeros(X.shape[0], dtype=float)
    for i, x in enumerate(X):
        c = som.winner(x)
        coords[i] = c
        distances[i] = float(np.linalg.norm(som.get_weights()[c[0], c[1]] - x))
    return coords, distances


def bmu_to_cluster_id(coords: np.ndarray, som_shape: tuple[int, int]) -> np.ndarray:
    """Convert (x, y) BMU coords to flat raw cluster IDs (row-major)."""
    _, n_y = som_shape
    return coords[:, 0] * n_y + coords[:, 1]


def node_label(x: int, y: int) -> str:
    return config.cluster_node_label(x, y)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def quantization_error(som: "object", X: np.ndarray) -> float:
    """MiniSom official quantization error (mean Euclidean distance to BMU)."""
    return float(som.quantization_error(X))


def topographic_error(som: "object", X: np.ndarray) -> float:
    """MiniSom topographic error (fraction of samples whose 1st and 2nd BMU are not adjacent)."""
    try:
        return float(som.topographic_error(X))
    except Exception:  # pragma: no cover - depends on minisom version
        return float("nan")


# ----------------------------------------------------------------------------
# Cluster statistics
# ----------------------------------------------------------------------------
def summarize_clusters(
    sample_ids: list[str],
    coords: np.ndarray,
    distances: np.ndarray,
    cluster_ids: np.ndarray,
    X: np.ndarray,
    X_interpolated: np.ndarray,
    y_normalized: np.ndarray,
    max_pce_200h: list[float],
    som_shape: tuple[int, int],
    som=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two DataFrames:

    - `assignments`: per-sample BMU / cluster info
    - `summary`: per-cluster aggregates
    """
    import pandas as pd
    n_samples = len(sample_ids)
    n_x, n_y = som_shape

    rows = []
    index_of_time = lambda t_hr: int(round(t_hr / (config.GRID_STEP_HOURS)))  # noqa: E731
    t_idx = {
        "pce_norm_0h": index_of_time(0),
        "pce_norm_10h": index_of_time(10),
        "pce_norm_50h": index_of_time(50),
        "pce_norm_100h": index_of_time(100),
        "pce_norm_150h": index_of_time(150),
        "pce_norm_200h": index_of_time(200),
    }

    for i in range(n_samples):
        x_coord, y_coord = int(coords[i, 0]), int(coords[i, 1])
        cl = int(cluster_ids[i])
        row = {
            "sample_id": sample_ids[i],
            "raw_cluster_id": cl,
            "som_node_x": x_coord,
            "som_node_y": y_coord,
            "node_label": node_label(x_coord, y_coord),
            "bmu_distance": float(distances[i]),
            "max_pce_0_200h": float(max_pce_200h[i]),
            "pce_norm_0h": float(y_normalized[i, t_idx["pce_norm_0h"]]),
            "pce_norm_10h": float(y_normalized[i, t_idx["pce_norm_10h"]]),
            "pce_norm_50h": float(y_normalized[i, t_idx["pce_norm_50h"]]),
            "pce_norm_100h": float(y_normalized[i, t_idx["pce_norm_100h"]]),
            "pce_norm_150h": float(y_normalized[i, t_idx["pce_norm_150h"]]),
            "pce_norm_200h": float(y_normalized[i, t_idx["pce_norm_200h"]]),
            "peak_time_h": float(np.argmax(y_normalized[i]) * config.GRID_STEP_HOURS),
            "qc_status": "included",
        }
        rows.append(row)

    assignments = pd.DataFrame(rows)

    # Summary
    summary_rows = []
    for x_coord in range(n_x):
        for y_coord in range(n_y):
            cl = x_coord * n_y + y_coord
            mask = cluster_ids == cl
            n_curves = int(mask.sum())
            if n_curves == 0:
                summary_rows.append({
                    "raw_cluster_id": cl,
                    "som_node_x": x_coord,
                    "som_node_y": y_coord,
                    "node_label": node_label(x_coord, y_coord),
                    "n_samples": 0,
                    "fraction": 0.0,
                    "mean_curve": [],
                    "median_curve": [],
                    "std_curve": [],
                    "q25_curve": [],
                    "q75_curve": [],
                    "codebook_vector": [],
                    "mean_bmu_distance": float("nan"),
                    "median_bmu_distance": float("nan"),
                    "pce_norm_0h_mean": float("nan"),
                    "pce_norm_10h_mean": float("nan"),
                    "pce_norm_50h_mean": float("nan"),
                    "pce_norm_100h_mean": float("nan"),
                    "pce_norm_150h_mean": float("nan"),
                    "pce_norm_200h_mean": float("nan"),
                    "peak_time_h_mean": float("nan"),
                    "delta_0_10h_mean": float("nan"),
                    "delta_0_50h_mean": float("nan"),
                    "delta_0_200h_mean": float("nan"),
                    "slope_0_10h_per_h": float("nan"),
                    "slope_100_200h_per_h": float("nan"),
                    "auc_200h": float("nan"),
                    "pce_norm_remaining_200h_mean": float("nan"),
                })
                continue
            Xc = X[mask]
            ynormc = y_normalized[mask]
            distances_c = distances[mask]
            codebook = (
                som.get_weights()[x_coord, y_coord]
                if (som is not None and hasattr(som, "get_weights"))
                else Xc.mean(axis=0)
            )
            mean_curve = Xc.mean(axis=0)
            median_curve = np.median(Xc, axis=0)
            std_curve = Xc.std(axis=0)
            q25 = np.percentile(Xc, 25, axis=0)
            q75 = np.percentile(Xc, 75, axis=0)

            pce_norm_0h = float(ynormc[:, t_idx["pce_norm_0h"]].mean())
            pce_norm_10h = float(ynormc[:, t_idx["pce_norm_10h"]].mean())
            pce_norm_50h = float(ynormc[:, t_idx["pce_norm_50h"]].mean())
            pce_norm_100h = float(ynormc[:, t_idx["pce_norm_100h"]].mean())
            pce_norm_150h = float(ynormc[:, t_idx["pce_norm_150h"]].mean())
            pce_norm_200h = float(ynormc[:, t_idx["pce_norm_200h"]].mean())
            peak_time_h = float((np.argmax(ynormc, axis=1) * config.GRID_STEP_HOURS).mean())

            delta_0_10 = (pce_norm_10h - pce_norm_0h) / 10.0
            delta_0_50 = (pce_norm_50h - pce_norm_0h) / 50.0
            delta_0_200 = (pce_norm_200h - pce_norm_0h) / 200.0
            # Slope 100–200 h: Δ per hour
            slope_100_200 = (pce_norm_200h - pce_norm_100h) / (200.0 - 100.0)

            # AUC over the 0–200 h window for normalized curves
            auc = float(np.trapz(mean_curve, dx=config.GRID_STEP_HOURS))

            summary_rows.append({
                "raw_cluster_id": cl,
                "som_node_x": x_coord,
                "som_node_y": y_coord,
                "node_label": node_label(x_coord, y_coord),
                "n_samples": n_curves,
                "fraction": n_curves / n_samples,
                "mean_curve": mean_curve.tolist(),
                "median_curve": median_curve.tolist(),
                "std_curve": std_curve.tolist(),
                "q25_curve": q25.tolist(),
                "q75_curve": q75.tolist(),
                "codebook_vector": codebook.tolist(),
                "mean_bmu_distance": float(distances_c.mean()),
                "median_bmu_distance": float(np.median(distances_c)),
                "pce_norm_0h_mean": pce_norm_0h,
                "pce_norm_10h_mean": pce_norm_10h,
                "pce_norm_50h_mean": pce_norm_50h,
                "pce_norm_100h_mean": pce_norm_100h,
                "pce_norm_150h_mean": pce_norm_150h,
                "pce_norm_200h_mean": pce_norm_200h,
                "peak_time_h_mean": peak_time_h,
                "delta_0_10h_mean": delta_0_10,
                "delta_0_50h_mean": delta_0_50,
                "delta_0_200h_mean": delta_0_200,
                "slope_0_10h_per_h": delta_0_10,
                "slope_100_200h_per_h": slope_100_200,
                "auc_200h": auc,
                "pce_norm_remaining_200h_mean": pce_norm_200h,
            })
    summary = pd.DataFrame(summary_rows)
    return assignments, summary


def shape_metrics(included: list[dict], assignments: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc descriptive naming table for cluster shapes.

    Names are derived only from metrics; no hardcoded mapping from cluster
    index to physical behaviour.
    """
    import pandas as pd
    rows = []
    for cid in sorted(assignments["raw_cluster_id"].unique()):
        cluster = assignments[assignments["raw_cluster_id"] == cid]
        if cluster.empty:
            continue
        n = int(len(cluster))
        p200_mean = float(cluster["pce_norm_200h"].mean())
        p10_mean = float(cluster["pce_norm_10h"].mean())
        p0_mean = float(cluster["pce_norm_0h"].mean())
        delta_0_10 = p10_mean - p0_mean
        delta_0_200 = p200_mean - p0_mean
        slope_100_200 = float(cluster.apply(
            lambda r: (r["pce_norm_200h"] - r["pce_norm_100h"]) / 100.0, axis=1
        ).mean())
        # Tag components
        initial_change = "initial_gain" if delta_0_10 > 0.005 else ("initial_drop" if delta_0_10 < -0.005 else "no_initial_change")
        magnitude = "rapid_degradation" if abs(delta_0_200) > 0.4 else ("moderate_degradation" if abs(delta_0_200) > 0.1 else "stable")
        trajectory = "decelerating_loss" if slope_100_200 > -0.001 else "steady_loss"
        suggested = f"{initial_change}_{magnitude}_{trajectory}"
        basis = (
            f"PCE@200h={p200_mean:.3f}, slope_0_10h={delta_0_10/10.0:.5f}/h, "
            f"slope_100_200h={slope_100_200:.5f}/h, Δ200h={delta_0_200:.3f}, n={n}"
        )
        rows.append({
            "raw_cluster_id": cid,
            "n_samples": n,
            "pce_norm_0h": p0_mean,
            "pce_norm_10h": p10_mean,
            "pce_norm_200h": p200_mean,
            "delta_0_10h": delta_0_10,
            "delta_0_200h": delta_0_200,
            "slope_100_200h_per_h": slope_100_200,
            "suggested_shape_name": suggested,
            "naming_basis": basis,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Cluster-to-cluster matching across runs (Hungarian or simple cost matrix)
# ----------------------------------------------------------------------------
def match_clusters_across_runs(
    codebook_main: np.ndarray,
    codebook_other: np.ndarray,
) -> np.ndarray:
    """Return assignment: for each row in codebook_other find the row in main
    with minimum Euclidean distance.

    Uses scipy.optimize.linear_sum_assignment for the best one-to-one matching
    when both sets have the same length. Falls back to greedy otherwise.
    """
    from scipy.optimize import linear_sum_assignment
    n = codebook_main.shape[0]
    m = codebook_other.shape[0]
    if n == m:
        cost = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                cost[i, j] = float(np.linalg.norm(codebook_main[i] - codebook_other[j]))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = np.zeros(m, dtype=int)
        for ri, ci in zip(row_ind, col_ind):
            mapping[ci] = ri
        return mapping
    greedy = np.argmin(
        np.stack(
            [np.linalg.norm(codebook_main - c, axis=1) for c in codebook_other],
            axis=0
        ),
        axis=1,
    )
    return greedy


# ----------------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------------
def save_som(som: "object", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(som, fh)


def load_som(path: Path) -> "object":
    with open(path, "rb") as fh:
        return pickle.load(fh)
