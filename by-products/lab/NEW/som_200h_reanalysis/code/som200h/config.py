"""Configuration for the independent 200 h SOM reanalysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    project_root: Path
    dataset_root: Path
    work_md: Path
    supplementary_md: Path

    window_hours: float = 200.0
    grid_step_hours: float = 10.0 / 60.0
    main_min_unique_points: int = 10
    paper_min_unique_points: int = 4
    savgol_window: int = 71
    savgol_polyorder: int = 2
    savgol_mode: str = "interp"

    som_sigma: float = 0.5
    som_learning_rate: float = 0.1
    som_iterations: int = 50_000
    som_primary_seed: int = 42
    som_stability_seeds: tuple[int, ...] = (7, 21, 42, 84, 168)
    som_node_counts: tuple[int, ...] = tuple(range(2, 11))

    # SI-based cluster-number decision:
    # candidate n is 4-6; choose the smallest adequate n.
    si_candidate_nodes: tuple[int, ...] = (4, 5, 6)
    min_seed_stability_ari: float = 0.80
    max_centroid_correlation: float = 0.995
    min_centroid_rmse_for_distinct: float = 0.10
    min_cluster_fraction: float = 0.01

    kmeans_n_init: int = 20
    pca_components_for_validation: int = 20
    plot_max_curves_per_cluster: int = 250

    topology_by_n: dict[int, tuple[int, int]] = field(
        default_factory=lambda: {
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (1, 5),
            6: (2, 3),
            7: (1, 7),
            8: (2, 4),
            9: (3, 3),
            10: (2, 5),
            16: (4, 4),
        }
    )

    @property
    def output_root(self) -> Path:
        return self.project_root

    @property
    def n_grid_points(self) -> int:
        return int(round(self.window_hours / self.grid_step_hours)) + 1

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("project_root", "dataset_root", "work_md", "supplementary_md"):
            data[key] = str(data[key])
        data["topology_by_n"] = {
            str(k): list(v) for k, v in self.topology_by_n.items()
        }
        data["som_stability_seeds"] = list(self.som_stability_seeds)
        data["som_node_counts"] = list(self.som_node_counts)
        data["si_candidate_nodes"] = list(self.si_candidate_nodes)
        data["n_grid_points"] = self.n_grid_points
        return data
