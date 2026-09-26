"""Centralized configuration for the 200h SOM analysis.

All tunable parameters live here. No numbers are hardcoded elsewhere in the
pipeline. The audit document `reference_parameter_audit.md` tracks the lineage
of every parameter to the author code / paper / an implementation assumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path("/Users/shunhao/Desktop/ML")
DATASET_ROOT = REPO_ROOT / "lab" / "05_accepted_all copy"
MANIFEST_DIR = DATASET_ROOT / "curves_over_200h_full"
MANIFEST_PATH = MANIFEST_DIR / "manifest.csv"

OUTPUT_ROOT = REPO_ROOT / "lab" / "RESULTS_>200h" / "200h_som"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SRC_DIR = OUTPUT_ROOT / "src"
SCRIPTS_DIR = OUTPUT_ROOT / "scripts"
TESTS_DIR = OUTPUT_ROOT / "tests"

for _d in (OUTPUT_DIR, FIG_DIR, SRC_DIR, SCRIPTS_DIR, TESTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Time-window and grid
# ----------------------------------------------------------------------------
WINDOW_HOURS: float = 200.0          # User-requested departure from paper (150 h)
GRID_STEP_HOURS: float = 10.0 / 60.0  # 10 min in hour units
N_GRID_POINTS: int = int(round(WINDOW_HOURS / GRID_STEP_HOURS)) + 1  # 200*6+1 = 1201

TIME_GRID = [round(i * GRID_STEP_HOURS, 6) for i in range(N_GRID_POINTS)]
assert len(TIME_GRID) == N_GRID_POINTS, "Time grid length mismatch"


# ----------------------------------------------------------------------------
# Data loading configuration
# ----------------------------------------------------------------------------
DATA_CONFIG: Dict[str, Any] = {
    "id_column": "sample_id",          # constructed from file path
    "time_column": "x",                # raw column
    "pce_column": "y",                 # raw column
    "time_unit": "hour",               # canonical unit; per-row factors apply
    "time_unit_factor_by_dir": {
        "x_time_h": 1.0,
        "x_time_day": 1.0 / 24.0,      # x is in days → hours
        "x_time_min": 60.0,            # x is in minutes → hours
        "x_time_week_month_year": 1.0, # falls back to per-file validation JSON
    },
}


# ----------------------------------------------------------------------------
# Quality-control thresholds
# ----------------------------------------------------------------------------
QC_CONFIG: Dict[str, Any] = {
    "min_unique_points_200h": 4,       # Akima minimum + a small margin
    "min_coverage_fraction": 0.0,      # data must cover at least 90% of the window
    "require_min_span_h": 200.0,       # each curve must span >=200 h after shift
    "require_max_pce_positive": True,
    "strict_no_extrapolation": True,    # do not extend below min_obs or above max_obs
}


# ----------------------------------------------------------------------------
# Duplicate-handling
# ----------------------------------------------------------------------------
DUPLICATE_CONFIG: Dict[str, Any] = {
    "duplicate_strategy": "mean",      # for repeated timestamps take the mean
    "log_to": "quality_control",       # log duplicates to QC report
}


# ----------------------------------------------------------------------------
# Pre-processing pipeline order (locked)
# ----------------------------------------------------------------------------
PREPROCESS_STEPS: List[str] = [
    "convert_time_to_hours",
    "build_relative_ageing_time",
    "restrict_to_200h",
    "sort_by_time",
    "resolve_duplicate_timestamps",
    "akima_interpolate_to_grid",
    "normalize_by_200h_max",
    "apply_savgol_filter",
]


# ----------------------------------------------------------------------------
# SOM configuration
# ----------------------------------------------------------------------------
SOM_CONFIG: Dict[str, Any] = {
    "shape": [2, 2],
    "sigma": 0.5,
    "learning_rate": 0.1,
    "activation_distance": "euclidean",
    "neighborhood_function": "gaussian",   # MiniSom default
    "topology": "rectangular",             # MiniSom default
    "initialization": "random_weights_init",
    "training_method": "train",
    "iterations": 50000,
    "random_seed": 42,
    "input_len": N_GRID_POINTS,
}


# ----------------------------------------------------------------------------
# Parameter sensitivity
# ----------------------------------------------------------------------------
SENSITIVITY_CONFIGS: List[Dict[str, Any]] = [
    {"name": "sigma0.3_lr0.1", "sigma": 0.3, "learning_rate": 0.1},
    {"name": "sigma0.5_lr0.3", "sigma": 0.5, "learning_rate": 0.3},
]


# ----------------------------------------------------------------------------
# QE-sweep
# ----------------------------------------------------------------------------
QE_SWEEP: Dict[str, Any] = {
    "node_counts": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "fallback_topology": "1xn",        # when author topology unknown
    "always_include_main_topology": True,
    "main_topology": [2, 2],
    "mark_as_implementation_assumption": True,
}


# ----------------------------------------------------------------------------
# K-means validation
# ----------------------------------------------------------------------------
KMEANS_CONFIG: Dict[str, Any] = {
    "k_range": list(range(2, 11)),
    "random_state": 42,
    "n_init": 10,
    "max_iter": 300,
}


# ----------------------------------------------------------------------------
# Savitzky–Golay filter
# ----------------------------------------------------------------------------
SAVGOL_CONFIG: Dict[str, Any] = {
    "window_length": 71,
    "polyorder": 2,
    "mode": "interp",                  # implementation assumption (default)
}


# ----------------------------------------------------------------------------
# Akima interpolation
# ----------------------------------------------------------------------------
AKIMA_CONFIG: Dict[str, Any] = {
    "method": "scipy.interpolate.Akima1DInterpolator",
    "extrapolate": False,              # no out-of-range extrapolation
    "internal_nan_handling": "akima",  # Akima handles this naturally
}


# ----------------------------------------------------------------------------
# Mapping CSV
# ----------------------------------------------------------------------------
def cluster_node_label(x: int, y: int) -> str:
    return f"node_{x}_{y}"


def raw_cluster_id(x: int, y: int) -> int:
    """Convert (x, y) node coords to a flat raw cluster id (row-major)."""
    return x * SOM_CONFIG["shape"][1] + y


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
RUN_LOG_FILENAME = "run.log"
RUN_CONFIG_FILENAME = "run_config.json"

__all__ = [
    "REPO_ROOT",
    "DATASET_ROOT",
    "MANIFEST_DIR",
    "MANIFEST_PATH",
    "OUTPUT_ROOT",
    "OUTPUT_DIR",
    "FIG_DIR",
    "DATA_CONFIG",
    "QC_CONFIG",
    "DUPLICATE_CONFIG",
    "PREPROCESS_STEPS",
    "SOM_CONFIG",
    "SENSITIVITY_CONFIGS",
    "QE_SWEEP",
    "KMEANS_CONFIG",
    "SAVGOL_CONFIG",
    "AKIMA_CONFIG",
    "WINDOW_HOURS",
    "GRID_STEP_HOURS",
    "N_GRID_POINTS",
    "TIME_GRID",
    "RUN_LOG_FILENAME",
    "RUN_CONFIG_FILENAME",
    "cluster_node_label",
    "raw_cluster_id",
]
