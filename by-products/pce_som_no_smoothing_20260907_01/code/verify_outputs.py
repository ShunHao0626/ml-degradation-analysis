#!/usr/bin/env python3
"""Verify core artifacts and the no-smoothing/no-label contract."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
cfg = json.loads((PROJECT / "config.json").read_text())
checks = {}
checks["smoothing_disabled_in_config"] = cfg["smoothing"]["enabled"] is False
checks["labels_disabled_in_config"] = cfg["labels_used_during_learning"] is False
source = "\n".join((PROJECT / path).read_text().lower() for path in (
    "code/run_analysis.py", "code/run_shape_sensitivity.py"
))
checks["no_savgol_import_or_call"] = "savgol_filter" not in source
checks["no_smoothing_function"] = "def smooth" not in source
required = [
    "run_manifest.json", "04_k_selection/frozen_selection.json",
    "05_final_model/assignments.csv", "05_final_model/weights.npy",
    "06_posthoc_ifo/cluster_shape_descriptors.csv", "reports/final_report_cn.md",
    "08_shape_sensitive/final_model.json", "08_shape_sensitive/clusters.png",
    "08_shape_sensitive/posthoc_individual_ifo_candidate_gallery.png",
]
checks["required_files_exist"] = all((PROJECT / p).exists() for p in required)
if checks["required_files_exist"]:
    manifest = json.loads((PROJECT / "run_manifest.json").read_text())
    assignments = pd.read_csv(PROJECT / "05_final_model/assignments.csv")
    weights = np.load(PROJECT / "05_final_model/weights.npy")
    checks["manifest_contract"] = manifest["smoothing_enabled"] is False and manifest["labels_used"] is False
    checks["assignment_count_matches"] = len(assignments) == manifest["primary_n"]
    checks["cluster_count_matches"] = len(weights) == manifest["selected_k"] == assignments["cluster"].nunique()
    checks["no_stale_cluster_artifacts"] = len(list((PROJECT / "05_final_model/clusters").glob("cluster_[0-9][0-9].png"))) == manifest["selected_k"]
(PROJECT / "verification_results.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
