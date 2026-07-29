#!/usr/bin/env python3
"""
Custom PvkSOM Pipeline for User's Raw Data
==========================================

Adapt PvkSOM pipeline to user's own raw MPPT data (30 CSV files in 原始数据/).

Usage:
    python my_pipeline.py

Input folder:  /Users/shunhao/Desktop/thesis/原始数据/
Output folder: /Users/shunhao/Desktop/thesis/abc/result/my_data/
"""

from __future__ import annotations

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler
from scipy.signal import savgol_filter
from minisom import MiniSom


# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(os.path.dirname(PROJECT_ROOT), "原始数据")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "result", "my_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOUR_LIMIT    = 150
N_DATA_POINTS = HOUR_LIMIT * 6 + 1   # = 901
SAGGOL_WINDOW = 71
SAGGOL_ORDER  = 2
SOM_SIGMA     = 0.5
SOM_LR        = 0.1
SOM_ITER      = 50000
SOM_X, SOM_Y  = 2, 2


# ── Step 1: Load & unify CSV ───────────────────────────────────────────────────
def load_csv_auto(filepath: str):
    """Auto-detect CSV format and return DataFrame with ['timestamp','Pmax']."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2000)

        # Format B: 中文 metadata header + Step/Pmax/RecordTime
        is_format_b = "样品名称" in head or "MPPT追踪" in head or \
                      "RecordTime" in head

        if is_format_b:
            data_start = 0
            for i, line in enumerate(head.split("\n")):
                if "Step" in line and "Pmax" in line:
                    data_start = i
                    break
            df = pd.read_csv(filepath, skiprows=data_start, encoding="gbk")
            if "RecordTime" not in df.columns or "Pmax" not in df.columns:
                print(f"  Skip (missing cols): {os.path.basename(filepath)}")
                return None
            df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["RecordTime"])
            df = df[["timestamp", "Pmax"]].copy()

        # Format A: standard Year,Month,Day,Hour,Minute,Pmax,Vmpp,Impp
        else:
            df = pd.read_csv(filepath)
            needed = {"Year","Month","Day","Hour","Minute","Pmax"}
            if not needed.issubset(df.columns):
                print(f"  Skip (missing cols): {os.path.basename(filepath)}")
                return None
            # Coerce Pmax to numeric, coercing errors to NaN
            df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")
            df["timestamp"] = pd.to_datetime(
                df[["Year","Month","Day","Hour","Minute"]], errors="coerce")
            df = df[["timestamp", "Pmax"]].copy()

        df = df.dropna(subset=["timestamp","Pmax"])
        if len(df) == 0:
            return None
        return df

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


# ── Step 2: Resample + truncate ────────────────────────────────────────────────
def resample_and_truncate(df: pd.DataFrame):
    """Resample to 10-min intervals and truncate to N_DATA_POINTS (901)."""
    s = (df.set_index("timestamp")["Pmax"]
           .resample("10min").mean())
    if len(s) < N_DATA_POINTS:
        return None
    s = s.iloc[1:N_DATA_POINTS + 1]   # drop first NaN row
    return s.values


# ── Step 3: Akima interpolation ────────────────────────────────────────────────
def fill_nans_akima(arr: np.ndarray) -> np.ndarray:
    """Fill NaN using Akima interpolation (fallback: linear/ffill/bfill)."""
    if not np.isnan(arr).any():
        return arr
    s = pd.Series(arr)
    s = s.interpolate(method="akima", limit_direction="both")
    s = s.fillna(method="ffill").fillna(method="bfill")
    return s.values


# ── Step 4: Normalize (MaxAbsScaler) ───────────────────────────────────────────
def normalize(arr: np.ndarray) -> np.ndarray:
    return MaxAbsScaler().fit_transform(arr.reshape(-1, 1)).flatten()


# ── Step 5: Smooth (Savitzky-Golay) ────────────────────────────────────────────
def smooth(arr: np.ndarray) -> np.ndarray:
    return savgol_filter(arr, SAGGOL_WINDOW, SAGGOL_ORDER)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Custom PvkSOM pipeline on user's raw data")
    print("="*60)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"),
                                 recursive=True))
    print(f"Found {len(csv_files)} CSV files in {RAW_DIR}")

    smoothed_list, filenames = [], []

    for fp in csv_files:
        df = load_csv_auto(fp)
        if df is None:
            continue
        arr = resample_and_truncate(df)
        if arr is None:
            print(f"  Skip (too short): {os.path.basename(fp)}")
            continue
        arr = fill_nans_akima(arr)
        arr = normalize(arr)
        arr = smooth(arr)
        smoothed_list.append(arr)
        filenames.append(os.path.relpath(fp, RAW_DIR))
        print(f"  OK: {os.path.basename(fp)} ({len(arr)} pts)")

    data = np.array(smoothed_list)
    print(f"\nFinal dataset shape: {data.shape}")

    if len(data) < 4:
        print("ERROR: Not enough valid curves to train SOM (need >=4).")
        return

    # ── Train SOM ──────────────────────────────────────────────────────────────
    som = MiniSom(x=SOM_X, y=SOM_Y, input_len=data.shape[1],
                  sigma=SOM_SIGMA, learning_rate=SOM_LR, random_seed=42)
    som.random_weights_init(data)
    som.train(data, SOM_ITER, verbose=False)

    qe = som.quantization_error(data)
    te = som.topographic_error(data)
    print(f"Quantization error: {qe:.4f}")
    print(f"Topographic error : {te:.4f}")

    labels = np.array([som.winner(d) for d in data])

    # ── Cluster distribution ───────────────────────────────────────────────────
    win_map = {}
    for i, d in enumerate(data):
        w = som.winner(d)
        win_map.setdefault(w, []).append(i)

    print("\nCluster distribution:")
    for w, idxs in sorted(win_map.items()):
        print(f"  Neuron {w}: {len(idxs)} series")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "n_series": int(len(data)),
        "series_length": int(data.shape[1]),
        "quantization_error": float(qe),
        "topographic_error":  float(te),
        "cluster_sizes": {str(k): len(v) for k, v in win_map.items()},
        "filenames": filenames,
    }
    with open(os.path.join(OUTPUT_DIR, "my_som_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Plot clusters ──────────────────────────────────────────────────────────
    time_axis = np.linspace(0, HOUR_LIMIT, data.shape[1])
    fig, axs = plt.subplots(SOM_X, SOM_Y, figsize=(4*SOM_Y, 3*SOM_X),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    for neuron_id in range(SOM_X * SOM_Y):
        ax = axs[neuron_id]
        mask = np.array([som.winner(d) == (neuron_id // SOM_Y,
                                           neuron_id % SOM_Y)
                         for d in data])
        n_in_cluster = mask.sum()
        if n_in_cluster > 0:
            for series in data[mask]:
                ax.plot(time_axis, series, alpha=0.1, linewidth=0.5,
                        color="gray")
            ax.plot(time_axis, data[mask].mean(axis=0),
                    color="red", linewidth=2, label=f"Mean (n={n_in_cluster})")
        ax.set_title(f"Neuron ({neuron_id // SOM_Y},{neuron_id % SOM_Y}) "
                     f"n={n_in_cluster}")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized Pmax")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "my_som_clusters.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/my_som_clusters.png")

    # ── Save labels ────────────────────────────────────────────────────────────
    label_df = pd.DataFrame({
        "filename": filenames,
        "cluster_x": [som.winner(d)[0] for d in data],
        "cluster_y": [som.winner(d)[1] for d in data],
    })
    label_df.to_csv(os.path.join(OUTPUT_DIR, "my_cluster_assignments.csv"),
                    index=False)
    print(f"Saved: {OUTPUT_DIR}/my_cluster_assignments.csv")
    print(f"Saved: {OUTPUT_DIR}/my_som_summary.json")

    # ── Save cluster traceability (溯源功能) ───────────────────────────────────
    print("\nSaving cluster traceability (溯源信息)...")
    
    # Build traceability JSON
    traceability = {
        "metadata": {
            "n_total_series": len(data),
            "som_grid": f"{SOM_X}x{SOM_Y}",
            "description": "Cluster traceability - maps each cluster to original files and indices"
        },
        "clusters": {}
    }
    
    for neuron_id in range(SOM_X * SOM_Y):
        coord_x = neuron_id // SOM_Y
        coord_y = neuron_id % SOM_Y
        coord = (coord_x, coord_y)
        
        mask = np.array([tuple(som.winner(d)) == coord for d in data])
        indices = np.where(mask)[0].tolist()
        cluster_files = [filenames[i] for i in indices]
        cluster_data = data[mask]
        
        traceability["clusters"][str(coord)] = {
            "neuron_coordinates": f"({coord_x}, {coord_y})",
            "neuron_id": neuron_id,
            "n_series": len(indices),
            "filenames": cluster_files,
            "data_indices": indices,
            "mean_trajectory": cluster_data.mean(axis=0).tolist() if len(indices) > 0 else [],
            "std_trajectory": cluster_data.std(axis=0).tolist() if len(indices) > 0 else [],
        }
    
    out_trace = os.path.join(OUTPUT_DIR, "my_cluster_traceability.json")
    with open(out_trace, "w", encoding="utf-8") as f:
        json.dump(traceability, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_trace}")
    
    # Save per-cluster CSV files
    for coord_key, info in traceability["clusters"].items():
        indices = info["data_indices"]
        if len(indices) > 0:
            cluster_df = pd.DataFrame({
                "sample_index": indices,
                "filename": [filenames[i] for i in indices],
                "time_series_data": [data[i].tolist() for i in indices]
            })
            cluster_csv_path = os.path.join(OUTPUT_DIR, 
                f"my_cluster_{coord_key.replace(' ', '')}_samples.csv")
            cluster_df.to_csv(cluster_csv_path, index=False)
            print(f"Saved: {cluster_csv_path}")

    # Print traceability summary
    print("\nCluster traceability summary:")
    for coord in sorted(traceability["clusters"].keys(), 
                       key=lambda x: eval(x) if isinstance(x, str) else x):
        info = traceability["clusters"][coord]
        print(f"  Cluster {coord}: {info['n_series']} files")
        for f in info["filenames"]:
            print(f"    - {f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()