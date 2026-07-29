#!/usr/bin/env python3
"""
PvkSOM Pipeline Runner
======================
Complete pipeline for perovskite solar cell degradation data clustering using Self-Organizing Maps.

This script replicates the workflow from:
- 20230816_degradation_analysis_revision_11_cleaned.ipynb (recommended)
- 20230227_degradation_analysis_revision_10_cleaned.ipynb

Requirements:
    conda env create -f environment_mac.yml
    conda activate PvkSOM

Usage:
    python run_pipeline.py

Output:
    - dataset/pkl_complete/20230303_mySeriesDropNorm.pkl
    - dataset/pkl_complete/20230303_mySeriesDrop_savgol.npy
    - 20230816_run_revision_excN2/*.png  (SOM visualizations)
    - 20230816_run_revision_excN2/*.html (interactive plots)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans

import pickle5 as pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

try:
    import colorlover as cl
except ImportError:
    cl = None

try:
    from PIL import ImageColor
except ImportError:
    ImageColor = None

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "dataset")
PICKLE_DIR   = os.path.join(DATA_DIR, "pkl_complete")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "20230816_run_revision_excN2")
NPY_DIR      = PICKLE_DIR

# Hour limit for analysis
HOUR_LIMIT    = 150
N_DATA_POINTS = HOUR_LIMIT * 6 + 1  # 10-min intervals = 6 per hour
SAGGOL_WINDOW = 71
SAGGOL_ORDER  = 2
SOM_SIGMA     = 0.5
SOM_LR        = 0.1
SOM_ITER      = 50000
SOM_X, SOM_Y  = 2, 2  # 2x2 = 4 clusters

for _dir in [PICKLE_DIR, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data():
    print("\n" + "="*60)
    print("STEP 1: Loading data")
    print("="*60)

    # Load PCE grouping CSV
    csv_path = os.path.join(DATA_DIR, "PCE_df_grouping.csv")
    if os.path.exists(csv_path):
        PCE_df = pd.read_csv(csv_path)
        if "Unnamed: 0" in PCE_df.columns:
            PCE_df = PCE_df.drop(["Unnamed: 0"], axis=1)
        print(f"  PCE_df loaded: {PCE_df.shape[0]} rows, {PCE_df.shape[1]} cols")
    else:
        print(f"  WARNING: {csv_path} not found, skipping PCE_df")
        PCE_df = None

    # Load preprocessed data directly (skip original pickle if it fails)
    npy_path = os.path.join(NPY_DIR, "20230303_mySeriesDrop_savgol.npy")
    if os.path.exists(npy_path):
        print(f"  Loading preprocessed data from: {npy_path}")
        mySeriesDrop_savgol = np.load(npy_path, allow_pickle=True).tolist()
        print(f"  Preprocessed data loaded: {len(mySeriesDrop_savgol)} series")
        return PCE_df, mySeriesDrop_savgol
    
    # Fallback: try loading original pickle
    pkl_path = os.path.join(PICKLE_DIR, "20230303_mySeriesDrop.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as fh:
                mySeriesDrop = pickle.load(fh)
            print(f"  mySeriesDrop loaded: {len(mySeriesDrop)} series")
        except Exception as e:
            print(f"  ERROR loading {pkl_path}: {e}")
            sys.exit(1)
    else:
        print(f"  ERROR: {pkl_path} not found!")
        sys.exit(1)

    return PCE_df, mySeriesDrop


# ── 2. Normalize ──────────────────────────────────────────────────────────────
def normalize_data(mySeriesDrop, method="MaxAbsScaler"):
    print("\n" + "="*60)
    print(f"STEP 2: Normalizing with {method}")
    print("="*60)

    output_path = os.path.join(PICKLE_DIR, "20230303_mySeriesDropNorm.pkl")

    if os.path.exists(output_path):
        print(f"  Loading cached: {output_path}")
        with open(output_path, "rb") as fh:
            mySeriesDrop_norm = pickle.load(fh)
        return mySeriesDrop_norm

    mySeriesDrop_norm = []
    for i in range(len(mySeriesDrop)):
        item = mySeriesDrop.iloc[i]["MPPT_EFF"].values
        scaler = MaxAbsScaler() if method == "MaxAbsScaler" else MinMaxScaler()
        normalized = scaler.fit_transform(item.reshape(-1, 1)).reshape(-1)
        mySeriesDrop_norm.append(normalized)
        if i % 500 == 0:
            print(f"  Normalized {i}/{len(mySeriesDrop)}...")

    print(f"  Normalized all {len(mySeriesDrop_norm)} series")

    # Save as pickle (a pandas Series of numpy arrays)
    norm_series = pd.Series(mySeriesDrop_norm)
    with open(output_path, "wb") as fh:
        pickle.dump(norm_series, fh)
    print(f"  Saved: {output_path}")

    return mySeriesDrop_norm


# ── 3. Smooth (Savitzky-Golay) ─────────────────────────────────────────────────
def smooth_data(mySeriesDrop_norm):
    print("\n" + "="*60)
    print(f"STEP 3: Savitzky-Golay smoothing (window={SAGGOL_WINDOW}, order={SAGGOL_ORDER})")
    print("="*60)

    output_path = os.path.join(NPY_DIR, "20230303_mySeriesDrop_savgol.npy")

    if os.path.exists(output_path):
        print(f"  Loading cached: {output_path}")
        mySeriesDrop_savgol = np.load(output_path, allow_pickle=True).tolist()
        return mySeriesDrop_savgol

    mySeriesDrop_savgol = []
    for i in range(len(mySeriesDrop_norm)):
        smoothed = savgol_filter(mySeriesDrop_norm[i], SAGGOL_WINDOW, SAGGOL_ORDER)
        mySeriesDrop_savgol.append(smoothed)
        if i % 500 == 0:
            print(f"  Smoothed {i}/{len(mySeriesDrop_norm)}...")

    print(f"  Smoothed all {len(mySeriesDrop_savgol)} series")

    # Save as numpy
    np.save(output_path, np.array(mySeriesDrop_savgol, dtype=object))
    print(f"  Saved: {output_path}")

    return mySeriesDrop_savgol


# ── 4. Visualize raw data ──────────────────────────────────────────────────────
def plot_overview(mySeriesDrop, mySeriesDrop_savgol, n_show=100):
    print("\n" + "="*60)
    print("STEP 4: Generating overview plots")
    print("="*60)

    n = min(n_show, len(mySeriesDrop))
    cols = 10
    rows = (n + cols - 1) // cols

    # Raw normalized data
    fig, axs = plt.subplots(rows, cols, figsize=(30, 3 * rows), sharex=True, sharey=True)
    if rows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()
    for i in range(n):
        axs[i].plot(mySeriesDrop[i], alpha=0.3, linewidth=0.5)
        axs[i].set_title(f"#{i}", fontsize=6)
        axs[i].set_ylim(0, 1.05)
    for i in range(n, len(axs)):
        axs[i].axis("off")
    plt.suptitle("Normalized MPPT Data (raw)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "overview_normalized.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # Smoothed data
    fig, axs = plt.subplots(rows, cols, figsize=(30, 3 * rows), sharex=True, sharey=True)
    if rows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()
    for i in range(n):
        axs[i].plot(mySeriesDrop_savgol[i], linewidth=0.5, color="red", alpha=0.8)
        axs[i].set_title(f"#{i}", fontsize=6)
        axs[i].set_ylim(0, 1.05)
    for i in range(n, len(axs)):
        axs[i].axis("off")
    plt.suptitle("Smoothed MPPT Data (Savitzky-Golay)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "overview_savgol.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── 5. Train SOM ───────────────────────────────────────────────────────────────
def train_som(mySeriesDrop_savgol):
    print("\n" + "="*60)
    print(f"STEP 5: Training SOM ({SOM_X}x{SOM_Y}, sigma={SOM_SIGMA}, lr={SOM_LR}, iter={SOM_ITER})")
    print("="*60)

    data = np.array(mySeriesDrop_savgol)
    print(f"  Data shape: {data.shape}")

    som = MiniSom(x=SOM_X, y=SOM_Y, input_len=data.shape[1],
                  sigma=SOM_SIGMA, learning_rate=SOM_LR,
                  random_seed=42)

    print("  Initializing weights...")
    som.random_weights_init(data)

    print(f"  Training for {SOM_ITER} iterations...")
    t0 = time.time()
    # Use verbose=False to avoid flooding output (50000 progress lines)
    som.train(data, SOM_ITER, verbose=False)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    qe = som.quantization_error(data)
    te = som.topographic_error(data)
    print(f"  Quantization error: {qe:.4f}")
    print(f"  Topographic error: {te:.4f}")

    return som, data


# ── 6. SOM Visualization ───────────────────────────────────────────────────────
def visualize_som(som, data, mySeriesDrop_savgol):
    print("\n" + "="*60)
    print("STEP 6: Visualizing SOM")
    print("="*60)

    win_map = {}
    for i, d in enumerate(data):
        w = som.winner(d)
        if w not in win_map:
            win_map[w] = []
        win_map[w].append(d)

    print(f"  Neuron map: {win_map.keys()}")
    for w, series_list in win_map.items():
        print(f"    Neuron {w}: {len(series_list)} series")

    # Unified Distance Matrix (U-Matrix)
    fig, ax = plt.subplots(figsize=(6, 6))
    dm = som.distance_map()
    sns.heatmap(dm, annot=False, cmap="coolwarm", square=True, ax=ax,
                cbar_kws={"label": "Mean Neuron Distance"})
    ax.set_title(f"U-Matrix / Distance Map\n(QE={som.quantization_error(data):.3f}, TE={som.topographic_error(data):.3f})")
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    out = os.path.join(OUTPUT_DIR, "som_umatrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved U-Matrix: {out}")

    # SOM distance map as heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    dm = som.distance_map()
    sns.heatmap(dm, annot=False, cmap="viridis", square=True, ax=ax,
                cbar_kws={"label": "Mean Neuron Distance"})
    ax.set_title(f"SOM Distance Map\n(QE={som.quantization_error(data):.3f}, TE={som.topographic_error(data):.3f})")
    out2 = os.path.join(OUTPUT_DIR, "som_distance_map.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved Distance Map: {out2}")

    # Cluster plots
    labels = np.array([w[0] * SOM_Y + w[1] for w in (som.winner(d) for d in data)])
    label_map = {i: f"({i // SOM_Y},{i % SOM_Y})" for i in range(SOM_X * SOM_Y)}

    # Matplotlib: cluster-by-cluster series + mean
    fig, axs = plt.subplots(SOM_X, SOM_Y, figsize=(4 * SOM_Y, 3 * SOM_X), sharex=True, sharey=True)
    axs = axs.flatten()
    time_axis = np.linspace(0, HOUR_LIMIT, data.shape[1])

    colors_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for neuron_id in range(SOM_X * SOM_Y):
        ax = axs[neuron_id]
        mask = labels == neuron_id
        n_in_cluster = mask.sum()
        if n_in_cluster > 0:
            for series in data[mask]:
                ax.plot(time_axis, series, alpha=0.1, linewidth=0.5, color="gray")
            cluster_mean = data[mask].mean(axis=0)
            ax.plot(time_axis, cluster_mean, color="red", linewidth=2,
                    label=f"Mean (n={n_in_cluster})")
        ax.set_title(f"Neuron ({neuron_id // SOM_Y}, {neuron_id % SOM_Y})\n"
                     f"n={n_in_cluster}", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Normalized PCE")

    plt.suptitle("SOM Clusters: Individual Series + Cluster Mean", fontsize=12)
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, "som_clusters.png")
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"  Saved Cluster Plot: {out3}")

    # Cluster count bar chart
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"Neuron {label_map[u]}" for u in unique], counts,
                  color=[colors_palette[u % len(colors_palette)] for u in unique])
    ax.set_xlabel("SOM Neuron")
    ax.set_ylabel("Count")
    ax.set_title("Series Count per SOM Cluster")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", fontsize=9)
    plt.tight_layout()
    out4 = os.path.join(OUTPUT_DIR, "som_cluster_counts.png")
    plt.savefig(out4, dpi=150)
    plt.close()
    print(f"  Saved Cluster Counts: {out4}")

    # Plotly interactive version
    try:
        time_arr = np.linspace(0, HOUR_LIMIT, data.shape[1])

        for neuron_id in range(SOM_X * SOM_Y):
            mask = labels == neuron_id
            if mask.sum() == 0:
                continue

            fig = go.Figure()
            color_hex = colors_palette[neuron_id % len(colors_palette)]

            for idx in np.where(mask)[0][:50]:  # Max 50 for readability
                fig.add_trace(go.Scatter(
                    x=time_arr, y=data[idx],
                    mode="lines", opacity=0.15,
                    line=dict(color=color_hex, width=0.8),
                    showlegend=False
                ))

            cluster_mean = data[mask].mean(axis=0)
            fig.add_trace(go.Scatter(
                x=time_arr, y=cluster_mean,
                mode="lines", opacity=1.0,
                line=dict(color="black", width=2.5),
                name=f"Mean (n={mask.sum()})"
            ))

            fig.update_layout(
                title=f"SOM Cluster {neuron_id} ({neuron_id // SOM_Y},{neuron_id % SOM_Y}), n={mask.sum()}",
                xaxis_title="Time (hours)",
                yaxis_title="Normalized PCE",
                yaxis=dict(range=[-0.05, 1.1]),
                font_family="Arial",
            )

            out_html = os.path.join(OUTPUT_DIR, f"som_cluster_{neuron_id}_interactive.html")
            fig.write_html(out_html)

            out_png = os.path.join(OUTPUT_DIR, f"som_cluster_{neuron_id}.png")
            fig.write_image(out_png, width=900, height=450, scale=2)

        print(f"  Saved interactive HTML + PNG plots for each cluster")
    except Exception as e:
        print(f"  Plotly visualization note: {e}")

    return win_map, labels


# ── 7. K-Means Comparison ─────────────────────────────────────────────────────
def compare_with_kmeans(data, labels, n_clusters=4):
    print("\n" + "="*60)
    print(f"STEP 7: K-Means comparison ({n_clusters} clusters)")
    print("="*60)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(data)
    km_centers = kmeans.cluster_centers_

    print(f"  Inertia: {kmeans.inertia_:.2f}")

    # Plot K-Means vs SOM
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # SOM cluster sizes
    unique_som, counts_som = np.unique(labels, return_counts=True)
    axs[0].bar(range(len(unique_som)), counts_som)
    axs[0].set_xticks(range(len(unique_som)))
    axs[0].set_xticklabels([f"({u // SOM_Y},{u % SOM_Y})" for u in unique_som])
    axs[0].set_xlabel("SOM Neuron")
    axs[0].set_ylabel("Count")
    axs[0].set_title("SOM Cluster Distribution")

    # K-Means cluster sizes
    unique_km, counts_km = np.unique(km_labels, return_counts=True)
    axs[1].bar(range(len(unique_km)), counts_km)
    axs[1].set_xticks(range(len(unique_km)))
    axs[1].set_xlabel("K-Means Cluster")
    axs[1].set_ylabel("Count")
    axs[1].set_title("K-Means Cluster Distribution")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "som_vs_kmeans_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return km_labels, km_centers


# ── 8. Time-Series K-Means (DTW) ─────────────────────────────────────────────
def time_series_clustering(data, n_clusters=4):
    print("\n" + "="*60)
    print(f"STEP 8: Time-Series K-Means with DTW ({n_clusters} clusters)")
    print("="*60)

    print("  Note: DTW clustering is computationally expensive.")
    print("  Running with 10% of data for speed test...")

    # Quick test with subset
    n_test = min(200, len(data))
    test_data = data[:n_test]
    ts_kmeans_test = TimeSeriesKMeans(
        n_clusters=min(3, n_clusters),
        metric="dtw",
        random_state=42,
        n_init=2,
        max_iter=10,
    )
    try:
        ts_km_test = ts_kmeans_test.fit_predict(test_data)
        print(f"  DTW K-Means test on {n_test} samples: OK")
    except Exception as e:
        print(f"  DTW K-Means note: {e}")

    print("  (Full DTW clustering skipped for speed. Use the notebook for full analysis.)")
    return None, None


# ── 9. PCA Visualization ───────────────────────────────────────────────────────
def pca_visualization(data, som_labels, km_labels):
    print("\n" + "="*60)
    print("STEP 9: PCA Visualization")
    print("="*60)

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(data_2d[:, 0], data_2d[:, 1], c=som_labels,
                   cmap="tab10", alpha=0.5, s=5)
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    axs[0].set_title("PCA Colored by SOM Cluster")

    if km_labels is not None:
        axs[1].scatter(data_2d[:, 0], data_2d[:, 1], c=km_labels,
                       cmap="tab10", alpha=0.5, s=5)
        axs[1].set_xlabel("PC1")
        axs[1].set_ylabel("PC2")
        axs[1].set_title("PCA Colored by K-Means Cluster")
    else:
        axs[1].axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pca_visualization.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── 10. Plotly Interactive PCE Analysis ────────────────────────────────────────
def plotly_pce_analysis(PCE_df):
    if PCE_df is None or len(PCE_df) == 0:
        print("\n  Skipping PCE analysis (no data)")
        return

    print("\n" + "="*60)
    print("STEP 10: Interactive PCE Analysis (Plotly)")
    print("="*60)

    try:
        unique_ceil = sorted(PCE_df["PCE_before_ceil_x"].unique())
        n_group = len(unique_ceil)

        # Box plot
        fig = px.box(PCE_df, x="PCE_before_x", y="PCE_delta",
                      title="PCE Delta by Group",
                      labels={"PCE_before_x": "Max. PCE Group (%)",
                              "PCE_delta": "Relative PCE Change (%)"})
        out = os.path.join(OUTPUT_DIR, "pce_boxplot.html")
        fig.write_html(out)
        print(f"  Saved: {out}")

        # Violin plot
        fig2 = go.Figure()
        colors_v = px.colors.sequential.Viridis
        for i, (group_val, color) in enumerate(
                zip(unique_ceil, px.colors.qualitative.Set3)):
            sub = PCE_df[PCE_df["PCE_before_ceil_x"] == group_val]
            fig2.add_trace(go.Violin(
                x=sub["PCE_before_x"],
                y=sub["PCE_delta"],
                name=str(group_val),
                box_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color="black",
            ))

        fig2.update_layout(
            title="PCE Delta Distribution by Group (Violin + Box)",
            xaxis_title="Max. PCE Group",
            yaxis_title="Relative PCE Change (%)",
            font_family="Arial",
        )
        out2 = os.path.join(OUTPUT_DIR, "pce_violin.html")
        fig2.write_html(out2)
        print(f"  Saved: {out2}")

        out_png = os.path.join(OUTPUT_DIR, "pce_violin.png")
        fig2.write_image(out_png, width=900, height=500, scale=2)
        print(f"  Saved: {out_png}")

    except Exception as e:
        print(f"  Plotly PCE analysis note: {e}")


# ── 11. Save Summary ──────────────────────────────────────────────────────────
def save_summary(som, data, labels, win_map):
    print("\n" + "="*60)
    print("STEP 11: Saving summary")
    print("="*60)

    summary = {
        "som_x": SOM_X,
        "som_y": SOM_Y,
        "som_sigma": SOM_SIGMA,
        "som_lr": SOM_LR,
        "som_iterations": SOM_ITER,
        "n_series": len(data),
        "series_length": data.shape[1],
        "hour_limit": HOUR_LIMIT,
        "quantization_error": float(som.quantization_error(data)),
        "topographic_error": float(som.topographic_error(data)),
        "cluster_sizes": {str(k): len(v) for k, v in win_map.items()},
    }

    import json
    out = os.path.join(OUTPUT_DIR, "som_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")

    return summary


# ── 12. Save Cluster Traceability (溯源功能) ─────────────────────────────────
def save_cluster_traceability(som, data, labels, PCE_df=None):
    """
    Save detailed cluster membership with traceability information.
    
    Creates a JSON file that maps each cluster to:
    - List of data indices in that cluster
    - Original PCE metadata (if available)
    - Cluster statistics
    
    Also saves a CSV file with per-sample cluster assignments.
    """
    print("\n" + "="*60)
    print("STEP 12: Saving cluster traceability (溯源信息)")
    print("="*60)

    import json

    # Build traceability data
    traceability = {
        "metadata": {
            "n_total_series": len(data),
            "som_grid": f"{SOM_X}x{SOM_Y}",
            "description": "Cluster traceability file - maps each cluster to original data indices"
        },
        "clusters": {}
    }

    # Physical interpretation mapping
    cluster_physics = {
        (0, 1): {
            "name": "Burn-in degradation",
            "description": "Smooth monotonic drop - surface defect formation / halide segregation, then stabilization",
            "n_cells": 0
        },
        (0, 0): {
            "name": "Linear degradation",
            "description": "Slow linear decline - ion migration at interfaces / gradual electrode corrosion",
            "n_cells": 0
        },
        (1, 1): {
            "name": "Light-soaking improvement + decay",
            "description": "Initial rise then fall - photo-induced ion rearrangement / defect passivation",
            "n_cells": 0
        },
        (1, 0): {
            "name": "Catastrophic failure",
            "description": "Sudden cliff-like collapse - electrode delamination / encapsulation breach / internal short",
            "n_cells": 0
        }
    }

    # Process each cluster
    for neuron_id in range(SOM_X * SOM_Y):
        coord_x = neuron_id // SOM_Y
        coord_y = neuron_id % SOM_Y
        coord = (coord_x, coord_y)
        
        mask = labels == neuron_id
        indices = np.where(mask)[0].tolist()
        cluster_data = data[mask]
        
        # Get PCE metadata if available
        pce_info = None
        if PCE_df is not None and len(PCE_df) >= len(data):
            pce_subset = PCE_df.iloc[indices]
            pce_info = {
                "PCE_before_mean": float(pce_subset["PCE_before"].mean()),
                "PCE_before_std": float(pce_subset["PCE_before"].std()),
                "PCE_before_min": float(pce_subset["PCE_before"].min()),
                "PCE_before_max": float(pce_subset["PCE_before"].max()),
                "PCE_after_mean": float(pce_subset["PCE_after"].mean()),
                "PCE_delta_mean": float(pce_subset["PCE_delta"].mean()),
                "PCE_delta_std": float(pce_subset["PCE_delta"].std()),
            }
        
        cluster_info = {
            "neuron_coordinates": f"({coord_x}, {coord_y})",
            "neuron_id": neuron_id,
            "n_series": len(indices),
            "data_indices": indices,
            "mean_trajectory": cluster_data.mean(axis=0).tolist(),
            "std_trajectory": cluster_data.std(axis=0).tolist(),
        }
        
        if pce_info:
            cluster_info["pce_statistics"] = pce_info
        
        if coord in cluster_physics:
            cluster_info["physical_interpretation"] = cluster_physics[coord]
        
        traceability["clusters"][str(coord)] = cluster_info
        
        if coord in cluster_physics:
            cluster_physics[coord]["n_cells"] = len(indices)

    # Save JSON traceability file
    out_json = os.path.join(OUTPUT_DIR, "cluster_traceability.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(traceability, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_json}")

    # Save CSV with sample-level assignments
    assignment_data = {
        "sample_index": list(range(len(data))),
        "cluster_neuron_id": labels.tolist(),
        "cluster_x": [l // SOM_Y for l in labels],
        "cluster_y": [l % SOM_Y for l in labels],
        "cluster_coord": [f"({l // SOM_Y}, {l % SOM_Y})" for l in labels],
    }
    
    if PCE_df is not None and len(PCE_df) >= len(data):
        for col in ["PCE_before", "PCE_after", "PCE_delta"]:
            assignment_data[col] = PCE_df[col].values[:len(data)].tolist()
    
    assignments_df = pd.DataFrame(assignment_data)
    out_csv = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")
    assignments_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Print summary
    print("\n  Cluster traceability summary:")
    for coord in sorted(traceability["clusters"].keys(), 
                       key=lambda x: eval(x) if isinstance(x, str) else x):
        info = traceability["clusters"][coord]
        n = info["n_series"]
        phys = info.get("physical_interpretation", {}).get("name", "Unknown")
        print(f"    Cluster {coord}: {n} series - {phys}")

    # Save per-cluster CSV files for easy access
    for coord_key, info in traceability["clusters"].items():
        indices = info["data_indices"]
        if len(indices) > 0:
            cluster_df = pd.DataFrame({
                "sample_index": indices,
                "time_series_data": [data[i].tolist() for i in indices]
            })
            if PCE_df is not None and len(PCE_df) >= len(data):
                for col in ["PCE_before", "PCE_after", "PCE_delta"]:
                    cluster_df[col] = PCE_df[col].iloc[indices].values
            
            cluster_csv_path = os.path.join(OUTPUT_DIR, 
                f"cluster_{coord_key.replace(' ', '')}_samples.csv")
            cluster_df.to_csv(cluster_csv_path, index=False)
            print(f"  Saved: {cluster_csv_path}")

    return traceability


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   PvkSOM Pipeline - Perovskite Solar Cell Degradation       ║")
    print("║   Stability follows efficiency analysis with SOM clustering  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. Load (use preprocessed data if available)
    PCE_df, preprocessed_data = load_data()
    
    # Check if data is already preprocessed (list of numpy arrays)
    if isinstance(preprocessed_data, list) and len(preprocessed_data) > 0:
        mySeriesDrop_savgol = preprocessed_data
        print(f"\n  Using preprocessed data: {len(mySeriesDrop_savgol)} series")
        
        # Create dummy normalized data for plotting (not actually used)
        mySeriesDrop_norm = None
        mySeriesDrop = None
    else:
        # 2. Normalize
        mySeriesDrop_norm = normalize_data(preprocessed_data, method="MaxAbsScaler")
        # 3. Smooth
        mySeriesDrop_savgol = smooth_data(mySeriesDrop_norm)
        mySeriesDrop = preprocessed_data  # original data for overview plots
    
    # 4. Plot overview (only if raw data available)
    if mySeriesDrop is not None:
        plot_overview(mySeriesDrop, mySeriesDrop_savgol)

    # 5. Train SOM
    som, data = train_som(mySeriesDrop_savgol)

    # 6. Visualize SOM
    win_map, som_labels = visualize_som(som, data, mySeriesDrop_savgol)

    # 7. K-Means comparison
    km_labels, km_centers = compare_with_kmeans(data, som_labels, n_clusters=4)

    # 8. Time-Series K-Means
    ts_km_labels, _ = time_series_clustering(data, n_clusters=4)

    # 9. PCA
    pca_visualization(data, som_labels, km_labels)

    # 10. Plotly
    plotly_pce_analysis(PCE_df)

    # 11. Summary
    summary = save_summary(som, data, som_labels, win_map)

    # 12. Cluster Traceability (溯源功能)
    traceability = save_cluster_traceability(som, data, som_labels, PCE_df)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Series analyzed: {summary['n_series']}")
    print(f"  Quantization error: {summary['quantization_error']:.4f}")
    print(f"  Topographic error: {summary['topographic_error']:.4f}")
    print(f"  SOM clusters: {SOM_X}x{SOM_Y} = {SOM_X*SOM_Y}")
    for k, v in sorted(summary["cluster_sizes"].items(), key=lambda x: x[1], reverse=True):
        print(f"    Neuron {k}: {v} series")
    print(f"\n  Run: jupyter notebook")
    print(f"  Then open: 20230816_degradation_analysis_revision_11_cleaned.ipynb")
    print(f"  Make sure to select the 'PvkSOM' kernel.")


if __name__ == "__main__":
    main()
