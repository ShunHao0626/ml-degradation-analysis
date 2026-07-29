#!/usr/bin/env python3
"""
Data Normalization Script for SOM Analysis
==========================================
Converts all data to unified format:
- Time unit: hours (convert from day/min/week/month/year)
- Time range: 0-200 hours
- Data points: 1200 (10-min intervals)
- PCE normalization: MaxAbsScaler

Output:
    /Users/shunhao/Desktop/ML/lab/normalized_200h/
"""

import os
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

warnings.filterwarnings("ignore")

# Configuration
TARGET_HOURS = 200
DATA_POINTS = TARGET_HOURS * 6  # 10-min intervals = 6 per hour
OUTPUT_DIR = "/Users/shunhao/Desktop/ML/lab/normalized_200h"
INPUT_BASE = "/Users/shunhao/Desktop/ML/lab/05_accepted_all copy"

# Time unit conversion factors (to hours)
TIME_CONVERSION = {
    "x_time_min": 1 / 60,           # min -> hours
    "x_time_h": 1,                   # hours -> hours
    "x_time_day": 24,                # days -> hours
    "x_time_week_month_year": None,  # need to detect per-file
}

# For week/month/year conversion
WEEK_TO_HOURS = 24 * 7
MONTH_TO_HOURS = 24 * 30
YEAR_TO_HOURS = 24 * 365


def detect_time_unit_from_content(filepath):
    """Detect time unit from CSV content by analyzing time values."""
    try:
        df = pd.read_csv(filepath)
        if 'x' not in df.columns:
            return None
        
        x_values = df['x'].values
        if len(x_values) < 2:
            return None
        
        # Check range of time values
        x_max = np.max(x_values)
        x_min = np.min(x_values)
        
        # If max is small (< 100), likely minutes
        if x_max < 100:
            return "min"
        # If max is moderate (< 1000), likely hours
        elif x_max < 1000:
            return "h"
        # If max is moderate-large (< 10000), likely days
        elif x_max < 10000:
            return "day"
        else:
            return "h"  # default to hours for large values
    except:
        return None


def get_time_conversion_factor(filepath, x_values=None):
    """
    Determine the time conversion factor.
    Priority: 1) actual data range analysis, 2) directory name
    """
    # First, try to detect from data values
    if x_values is not None and len(x_values) > 0:
        x_max = np.max(x_values)
        x_range = np.max(x_values) - np.min(x_values)
        
        # If max value is small (< 100) and range is small, likely minutes
        if x_max < 100 and x_range < 100:
            return 1 / 60  # minutes -> hours
        # If max is moderate (< 500), could be hours
        elif x_max < 500:
            return 1  # already hours
        # If max is large (500-2000), likely days
        elif x_max < 2000:
            return 24  # days -> hours
        # Very large values, could be weeks
        elif x_max < 15000:
            return 24 * 7  # weeks -> hours
        else:
            return 1  # default to hours
    
    # Fallback: use directory name
    normalized = filepath.replace("\\", "/")
    parts = normalized.split("/")
    
    category = None
    for part in parts:
        if part.startswith("x_time_"):
            category = part
            break
    
    if category == "x_time_min":
        return 1 / 60
    elif category == "x_time_h":
        return 1
    elif category == "x_time_day":
        return 24
    elif category == "x_time_week_month_year":
        return 1  # will be detected from data
    else:
        return 1


def convert_to_hours(filepath, x_values):
    """Convert time values to hours."""
    factor = get_time_conversion_factor(filepath, x_values)
    return x_values * factor


def process_single_curve(df, filepath=None, target_points=DATA_POINTS, target_hours=TARGET_HOURS):
    """
    Process a single curve to unified format.
    
    Steps:
    1. Convert time to hours
    2. Interpolate to target length
    3. Truncate to target duration (200h)
    """
    if 'x' not in df.columns or 'y' not in df.columns:
        return None
    
    x = df['x'].values.astype(float)
    y = df['y'].values.astype(float)
    
    # Remove NaN
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        return None
    
    # Convert time to hours (auto-detect based on data range)
    x_hours = convert_to_hours(filepath, x)
    
    # Remove duplicates by averaging
    unique_x, unique_idx = np.unique(x_hours, return_index=True)
    if len(unique_x) < len(x_hours):
        # Average y values for duplicate x
        y_dict = {}
        for xi, yi in zip(x_hours, y):
            if xi not in y_dict:
                y_dict[xi] = []
            y_dict[xi].append(yi)
        x_hours = np.array(list(y_dict.keys()))
        y = np.array([np.mean(y_dict[k]) for k in x_hours])
    
    # Sort by time
    sort_idx = np.argsort(x_hours)
    x_hours = x_hours[sort_idx]
    y = y[sort_idx]
    
    # Skip if data is too short (< 50 hours after filtering)
    if x_hours[-1] < 50:
        return None
    
    # Interpolate to standard grid (0 to target_hours, target_points points)
    target_x = np.linspace(0, target_hours, target_points)
    
    # Use linear interpolation, fall back to nearest for edge cases
    try:
        # Create interpolation function
        f = interpolate.interp1d(x_hours, y, kind='linear', 
                                  bounds_error=False, fill_value=(y[0], y[-1]))
        y_interp = f(target_x)
    except:
        try:
            f = interpolate.interp1d(x_hours, y, kind='nearest',
                                      bounds_error=False, fill_value=(y[0], y[-1]))
            y_interp = f(target_x)
        except:
            return None
    
    # Normalize PCE using MaxAbsScaler
    y_max = np.max(np.abs(y_interp))
    if y_max > 0:
        y_norm = y_interp / y_max
    else:
        return None
    
    return y_norm


def process_all_data():
    """Process all CSV files and save normalized data."""
    print("=" * 60)
    print("Data Normalization for SOM Analysis (200h)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all accepted CSV files
    pattern = os.path.join(INPUT_BASE, "**/accepted/*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Process each file
    all_series = []
    metadata = []
    errors = []
    
    for i, filepath in enumerate(csv_files):
        try:
            df = pd.read_csv(filepath)
            
            # Get relative path for tracking
            rel_path = os.path.relpath(filepath, INPUT_BASE)
            
            # Process each row as potential series (some files have multiple series)
            # The y column contains the PCE values, x is time
            processed = process_single_curve(df, filepath=filepath)
            
            if processed is not None and len(processed) == DATA_POINTS:
                all_series.append(processed)
                metadata.append({
                    "index": len(all_series) - 1,
                    "source_file": rel_path,
                    "original_filename": os.path.basename(filepath),
                    "doi": rel_path.split('/')[1] if len(rel_path.split('/')) > 1 else "unknown",
                })
            else:
                errors.append({"file": rel_path, "reason": "processing_failed"})
                
        except Exception as e:
            rel_path = os.path.relpath(filepath, INPUT_BASE)
            errors.append({"file": rel_path, "reason": str(e)})
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(csv_files)} files...")
    
    print(f"\n  Successfully processed: {len(all_series)} curves")
    print(f"  Failed/skipped: {len(errors)} files")
    
    if len(all_series) == 0:
        print("ERROR: No valid data found!")
        return None
    
    # Convert to numpy array
    data_array = np.array(all_series)
    print(f"\n  Data shape: {data_array.shape}")
    
    # Save as numpy array
    npy_path = os.path.join(OUTPUT_DIR, "normalized_200h.npy")
    np.save(npy_path, data_array)
    print(f"  Saved: {npy_path}")
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "n_series": len(all_series),
            "n_points": DATA_POINTS,
            "target_hours": TARGET_HOURS,
            "interval_minutes": 10,
            "metadata": metadata,
            "errors": errors[:100]  # Save first 100 errors
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {metadata_path}")
    
    # Save as CSV for easy viewing
    df_output = pd.DataFrame(all_series)
    csv_path = os.path.join(OUTPUT_DIR, "normalized_200h.csv")
    df_output.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Save summary statistics
    summary = {
        "total_files_found": len(csv_files),
        "successfully_processed": len(all_series),
        "failed_skipped": len(errors),
        "data_shape": list(data_array.shape),
        "target_hours": TARGET_HOURS,
        "data_points": DATA_POINTS,
        "interval_minutes": 10,
        "mean_by_timepoint": data_array.mean(axis=0).tolist(),
        "std_by_timepoint": data_array.std(axis=0).tolist(),
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Print sample statistics
    print(f"\n  Statistics:")
    print(f"    Mean PCE at t=0h: {data_array[:, 0].mean():.4f}")
    print(f"    Mean PCE at t=100h: {data_array[:, 600].mean():.4f}")
    print(f"    Mean PCE at t=200h: {data_array[:, -1].mean():.4f}")
    
    return data_array, metadata


if __name__ == "__main__":
    result = process_all_data()
    if result is not None:
        data, meta = result
        print("\n" + "=" * 60)
        print("Normalization complete!")
        print("=" * 60)
