#!/usr/bin/env python3
"""Build a variable-duration, realistic four-shape curve dataset from eg references.

The reference PNGs are used qualitatively: no pixels are digitized or copied.
Every generated curve has its own duration, irregular observation grid, phase
timing, amplitudes, drift, correlated noise, nuisance events and outliers.
This script only constructs and validates the dataset; it performs no clustering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


CLASSES = ("bridge", "hill", "slope", "valley")
CORE_DEFINITION_VERSION = "ifo_presets_v2"
COLORS = {
    "bridge": "#2f6db0",
    "hill": "#d95f02",
    "slope": "#2a9d58",
    "valley": "#8e5bb7",
}
REGIMES = {
    "subhour_to_2h": (0.5, 2.0, 10),
    "short_8_to_80h": (8.0, 80.0, 35),
    "medium_120_to_1000h": (120.0, 1000.0, 35),
    "long_1200_to_4500h": (1200.0, 4500.0, 20),
}
REFERENCE_BY_REGIME = {
    "subhour_to_2h": "1299.PNG",
    "short_8_to_80h": "1297.PNG;1300.PNG;1301.PNG",
    "medium_120_to_1000h": "1294.PNG;1298.PNG;1302.PNG",
    "long_1200_to_4500h": "1295.PNG;1296.PNG;1304.PNG",
}


@dataclass
class CurveRecord:
    curve_id: str
    target_class: str
    core_definition_version: str
    duration_regime: str
    duration_h: float
    n_observations: int
    sampling_pattern: str
    reference_examples: str
    random_seed: int
    min_normalized_pce: float
    max_normalized_pce: float
    final_normalized_pce: float
    time_of_min_h: float
    time_of_max_h: float
    morphology_qc_pass: int


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def regime_sequence(class_size: int, rng: np.random.Generator) -> list[str]:
    """Allocate matched duration-regime proportions for any class size.

    Largest-remainder rounding keeps each class close to the common
    10%/35%/35%/20% target instead of adding random class-specific duration
    imbalance that a downstream model could exploit as label leakage.
    """
    regime_names = list(REGIMES)
    weights = np.asarray([item[2] for item in REGIMES.values()], dtype=float)
    weights /= weights.sum()
    expected = weights * class_size
    counts = np.floor(expected).astype(int)
    remainder = class_size - int(counts.sum())
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    labels = [name for name, count in zip(regime_names, counts) for _ in range(int(count))]
    rng.shuffle(labels)
    return labels


def sample_times(
    duration: float, regime: str, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    count_ranges = {
        "subhour_to_2h": (35, 90),
        "short_8_to_80h": (25, 75),
        "medium_120_to_1000h": (15, 50),
        "long_1200_to_4500h": (10, 35),
    }
    low, high = count_ranges[regime]
    count = int(rng.integers(low, high + 1))
    pattern = str(rng.choice(
        ["early_dense", "jittered_regular", "phase_campaigns", "hybrid"],
        p=[0.30, 0.25, 0.20, 0.25],
    ))
    if pattern == "early_dense":
        interior = np.geomspace(max(duration * 1e-4, 1e-6), duration, count - 1)
        interior *= rng.lognormal(0.0, 0.035, interior.size)
        times = np.r_[0.0, interior]
    elif pattern == "jittered_regular":
        times = np.linspace(0.0, duration, count)
        jitter = rng.normal(0.0, duration / max(count - 1, 1) * 0.20, count)
        jitter[[0, -1]] = 0.0
        times += jitter
    elif pattern == "phase_campaigns":
        campaign_count = int(rng.integers(4, 9))
        centers = np.linspace(0.04, 0.96, campaign_count)
        centers = np.clip(centers + rng.normal(0.0, 0.025, campaign_count), 0.02, 0.98)
        assignments = np.r_[centers, rng.choice(centers, size=max(0, count - 2 - campaign_count))]
        rng.shuffle(assignments)
        normalized = assignments + rng.normal(0.0, 0.018, count - 2)
        times = np.r_[0.0, np.clip(normalized, 0.0, 1.0) * duration, duration]
    else:
        early_count = max(5, count // 3)
        later_count = count - early_count - 1
        early = np.geomspace(max(duration * 1e-4, 1e-6), duration * 0.18, early_count)
        later = rng.uniform(duration * 0.15, duration, later_count)
        times = np.r_[0.0, early, later, duration]
    times = np.clip(times, 0.0, duration)
    # Round with scale-aware precision, then restore endpoints and uniqueness.
    decimals = 5 if duration < 2 else 3 if duration < 100 else 2
    times = np.unique(np.round(times, decimals=decimals))
    times = np.unique(np.r_[0.0, times, duration])
    return np.sort(times), pattern


def core_shape(
    target: str, u: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    gamma = float(rng.uniform(0.82, 1.20))
    warped = np.power(np.clip(u, 0.0, 1.0), gamma)
    baseline = float(rng.normal(1.0, 0.025))
    drift = float(rng.uniform(-0.055, 0.025))
    values = baseline + drift * warped
    params: dict[str, float] = {"time_warp_gamma": gamma, "baseline": baseline, "drift": drift}

    if target == "bridge":
        # IFO-Bridge: rapid increase -> high/broad platform -> slow decay.
        rise_gain = float(rng.uniform(0.18, 0.50))
        rise_tau = float(rng.uniform(0.025, 0.115))
        slow_loss = float(rng.uniform(0.10, 0.38))
        decay_start = float(rng.uniform(0.30, 0.58))
        decay_exponent = float(rng.uniform(0.70, 1.45))
        values += rise_gain * (1.0 - np.exp(-warped / rise_tau))
        decay_progress = np.clip((warped - decay_start) / (1.0 - decay_start), 0.0, 1.0)
        values -= slow_loss * np.power(decay_progress, decay_exponent)
        params.update(
            rise_gain=rise_gain,
            rise_tau=rise_tau,
            slow_loss=slow_loss,
            decay_start=decay_start,
            decay_exponent=decay_exponent,
        )
    elif target == "hill":
        # IFO-Hill: rapid increase -> clear peak -> rapid fall -> slow tail decay.
        height = float(rng.uniform(0.22, 0.52))
        peak_time = float(rng.uniform(0.20, 0.46))
        rise_width = float(rng.uniform(0.075, 0.18))
        rapid_decay_tau = float(rng.uniform(0.055, 0.16))
        slow_loss = float(rng.uniform(0.08, 0.32))
        slow_start = float(rng.uniform(max(peak_time + 0.15, 0.48), 0.72))
        before_peak = np.exp(-0.5 * np.square((warped - peak_time) / rise_width))
        after_peak = np.exp(-(warped - peak_time) / rapid_decay_tau)
        peak = np.where(warped <= peak_time, before_peak, after_peak)
        values += height * peak
        slow_progress = np.clip((warped - slow_start) / (1.0 - slow_start), 0.0, 1.0)
        values -= slow_loss * np.power(slow_progress, rng.uniform(0.75, 1.45))
        params.update(
            height=height,
            peak_time=peak_time,
            rise_width=rise_width,
            rapid_decay_tau=rapid_decay_tau,
            slow_loss=slow_loss,
            slow_start=slow_start,
        )
    elif target == "slope":
        burn_in = float(rng.uniform(0.08, 0.34))
        steady_loss = float(rng.uniform(0.14, 0.48))
        tau = float(rng.uniform(0.018, 0.20))
        exponent = float(rng.uniform(0.65, 1.55))
        values -= burn_in * (1.0 - np.exp(-warped / tau))
        values -= steady_loss * np.power(warped, exponent)
        if rng.random() < 0.55:
            values += rng.uniform(0.01, 0.07) * np.exp(
                -0.5 * ((warped - rng.uniform(0.25, 0.80)) / rng.uniform(0.03, 0.13)) ** 2
            )
        params.update(burn_in=burn_in, steady_loss=steady_loss, tau=tau, exponent=exponent)
    elif target == "valley":
        # IFO-Valley: rapid decay -> power-law recovery -> slow late decay.
        drop = float(rng.uniform(0.25, 0.52))
        drop_tau = float(rng.uniform(0.025, 0.12))
        recovery = float(rng.uniform(0.28, 0.50))
        recovery_start = float(rng.uniform(0.10, 0.25))
        recovery_end = float(rng.uniform(max(recovery_start + 0.25, 0.45), 0.72))
        recovery_exponent = float(rng.uniform(0.30, 0.78))
        # Tie late loss to the recovered amount: the final decay remains
        # visible but normally does not fall below the early valley minimum.
        late_loss = float(recovery * rng.uniform(0.25, 0.55))
        late_exponent = float(rng.uniform(0.72, 1.40))
        values -= drop * (1.0 - np.exp(-warped / drop_tau))
        recovery_progress = np.clip(
            (warped - recovery_start) / (recovery_end - recovery_start), 0.0, 1.0
        )
        values += recovery * np.power(recovery_progress, recovery_exponent)
        late_progress = np.clip(
            (warped - recovery_end) / (1.0 - recovery_end), 0.0, 1.0
        )
        values -= late_loss * np.power(late_progress, late_exponent)
        params.update(
            drop=drop,
            drop_tau=drop_tau,
            recovery=recovery,
            recovery_start=recovery_start,
            recovery_end=recovery_end,
            recovery_exponent=recovery_exponent,
            late_loss=late_loss,
            late_exponent=late_exponent,
        )
    else:
        raise ValueError(target)
    return values, params


def add_realistic_variation(
    values: np.ndarray, u: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float | int]]:
    result = values.copy()
    params: dict[str, float | int] = {}
    nuisance_count = int(rng.integers(1, 5))
    for index in range(nuisance_count):
        amplitude = float(rng.uniform(-0.055, 0.055))
        center = float(rng.uniform(0.04, 0.96))
        width = float(rng.uniform(0.008, 0.10))
        result += amplitude * np.exp(-0.5 * ((u - center) / width) ** 2)
        params[f"nuisance_{index}_amplitude"] = amplitude
        params[f"nuisance_{index}_center"] = center

    cycle_amplitude = 0.0
    if rng.random() < 0.28:
        cycle_amplitude = float(rng.uniform(0.008, 0.065))
        cycles = float(rng.uniform(1.5, 8.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        envelope = 0.55 + 0.45 * np.exp(-u / rng.uniform(0.15, 0.70))
        result += cycle_amplitude * envelope * np.sin(2.0 * math.pi * cycles * u + phase)
        params.update(cycles=cycles, cycle_phase=phase)

    step_amplitude = 0.0
    if rng.random() < 0.25:
        step_amplitude = float(rng.uniform(-0.10, 0.10))
        step_time = float(rng.uniform(0.12, 0.88))
        result += step_amplitude * sigmoid((u - step_time) / rng.uniform(0.002, 0.018))
        params["step_time"] = step_time

    rho = float(rng.uniform(0.72, 0.97))
    innovation_scale = float(rng.uniform(0.0015, 0.009))
    ar = np.zeros_like(result)
    innovations = rng.normal(0.0, innovation_scale, result.size)
    for index in range(1, result.size):
        ar[index] = rho * ar[index - 1] + innovations[index]
    result += ar
    params.update(
        nuisance_count=nuisance_count,
        cycle_amplitude=cycle_amplitude,
        step_amplitude=step_amplitude,
        ar_rho=rho,
        ar_innovation_scale=innovation_scale,
    )
    return result, params


def make_curve(
    target: str,
    regime: str,
    curve_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(curve_seed)
    low, high, _ = REGIMES[regime]
    duration = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    times, sampling_pattern = sample_times(duration, regime, rng)
    dense_u = np.linspace(0.0, 1.0, 1601)
    latent, core_parameters = core_shape(target, dense_u, rng)
    latent, variation_parameters = add_realistic_variation(latent, dense_u, rng)
    observed = np.interp(times / duration, dense_u, latent)
    measurement_scale = float(rng.uniform(0.003, 0.025))
    observed += rng.normal(0.0, measurement_scale * (0.75 + 0.35 * np.abs(observed)), observed.size)
    outlier_flags = np.zeros(observed.size, dtype=int)
    if observed.size >= 15 and rng.random() < 0.45:
        outlier_count = int(rng.integers(1, max(2, min(4, observed.size // 12 + 1))))
        candidates = np.arange(1, observed.size - 1)
        chosen = rng.choice(candidates, size=min(outlier_count, candidates.size), replace=False)
        observed[chosen] += rng.choice([-1.0, 1.0], chosen.size) * rng.uniform(0.035, 0.15, chosen.size)
        outlier_flags[chosen] = 1
    if rng.random() < 0.18:
        quantum = float(rng.uniform(0.005, 0.025))
        observed = np.round(observed / quantum) * quantum
    else:
        quantum = 0.0
    # Normalized PCE is defined per curve by its observed maximum.  This keeps
    # every output in [0, 1] and makes the largest measured PCE exactly 1.
    observed = np.clip(observed, 0.0, None)
    reference = float(np.max(observed))
    if reference <= 1e-12:
        reference = 1.0
    observed = np.clip(observed / reference, 0.0, 1.0)
    observed[int(np.argmax(observed))] = 1.0
    parameters: dict[str, object] = {
        "duration_h": duration,
        "sampling_pattern": sampling_pattern,
        "measurement_scale": measurement_scale,
        "quantization_step": quantum,
        "normalization": "per_curve_observed_max_equals_1",
        "normalization_reference_raw": reference,
        **core_parameters,
        **variation_parameters,
    }
    return times, observed, outlier_flags, parameters


def morphology_qc_pass(target: str, times: np.ndarray, values: np.ndarray) -> bool:
    """Strict class-shape check after sampling/noise; not a learned classifier."""
    normalized_time = times / times[-1]
    grid = np.linspace(0.0, 1.0, 101)
    smooth = savgol_filter(np.interp(grid, normalized_time, values), 21, 3)
    maximum = int(np.argmax(smooth))
    minimum = int(np.argmin(smooth))
    if target == "bridge":
        return bool(
            5 <= maximum <= 75
            and smooth[maximum] - smooth[0] >= 0.10
            and smooth[maximum] - smooth[-1] >= 0.05
        )
    if target == "hill":
        return bool(
            7 <= maximum <= 65
            and smooth[maximum] - smooth[0] >= 0.08
            and smooth[maximum] - smooth[-1] >= 0.14
        )
    if target == "slope":
        return bool(
            smooth[-1] <= smooth[0] - 0.15
            and np.mean(smooth[:20]) > np.mean(smooth[-20:])
        )
    post_recovery = minimum + int(np.argmax(smooth[minimum:]))
    return bool(
        3 <= minimum <= 50
        and smooth[post_recovery] - smooth[minimum] >= 0.10
        and smooth[post_recovery] - smooth[-1] >= 0.025
    )


def write_reference_catalog(output: Path) -> None:
    rows = [
        ("IFO preset schematic", "relative early/late phases", "Bridge/Hill/Slope/Valley core definitions supplied by user"),
        ("1294.PNG", "0-1000 h", "initial decay; recovery; broad maximum; permanent slow decay"),
        ("1295.PNG", "0-3000 h", "sparse multi-phase ageing; recovery and later severe loss; axis break"),
        ("1296.PNG", "1-3000 h", "log-time multi-phase traces; plateaus, recovery and abrupt late degradation"),
        ("1297.PNG", "0-25 h", "dense initial loss; rapid recovery; broad hill; terminal decline"),
        ("1298.PNG", "0-1000 h", "very sparse initial drop; partial recovery; slow long-term decline"),
        ("1299.PNG", "0-60 min = 0-1 h", "burn-in followed by noisy stepwise degradation and periodic forcing"),
        ("1300.PNG", "0-50 h", "sharp valley; gradual recovery; late plateau"),
        ("1301.PNG", "0-42 h", "phase switch; mixed plateau/hill/rapid failure behaviours"),
        ("1302.PNG", "0-1000 h", "sparse campaign measurements; staged performance loss"),
        ("1304.PNG", "0-3500 h", "sparse long-term non-monotonic trajectories with recovery and decay"),
    ]
    with (output / "reference_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference_image", "visible_time_range", "qualitative_features_used"])
        writer.writerows(rows)


def build_dataset(output: Path, requested_class_counts: dict[str, int], seed: int) -> None:
    if set(requested_class_counts) != set(CLASSES):
        raise ValueError(f"class counts must be supplied for exactly: {', '.join(CLASSES)}")
    if any(count <= 0 for count in requested_class_counts.values()):
        raise ValueError("all requested class counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    curves_root = output / "curves"
    rng = np.random.default_rng(seed)
    records: list[CurveRecord] = []
    parameters: list[dict[str, object]] = []
    morphology_passes = {target: 0 for target in CLASSES}
    plotted: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {
        (target, regime): [] for target in CLASSES for regime in REGIMES
    }

    with (output / "curve_data_long.csv").open("w", encoding="utf-8", newline="") as long_handle:
        long_writer = csv.DictWriter(
            long_handle,
            fieldnames=[
                "curve_id",
                "target_class",
                "observation_index",
                "time_h",
                "normalized_pce",
                "is_injected_outlier",
            ],
        )
        long_writer.writeheader()
        for class_index, target in enumerate(CLASSES):
            regimes = regime_sequence(requested_class_counts[target], rng)
            class_dir = curves_root / target
            class_dir.mkdir(parents=True, exist_ok=True)
            for index, regime in enumerate(regimes):
                curve_seed = int(seed + class_index * 1_000_000 + index * 7_919)
                curve_id = f"EGSYN_{target}_{index:03d}"
                times, values, outliers, curve_parameters = make_curve(target, regime, curve_seed)
                shape_pass = morphology_qc_pass(target, times, values)
                morphology_passes[target] += int(shape_pass)
                individual_path = class_dir / f"{curve_id}.csv"
                with individual_path.open("w", encoding="utf-8", newline="") as curve_handle:
                    writer = csv.writer(curve_handle)
                    writer.writerow(["time_h", "normalized_pce", "is_injected_outlier"])
                    for observation_index, (time_h, y_value, outlier) in enumerate(
                        zip(times, values, outliers)
                    ):
                        writer.writerow([f"{time_h:.12g}", f"{y_value:.12g}", int(outlier)])
                        long_writer.writerow(
                            {
                                "curve_id": curve_id,
                                "target_class": target,
                                "observation_index": observation_index,
                                "time_h": f"{time_h:.12g}",
                                "normalized_pce": f"{y_value:.12g}",
                                "is_injected_outlier": int(outlier),
                            }
                        )
                minimum = int(np.argmin(values))
                maximum = int(np.argmax(values))
                records.append(
                    CurveRecord(
                        curve_id=curve_id,
                        target_class=target,
                        core_definition_version=CORE_DEFINITION_VERSION,
                        duration_regime=regime,
                        duration_h=float(times[-1]),
                        n_observations=int(times.size),
                        sampling_pattern=str(curve_parameters["sampling_pattern"]),
                        reference_examples=REFERENCE_BY_REGIME[regime],
                        random_seed=curve_seed,
                        min_normalized_pce=float(values[minimum]),
                        max_normalized_pce=float(values[maximum]),
                        final_normalized_pce=float(values[-1]),
                        time_of_min_h=float(times[minimum]),
                        time_of_max_h=float(times[maximum]),
                        morphology_qc_pass=int(shape_pass),
                    )
                )
                parameters.append(
                    {
                        "curve_id": curve_id,
                        "target_class": target,
                        "core_definition_version": CORE_DEFINITION_VERSION,
                        "duration_regime": regime,
                        "reference_examples": REFERENCE_BY_REGIME[regime],
                        "random_seed": curve_seed,
                        "parameters": curve_parameters,
                    }
                )
                if len(plotted[(target, regime)]) < 7:
                    plotted[(target, regime)].append((times, values))

    with (output / "curve_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    with (output / "generation_parameters.jsonl").open("w", encoding="utf-8") as handle:
        for item in parameters:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_reference_catalog(output)

    # Structural quality controls.
    class_counts = {target: sum(record.target_class == target for record in records) for target in CLASSES}
    regime_counts = {
        target: {
            regime: sum(record.target_class == target and record.duration_regime == regime for record in records)
            for regime in REGIMES
        }
        for target in CLASSES
    }
    target_regime_proportions = {
        regime: weight / sum(item[2] for item in REGIMES.values())
        for regime, (_, _, weight) in REGIMES.items()
    }
    actual_regime_proportions = {
        target: {
            regime: regime_counts[target][regime] / class_counts[target]
            for regime in REGIMES
        }
        for target in CLASSES
    }
    maximum_regime_share_deviation = max(
        abs(actual_regime_proportions[target][regime] - target_regime_proportions[regime])
        for target in CLASSES
        for regime in REGIMES
    )
    validation = {
        "seed": seed,
        "core_definition_version": CORE_DEFINITION_VERSION,
        "curves_total": len(records),
        "curves_per_class": class_counts,
        "class_proportions": {
            target: class_counts[target] / len(records) for target in CLASSES
        },
        "class_imbalance_design": {
            "intentional": len(set(class_counts.values())) > 1,
            "purpose": "stress-test recovery of unequal target classes; not an estimate of real-world prevalence",
        },
        "duration_regime_counts_per_class": regime_counts,
        "duration_regime_target_proportions": target_regime_proportions,
        "duration_regime_actual_proportions_per_class": actual_regime_proportions,
        "duration_h_min": min(record.duration_h for record in records),
        "duration_h_median": float(np.median([record.duration_h for record in records])),
        "duration_h_max": max(record.duration_h for record in records),
        "observations_min": min(record.n_observations for record in records),
        "observations_median": float(np.median([record.n_observations for record in records])),
        "observations_max": max(record.n_observations for record in records),
        "all_time_units": "hour",
        "y_axis": "normalized_pce",
        "normalization": "each curve divided by its own observed maximum; maximum equals 1",
        "normalized_pce_range": [
            min(record.min_normalized_pce for record in records),
            max(record.max_normalized_pce for record in records),
        ],
        "fixed_analysis_window": False,
        "clustering_performed": False,
        "morphology_qc": {
            "type": "fixed rule-based post-generation check; no clustering or learned model",
            "pass_counts": morphology_passes,
            "pass_rates": {
                target: morphology_passes[target] / class_counts[target] for target in CLASSES
            },
        },
        "quality_checks": {
            "unique_curve_ids": len({record.curve_id for record in records}) == len(records),
            "positive_duration": all(record.duration_h > 0 for record in records),
            "at_least_10_observations": all(record.n_observations >= 10 for record in records),
            "normalized_pce_within_0_to_1": all(
                0.0 <= record.min_normalized_pce <= record.max_normalized_pce <= 1.0
                for record in records
            ),
            "each_curve_max_normalized_pce_equals_1": all(
                math.isclose(record.max_normalized_pce, 1.0, abs_tol=1e-12)
                for record in records
            ),
            "class_counts_match_request": class_counts == requested_class_counts,
            "intentionally_imbalanced_classes": len(set(class_counts.values())) > 1,
            "duration_regime_proportions_matched": maximum_regime_share_deviation <= 0.006,
            "maximum_duration_regime_share_deviation": maximum_regime_share_deviation,
        },
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Plot rows=class and columns=time regime, preserving actual hour axes.
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), squeeze=False)
    for row, target in enumerate(CLASSES):
        for column, regime in enumerate(REGIMES):
            axis = axes[row, column]
            for times, values in plotted[(target, regime)]:
                axis.plot(times, values, marker="o", markersize=1.8, linewidth=0.9, alpha=0.58, color=COLORS[target])
            if row == 0:
                axis.set_title(regime.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{target.capitalize()}\nnormalized PCE")
            if row == len(CLASSES) - 1:
                axis.set_xlabel("time (h)")
            axis.grid(alpha=0.18)
    fig.suptitle("EG-reference semi-synthetic dataset: variable real-hour durations", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "dataset_preview.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.8))
    bins = np.geomspace(
        min(record.duration_h for record in records) * 0.95,
        max(record.duration_h for record in records) * 1.05,
        28,
    )
    for target in CLASSES:
        durations = [record.duration_h for record in records if record.target_class == target]
        weights = np.full(len(durations), 1.0 / len(durations))
        axis.hist(
            durations,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.0,
            label=target.capitalize(),
            color=COLORS[target],
        )
    axis.set_xscale("log")
    axis.set_xlabel("curve duration (h, log scale)")
    axis.set_ylabel("fraction within class")
    axis.set_title("Matched duration distribution across the four target classes")
    axis.grid(alpha=0.2, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "duration_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 5.6))
    labels = [target.capitalize() for target in CLASSES]
    counts = [class_counts[target] for target in CLASSES]
    bars = axis.bar(labels, counts, color=[COLORS[target] for target in CLASSES], width=0.68)
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(counts) * 0.018,
            f"{count} ({count / len(records):.0%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axis.set_ylim(0, max(counts) * 1.13)
    axis.set_ylabel("curves")
    axis.set_title("Intentional class imbalance for stress testing")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "eg_based_synthetic_dataset",
    )
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument(
        "--class-counts",
        type=int,
        nargs=4,
        metavar=("BRIDGE", "HILL", "SLOPE", "VALLEY"),
        help="explicit counts in Bridge/Hill/Slope/Valley order; overrides --per-class",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.class_counts is None:
        requested_class_counts = {target: args.per_class for target in CLASSES}
    else:
        requested_class_counts = dict(zip(CLASSES, args.class_counts))
    build_dataset(args.output, requested_class_counts, args.seed)
    print((args.output / "dataset_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
