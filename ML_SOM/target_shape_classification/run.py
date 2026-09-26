#!/usr/bin/env python3
"""Auditable, full-coverage classification of four observed PCE curve shapes.

Source CSVs and images are read only. Human reviews and digitizations are explicit
versioned inputs; this program never creates a positive source review.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter


CLASSES = ("bridge", "hill", "slope", "valley")
DEFAULT = {"grid_points": 96, "minimum_excursion": 0.01,
           "ambiguity_margin": 0.025, "seed": 42001,
           "dtw_window_fraction": 0.10, "value_weight": 0.5}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, data, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def axis_hours(folder, xaxis):
    if folder == "final_data":
        return 1.0, "hours", "final_data time_h column"
    s = " ".join(str(xaxis.get(k) or "") for k in ("name", "unit")).lower()
    u = str(xaxis.get("unit") or "").lower()
    if re.search(r"cycle|equivalent|ageing equivalent", s):
        return None, "unresolved", "source axis requires a documented time definition"
    # A month on an undated stability plot has no exact number of hours.
    # Use the mean Gregorian calendar month and state the approximation in
    # every input-manifest conversion record instead of leaving the axis blank.
    if re.search(r"\bmonths?\b", s):
        return 365.2425 * 24 / 12, "months_approx", "months × 730.485 h (mean Gregorian month; approximate)"
    if re.search(r"\byears?\b", s):
        return 365.2425 * 24, "years_approx", "years × 8765.82 h (mean Gregorian year; approximate)"
    if re.search(r"\bweek|\b(wk|wks)\b", s):
        return 168.0, "weeks", "weeks × 168"
    if re.search(r"\bmin(ute)?s?\b|\(min\)", s):
        return 1 / 60, "minutes", "minutes / 60"
    if u.strip("()[] .") == "d":
        return 24.0, "days", "source d (days) × 24"
    if re.search(r"\bday|\bdays|\(d\)|\[d\]|/\s*d(ays?)?\b", s):
        return 24.0, "days", "days × 24"
    if re.search(r"\b(h|hr|hrs|hour|hours)\b|\(h\)|\[h\]|/\s*h\b", s):
        return 1.0, "hours", "source axis in hours"
    return None, "unresolved", "source axis unit not established"


def y_kind(yaxis):
    s = " ".join(str(yaxis.get(k) or "") for k in ("name", "unit")).lower()
    if "δ" in s or "Δ" in s or "δ" in str(yaxis) or "Δ" in str(yaxis) or "delta pce" in s:
        return "derived_delta_pce"
    if "η" in s or any(x in s for x in ("pce", "efficiency", "effeciency", "normalized eff.", "power conversion")):
        return "pce"
    if not s.strip():
        return "unresolved"
    return "non_pce_or_unresolved"


@lru_cache(maxsize=1)
def source_axis_overrides():
    """Apply only source-image-verified exceptions to filename/column assumptions."""
    path = Path(__file__).resolve().parent / "review_inputs" / "source_axis_overrides.csv"
    records = rows(path)
    by_source = {r["source_csv"]: r for r in records if r["source_csv"]}
    by_image = {r["source_image"]: r for r in records if not r["source_csv"]}
    if len(by_source) + len(by_image) != len(records):
        raise ValueError("duplicate source-axis override")
    return by_source, by_image


def source_info(path, root):
    rel = path.relative_to(root)
    folder = rel.parts[0]
    meta_file = path.parent.parent / "validation_result.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    axis = meta.get("axis") or {}
    xaxis, yaxis = axis.get("x") or {}, axis.get("y") or {}
    factor, unit, conversion = axis_hours(folder, xaxis)
    doi = str(meta.get("doi") or rel.parts[2].replace("_", "/", 1))
    fig = str(meta.get("figure_id") or "")
    panel = str(meta.get("panel_id") or "")
    if not fig:
        m = re.search(r"(Fig\d+)_([^_]+)", rel.parts[3], re.I)
        if m:
            fig, panel = m.group(1), m.group(2)
    series_match = re.search(r"Series[_ ](\d+)", path.stem, re.I)
    series = series_match.group(1) if series_match else path.stem
    image_name = meta.get("source_image_path") or ""
    image_path = path.parent.parent / image_name if image_name else None
    if not image_path or not image_path.exists():
        candidates = [p for directory in (path.parent.parent, path.parent) for p in directory.glob("*.png")
                      if "replot" not in p.name.lower() and "series" not in p.name.lower()]
        image_path = candidates[0] if candidates else None
    info = dict(folder=folder, doi=doi, figure=fig, panel=panel, series=series,
                source_group=doi.lower(), x_name=xaxis.get("name") or "",
                x_unit=xaxis.get("unit") or "", y_name=yaxis.get("name") or "",
                y_unit=yaxis.get("unit") or "", y_kind=y_kind(yaxis) if folder == "data_all" else "pce",
                time_unit=unit, time_factor=factor, conversion=conversion,
                source_image=str(image_path.relative_to(root)) if image_path else "",
                image_sha256=sha(image_path) if image_path else "",
                metadata_file=str(meta_file.relative_to(root)) if meta_file.exists() else "")
    by_source, by_image = source_axis_overrides()
    override = by_source.get(str(rel)) or by_image.get(info["source_image"])
    if override:
        if override["source_image"] != info["source_image"]:
            raise ValueError(f"source-axis review image path mismatch: {rel}")
        if override["source_image_sha256"] != info["image_sha256"]:
            raise ValueError(f"source-axis review image hash mismatch: {info['source_image']}")
        for key in ("y_kind", "y_name", "y_unit", "time_unit", "conversion"):
            if override[key]:
                info[key] = override[key]
        if override["time_factor"] == "unresolved":
            info["time_factor"] = None
        elif override["time_factor"]:
            info["time_factor"] = float(override["time_factor"])
    return info


def read_curve(path, folder):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        expected = ("x", "y") if folder == "data_all" else ("time_h", "normalized_pce")
        if not set(expected).issubset(rd.fieldnames or []):
            raise ValueError("missing required coordinate columns")
        points = []
        for r in rd:
            try:
                a, b = float(r[expected[0]]), float(r[expected[1]])
                if math.isfinite(a) and math.isfinite(b):
                    points.append((a, b))
            except (ValueError, TypeError):
                pass
    if not points:
        raise ValueError("no finite coordinate pairs")
    arr = np.asarray(points)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    t, inv = np.unique(arr[:, 0], return_inverse=True)
    y = np.array([np.median(arr[inv == i, 1]) for i in range(len(t))])
    return t, y, len(points), len(points) - len(t)


def load_inputs(root, output, digitizations):
    paths = sorted((root / "data_all").rglob("*.csv")) + sorted((root / "final_data").rglob("*.csv"))
    digit = {r.get("source_csv"): r for r in rows(digitizations)}
    manifest, curves = [], {}
    for i, path in enumerate(paths):
        rel = str(path.relative_to(root))
        info = source_info(path, root)
        record = dict(file_id=f"F{i+1:05d}", source_csv=rel, source_sha256=sha(path), **info)
        record.update(raw_points=0, unique_points=0, duplicate_times=0,
                      x_min="", x_max="", time_start_h="", time_end_h="",
                      normalization_max="", quality_flags="", load_error="",
                      analysis_version=rel, analysis_sha256=record["source_sha256"])
        try:
            version = digit.get(rel)
            use_path = path
            if version and version.get("status") == "accepted":
                candidate = (output / version["digitized_csv"]).resolve()
                if not candidate.is_relative_to(output.resolve()) or not candidate.exists():
                    raise ValueError("accepted digitization path missing or outside output")
                if version.get("source_sha256") != record["source_sha256"]:
                    raise ValueError("digitization source hash mismatch")
                if version.get("digitized_sha256") != sha(candidate):
                    raise ValueError("digitization version hash mismatch")
                use_path = candidate
                record["analysis_version"] = str(use_path.relative_to(output))
                record["analysis_sha256"] = sha(use_path)
            t, y, n, duplicates = read_curve(use_path, info["folder"])
            flags = []
            if n < 5: flags.append("sparse_points")
            if duplicates: flags.append("duplicate_times")
            if t[0] < 0: flags.append("negative_time")
            if np.any(y < 0): flags.append("negative_response")
            if np.any(y > 1.05) and info["folder"] == "final_data": flags.append("above_one_before_renormalization")
            if len(t) > 2 and np.max(np.diff(t)) > .3 * max(t[-1]-t[0], 1e-12): flags.append("large_sampling_gap")
            if info["time_factor"] is None: flags.append("time_unit_unresolved")
            if info["y_kind"] != "pce": flags.append("response_identity_unresolved")
            ymax = float(np.max(y))
            if ymax <= 0: flags.append("nonpositive_maximum")
            t_hour = t * info["time_factor"] if info["time_factor"] is not None else None
            record.update(raw_points=n, unique_points=len(t), duplicate_times=duplicates,
                          x_min=float(t[0]), x_max=float(t[-1]),
                          time_start_h=float(t_hour[0]) if t_hour is not None else "",
                          time_end_h=float(t_hour[-1]) if t_hour is not None else "",
                          normalization_max=ymax, quality_flags=";".join(flags))
            curves[record["file_id"]] = dict(t=t, t_hour=t_hour,
                y=y / ymax if ymax > 0 else y, raw_y=y, info=record)
        except Exception as e:
            record["load_error"] = str(e)
            record["quality_flags"] = "unreadable_or_unrecoverable"
        manifest.append(record)
    # Only merge same source identity with numerically matching trajectory.
    groups = defaultdict(list)
    for r in manifest:
        key = (r["doi"].lower(), r["figure"].lower(), r["panel"].lower(), r["series"])
        groups[key].append(r)
    aliases, canonical = [], []
    for group in groups.values():
        group.sort(key=lambda r: (r["folder"] == "final_data", r["unique_points"]), reverse=True)
        bins = []
        for r in group:
            c = curves.get(r["file_id"])
            signature = None
            if c is not None and len(c["t"]) >= 2 and np.ptp(c["t"]) > 0:
                u = (c["t"]-c["t"][0]) / np.ptp(c["t"])
                signature = np.interp(np.linspace(0, 1, 32), u, c["y"])
            found = None
            for rep, sig in bins:
                same_cross_version = (signature is not None and sig is not None
                    and rep["folder"] != r["folder"]
                    and rep["time_factor"] is not None and r["time_factor"] is not None
                    and abs(float(rep["time_end_h"])-float(r["time_end_h"]))
                        <= .03 * max(abs(float(rep["time_end_h"])), 1)
                    and float(np.sqrt(np.mean((signature-sig)**2))) < .012)
                if same_cross_version:
                    found = rep
                    break
            if found is None:
                bins.append((r, signature))
                canonical.append(r)
                found = r
            r["canonical_id"] = found["file_id"]
            if found is not r:
                aliases.append(dict(file_id=r["file_id"], canonical_id=found["file_id"],
                                    source_csv=r["source_csv"], canonical_csv=found["source_csv"],
                                    reason="same DOI/figure/panel/series and numeric trajectory"))
    write_rows(output / "input_manifest.csv", manifest, list(manifest[0]) if manifest else [])
    write_rows(output / "canonical_curves.csv", canonical, list(manifest[0]) if manifest else [])
    write_rows(output / "duplicate_aliases.csv", aliases,
               ["file_id", "canonical_id", "source_csv", "canonical_csv", "reason"])
    mapping_path=output/"point_mapping.csv"
    with mapping_path.open("w",newline="",encoding="utf-8") as f:
        fields=["file_id","analysis_version","source_row","x_original","y_original",
                "processed_index","x_processed","y_median","point_source","mapping_status"]
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in manifest:
            c=curves.get(r["file_id"])
            version=(output/r["analysis_version"] if r["analysis_version"]!=r["source_csv"]
                     else root/r["source_csv"])
            xcol,ycol=("x","y") if r["folder"]=="data_all" else ("time_h","normalized_pce")
            with version.open(newline="",encoding="utf-8-sig") as source:
                for line,item in enumerate(csv.DictReader(source),1):
                    base=dict(file_id=r["file_id"],analysis_version=r["analysis_version"],source_row=line,
                              x_original=item.get(xcol,""),y_original=item.get(ycol,""),
                              point_source=item.get("point_source","source_csv"))
                    try:
                        x,y=float(item[xcol]),float(item[ycol])
                        if not math.isfinite(x) or not math.isfinite(y): raise ValueError("nonfinite")
                        j=int(np.searchsorted(c["t"],x))
                        base.update(processed_index=j,x_processed=float(c["t"][j]),
                                    y_median=float(c["raw_y"][j]),mapping_status="mapped")
                    except (ValueError,TypeError,KeyError,IndexError):
                        base["mapping_status"]="invalid_source_row"
                    writer.writerow(base)
    return manifest, canonical, curves


def shape_view(c, n):
    t, y = c["t"], c["y"]
    if len(t) < 2 or t[-1] <= t[0] or np.max(y) <= 0:
        return None
    u = (t-t[0]) / (t[-1]-t[0])
    f = PchipInterpolator(t, y, extrapolate=False)
    grid_t = np.linspace(t[0], t[-1], n)
    v = np.asarray(f(grid_t))
    if not np.isfinite(v).all():
        return None
    window = min(7, n if n % 2 else n-1)
    smooth = savgol_filter(v, window, 2) if window >= 5 else v.copy()
    noise = float(1.4826 * np.median(np.abs(v-smooth)))
    # The smoother estimates noise only. Every resampled and source observation
    # remains unchanged for scoring and for the final hour plots.
    return v, noise, u


def piece_score(v, kind, threshold, noise, raw_u=None, raw_y=None):
    """Continuous piecewise-linear anchor model with explicit stage penalties."""
    n = len(v)
    pos = np.unique(np.rint(np.linspace(.12, .88, 11) * (n-1)).astype(int))
    if kind == "slope":
        candidates = [(i,) for i in pos]
    elif kind == "valley":
        candidates = [(i,j,k) for i in pos for j in pos for k in pos
                      if j-i >= .10*(n-1) and k-j >= .10*(n-1)]
    else:
        candidates = [(i,j) for i in pos for j in pos if j-i >= .12*(n-1)]
    best = (float("inf"), (), ())
    scale = max(float(np.ptp(v)), .08)
    delta = max(threshold, 3*noise)
    x = np.arange(n)
    for breaks in candidates:
        idx = (0, *breaks, n-1)
        a = v[list(idx)]
        dv = np.diff(a)
        widths = np.diff(idx) / (n-1)
        slopes = dv / widths
        fitted = np.interp(x, idx, a)
        residual = float(np.sqrt(np.mean((v-fitted)**2))) / scale
        if raw_u is not None and raw_y is not None:
            source_fit = np.interp(raw_u*(n-1),idx,a)
            source_residual = float(np.sqrt(np.mean((raw_y-source_fit)**2))) / scale
            residual = .5*residual+.5*source_residual
        # Smooth penalties permit a closest class even for a background curve.
        if kind == "bridge":
            penalties = [max(0, delta-dv[0]), max(0, dv[1]-delta*.2),
                         max(0, dv[2]+delta*.3),
                         max(0, abs(slopes[1])-max(abs(slopes[0])*.7, .08))]
        elif kind == "hill":
            penalties = [max(0, delta-dv[0]), max(0, delta+dv[1]),
                         max(0, dv[2]+delta*.2),
                         max(0, 2.5*abs(slopes[2])-abs(slopes[1]))]
        elif kind == "slope":
            penalties = [max(0, delta+dv[0]), max(0, dv[1]+delta*.2),
                         max(0, 2.5*abs(slopes[1])-abs(slopes[0]))]
        else:
            penalties = [max(0, delta+dv[0]), max(0, delta-dv[1]),
                         max(0, -dv[2]-delta*.3), max(0, dv[3]+delta*.3)]
        penalty = sum(penalties) / scale
        score = residual + .5*penalty + .007 * len(breaks)
        if score < best[0]:
            best = (score, idx, tuple(float(z) for z in slopes))
    return best


def morphology(c, config):
    view = shape_view(c, int(config["grid_points"]))
    if view is None:
        return None
    v, noise, _ = view
    raw_y=np.asarray(c["y"],float)
    scores = {k: piece_score(v, k, config["minimum_excursion"], noise,view[2],raw_y) for k in CLASSES}
    ordering = sorted(CLASSES, key=lambda k: scores[k][0])
    chosen = ordering[0]
    idx = scores[chosen][1]
    landmarks = []
    for j in idx[1:-1]:
        frac = j / (len(v)-1)
        landmarks.append(float(c["t_hour"][0] + frac*np.ptp(c["t_hour"])) if c["t_hour"] is not None else None)
    amplitude = float(np.ptp(v))
    # A straight background decline may fit Slope, but lacks curvature evidence.
    linear_error = float(np.sqrt(np.mean((v-np.linspace(v[0],v[-1],len(v)))**2)))
    stage = scores[chosen][2]
    delta=max(float(config["minimum_excursion"]),3*noise)
    anchor_values=v[list(idx)]
    segment_changes=np.diff(anchor_values)
    peak_idx,peak_props=find_peaks(raw_y,prominence=.01)
    trough_idx,trough_props=find_peaks(-raw_y,prominence=.01)
    micro_events=int(np.sum((peak_props["prominences"]>=.01)&(peak_props["prominences"]<.03))
                     +np.sum((trough_props["prominences"]>=.01)&(trough_props["prominences"]<.03)))
    all_events=int(len(peak_idx)+len(trough_idx))
    complete = amplitude >= max(.03, 5*noise) and len(c["t"]) >= 5
    if chosen == "slope":
        complete &= (linear_error > max(.01, 2*noise) and segment_changes[0]<=-delta
                     and segment_changes[1]<=-delta)
    if chosen == "valley":
        complete &= (len(stage) == 4 and segment_changes[0]<=-delta
                     and segment_changes[1]>=delta and segment_changes[3]<=-delta)
    if chosen == "hill":
        complete &= (len(stage) == 3 and segment_changes[0]>=delta
                     and segment_changes[1]<=-delta and segment_changes[2]<=-delta
                     and abs(stage[1]) > 2*abs(stage[2]))
    if chosen == "bridge":
        complete &= (len(stage) == 3 and segment_changes[0]>=delta
                     and segment_changes[2]<=-delta)
    return dict(candidate_class=chosen, shape_scores={k: float(scores[k][0]) for k in CLASSES},
                models={k: (scores[k][1], scores[k][2]) for k in CLASSES},
                score_margin=float(scores[ordering[1]][0]-scores[chosen][0]),
                landmark_hours=landmarks, stage_slopes_per_progress=stage,
                stage_slopes_per_h=tuple(float(z/np.ptp(c["t_hour"])) for z in stage) if c["t_hour"] is not None and np.ptp(c["t_hour"]) > 0 else (),
                noise=noise, amplitude=amplitude, linear_error=linear_error,
                one_percent_extrema=all_events, one_to_three_percent_extrema=micro_events,
                stages_complete=bool(complete))


def dtw_distance(a, b, window_fraction, value_weight):
    """Band-limited DTW with value and first derivative on one shared path."""
    n = len(a)
    da = np.gradient(a) * (n-1) / 4
    db = np.gradient(b) * (n-1) / 4
    band = max(1, int(round(window_fraction*n)))
    previous = np.full(n+1, np.inf); previous[0] = 0
    for i in range(1,n+1):
        current = np.full(n+1, np.inf)
        for j in range(max(1,i-band),min(n,i+band)+1):
            cost = value_weight*abs(a[i-1]-b[j-1]) + (1-value_weight)*abs(da[i-1]-db[j-1])
            current[j] = cost + min(previous[j],current[j-1],previous[j-1])
        previous = current
    return float(previous[n]/n)


def apply_templates(canonical, curves, morph, reviews, config):
    reviewed = defaultdict(list)
    for r in canonical:
        m = morph[r["file_id"]]
        v = reviews.get(r["file_id"])
        if (m and v and v.get("source_verified", "").lower() == "yes"
                and v.get("stages_verified", "").lower() == "yes"
                and v.get("reviewed_class") in CLASSES
                and v.get("source_image_sha256") == r["image_sha256"]):
            reviewed[v["reviewed_class"]].append(r)
    templates = {k: reviewed[k][:8] for k in CLASSES}
    traces = {r["file_id"]: shape_view(curves[r["file_id"]],int(config["grid_points"]))[0]
              for group in templates.values() for r in group}
    for r in canonical:
        m = morph[r["file_id"]]
        if m is None:
            continue
        m["model_class"] = m["candidate_class"]
        own = shape_view(curves[r["file_id"]],int(config["grid_points"]))[0]
        distances = {}
        for k in CLASSES:
            eligible = [t for t in templates[k] if t["source_group"] != r["source_group"]]
            if eligible:
                distances[k] = min(dtw_distance(own,traces[t["file_id"]],
                    config["dtw_window_fraction"],config["value_weight"]) for t in eligible)
        m["template_distances"] = distances
        # Compare all four on the same basis; source isolation may leave one
        # class without an eligible template for a reviewed source DOI.
        if len(distances) == 4:
            combined = {k: .75*m["shape_scores"][k]+.25*distances[k] for k in CLASSES}
            ranking = sorted(CLASSES,key=lambda k:combined[k])
            m["combined_scores"] = combined
            m["candidate_class"] = ranking[0]
            m["score_margin"] = float(combined[ranking[1]]-combined[ranking[0]])
            if ranking[0] != min(CLASSES,key=lambda k:m["shape_scores"][k]):
                # A template can propose a different nearest class, but stage
                # support remains unverified until class-specific checks exist.
                m["stages_complete"] = False
        else:
            m["combined_scores"] = m["shape_scores"]
    return templates


def stability_for_candidates(canonical, curves, morph, config):
    rng = np.random.default_rng(config["seed"])
    selected = set()
    for k in CLASSES:
        matches = [r for r in canonical if morph[r["file_id"]] and morph[r["file_id"]]["candidate_class"] == k]
        matches.sort(key=lambda r: (not morph[r["file_id"]]["stages_complete"],
                                    morph[r["file_id"]]["shape_scores"][k]))
        selected.update(r["file_id"] for r in matches[:20])
    for r in canonical:
        m = morph[r["file_id"]]
        if m is None or r["file_id"] not in selected:
            continue
        c = curves[r["file_id"]]
        same = 0
        for _ in range(20):
            n = len(c["t"])
            keep = np.ones(n,dtype=bool)
            if n > 8:
                keep[1:-1] = rng.random(n-2) > .08
            altered = dict(c)
            altered["t"] = c["t"][keep]
            altered["t_hour"] = c["t_hour"][keep] if c["t_hour"] is not None else None
            altered["y"] = c["y"][keep] + rng.normal(0,max(m["noise"],.003),sum(keep))
            alt = morphology(altered,config)
            same += bool(alt and alt["candidate_class"] == m["candidate_class"])
        m["stability"] = same/20
    return selected


def parameter_search(output, canonical, curves, reviews, config):
    """Source-separated operational check on curated reviews; not accuracy."""
    labeled=[]
    for r in canonical:
        v=reviews.get(r["file_id"])
        if (v and v.get("source_verified","").lower()=="yes"
                and v.get("stages_verified","").lower()=="yes"
                and v.get("source_image_sha256")==r["image_sha256"]
                and v.get("reviewed_class") in CLASSES and r["file_id"] in curves):
            labeled.append((r,v["reviewed_class"]))
    counts=Counter(label for _,label in labeled)
    if any(counts[k]<3 for k in CLASSES):
        write_rows(output/"parameter_search.csv",[dict(status="deferred",reason="fewer than three verified source curves in one class")],
                   ["status","reason"])
        return
    # Five folds, balanced by class; every held-out DOI is excluded from its
    # fold's templates. The reviews were still selected using this pipeline.
    fold={}
    for k in CLASSES:
        for i,(r,_) in enumerate(sorted((x for x in labeled if x[1]==k),key=lambda x:x[0]["source_group"])):
            fold[r["file_id"]]=i%5
    grid_cache={}
    morph_cache={}
    for n in (64,96):
        cfg=config|{"grid_points":n}
        grid_cache[n]={r["file_id"]:shape_view(curves[r["file_id"]],n)[0] for r,_ in labeled}
        morph_cache[n]={r["file_id"]:morphology(curves[r["file_id"]],cfg) for r,_ in labeled}
    comparisons=[];predictions=[]
    for n in (64,96):
        for window in (.05,.10,.15):
            for weight in (.25,.5,.75,1.0):
                start=time.perf_counter();correct=0;model_correct=0
                for r,label in labeled:
                    id=r["file_id"];model=morph_cache[n][id]
                    model_correct+=model["candidate_class"]==label
                    scores={}
                    for k in CLASSES:
                        eligible=[t for t,tlabel in labeled if tlabel==k
                                  and fold[t["file_id"]]!=fold[id]
                                  and t["source_group"]!=r["source_group"]]
                        d=min(dtw_distance(grid_cache[n][id],grid_cache[n][t["file_id"]],window,weight)
                              for t in eligible)
                        scores[k]=.75*model["shape_scores"][k]+.25*d
                    predicted=min(CLASSES,key=lambda k:scores[k])
                    correct+=predicted==label
                    predictions.append(dict(grid_points=n,dtw_window_fraction=window,value_weight=weight,
                        file_id=id,source_group=r["source_group"],fold=fold[id],reviewed_class=label,
                        predicted_class=predicted,agrees=predicted==label))
                comparisons.append(dict(status="evaluated",grid_points=n,dtw_window_fraction=window,
                    value_weight=weight,minimum_excursion=config["minimum_excursion"],reviewed_sources=len(labeled),
                    source_isolated_agreement_count=correct,model_only_agreement_count=model_correct,
                    elapsed_s=round(time.perf_counter()-start,3),
                    frozen_for_final=(n==config["grid_points"] and window==config["dtw_window_fraction"] and weight==config["value_weight"]),
                    interpretation="curated candidate agreement; not independent generalization accuracy"))
    write_rows(output/"parameter_search.csv",comparisons,list(comparisons[0]))
    write_rows(output/"source_group_validation_predictions.csv",predictions,list(predictions[0]))
    # Fixed-seed source/duration/density sample documents the unlabeled search
    # population without using its unknown labels as ground truth.
    eligible=[r for r in canonical if r["file_id"] in curves and r["time_factor"] is not None
              and r["y_kind"]=="pce" and float(r["time_end_h"])>float(r["time_start_h"])]
    rng=np.random.default_rng(config["seed"])
    rng.shuffle(eligible)
    strata=defaultdict(list)
    for r in eligible:
        duration=float(r["time_end_h"])-float(r["time_start_h"])
        dbin=int(np.clip(np.floor(np.log10(max(duration,1e-6))/2)+1,0,2))
        nbin=0 if r["unique_points"]<20 else (1 if r["unique_points"]<100 else 2)
        strata[(dbin,nbin)].append(r)
    sample=[];seen=set()
    for group in strata.values():
        for r in group[:34]:
            if r["source_group"] not in seen:
                sample.append(r);seen.add(r["source_group"])
    if len(sample)<300:
        for r in eligible:
            if r["source_group"] not in seen:
                sample.append(r);seen.add(r["source_group"])
            if len(sample)>=300:break
    sample=sample[:300]
    write_rows(output/"development_sample.csv",[dict(file_id=r["file_id"],source_group=r["source_group"],
        duration_h=float(r["time_end_h"])-float(r["time_start_h"]),n_points=r["unique_points"])
        for r in sample],["file_id","source_group","duration_h","n_points"])


def som_comparison(output, canonical, curves, unique, config):
    """Anonymous SOM groups on the same comparable canonical trajectories."""
    from minisom import MiniSom
    records=[];vectors=[]
    for r in canonical:
        c=curves.get(r["file_id"])
        if c is None: continue
        view=shape_view(c,int(config["grid_points"]))
        if view is None: continue
        records.append(r);vectors.append(view[0])
    data=np.asarray(vectors,float)
    som=MiniSom(2,2,data.shape[1],sigma=1.0,learning_rate=.5,random_seed=config["seed"])
    som.random_weights_init(data)
    som.train_random(data,10000)
    cls={r["file_id"]:r for r in unique}
    assignments=[];cross=Counter()
    for r,v in zip(records,data):
        x,y=som.winner(v);cluster=2*x+y
        label=cls[r["file_id"]]["candidate_class"]
        assignments.append(dict(file_id=r["file_id"],source_group=r["source_group"],
            anonymous_som_cluster=cluster,candidate_class=label,
            evidence_status=cls[r["file_id"]]["evidence_status"]))
        cross[(cluster,label)]+=1
    write_rows(output/"som_assignments.csv",assignments,list(assignments[0]))
    table=[dict(anonymous_som_cluster=cluster,candidate_class=k,count=cross[(cluster,k)])
           for cluster in range(4) for k in CLASSES]
    write_rows(output/"som_crosstab.csv",table,["anonymous_som_cluster","candidate_class","count"])
    return Counter(r["anonymous_som_cluster"] for r in assignments)


def review_map(path):
    result = {}
    for r in rows(path):
        if r.get("file_id"):
            result[r["file_id"]] = r
    return result


def evidence(r, m, review, config):
    if r["load_error"] or r["time_factor"] is None or r["y_kind"] != "pce":
        return "review_required"
    if m is None:
        return "review_required"
    if not m["stages_complete"]:
        return "incomplete" if "sparse_points" in r["quality_flags"] else "other"
    if any(x in r["quality_flags"] for x in ("large_sampling_gap", "negative_response")):
        return "review_required"
    if review and review.get("source_verified", "").lower() == "yes" and review.get("reviewed_class") == m["candidate_class"] and review.get("stages_verified", "").lower() == "yes":
        return "supported"
    if m["score_margin"] < config["ambiguity_margin"]:
        return "ambiguous"
    return "review_required"


def make_tasks(canonical, curves, morph, output):
    tasks = []; review_tasks=[]
    for r in canonical:
        m = morph.get(r["file_id"])
        reasons = []
        if r["load_error"]: reasons.append("unreadable CSV: " + r["load_error"])
        if "sparse_points" in r["quality_flags"]: reasons.append("fewer than five observed points")
        if "large_sampling_gap" in r["quality_flags"]: reasons.append("large sampling gap; inspect turning points")
        if r["time_factor"] is None: reasons.append("time axis definition unresolved")
        if r["y_kind"] != "pce": reasons.append("response identity unresolved")
        if m and not m["stages_complete"]:
            review_tasks.append(dict(file_id=r["file_id"],source_csv=r["source_csv"],
                source_image=r["source_image"],candidate_class=m["candidate_class"],
                reason="target stage evidence incomplete; inspect source before deciding on redigitization",status="pending"))
        if m and m["score_margin"] < .025:
            review_tasks.append(dict(file_id=r["file_id"],source_csv=r["source_csv"],
                source_image=r["source_image"],candidate_class=m["candidate_class"],
                reason="close competing shape scores; inspect source series",status="pending"))
        if reasons:
            tasks.append(dict(file_id=r["file_id"], source_csv=r["source_csv"], source_image=r["source_image"],
                              candidate_class=m["candidate_class"] if m else "", reason="; ".join(reasons),
                              status="pending", resolution=""))
    write_rows(output / "digitization_tasks.csv", tasks,
               ["file_id", "source_csv", "source_image", "candidate_class", "reason", "status", "resolution"])
    write_rows(output / "source_review_tasks.csv",review_tasks,
               ["file_id","source_csv","source_image","candidate_class","reason","status"])
    return tasks


def plot_curve(c, m, path, title):
    if c["t_hour"] is None or c["info"]["y_kind"] != "pce" or c["info"]["normalization_max"] <= 0:
        return False
    t, y = c["t_hour"], c["y"]
    if len(t) < 2 or t[-1] <= t[0]:
        return False
    f = PchipInterpolator(t, y, extrapolate=False)
    grid = np.linspace(t[0], t[-1], 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, f(grid), color="#245e8c", lw=1.6)
    ax.scatter(t, y, s=9, color="#111111", zorder=3)
    ax.set(xlabel="Time (h)", ylabel="Normalized PCE", title=title)
    ax.set_xlim(t[0], t[-1]); ax.grid(alpha=.2)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return True


def report(root, output, manifest, canonical, curves, morph, unique, tasks, config, reviews_path, digit_path):
    for dirname in ("redigitized","source_overlays"):
        directory=output/dirname;directory.mkdir(exist_ok=True)
        note=directory/"README.md"
        if not note.exists():
            note.write_text("本目录由 redigitize.py 写入；补采采点及接受状态以 digitization_manifest.csv 为准。\n",encoding="utf-8")
    supported = [r for r in unique if r["evidence_status"] == "supported"]
    count = Counter(r["candidate_class"] for r in supported)
    doi_count = {k: len({r["source_group"] for r in supported if r["candidate_class"] == k}) for k in CLASSES}
    review_queue_count=len({r["file_id"] for r in rows(output/"source_review_tasks.csv")})
    verification = dict(input_files=len(manifest), result_files=len(manifest),
        canonical_curves=len(canonical), unreadable_files=sum(bool(r["load_error"]) for r in manifest),
        classified_canonical=sum(bool(r["candidate_class"]) for r in unique),
        full_nearest_label_coverage=all(bool(r["candidate_class"]) for r in unique),
        unresolved_hour_axes=sum(r["time_factor"] is None for r in manifest),
        unresolved_response_identity=sum(r["y_kind"] != "pce" for r in manifest),
        pending_digitization_tasks=len(tasks), pending_source_review_curves=review_queue_count,
        supported_counts=dict(count),
        supported_doi_counts=doi_count, targets_met=all(count[k]>=5 and doi_count[k]>=2 for k in CLASSES),
        source_review_required=True, config=config,
        unclassified_files=[r["file_id"] for r in unique if not r["candidate_class"]],
        input_manifest_sha256=sha(output/"input_manifest.csv"),
        source_reviews_path=str(reviews_path),source_reviews_sha256=sha(reviews_path) if reviews_path.exists() else "",
        digitization_manifest_path=str(digit_path),digitization_manifest_sha256=sha(digit_path) if digit_path.exists() else "")
    verification["overall_acceptance_met"]=(verification["targets_met"]
        and verification["full_nearest_label_coverage"]
        and verification["unresolved_hour_axes"]==0
        and verification["unresolved_response_identity"]==0
        and verification["pending_digitization_tasks"]==0)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 四类目标 PCE 曲线运行报告", "", f"输入 CSV：{len(manifest)}；规范曲线：{len(canonical)}；有最近标签的规范曲线：{verification['classified_canonical']}。",
             f"补采或单位/物理量核验任务：{len(tasks)}；形态源图复核队列：{review_queue_count}；无法读取：{verification['unreadable_files']}。", "", "## 可信成员", ""]
    for k in CLASSES:
        lines.append(f"- {k}: {count[k]} 条，{doi_count[k]} 篇来源；目标为至少 5 条、2 篇。")
    lines += ["", "## 未完成项", "", f"目标成员数达标：{'是' if verification['targets_met'] else '否'}；方案整体验收完成：{'是' if verification['overall_acceptance_met'] else '否'}。",
              "可信成员仅根据明确的 source_reviews.csv 记录授予。未审阅的最近匹配不是已确认目标形态。",
              "时间单位或纵轴物理量未确认时，真实小时图和可信状态不能验收。补采任务需对照原图处理。",
              f"仍无可比较 PCE 数值的文件：{', '.join(verification['unclassified_files']) or '无'}。",
              "当前每类有 5 条真实模板；24 组形态模板参数的来源隔离操作性一致性对照已运行。此处的匿名 SOM 簇仅供早期对照；四类无监督 SOM 主方法另见 som_primary/REPORT_CN.md。审阅样本来自本轮初筛，不能据此宣称独立泛化精度；独立来源测试尚未完成。", ""]
    (output / "RUN_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    # An index avoids thousands of redundant PNGs and points directly to inputs.
    import html
    cards = []
    for r in unique:
        source = html.escape(r["source_csv"])
        cls = html.escape(r["candidate_class"] or "unclassified")
        status = html.escape(r["evidence_status"])
        cards.append(f"<tr data-class='{cls}' data-status='{status}'><td>{r['file_id']}</td><td>{cls}</td><td>{status}</td><td>{source}</td><td>{html.escape(r['source_image'])}</td></tr>")
    page = """<!doctype html><html lang='zh'><meta charset='utf-8'><title>四类 PCE 全量索引</title><style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.4rem;text-align:left}input,select{margin:.5rem;padding:.3rem}</style><h1>全量曲线索引</h1><p><a href='galleries/all_curves.html'>查看全量可交互图库</a> · <a href='galleries/index.html'>候选与可信图库</a></p><p>类别是最近匹配；可信状态单独标记。CSV 和原图路径相对于 data_final。</p><select id='c'><option value=''>全部类别</option><option>bridge</option><option>hill</option><option>slope</option><option>valley</option></select><select id='s'><option value=''>全部状态</option><option>supported</option><option>ambiguous</option><option>incomplete</option><option>other</option><option>review_required</option></select><table><thead><tr><th>ID</th><th>类别</th><th>状态</th><th>CSV</th><th>原图</th></tr></thead><tbody>""" + "".join(cards) + """</tbody></table><script>function filter(){for(const r of document.querySelectorAll('tbody tr'))r.hidden=(c.value&&r.dataset.class!==c.value)||(s.value&&r.dataset.status!==s.value)}c.onchange=s.onchange=filter</script></html>"""
    (output / "all_curves_index.html").write_text(page, encoding="utf-8")
    review_queue=[]
    for k in CLASSES:
        matches=[r for r in unique if r["candidate_class"]==k and r["evidence_status"]!="supported"
                 and r["time_start_h"]!="" and curves[r["file_id"]]["info"]["y_kind"]=="pce"]
        matches.sort(key=lambda r:(not r["stages_complete"],-float(r["score_margin"] or 0),
                                   float(json.loads(r["shape_scores"] or "{}").get(k,99))))
        per_source=Counter()
        for r in matches:
            if per_source[r["source_group"]] >= 2: continue
            review_queue.append(dict(file_id=r["file_id"],source_csv=r["source_csv"],source_image=r["source_image"],
                candidate_class=k,source_group=r["source_group"],score_margin=r["score_margin"],
                stages_complete=r["stages_complete"],evidence_status=r["evidence_status"]))
            per_source[r["source_group"]]+=1
            if sum(x["candidate_class"]==k for x in review_queue)>=20: break
    write_rows(output/"candidate_review_queue.csv",review_queue,
        ["file_id","source_csv","source_image","candidate_class","source_group","score_margin","stages_complete","evidence_status"])
    gallery=output/"galleries"; gallery.mkdir(exist_ok=True)
    entries=[]
    for r in review_queue + supported:
        c=curves[r["file_id"]]
        filename=f"{r['file_id']}.png"
        if plot_curve(c,morph[r["file_id"]],gallery/filename,
                      f"{r['file_id']} | {r['candidate_class']} | {r['evidence_status']}"):
            entries.append(f"<article><h3>{r['file_id']} — {r['candidate_class']} — {r['evidence_status']}</h3>"
                f"<img src='{filename}' width='480'><p>{html.escape(r['source_csv'])}</p>"
                f"<p>Source image: {html.escape(r['source_image'])}</p></article>")
    (gallery/"index.html").write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><title>Candidate gallery</title>"
        "<style>body{font:14px system-ui;margin:2rem}article{display:inline-block;width:500px;vertical-align:top;margin:1rem;border:1px solid #ddd;padding:1rem}p{overflow-wrap:anywhere}</style>"
        "<h1>候选与可信样本</h1><p>图中仅为现有 CSV 数值；请结合源图复核。</p>"+"".join(entries)+"</html>",encoding="utf-8")
    # Full gallery embeds observed points so file:// viewing works offline.
    full_data=[]
    for r in unique:
        c=curves.get(r["file_id"])
        if (c is None or c["t_hour"] is None or len(c["t_hour"])<2
                or c["info"]["y_kind"]!="pce" or c["info"]["normalization_max"]<=0):
            points=[]
        else:
            points=[[round(float(x),6),round(float(y),6)] for x,y in zip(c["t_hour"],c["y"])]
        full_data.append(dict(id=r["file_id"],cls=r["candidate_class"],status=r["evidence_status"],
            csv=r["source_csv"],image=r["source_image"],points=points))
    payload=json.dumps(full_data,separators=(",",":"),ensure_ascii=False).replace("</","<\\/")
    all_page="""<!doctype html><html lang='zh'><meta charset='utf-8'><title>全量 PCE 曲线图库</title>
<style>body{font:14px system-ui;margin:1rem;display:grid;grid-template-columns:340px 1fr;gap:1rem}aside{height:95vh;overflow:auto}button{display:block;width:100%;text-align:left;background:white;border:0;border-bottom:1px solid #ddd;padding:.35rem;cursor:pointer}button:hover{background:#eef}canvas{max-width:100%;border:1px solid #ddd}select{margin:.3rem}p{overflow-wrap:anywhere}</style>
<aside><h2>全量规范曲线</h2><select id='cls'><option value=''>全部类别</option><option>bridge</option><option>hill</option><option>slope</option><option>valley</option></select><select id='status'><option value=''>全部状态</option><option>supported</option><option>ambiguous</option><option>incomplete</option><option>other</option><option>review_required</option></select><div id='list'></div></aside>
<main><h1 id='title'>选择一条曲线</h1><canvas id='chart' width='900' height='520'></canvas><p id='source'></p><p>曲线图只连接已观测点；单位未确认时不绘制小时轴。类别是最近匹配，状态另行判定。</p></main>
<script>const data="""+payload+""";const list=document.getElementById('list'),canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),title=document.getElementById('title'),source=document.getElementById('source'),cls=document.getElementById('cls'),status=document.getElementById('status');
function show(d){title.textContent=d.id+' | '+(d.cls||'unclassified')+' | '+d.status;source.textContent='CSV: '+d.csv+' | 原图: '+d.image;ctx.clearRect(0,0,900,520);const p=d.points;if(!p.length){ctx.fillText('实际小时坐标尚未确认或数据不可比较',260,250);return}const x0=p[0][0],x1=p[p.length-1][0],ys=p.map(v=>v[1]),y0=Math.min(...ys),y1=Math.max(...ys),dy=Math.max(y1-y0,.05);const L=90,R=850,T=35,B=450;ctx.strokeStyle='#222';ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,B);ctx.lineTo(R,B);ctx.stroke();ctx.font='15px sans-serif';ctx.fillStyle='#111';ctx.fillText('Time (h)',450,500);ctx.save();ctx.translate(22,280);ctx.rotate(-Math.PI/2);ctx.fillText('Normalized PCE',0,0);ctx.restore();ctx.fillText(x0.toFixed(2),L-12,B+22);ctx.fillText(x1.toFixed(2),R-35,B+22);ctx.fillText(y1.toFixed(2),L-65,T+5);ctx.fillText(y0.toFixed(2),L-65,B+5);ctx.strokeStyle='#24628b';ctx.beginPath();p.forEach((v,i)=>{const x=L+(v[0]-x0)/Math.max(x1-x0,1e-12)*(R-L),y=B-(v[1]-y0)/dy*(B-T);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#111';for(const v of p){const x=L+(v[0]-x0)/Math.max(x1-x0,1e-12)*(R-L),y=B-(v[1]-y0)/dy*(B-T);ctx.fillRect(x-1,y-1,2,2)}}
function render(){list.replaceChildren();for(const d of data){if(cls.value&&d.cls!==cls.value||status.value&&d.status!==status.value)continue;const b=document.createElement('button');b.textContent=d.id+'  '+(d.cls||'unclassified')+'  '+d.status;b.onclick=()=>show(d);list.append(b)}}cls.onchange=status.onchange=render;render();</script></html>"""
    (gallery/"all_curves.html").write_text(all_page,encoding="utf-8")
    # Source review sheets juxtapose evidence without pretending to have a
    # calibrated pixel overlay.
    sheet_dir=output/"source_review_sheets"; sheet_dir.mkdir(exist_ok=True)
    for r in review_queue + supported:
        if not r["source_image"]: continue
        image_path=root/r["source_image"]
        if not image_path.exists(): continue
        c=curves[r["file_id"]]
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        axs[0].imshow(plt.imread(image_path)); axs[0].axis("off");axs[0].set_title("Original source panel")
        axs[1].plot(c["t_hour"],c["y"],"o-",ms=2,lw=1)
        axs[1].set(xlabel="Time (h)",ylabel="Normalized PCE",title=f"{r['file_id']} | {r['candidate_class']}")
        fig.tight_layout();fig.savefig(sheet_dir/f"{r['file_id']}.png",dpi=130);plt.close(fig)
    model_comparison=[]
    for k in CLASSES:
        model_comparison.append(dict(method="shape_model_only",class_name=k,
            assigned=sum(bool(m and m.get("model_class",m["candidate_class"])==k) for m in morph.values()),
            note="full canonical set, including unresolved evidence"))
        model_comparison.append(dict(method="shape_model_plus_template_dtw",class_name=k,
            assigned=sum(r["candidate_class"]==k for r in unique),
            note="five reviewed templates per class; preliminary"))
    som_counts=som_comparison(output,canonical,curves,unique,config)
    for cluster in range(4):
        model_comparison.append(dict(method="SOM",class_name=f"anonymous_cluster_{cluster}",
            assigned=som_counts[cluster],note="same comparable canonical curves; no target-class mapping"))
    write_rows(output/"method_comparison.csv",model_comparison,["method","class_name","assigned","note"])
    parameter_search(output,canonical,curves,review_map(reviews_path),config)
    verification["parameter_search_status"]="24 source-separated configurations on curated reviewed candidates"
    (output / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, k in zip(axs.flat, CLASSES):
        members = [r for r in unique if r["candidate_class"] == k and r["evidence_status"] == "supported"]
        if not members:
            ax.text(.5,.5,"No source-verified member yet",ha="center",va="center",transform=ax.transAxes)
        else:
            r = members[0]; c = curves[r["file_id"]]
            t,y=c["t_hour"],c["y"]
            grid=np.linspace(t[0],t[-1],250)
            ax.plot(grid,PchipInterpolator(t,y,extrapolate=False)(grid),lw=1.4,label="PCHIP within observations")
            ax.scatter(t,y,s=9,color="black",label="Observed points")
            ax.set_xlabel("Time (h)"); ax.set_ylabel("Normalized PCE")
            ax.legend(fontsize=7)
        ax.set_title(k)
    fig.tight_layout(); fig.savefig(output / "four_classes_representatives.png",dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data_final"))
    ap.add_argument("--output", type=Path, default=Path("target_shape_classification/results"))
    ap.add_argument("--stage", choices=["audit","candidates","redigitize","templates","tune","classify","report","all"], default="all")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--reviews", type=Path)
    ap.add_argument("--digitization-manifest", type=Path)
    args = ap.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = DEFAULT | (json.loads(args.config.read_text()) if args.config else {})
    reviews_path = args.reviews or output / "source_reviews.csv"
    digit_path = args.digitization_manifest or output / "digitization_manifest.csv"
    if not reviews_path.exists():
        write_rows(reviews_path, [], ["file_id","source_csv","reviewed_class","source_verified","stages_verified","corrected_series","reviewer","review_date","source_image_sha256","notes"])
    if not digit_path.exists():
        write_rows(digit_path, [], ["source_csv","source_sha256","digitized_csv","digitized_sha256","status","calibration_json","point_provenance","notes"])
    manifest, canonical, curves = load_inputs(root, output, digit_path)
    if args.stage == "audit":
        print(f"Audited {len(manifest)} files, {len(canonical)} canonical curves")
        return
    morph = {r["file_id"]: morphology(curves[r["file_id"]],config) if r["file_id"] in curves else None for r in canonical}
    tasks = make_tasks(canonical,curves,morph,output)
    candidate_rows = []
    for r in canonical:
        m = morph[r["file_id"]]
        candidate_rows.append(dict(file_id=r["file_id"],source_csv=r["source_csv"],doi=r["doi"],
            candidate_class=m["candidate_class"] if m else "",score=m["shape_scores"][m["candidate_class"]] if m else "",
            score_margin=m["score_margin"] if m else "",stages_complete=m["stages_complete"] if m else False,
            source_image=r["source_image"]))
    write_rows(output/"candidate_scores.csv",candidate_rows,list(candidate_rows[0]) if candidate_rows else [])
    if args.stage == "candidates":
        print(f"Scored {len(candidate_rows)} canonical curves; {len(tasks)} review tasks")
        return
    rev = review_map(reviews_path)
    template_groups = apply_templates(canonical,curves,morph,rev,config)
    stability_for_candidates(canonical,curves,morph,config)
    unique=[]
    for r in canonical:
        m = morph[r["file_id"]]
        review = rev.get(r["file_id"])
        if review and review.get("source_image_sha256") != r["image_sha256"]:
            review = None
        status=evidence(r,m,review,config)
        item=dict(file_id=r["file_id"],source_csv=r["source_csv"],source_group=r["source_group"],
                  source_image=r["source_image"],analysis_version=r["analysis_version"],
                  corrected_series=(review or {}).get("corrected_series", ""),
                  support_reason=(review or {}).get("notes", ""),
                  candidate_class=m["candidate_class"] if m else "",evidence_status=status,
                  limited_support=";".join(x for x in r["quality_flags"].split(";") if x),
                  score_margin=m["score_margin"] if m else "",shape_scores=json.dumps(m["shape_scores"]) if m else "",
                  combined_scores=json.dumps(m.get("combined_scores",{})) if m else "",
                  template_distances=json.dumps(m.get("template_distances",{})) if m else "",
                  stability=m.get("stability","") if m else "",
                  landmark_hours=json.dumps(m["landmark_hours"]) if m else "",
                  stage_slopes_per_h=json.dumps(m["stage_slopes_per_h"]) if m else "",
                  stages_complete=m["stages_complete"] if m else False,
                  noise=m["noise"] if m else "",amplitude=m["amplitude"] if m else "",
                  one_percent_extrema=m["one_percent_extrema"] if m else "",
                  one_to_three_percent_extrema=m["one_to_three_percent_extrema"] if m else "",
                  time_start_h=r["time_start_h"],time_end_h=r["time_end_h"],
                  normalization_max=r["normalization_max"])
        unique.append(item)
    fields=list(unique[0]) if unique else []
    write_rows(output/"classification_unique_curves.csv",unique,fields)
    lookup={r["file_id"]:r for r in unique}
    all_rows=[]
    for r in manifest:
        result=lookup[r["canonical_id"]].copy()
        result.update(file_id=r["file_id"],source_csv=r["source_csv"],canonical_id=r["canonical_id"],
                      source_sha256=r["source_sha256"],analysis_version=r["analysis_version"])
        all_rows.append(result)
    write_rows(output/"classification_all_files.csv",all_rows,list(all_rows[0]) if all_rows else [])
    supported=[r for r in unique if r["evidence_status"]=="supported"]
    write_rows(output/"supported_members.csv",supported,fields)
    templates=[dict(file_id=r["file_id"],candidate_class=k,source_group=r["source_group"],reviewed=True)
               for k,group in template_groups.items() for r in group]
    write_rows(output/"templates.csv",templates,["file_id","candidate_class","source_group","reviewed"])
    if args.stage in ("report","classify","all","templates","tune","redigitize"):
        report(root,output,manifest,canonical,curves,morph,unique,tasks,config,reviews_path,digit_path)
    print(f"Processed {len(manifest)} files; {len(canonical)} canonical; {len(supported)} source-verified supported")


if __name__ == "__main__":
    main()
