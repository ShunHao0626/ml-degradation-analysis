#!/usr/bin/env python3
"""Auditable four-shape rule based on observed stages on both sides of 200 h.

This is a semantic rule baseline, separate from the frozen anonymous SOM and
from human reference labels. It never interpolates a missing stage into evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from early_data import HERE, digest, read_observations, rows, write_rows


OUT = HERE / "diagram_stage_20260925"
SOURCE = HERE.parents[1] / "original_curves_2250"
WINDOW = 200.0
MIN_EVENT = 0.01  # An audit trigger on the normalized scale, not significance.


def unique_observations(points, divisor):
    by_time = defaultdict(list)
    for h, value, _ in points:
        if h >= 0:
            by_time[h].append(value / divisor)
    t = np.array(sorted(by_time), dtype=float)
    y = np.array([np.median(by_time[h]) for h in t], dtype=float)
    return t, y


def rate(t, y, i, j):
    return float((y[j] - y[i]) / (t[j] - t[i])) if j > i and t[j] > t[i] else 0.0


def decay_transition(t, y, first):
    """Locate a supported fast-to-slow decline; use actual samples only."""
    if len(t) - first < 5:
        return None
    best = None
    for split in range(first + 2, len(t) - 2):
        fast = rate(t, y, first, split)
        slow = rate(t, y, split, len(t) - 1)
        if fast >= -1e-9 or slow > 1e-9 or abs(fast) < 1.5 * max(abs(slow), 1e-9):
            continue
        if y[first] - y[split] < MIN_EVENT:
            continue
        # Both pieces should roughly follow their endpoint trend. The penalty
        # prevents a single noisy point from becoming the transition.
        a = np.interp(t[first:split + 1], [t[first], t[split]], [y[first], y[split]])
        b = np.interp(t[split:], [t[split], t[-1]], [y[split], y[-1]])
        error = float(np.mean((y[first:split + 1] - a) ** 2) +
                      np.mean((y[split:] - b) ** 2))
        score = error + 0.00001 / max(t[split] - t[first], 1)
        if best is None or score < best[0]:
            best = (score, split, fast, slow)
    return best


def classify(t, y):
    """Return four-way candidate, observation status, and real-point events."""
    strict = np.flatnonzero(t <= WINDOW)
    if len(strict) < 2:
        return dict(candidate_class="", evidence_status="data_issue", reason="too_few_strict_points")
    n = len(strict)
    tw, yw = t[:n], y[:n]
    post = np.flatnonzero(t > WINDOW)
    direction_hits = np.flatnonzero(np.abs(yw - yw[0]) >= MIN_EVENT)
    direction = float(yw[direction_hits[0]] - yw[0]) if len(direction_hits) else float(yw[-1] - yw[0])
    weak_start = not len(direction_hits)
    result = dict(candidate_class="", evidence_status="stage_candidate", reason="",
                  initial_direction="increase" if direction >= 0 else "decay",
                  initial_h=float(tw[0]), initial_pce=float(yw[0]),
                  primary_turn_h="", primary_turn_pce="", secondary_turn_h="",
                  secondary_turn_pce="", decay_transition_h="",
                  initial_amplitude="", reversal_amplitude="", initial_rate_per_h="",
                  reversal_rate_per_h="", slow_decay_rate_per_h="",
                  post_200_direction="unobserved", post_200_points=len(post),
                  post_200_pattern="unobserved", post_peak_h="", post_peak_pce="",
                  post_decay_amplitude="", post_200_matches_diagram="undetermined",
                  strict_points=n, strict_end_h=float(tw[-1]),
                  max_strict_gap_h=float(np.max(np.diff(tw))),
                  feature_time_basis="observed_source_points")
    if direction >= 0:
        turn = int(np.argmax(yw))
        rise = float(yw[turn] - yw[0])
        down = int(turn + np.argmin(yw[turn:])) if turn < n - 1 else turn
        drop = float(yw[turn] - yw[down])
        result.update(primary_turn_h=float(tw[turn]), primary_turn_pce=float(yw[turn]),
                      secondary_turn_h=float(tw[down]) if down > turn else "",
                      secondary_turn_pce=float(yw[down]) if down > turn else "",
                      initial_amplitude=rise, reversal_amplitude=drop,
                      initial_rate_per_h=rate(tw, yw, 0, turn),
                      reversal_rate_per_h=rate(tw, yw, turn, down))
        transition = decay_transition(tw, yw, turn)
        if transition is not None:
            _, k, fast, slow = transition
            result.update(decay_transition_h=float(t[k]),
                          reversal_rate_per_h=fast, slow_decay_rate_per_h=slow)
        # Hill needs a visible post-peak rapid loss. A broad high region is Bridge.
        hill = (drop >= MIN_EVENT and rise >= MIN_EVENT and
                (transition is not None or drop >= 0.35 * rise))
        result["candidate_class"] = "hill" if hill else "bridge"
        result["reason"] = "rise_peak_fast_decay" if hill else "rise_high_region"
        supported = turn >= 1 and down > turn and rise >= MIN_EVENT
        if hill:
            supported = supported and drop >= MIN_EVENT and (down - turn) >= 2
    else:
        turn = int(np.argmin(yw))
        fall = float(yw[0] - yw[turn])
        up = int(turn + np.argmax(yw[turn:])) if turn < n - 1 else turn
        recovery = float(yw[up] - yw[turn])
        result.update(primary_turn_h=float(tw[turn]), primary_turn_pce=float(yw[turn]),
                      secondary_turn_h=float(tw[up]) if up > turn else "",
                      secondary_turn_pce=float(yw[up]) if up > turn else "",
                      initial_amplitude=fall, reversal_amplitude=recovery,
                      initial_rate_per_h=rate(tw, yw, 0, turn),
                      reversal_rate_per_h=rate(tw, yw, turn, up))
        valley = fall >= MIN_EVENT and recovery >= MIN_EVENT and up > turn
        result["candidate_class"] = "valley" if valley else "slope"
        result["reason"] = "decay_trough_recovery" if valley else "decay_without_strict_recovery"
        supported = turn >= 2 and fall >= MIN_EVENT
        if valley:
            supported = supported and (up - turn) >= 2
        else:
            transition = decay_transition(tw, yw, 0)
            if transition is not None:
                _, k, fast, slow = transition
                result.update(decay_transition_h=float(t[k]),
                              initial_rate_per_h=fast, slow_decay_rate_per_h=slow)
    if len(post):
        # The first post-200 sample describes the actual boundary; no 200 h
        # value is invented if the source lacks it.
        p0, p1 = int(post[0]), int(post[-1])
        if p1 > p0 and y[p1] - y[p0] >= MIN_EVENT:
            result["post_200_direction"] = "recovery"
        elif p1 > p0 and y[p0] - y[p1] >= MIN_EVENT:
            result["post_200_direction"] = "decay"
            if result["slow_decay_rate_per_h"] == "":
                result["slow_decay_rate_per_h"] = rate(t, y, p0, p1)
        else:
            result["post_200_direction"] = "flat_or_insufficient"
        if len(post) >= 2:
            if result["candidate_class"] == "valley":
                crest = int(n - 1 + np.argmax(y[n - 1:]))
                result["post_peak_h"] = float(t[crest])
                result["post_peak_pce"] = float(y[crest])
                result["post_decay_amplitude"] = float(y[crest] - y[-1])
                recovering = crest > n - 1 and y[crest] - y[n - 1] >= MIN_EVENT
                decaying = crest < len(t) - 1 and y[crest] - y[-1] >= MIN_EVENT
                result["post_200_pattern"] = ("recovery_then_decay" if recovering and decaying else
                                                "recovery_ongoing" if recovering else
                                                "decay_after_recovery" if decaying else "flat_or_complex")
                result["post_200_matches_diagram"] = result["post_200_pattern"] != "flat_or_complex"
            elif result["candidate_class"] in ("hill", "slope"):
                transition = decay_transition(t, y, turn if result["candidate_class"] == "hill" else 0)
                late_rate = rate(t, y, p0, p1)
                rapid_rate = abs(result["reversal_rate_per_h"] if result["candidate_class"] == "hill"
                                 else result["initial_rate_per_h"])
                if transition is not None:
                    _, k, fast, slow = transition
                    result["decay_transition_h"] = float(t[k])
                    result["slow_decay_rate_per_h"] = slow
                if y[p1] - y[p0] >= MIN_EVENT:
                    result["post_200_pattern"] = "recovery_conflicts_with_decay"
                    result["post_200_matches_diagram"] = False
                elif late_rate < 0 and rapid_rate >= 1.5 * abs(late_rate):
                    result["post_200_pattern"] = ("fast_decay_then_slow_decay" if transition
                                                  else "slow_decay_after_initial_fast_decay")
                    result["post_200_matches_diagram"] = True
                else:
                    result["post_200_pattern"] = result["post_200_direction"]
                    result["post_200_matches_diagram"] = "undetermined"
            else:
                slow = rate(t, y, p0, p1)
                result["post_200_pattern"] = "slow_decay" if slow < 0 else result["post_200_direction"]
                result["post_200_matches_diagram"] = bool(slow <= 0 and
                    (result["initial_rate_per_h"] == 0 or abs(slow) < abs(result["initial_rate_per_h"])))
    if weak_start or not supported or n < 5 or tw[-1] < 100 or result["max_strict_gap_h"] > 40:
        result["evidence_status"] = "insufficient_evidence"
    if len(post) < 2:
        result["post_200_status"] = "insufficient_evidence"
    else:
        result["post_200_status"] = "observed"
    # A post-200 recovery can suggest a boundary Valley. It does not rewrite
    # the strict-window category or pretend that the trough was seen by 200 h.
    result["boundary_valley_hint"] = bool(result["candidate_class"] == "slope" and
                                         len(post) and len(t) > n and
                                         float(np.max(y[n:]) - yw[turn]) >= MIN_EVENT)
    if result["post_200_matches_diagram"] is False and result["evidence_status"] == "stage_candidate":
        result["evidence_status"] = "pattern_conflict"
    return result


def plot(path, t, y, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="410" viewBox="0 0 1040 410">',
             '<rect width="1040" height="410" fill="white"/>',
             f'<text x="30" y="28" font-family="sans-serif" font-size="17">{result["curve_id"]} | {result["candidate_class"]} | {result["evidence_status"]}</text>']
    for panel, limits in enumerate(((0, 200), (200, float(t[-1])))):
        left = 55 + 510 * panel
        top, width, height = 62, 450, 278
        low, high = limits
        high = max(high, low + 1)
        mask = (t >= low) & (t <= high)
        tt, yy = t[mask], y[mask]
        if len(tt) > 500:
            keep = np.unique(np.linspace(0, len(tt) - 1, 500).astype(int))
            tt, yy = tt[keep], yy[keep]
        ymin, ymax = (float(np.min(yy)), float(np.max(yy))) if len(yy) else (0., 1.)
        pad = max((ymax - ymin) * .1, .02)
        ymin -= pad
        ymax += pad
        xpx = lambda h: left + width * (h - low) / (high - low)
        ypx = lambda v: top + height * (ymax - v) / (ymax - ymin)
        parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fbfb" stroke="#9aabad"/>')
        title = "0–200 h initial stages" if panel == 0 else "After 200 h observed stages"
        parts.append(f'<text x="{left}" y="53" font-family="sans-serif" font-size="13">{title}</text>')
        parts.append(f'<text x="{left}" y="365" font-family="sans-serif" font-size="12">{low:.0f} h</text>')
        parts.append(f'<text x="{left + width - 45}" y="365" font-family="sans-serif" font-size="12">{high:.0f} h</text>')
        parts.append(f'<text x="{left}" y="390" font-family="sans-serif" font-size="12">Hour (h) · Normalized PCE {ymin:.2f}–{ymax:.2f}</text>')
        if len(tt):
            coords = " ".join(f"{xpx(h):.1f},{ypx(v):.1f}" for h, v in zip(tt, yy))
            parts.append(f'<polyline points="{coords}" fill="none" stroke="#286381" stroke-width="1.5"/>')
            for h, v in zip(tt, yy):
                parts.append(f'<circle cx="{xpx(h):.1f}" cy="{ypx(v):.1f}" r="1.5" fill="#286381"/>')
        for name, color in (("primary", "#c84f39"), ("secondary", "#957431")):
            h, v = result.get(name + "_turn_h", ""), result.get(name + "_turn_pce", "")
            if h != "" and low <= h <= high:
                parts.append(f'<circle cx="{xpx(h):.1f}" cy="{ypx(v):.1f}" r="5" fill="{color}"/>')
        h = result.get("decay_transition_h", "")
        if h != "" and low <= h <= high:
            parts.append(f'<line x1="{xpx(h):.1f}" x2="{xpx(h):.1f}" y1="{top}" y2="{top + height}" stroke="#a862a7" stroke-dasharray="4 4"/>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_point_csv(path, points, divisor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hour", "normalized_pce", "source_row", "region"])
        for h, v, source_row in points:
            if h >= 0:
                writer.writerow([f"{h:.12g}", f"{v / divisor:.12g}", source_row,
                                 "strict_200h" if h <= 200 else "after_200h"])


def gallery(path, records):
    payload = [{k: r.get(k, "") for k in ("curve_id", "candidate_class", "evidence_status",
                "source_group", "source_image", "plot_path", "curve_csv_path", "primary_turn_h",
                "secondary_turn_h", "decay_transition_h", "post_200_pattern",
                "post_200_matches_diagram", "boundary_valley_hint")} for r in records]
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>200 h 分段形态判别</title>
<style>body{font:14px system-ui;margin:22px;color:#263840}main{display:flex;gap:20px}aside{width:38%;height:85vh;overflow:auto}article{flex:1;position:sticky;top:15px;height:85vh}input,select{padding:7px;margin-right:8px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}tr:hover{background:#eaf2f4;cursor:pointer}object{width:100%;height:450px;border:1px solid #ddd}a{margin-right:12px}</style>
<h1>200 h 分段形态判别（自动基线）</h1><p><a href="../final_integrated_20260925/all_curves_index.html">查看已整合人工意见的最终图库</a></p><p>类别是自动规则候选；红点为初期主转折，黄点为后续转折，紫线为下降变缓位置。200 h 后阶段单独检查；缺失观测不补造。</p>
<input id="search" placeholder="曲线 ID 或 DOI"><select id="cls"><option value="">全部类别</option><option>bridge</option><option>hill</option><option>slope</option><option>valley</option></select><select id="status"><option value="">全部证据状态</option><option>stage_candidate</option><option>insufficient_evidence</option><option>pattern_conflict</option></select><span id="count"></span>
<main><aside><table><thead><tr><th>ID</th><th>候选</th><th>状态</th><th>200 h 后</th></tr></thead><tbody id="body"></tbody></table></aside><article><h2 id="title">选择一条曲线</h2><object id="chart" type="image/svg+xml"></object><div id="links"></div><p id="features"></p></article></main>
<script>const rows=__DATA__,body=document.querySelector('#body');function show(r){document.querySelector('#title').textContent=r.curve_id+' | '+r.candidate_class;document.querySelector('#chart').data=r.plot_path;document.querySelector('#features').textContent='初期转折 '+(r.primary_turn_h||'—')+' h；后续转折 '+(r.secondary_turn_h||'—')+' h；下降变缓 '+(r.decay_transition_h||'—')+' h；200 h 后：'+r.post_200_pattern;let links=document.querySelector('#links');links.replaceChildren();for(const [label,path] of [['逐点 CSV',r.curve_csv_path],['曲线图',r.plot_path],['论文原图',r.source_image?'../../../data_final/'+r.source_image:'']])if(path){let a=document.createElement('a');a.href=path;a.textContent=label;a.target='_blank';links.append(a)}}function render(){body.replaceChildren();let q=document.querySelector('#search').value.toLowerCase(),c=document.querySelector('#cls').value,s=document.querySelector('#status').value;let shown=rows.filter(r=>(!c||r.candidate_class===c)&&(!s||r.evidence_status===s)&&(!q||(r.curve_id+r.source_group).toLowerCase().includes(q)));document.querySelector('#count').textContent=shown.length+' / '+rows.length;for(const r of shown){let tr=document.createElement('tr');for(const key of ['curve_id','candidate_class','evidence_status','post_200_pattern']){let td=document.createElement('td');td.textContent=r[key];tr.append(td)}tr.onclick=()=>show(r);body.append(tr)}if(shown.length)show(shown[0])}for(const id of ['search','cls','status'])document.querySelector('#'+id).addEventListener('input',render);render();</script>'''.replace("__DATA__", data)
    path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true", help="Compute feature audit before image rendering")
    args = parser.parse_args()
    source = {r["file_id"]: r for r in rows(HERE / "manifests/frozen_inputs.csv")}
    old = rows(HERE / "delivery/class_index.csv")
    records = []
    for index, r in enumerate(old):
        fid = r["curve_id"]
        record = dict(curve_id=fid, source_group=r["source_group"], split=r["split"],
                      source_csv=r["source_csv"], source_sha256=r["source_sha256"],
                      source_image=r["source_image"],
                      som_posthoc_class=r["som_posthoc_class"],
                      previous_rule_class=r["rule_candidate_class"],
                      plot_path="")
        src = source[fid]
        path = SOURCE / src["source_csv"]
        if digest(path) != src["source_sha256"]:
            raise ValueError("source hash changed: " + fid)
        points = read_observations(src, SOURCE)
        divisor = float(r["normalization_divisor"])
        t, y = unique_observations(points, divisor)
        current = classify(t, y)
        record.update(current)
        if current["candidate_class"]:
            filename = Path(current["candidate_class"]) / (fid + ".svg")
            record["plot_path"] = str(filename)
            record["curve_csv_path"] = str(filename.with_suffix(".csv"))
            write_point_csv(OUT / record["curve_csv_path"], points, divisor)
            if not args.skip_plots:
                plot(OUT / filename, t, y, dict(record))
        records.append(record)
        if index % 300 == 0:
            print(index, fid, flush=True)
    write_rows(OUT / "stage_features.csv", records, list(dict.fromkeys(k for r in records for k in r)))
    write_rows(OUT / "priority_review.csv", [r for r in records if r["evidence_status"] != "stage_candidate" or
                r["candidate_class"] != r["previous_rule_class"] or r["boundary_valley_hint"]],
               list(dict.fromkeys(k for r in records for k in r)))
    gallery(OUT / "all_curves_index.html", records)
    counts = Counter(r["candidate_class"] for r in records)
    status = Counter(r["evidence_status"] for r in records)
    report = dict(input_count=len(records), classes=counts, statuses=status,
                  disagreements_with_previous=sum(r["candidate_class"] != r["previous_rule_class"] for r in records),
                  no_post_200=sum(r.get("post_200_status") == "insufficient_evidence" for r in records),
                  protocol="diagram_stage_rule_v1", event_threshold_normalized=MIN_EVENT,
                  interpretation="Automatic rule candidates, not human reference labels or measured accuracy")
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
