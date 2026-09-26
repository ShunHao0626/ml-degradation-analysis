#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent 0–200 h SOM classification of canonical PCE curves.

The SOM fit uses window observations only. Its anonymous BMU, post-hoc neuron
name, and the final Valley-event decision are reported in separate columns.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import run
import unsupervised_som as som


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIELDS = [
    "file_id", "canonical_id", "source_csv", "analysis_version", "source_image",
    "source_group", "source_sha256", "analysis_sha256", "image_sha256",
    "y_kind", "time_unit", "time_factor", "classification_status", "window_reason",
    "final_class", "decision_basis", "som_candidate_class", "som_direct_group",
    "branch", "best_matching_neuron", "som_candidate_neuron", "som_stability",
    "ensemble_agreement", "quantization_distance", "window_points", "duplicate_times",
    "window_start_h", "window_end_h", "window_max_pce", "one_percent_source_extrema",
    "one_to_three_percent_extrema", "valley_event", "valley_trough_h",
    "valley_depth", "valley_recovery", "window_stage_class", "window_stages_complete",
    "boundary_valley_candidate", "boundary_valley_trough_h", "boundary_valley_depth",
    "boundary_valley_recovery_by_250h",
    "quality_flags", "source_review_200h_status", "source_review_200h_class",
    "source_review_200h_corrected_series", "source_review_200h_note",
]


def analysis_path(root: Path, row: dict) -> Path:
    path = root / row["analysis_version"]
    if not path.exists() and row["analysis_version"] != row["source_csv"]:
        path = RESULTS / row["analysis_version"]
    return path


def window_observations(root: Path, row: dict, hour_limit: float = 200):
    """Select source time points first, then normalize by the selected maximum."""
    if row["y_kind"] != "pce":
        return None, "response_identity_" + row["y_kind"]
    try:
        factor = float(row["time_factor"])
    except (TypeError, ValueError):
        return None, "unresolved_hour_axis"
    if not math.isfinite(factor) or factor <= 0:
        return None, "unresolved_hour_axis"
    try:
        t, y, _, duplicates = run.read_curve(analysis_path(root, row), row["folder"])
    except (OSError, ValueError, KeyError) as exc:
        return None, "unusable_coordinates:" + type(exc).__name__
    t = t * factor
    near_mask = (t >= 0) & (t <= hour_limit + 50)
    near_t, near_y = t[near_mask], y[near_mask]
    mask = (t >= 0) & (t <= hour_limit)
    t, y = t[mask], y[mask]
    if len(t) < 2:
        return None, "fewer_than_two_observations_0_200h"
    maximum = float(np.max(y))
    if not math.isfinite(maximum) or maximum <= 0:
        return None, "nonpositive_window_maximum"
    normalized = y / maximum
    view = run.shape_view({"t": t, "y": normalized}, som.N_GRID)
    if view is None:
        return None, "unusable_window_shape"
    return dict(t=t, y=normalized, near_t=near_t, near_y=near_y / maximum,
                view=view[0], divisor=maximum,
                duplicates=duplicates), ""


def valley_event(t: np.ndarray, y: np.ndarray):
    """Find a temporal fall and recovery inside the window, preserving 1% events.

    A >=3% V is accepted with one observation on each side. For a 1–3% V,
    two distinct declining and recovering steps are required, so a single
    digitization jitter point does not force a Valley label.
    """
    if len(y) < 3:
        return None
    troughs, props = find_peaks(-y, prominence=.01 - 1e-9)
    found = []
    for trough, prominence in zip(troughs, props["prominences"]):
        left = int(np.argmax(y[:trough]))
        right = trough + 1 + int(np.argmax(y[trough + 1:]))
        depth = float(y[left] - y[trough])
        recovery = float(y[right] - y[trough])
        strength = min(depth, recovery)
        if strength < .01 - 1e-9:
            continue
        if strength < .03 - 1e-9:
            # Distinct observations on each side support the small reversal.
            declines = int(np.sum(np.diff(y[left:trough + 1]) < -1e-6))
            rises = int(np.sum(np.diff(y[trough:right + 1]) > 1e-6))
            if declines < 2 or rises < 2:
                continue
        found.append(dict(trough_h=float(t[trough]), depth=depth,
                          recovery=recovery, prominence=float(prominence),
                          left_h=float(t[left]), right_h=float(t[right])))
    return max(found, key=lambda event: (min(event["depth"], event["recovery"]),
                                          event["prominence"])) if found else None


def boundary_valley_event(t: np.ndarray, y: np.ndarray, hour_limit: float = 200):
    """Flag a trough just before 200 h whose recovery is seen by 250 h.

    This is a review candidate. Confirmation points never enter the SOM fit or
    the 0–200 h normalization denominator.
    """
    later = np.flatnonzero((t > hour_limit) & (t <= hour_limit + 50))
    if len(later) == 0:
        return None
    right = int(later[np.argmax(y[later])])
    found = []
    for j in np.flatnonzero((t >= hour_limit - 50) & (t <= hour_limit)):
        if j == 0:
            continue
        left = int(np.argmax(y[:j]))
        drop = float(y[left] - y[j])
        recovery = float(y[right] - y[j])
        if drop >= .01 - 1e-9 and recovery >= .01 - 1e-9:
            found.append(dict(trough_h=float(t[j]), depth=drop,
                              recovery=recovery, recovery_h=float(t[right])))
    return max(found, key=lambda event: (min(event["depth"], event["recovery"]),
                                          event["recovery"])) if found else None


def load_window(root: Path, canonical: list[dict], hour_limit: float):
    values, raw, metadata, details, excluded = [], [], [], {}, {}
    for row in canonical:
        observation, reason = window_observations(root, row, hour_limit)
        if observation is None:
            excluded[row["file_id"]] = reason
            continue
        t, y = observation["t"], observation["y"]
        phase = (t - t[0]) / (t[-1] - t[0])
        values.append(observation["view"])
        raw.append((phase, y))
        metadata.append(row)
        details[row["file_id"]] = observation
    return np.asarray(values), raw, metadata, details, excluded


def chart_points(observation):
    if observation is None:
        return []
    return [[round(float(t), 7), round(float(y), 7)]
            for t, y in zip(observation["t"], observation["y"])]


def load_source_reviews(root: Path, canonical: list[dict], path: Path):
    """Use only reviews tied to the current source image and canonical series."""
    known = {row["file_id"]: row for row in canonical}
    reviews = {}
    for review in run.rows(path):
        fid = review["file_id"]
        if fid in reviews or fid not in known:
            raise ValueError(f"unknown or duplicate 200 h source review: {fid}")
        status, label = review["review_status"], review["reviewed_class"]
        if (status == "verified" and label not in run.CLASSES) or (
                status in ("ambiguous", "identity_mismatch", "digitization_mismatch") and label) or (
                status not in ("verified", "ambiguous", "identity_mismatch", "digitization_mismatch")):
            raise ValueError(f"invalid 200 h source review: {fid}")
        source = known[fid]
        image = root / source["source_image"]
        if (review["source_image_sha256"] != source["image_sha256"] or
                run.sha(image) != review["source_image_sha256"]):
            raise ValueError(f"source image has changed for 200 h review: {fid}")
        reviews[fid] = review
    return reviews


def make_page(path: Path, all_rows: list[dict], details: dict, summary: dict):
    """One self-contained offline index; chart points remain the observed points."""
    payload = []
    for row in all_rows:
        d = details.get(row["canonical_id"])
        payload.append(dict(id=row["file_id"], canonical=row["canonical_id"],
                            cls=row["final_class"], status=row["classification_status"],
                            basis=row["decision_basis"], reason=row["window_reason"],
                            som=row["som_candidate_class"], direct=row["som_direct_group"],
                            branch=row["branch"], neuron=row["best_matching_neuron"],
                            agreement=row["ensemble_agreement"],
                            stability=row["som_stability"],
                            events=row["one_percent_source_extrema"],
                            source_review=row["source_review_200h_status"],
                            source_review_class=row["source_review_200h_class"],
                            corrected_series=row["source_review_200h_corrected_series"],
                            source_review_note=row["source_review_200h_note"],
                            valley=row["valley_event"], trough=row["valley_trough_h"],
                            depth=row["valley_depth"], recovery=row["valley_recovery"],
                            boundary=row["boundary_valley_candidate"],
                            boundary_trough=row["boundary_valley_trough_h"],
                            boundary_recovery=row["boundary_valley_recovery_by_250h"],
                            start=row["window_start_h"], end=row["window_end_h"],
                            n=row["window_points"], group=row["source_group"],
                            csv=row["source_csv"], image=row["source_image"],
                            points=chart_points(d)))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    counts = summary["classes"]
    cards = "".join(f"<span class='count'><b>{label.title()}</b> {counts.get(label, 0)}</span>"
                    for label in run.CLASSES)
    html = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>0–200 h PCE 曲线分类</title>
<style>
:root{font:14px system-ui,-apple-system,sans-serif;color:#183044;background:#f5f7f9}
*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;background:#143d50;color:white}
h1{margin:0 0 7px;font-size:23px}header p{margin:0;color:#dbe8ed}main{display:grid;grid-template-columns:minmax(390px,44%) 1fr;height:calc(100vh - 105px)}
aside{overflow:auto;border-right:1px solid #d9e2e7;background:white;padding:18px}section{padding:22px;overflow:auto}
.counts{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}.count{background:#e7f1f4;border-radius:20px;padding:6px 10px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}.controls input{grid-column:1/3}
input,select{font:inherit;padding:8px;border:1px solid #b9cbd2;border-radius:5px;background:white;min-width:0}
#shown{color:#537081;margin:8px 0}.list{display:grid;gap:5px}.item{display:grid;grid-template-columns:65px 65px 1fr;gap:8px;align-items:center;text-align:left;width:100%;padding:8px;border:1px solid #e2eaee;border-radius:6px;background:white;color:inherit;cursor:pointer}
.item:hover,.item.active{border-color:#1681a1;background:#f0f9fb}.item small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#577181}
.tag{font-weight:700;text-transform:capitalize}.tag.valley{color:#9c3c75}.tag.hill{color:#ad5a1c}.tag.bridge{color:#36765d}.tag.slope{color:#2565a5}
.muted{color:#667d8b}#chart{width:100%;max-width:950px;height:auto;background:white;border:1px solid #d9e2e7;border-radius:8px}
.meta{display:grid;grid-template-columns:145px 1fr;gap:6px 10px;max-width:900px;margin-top:18px}.meta dt{color:#637d8c}.meta dd{margin:0;overflow-wrap:anywhere}
a{color:#006d93}details{max-width:900px;margin-top:15px}details summary{cursor:pointer}
@media(max-width:800px){main{display:block;height:auto}aside{max-height:480px;border-right:0;border-bottom:1px solid #d9e2e7}}
</style>
<header><h1>0–200 h PCE 曲线分类</h1><p>仅使用真实 0–200 h 内的观测点和窗口内最大 PCE 归一化；1% 波动保留。点击左侧曲线查看原始点和判类依据。</p></header>
<main><aside><div class="counts">__CARDS__</div><div class="controls">
<input id="search" placeholder="搜索 ID、CSV 或 DOI">
<select id="classFilter"><option value="">全部类别</option><option>bridge</option><option>hill</option><option>slope</option><option>valley</option><option value="unresolved">未能分类</option></select>
<select id="statusFilter"><option value="">全部状态</option><option value="candidate">SOM 候选</option><option value="valley_event">Valley 事件</option><option value="source_verified_200h">原图核验</option><option value="source_review_ambiguous">原图仍有歧义</option><option value="source_identity_unresolved">原图系列身份不符</option><option value="source_digitization_unresolved">采点与原图不符</option><option value="unresolved">窗口数据不足/轴未确认</option></select>
<select id="reviewFilter"><option value="">全部复核标记</option><option value="override">Valley 改判</option><option value="boundary">200 h 边界 Valley 候选</option><option value="variable">跨种子不稳定</option><option value="unnamed">最佳神经元未命名</option><option value="short">≤4 个窗口点</option></select>
</div><div id="shown"></div><div id="list" class="list"></div></aside>
<section><h2 id="title">选择一条曲线</h2><canvas id="chart" width="1100" height="570"></canvas>
<dl class="meta" id="meta"></dl><details><summary>判读说明</summary><p>SOM 在窗口曲线上无监督训练；神经元的 Bridge/Hill/Slope/Valley 名称是在训练后依据形状赋予。最终类别优先记录窗口内有下降及回升证据的 Valley 事件。神经元、SOM 候选和最终类别分列展示。1–3% 的小幅 Valley 需要下降和回升各至少两个连续时间步；孤立的单点抖动仍计入波动数，但不会强制改判。数据不足时保留为未能分类。</p></details>
</section></main><script>const data=__DATA__;
const list=document.getElementById('list'),chart=document.getElementById('chart'),ctx=chart.getContext('2d');
const meta=document.getElementById('meta'),title=document.getElementById('title');
const search=document.getElementById('search'),classFilter=document.getElementById('classFilter');
const statusFilter=document.getElementById('statusFilter'),reviewFilter=document.getElementById('reviewFilter'),shown=document.getElementById('shown');
const byId=new Map(data.map(d=>[d.id,d]));let selected='';
function metric(value,digits=3){const n=Number(value);return value===''||value==null||!Number.isFinite(n)?'—':n.toFixed(digits)}
function link(path){if(!path)return '—';const a=document.createElement('a');a.href='../../../data_final/'+path.split('/').map(encodeURIComponent).join('/');a.textContent=path;a.target='_blank';return a}
function addMeta(label,value){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;if(value instanceof Node)dd.append(value);else dd.textContent=String(value??'—');meta.append(dt,dd)}
function draw(d){ctx.clearRect(0,0,1100,570);ctx.fillStyle='#fff';ctx.fillRect(0,0,1100,570);const p=d.points;if(p.length<2){ctx.fillStyle='#506c7c';ctx.font='22px system-ui';ctx.fillText('0–200 h 内无足够的可比 PCE 观测点',280,275);return}
const L=90,R=1040,T=40,B=490,x0=0,x1=200,yy=p.map(v=>v[1]),y0=Math.min(0,Math.floor(Math.min(...yy)*10)/10),y1=Math.max(1.05,Math.ceil(Math.max(...yy)*10)/10),X=x=>L+(x-x0)/(x1-x0)*(R-L),Y=y=>B-(y-y0)/(y1-y0)*(B-T);
ctx.font='15px system-ui';ctx.fillStyle='#465f6d';ctx.strokeStyle='#dbe4e8';ctx.lineWidth=1;
for(let i=0;i<=4;i++){let x=i*50;ctx.beginPath();ctx.moveTo(X(x),T);ctx.lineTo(X(x),B);ctx.stroke();ctx.fillText(String(x),X(x)-8,B+24)}
for(let i=0;i<=5;i++){let y=y0+(y1-y0)*i/5;ctx.beginPath();ctx.moveTo(L,Y(y));ctx.lineTo(R,Y(y));ctx.stroke();ctx.fillText(y.toFixed(2),L-54,Y(y)+5)}
ctx.strokeStyle='#304c5b';ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,B);ctx.lineTo(R,B);ctx.stroke();ctx.fillText('Time (h)',530,545);ctx.save();ctx.translate(24,300);ctx.rotate(-Math.PI/2);ctx.fillText('Normalized PCE',0,0);ctx.restore();
ctx.strokeStyle=d.cls==='valley'?'#a34179':'#177c9a';ctx.lineWidth=2;ctx.beginPath();p.forEach((v,i)=>i?ctx.lineTo(X(v[0]),Y(v[1])):ctx.moveTo(X(v[0]),Y(v[1])));ctx.stroke();ctx.fillStyle='#193c4d';for(const v of p)ctx.fillRect(X(v[0])-1.5,Y(v[1])-1.5,3,3);
if(d.valley==='True'||d.valley===true){ctx.fillStyle='#b31a69';ctx.beginPath();ctx.arc(X(Number(d.trough)),Y(p.reduce((a,v)=>Math.abs(v[0]-Number(d.trough))<Math.abs(a[0]-Number(d.trough))?v:a,p[0])[1]),5,0,2*Math.PI);ctx.fill()}}
function show(id){const d=byId.get(id);if(!d)return;selected=id;document.querySelectorAll('.item').forEach(b=>b.classList.toggle('active',b.dataset.id===id));title.textContent=d.id+' · '+(d.cls||'未能分类');draw(d);meta.replaceChildren();addMeta('判类依据',d.basis||d.reason);addMeta('SOM 候选',d.som||'—');addMeta('直接 BMU 组',d.direct||'—');addMeta('SOM 分支 / 神经元',(d.branch||'—')+' / '+(d.neuron===''?'—':d.neuron));addMeta('种子一致率',d.agreement===''?'—':Math.round(Number(d.agreement)*100)+'%');addMeta('窗口观测点',d.n||'0');addMeta('窗口时间',d.start===''?'—':metric(d.start,2)+'–'+metric(d.end,2)+' h');addMeta('≥1% 极值事件',d.events===''?'—':d.events);addMeta('Valley 深度 / 回升',d.valley==='True'?metric(d.depth)+' / '+metric(d.recovery):'—');if(d.boundary==='True'||d.boundary===true)addMeta('200 h 边界谷底 / 250 h 前回升',metric(d.boundary_trough,1)+' h / '+metric(d.boundary_recovery));addMeta('原图复核',d.source_review||'未核');if(d.corrected_series)addMeta('原图核对系列',d.corrected_series);if(d.source_review_note)addMeta('原图复核说明',d.source_review_note);addMeta('规范曲线',d.canonical);addMeta('来源',d.group);addMeta('原始 CSV',link(d.csv));addMeta('来源图',link(d.image))}
function render(){const q=search.value.trim().toLowerCase(),cls=classFilter.value,status=statusFilter.value,review=reviewFilter.value;list.replaceChildren();let count=0;const fragment=document.createDocumentFragment();for(const d of data){if(cls&&(cls==='unresolved'?!!d.cls:d.cls!==cls))continue;if(status&&d.status!==status)continue;if(review==='override'&&!(d.basis==='valley_event_within_200h'&&d.som!=='valley'))continue;if(review==='boundary'&&!(d.boundary==='True'||d.boundary===true))continue;if(review==='variable'&&d.stability!=='variable')continue;if(review==='unnamed'&&d.direct!=='unresolved')continue;if(review==='short'&&!(Number(d.n)>0&&Number(d.n)<=4))continue;if(q&&!(d.id+' '+d.csv+' '+d.group).toLowerCase().includes(q))continue;count++;const b=document.createElement('button');b.className='item'+(d.id===selected?' active':'');b.dataset.id=d.id;const id=document.createElement('b'),tag=document.createElement('span'),small=document.createElement('small');id.textContent=d.id;tag.className='tag '+d.cls;tag.textContent=d.cls||'—';small.textContent=d.csv;b.append(id,tag,small);b.onclick=()=>show(d.id);fragment.append(b)}list.append(fragment);shown.textContent='显示 '+count+' / '+data.length+' 个原始文件'}
search.oninput=classFilter.onchange=statusFilter.onchange=reviewFilter.onchange=render;render();if(data.length)show(data.find(d=>d.points.length)?.id||data[0].id);
</script></html>"""
    path.write_text(html.replace("__CARDS__", cards).replace("__DATA__", data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE.parent / "data_final")
    parser.add_argument("--output", type=Path, default=RESULTS / "som_200h")
    parser.add_argument("--hours", type=float, default=200)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    args = parser.parse_args()
    if args.hours <= 0 or not math.isfinite(args.hours):
        parser.error("--hours must be positive and finite")
    seeds = sorted({int(s) for s in args.seeds.split(",") if s.strip()})
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    args.output.mkdir(parents=True, exist_ok=True)
    canonical = run.rows(RESULTS / "canonical_curves.csv")
    manifest = run.rows(RESULTS / "input_manifest.csv")
    source_reviews = load_source_reviews(args.root, canonical,
                                         HERE / "review_inputs" / "source_reviews_200h.csv")
    values, raw, metadata, details, excluded = load_window(args.root, canonical, args.hours)
    print(f"window curves={len(metadata)} excluded={len(excluded)}", flush=True)
    x, _, branch, events = som.features(values, raw, derivative_weight=.3)
    # Duration information prevents a short observed segment from looking
    # identical to a full 200 h observation after phase alignment.
    spans = np.array([(details[r["file_id"]]["t"][-1] - details[r["file_id"]]["t"][0]) / args.hours
                      for r in metadata])
    starts = np.array([details[r["file_id"]]["t"][0] / args.hours for r in metadata])
    x = np.column_stack((x, .15 * spans, .08 * starts))
    stage_ref = {}
    for value, row in zip(values, metadata):
        m = run.morphology({"t": np.linspace(0, 1, som.N_GRID), "t_hour": None,
                            "y": value}, run.DEFAULT)
        stage_ref[row["file_id"]] = m
    stage_labels = [stage_ref[r["file_id"]]["candidate_class"]
                    if stage_ref[r["file_id"]]["stages_complete"] else "" for r in metadata]
    by_seed, models_by_seed, prototypes, quality = {}, {}, [], {}
    arrays = {"feature_matrix": x, "file_ids": np.array([r["file_id"] for r in metadata])}
    for seed in seeds:
        model, named = som.train(x, branch, np.ones(len(x), bool), seed, args.side,
                                 args.iterations, stage_labels=stage_labels,
                                 early_stage_balance_power=2)
        models_by_seed[seed] = model
        by_seed[seed] = som.assign(x, branch, metadata, events, model)
        quality[seed] = som.map_quality(x, branch, model, np.ones(len(x), bool), args.side)["fit"]
        prototypes.extend(dict(seed=seed, **r) for r in named)
        for b, name in ((0, "no_early_gain"), (1, "early_gain")):
            arrays[f"seed_{seed}_{name}_codebook"] = model[b][0]
        print(f"seed {seed}: {Counter(r['som_candidate_class'] for r in by_seed[seed])}", flush=True)
    # Select by topology and stage agreement when all four complete stages
    # occur in-window; otherwise use topology and quantization alone.
    reference = {r["file_id"]: s for r, s in zip(metadata, stage_labels)}
    if all(label in stage_labels for label in run.CLASSES):
        stage_scores = {s: som.balanced_stage_agreement(by_seed[s], reference)[0] for s in seeds}
        selected = som.select_seed(seeds, quality, stage_scores)
        selection = "topology_then_balanced_window_stage_agreement"
    else:
        stage_scores = {}
        selected = min(seeds, key=lambda s: (quality[s]["topographic_error"],
                                             quality[s]["quantization_error"], s))
        selection = "topology_then_quantization"
    assignments, votes = som.consensus(by_seed, selected)
    assignments_by_id = {r["file_id"]: r for r in assignments}
    for a in assignments:
        d = details[a["file_id"]]
        event = valley_event(d["t"], d["y"])
        boundary = boundary_valley_event(d["near_t"], d["near_y"], args.hours)
        m = stage_ref[a["file_id"]]
        peaks, peak_props = find_peaks(d["y"], prominence=.01)
        troughs, trough_props = find_peaks(-d["y"], prominence=.01)
        micro_events = int(np.sum(peak_props["prominences"] < .03)
                           + np.sum(trough_props["prominences"] < .03))
        a.update(final_class="valley" if event else a["som_candidate_class"],
                 decision_basis="valley_event_within_200h" if event else "som_window_candidate",
                 classification_status="valley_event" if event else "candidate",
                 window_stage_class=m["candidate_class"],
                 window_stages_complete=m["stages_complete"],
                 one_to_three_percent_extrema=micro_events,
                 valley_event=bool(event), valley_trough_h=event["trough_h"] if event else "",
                 valley_depth=event["depth"] if event else "",
                 valley_recovery=event["recovery"] if event else "",
                 boundary_valley_candidate=bool(boundary),
                 boundary_valley_trough_h=boundary["trough_h"] if boundary else "",
                 boundary_valley_depth=boundary["depth"] if boundary else "",
                 boundary_valley_recovery_by_250h=boundary["recovery"] if boundary else "",
                 window_points=len(d["t"]), window_start_h=float(d["t"][0]),
                 window_end_h=float(d["t"][-1]), window_max_pce=d["divisor"],
                 duplicate_times=d["duplicates"])
        review = source_reviews.get(a["file_id"])
        a.update(source_review_200h_status=review["review_status"] if review else "",
                 source_review_200h_class=review["reviewed_class"] if review else "",
                 source_review_200h_corrected_series=review.get("corrected_series", "") if review else "",
                 source_review_200h_note=review["notes"] if review else "")
        if review:
            if review["review_status"] == "verified":
                a.update(final_class=review["reviewed_class"],
                         decision_basis="source_review_200h",
                         classification_status="source_verified_200h")
            elif review["review_status"] == "ambiguous":
                a.update(final_class=a["som_candidate_class"],
                         decision_basis="source_review_ambiguous_som_candidate",
                         classification_status="source_review_ambiguous")
            elif review["review_status"] == "identity_mismatch":
                a.update(final_class="", decision_basis="source_identity_mismatch",
                         classification_status="source_identity_unresolved")
            else:
                a.update(final_class="", decision_basis="source_digitization_mismatch",
                         classification_status="source_digitization_unresolved")
    final_rows = []
    for source in manifest:
        canon = source["canonical_id"]
        a = assignments_by_id.get(canon)
        reason = (excluded.get(canon, "") if a is None else
                  "source_identity_mismatch" if a["classification_status"] == "source_identity_unresolved"
                  else "source_digitization_mismatch" if a["classification_status"] == "source_digitization_unresolved"
                  else "")
        row = {key: source.get(key, "") for key in FIELDS}
        row.update(canonical_id=canon,
                   classification_status=a["classification_status"] if a else "unresolved",
                   window_reason=reason, final_class=a["final_class"] if a else "",
                   decision_basis=a["decision_basis"] if a else "",
                   som_candidate_class=a["som_candidate_class"] if a else "",
                   som_direct_group=a["som_direct_group"] if a else "",
                   branch=a["branch"] if a else "",
                   best_matching_neuron=a["best_matching_neuron"] if a else "",
                   som_candidate_neuron=a["som_candidate_neuron"] if a else "",
                   som_stability=a["som_stability"] if a else "",
                   ensemble_agreement=a["ensemble_agreement"] if a else "",
                   quantization_distance=a["quantization_distance"] if a else "",
                   window_points=a["window_points"] if a else "",
                   duplicate_times=a["duplicate_times"] if a else "",
                   window_start_h=a["window_start_h"] if a else "",
                   window_end_h=a["window_end_h"] if a else "",
                   window_max_pce=a["window_max_pce"] if a else "",
                   one_percent_source_extrema=a["one_percent_source_extrema"] if a else "",
                   one_to_three_percent_extrema=a["one_to_three_percent_extrema"] if a else "",
                   valley_event=a["valley_event"] if a else "",
                   valley_trough_h=a["valley_trough_h"] if a else "",
                   valley_depth=a["valley_depth"] if a else "",
                   valley_recovery=a["valley_recovery"] if a else "",
                   boundary_valley_candidate=a["boundary_valley_candidate"] if a else "",
                   boundary_valley_trough_h=a["boundary_valley_trough_h"] if a else "",
                   boundary_valley_depth=a["boundary_valley_depth"] if a else "",
                   boundary_valley_recovery_by_250h=a["boundary_valley_recovery_by_250h"] if a else "",
                   window_stage_class=a["window_stage_class"] if a else "",
                   window_stages_complete=a["window_stages_complete"] if a else "",
                   source_review_200h_status=a["source_review_200h_status"] if a else "",
                   source_review_200h_class=a["source_review_200h_class"] if a else "",
                   source_review_200h_corrected_series=a["source_review_200h_corrected_series"] if a else "",
                   source_review_200h_note=a["source_review_200h_note"] if a else "")
        final_rows.append(row)
    run.write_rows(args.output / "all_files_200h.csv", final_rows, FIELDS)
    run.write_rows(args.output / "canonical_assignments_200h.csv", assignments,
                   list(assignments[0]))
    run.write_rows(args.output / "som_seed_votes.csv", votes, list(votes[0]))
    run.write_rows(args.output / "som_prototypes.csv", prototypes, list(prototypes[0]))
    run.write_rows(args.output / "excluded_canonical.csv",
                   [dict(file_id=k, reason=v) for k, v in excluded.items()], ["file_id", "reason"])
    source_by_id = {r["file_id"]: r for r in metadata}
    review_queue = []
    for a in assignments:
        if a["source_review_200h_status"] == "verified":
            continue
        reasons = []
        if a["source_review_200h_status"] == "ambiguous":
            reasons.append("source_ambiguous")
        if a["source_review_200h_status"] == "identity_mismatch":
            reasons.append("source_identity_mismatch")
        if a["source_review_200h_status"] == "digitization_mismatch":
            reasons.append("source_digitization_mismatch")
        if a["boundary_valley_candidate"] and not a["valley_event"]:
            reasons.append("boundary_valley_candidate")
        if a["decision_basis"] == "valley_event_within_200h" and a["som_candidate_class"] != "valley":
            reasons.append("valley_override")
        if a["som_stability"] == "variable":
            reasons.append("seed_variable")
        if a["som_direct_group"] == "unresolved":
            reasons.append("bmu_unnamed")
        if a["window_points"] <= 4:
            reasons.append("few_window_points")
        if reasons:
            source = source_by_id[a["file_id"]]
            review_queue.append(dict(priority=(0 if "source_ambiguous" in reasons or
                                               "source_identity_mismatch" in reasons or
                                               "source_digitization_mismatch" in reasons or
                                               "boundary_valley_candidate" in reasons or
                                               "valley_override" in reasons else
                                               1 if "seed_variable" in reasons else
                                               2 if "few_window_points" in reasons else 3),
                                     file_id=a["file_id"], source_csv=a["source_csv"],
                                     source_image=source["source_image"],
                                     final_class=a["final_class"],
                                     som_candidate_class=a["som_candidate_class"],
                                     best_matching_neuron=a["best_matching_neuron"],
                                     reasons=";".join(reasons),
                                     ensemble_agreement=a["ensemble_agreement"],
                                     window_points=a["window_points"],
                                     valley_trough_h=a["valley_trough_h"],
                                     valley_depth=a["valley_depth"],
                                     valley_recovery=a["valley_recovery"],
                                     boundary_valley_trough_h=a["boundary_valley_trough_h"],
                                     boundary_valley_depth=a["boundary_valley_depth"],
                                     boundary_valley_recovery_by_250h=a["boundary_valley_recovery_by_250h"]))
    review_queue.sort(key=lambda r: (r["priority"], r["file_id"]))
    run.write_rows(args.output / "review_priority_200h.csv", review_queue,
                   ["priority", "file_id", "source_csv", "source_image", "final_class",
                    "som_candidate_class", "best_matching_neuron", "reasons",
                    "ensemble_agreement", "window_points", "valley_trough_h",
                    "valley_depth", "valley_recovery", "boundary_valley_trough_h",
                    "boundary_valley_depth", "boundary_valley_recovery_by_250h"])
    np.savez_compressed(args.output / "som_model.npz", **arrays)
    summary = dict(hour_limit=args.hours, normalization="maximum PCE among original points in 0–200 h",
                   raw_input_files=len(manifest), canonical_curves=len(canonical),
                   comparable_canonical=len(metadata), unresolved_canonical=len(excluded),
                   four_class_raw_files=sum(bool(r["final_class"]) for r in final_rows),
                   unresolved_raw_files=sum(not r["final_class"] for r in final_rows),
                   classes=dict(Counter(r["final_class"] for r in final_rows if r["final_class"])),
                   statuses=dict(Counter(r["classification_status"] for r in final_rows)),
                   unresolved_reasons=dict(Counter(r["window_reason"] for r in final_rows if r["window_reason"])),
                   valley_overrides=sum(a["decision_basis"] == "valley_event_within_200h" and
                                        a["som_candidate_class"] != "valley" for a in assignments),
                   source_verified_200h_curves=sum(a["source_review_200h_status"] == "verified"
                                                   for a in assignments),
                   source_ambiguous_200h_curves=sum(a["source_review_200h_status"] == "ambiguous"
                                                    for a in assignments),
                   source_identity_mismatch_curves=sum(a["source_review_200h_status"] == "identity_mismatch"
                                                       for a in assignments),
                   source_digitization_mismatch_curves=sum(a["source_review_200h_status"] == "digitization_mismatch"
                                                           for a in assignments),
                   one_percent_event_curves=sum(a["one_percent_source_extrema"] > 0 for a in assignments),
                   boundary_valley_candidates=sum(bool(a["boundary_valley_candidate"])
                                                  for a in assignments),
                   selected_seed=selected, seed_selection=selection,
                   seeds=seeds, side=args.side, iterations=args.iterations,
                   seed_quality={str(k): v for k, v in quality.items()},
                   stage_agreement={str(k): v for k, v in stage_scores.items()},
                   branch_counts=dict(Counter("early_gain" if b else "no_early_gain" for b in branch)),
                   prototype_counts=dict(Counter(p["label"] or "unnamed" for p in prototypes)))
    summary["review_priority_canonical"] = len(review_queue)
    summary["review_priority_images"] = len({r["source_image"] for r in review_queue})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_page(args.output / "all_curves_index.html", final_rows, details, summary)
    (args.output / "README.md").write_text(
        "# 0–200 h PCE 曲线分类\n\n"
        "运行 `python3 target_shape_classification/classify_200h.py` 可重建本目录。"
        "读取 `results/input_manifest.csv` 与 `canonical_curves.csv` 所记录的版本；原始文件和旧的全程结果不修改。\n\n"
        "SOM 训练只选择实际 0–200 h 内的原始时间点，按窗口最大 PCE 归一化；不使用 200 h 后的点补训练窗口。"
        "相同时间点的不同读数按原管线的中位数合并，重复次数保留在 CSV；1% 及以上的时间极值被计数，"
        "并进入 SOM 的源点包络与事件特征。短于 200 h 的曲线只使用已有观测区间。\n\n"
        "两张 10×10 SOM 地图按早期增益分支独立训练；阶段平衡采样仅由窗口内曲线计算，"
        "训练不读取人工四类标签。神经元名称是训练后的形状解释，CSV 同时保留匿名 BMU、SOM 候选、"
        "种子投票和最终类别。最终 Valley 优先规则只根据窗口内的下降与回升事件触发；"
        "150–200 h 的谷底若在 200–250 h 有至少 1% 恢复，会另列边界复核候选；原图核实后可按 200 h 左右的 Valley 归类。"
        "有当前原图哈希支持的人工复核可覆盖暂定类别。未复核的最终类仍为候选，不能当作人工确认标签。\n\n"
        "`all_files_200h.csv` 和 `all_curves_index.html` 覆盖全部原始文件；"
        "非 PCE、小时轴未确认或窗口点数不足者明确列为 unresolved。"
        "`review_priority_200h.csv` 优先列出 Valley 改判、200 h 边界恢复、跨种子不稳定、神经元未命名与窗口点少的曲线，"
        "供后续对照原图复核；已确认的曲线退出该队列，有歧义者留在队列。"
        "HTML 可用本地文件方式直接打开；图连接原始观测点，横轴固定为真实 0–200 h。\n",
        encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("raw_input_files", "comparable_canonical",
                        "classes", "unresolved_raw_files", "valley_overrides", "selected_seed")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
