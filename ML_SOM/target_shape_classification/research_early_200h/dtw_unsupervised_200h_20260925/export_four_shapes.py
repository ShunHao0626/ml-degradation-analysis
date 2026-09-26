"""Join frozen anonymous DTW groups to separate, post hoc four-shape candidates.

The existing integrated four-class decisions are used only here. This script
does not refit the distance, neighbor graph, or clusters.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INTEGRATED = HERE.parent / "final_integrated_20260925" / "class_index.csv"
SHAPES = ("bridge", "hill", "valley", "slope")
SOURCE_ROOT = Path(__file__).resolve().parents[3] / "original_curves_2250"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reviewed_axis_points(record: dict) -> tuple[list[dict], float]:
    """Use source coordinates for two paper-verified PCE axes absent from DTW v1."""
    path = (SOURCE_ROOT / record["source_csv"]).resolve()
    assert path.is_relative_to(SOURCE_ROOT) and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["source_sha256"]
    factor = float(record["time_factor"])
    points = []
    for source_row, item in enumerate(read_csv(path), 2):
        try:
            hour, value = float(item["x"]) * factor, float(item["y"])
        except (ValueError, TypeError):
            continue
        if 0 <= hour <= 200:
            points.append((hour, value, source_row))
    divisor = max(value for _, value, _ in points)
    assert divisor > 0
    output = [dict(hour=f"{hour:.12g}", source_y=f"{value:.12g}",
                   normalized_pce=f"{value/divisor:.12g}", source_row=source_row)
              for hour, value, source_row in sorted(points)]
    return output, divisor


def main() -> None:
    index = read_csv(RESULTS / "curve_index.csv")
    assignments = {r["curve_id"]: r for r in read_csv(RESULTS / "anonymous_assignments.csv")}
    decisions = {r["curve_id"]: r for r in read_csv(INTEGRATED)}
    assert len(index) == len(assignments) == 2246
    assert len(decisions) == 2216

    fields = ["curve_id", "shape_candidate", "evidence_status", "classification_origin",
              "dtw_model_status", "anonymous_cluster_k8", "anonymous_cluster_k12",
              "original_points_0_200", "first_hour", "last_hour",
              "normalization_divisor", "source_csv", "source_sha256", "normalized_csv"]
    candidates: list[dict] = []
    point_rows: list[dict] = []
    viewer_rows: list[list] = []
    for record in index:
        curve_id = record["curve_id"]
        decision = decisions.get(curve_id)
        if not decision or decision["integrated_class"] not in SHAPES:
            continue
        # Image-only opinions with 0 or 1 early points are kept in the existing
        # integrated delivery, never promoted to numeric shape evidence here.
        if int(decision["numeric_strict_points"]) < 2:
            continue
        shape = decision["integrated_class"]
        assignment = assignments[curve_id]
        if record["normalized_csv"]:
            points = read_csv(RESULTS / record["normalized_csv"])
            normalized_csv = record["normalized_csv"]
            divisor = record["normalization_divisor"]
        else:
            # F01223/F01224 were verified as normalized PCE against Fig. 6c
            # after the anonymous DTW v1 inputs had been frozen. They enter
            # only this post hoc export, with no fabricated DTW cluster.
            assert curve_id in {"F01223", "F01224"}
            assert decision["classification_origin"] == "paper_axis_check_plus_automatic_stage_rule"
            points, scale = reviewed_axis_points(record)
            normalized_csv = f"posthoc_points/{curve_id}.csv"
            (RESULTS / "posthoc_points").mkdir(exist_ok=True)
            write_csv(RESULTS / normalized_csv, points,
                      ["hour", "source_y", "normalized_pce", "source_row"])
            divisor = f"{scale:.12g}"
        row = dict(curve_id=curve_id, shape_candidate=shape,
                   evidence_status=decision["evidence_status"],
                   classification_origin=decision["classification_origin"],
                   dtw_model_status=record["model_status"],
                   anonymous_cluster_k8=assignment["cluster_fused_k8"],
                   anonymous_cluster_k12=assignment["cluster_fused_k12"],
                   original_points_0_200=len(points),
                   first_hour=points[0]["hour"], last_hour=points[-1]["hour"],
                   normalization_divisor=divisor,
                   source_csv=record["source_csv"], source_sha256=record["source_sha256"],
                   normalized_csv=normalized_csv)
        candidates.append(row)
        # numeric_strict_points counts distinct hours; source CSVs can contain
        # multiple measured values at exactly the same hour.
        assert len({float(p["hour"]) for p in points}) == int(decision["numeric_strict_points"]), (
            curve_id, len(points), decision["numeric_strict_points"])
        xy = []
        for point in points:
            hour = float(point["hour"])
            value = float(point["normalized_pce"])
            assert 0 <= hour <= 200
            point_rows.append(dict(curve_id=curve_id, shape_candidate=shape,
                                   hour=point["hour"], normalized_pce=point["normalized_pce"],
                                   source_row=point["source_row"]))
            xy.append([round(hour, 8), round(value, 8)])
        assert abs(max(v for _, v in xy) - 1) < 1e-7
        viewer_rows.append([curve_id, shape, decision["evidence_status"],
                            assignment["cluster_fused_k8"], xy])

    counts = Counter(r["shape_candidate"] for r in candidates)
    assert counts == dict(bridge=87, hill=176, slope=1868, valley=76), counts
    assert len(candidates) == 2207
    write_csv(RESULTS / "four_shape_candidates.csv", candidates, fields)
    write_csv(RESULTS / "four_shape_points.csv", point_rows,
              ["curve_id", "shape_candidate", "hour", "normalized_pce", "source_row"])
    for shape in SHAPES:
        write_csv(RESULTS / f"{shape}_candidates.csv",
                  [r for r in candidates if r["shape_candidate"] == shape], fields)

    payload = json.dumps(viewer_rows, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>0–200 h PCE shape candidates</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1150px;margin:25px auto;padding:0 16px;color:#18302d}
h1{font-size:25px}p{line-height:1.45}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}
label{display:grid;gap:4px}select,input{padding:7px;font:inherit}main{display:grid;grid-template-columns:300px 1fr;gap:18px}
#list{height:560px;overflow:auto;border:1px solid #cad5d0}button{display:block;width:100%;padding:7px 12px;background:white;border:0;border-bottom:1px solid #e4e9e6;text-align:left;cursor:pointer}
button:hover,button.active{background:#e5f1e9}svg{width:100%;height:450px;border:1px solid #cad5d0;background:white}
#details{margin:10px 0}a{color:#12688c}@media(max-width:700px){main{display:block}#list{height:230px}}
</style><h1>0–200 h Normalized PCE: four shape candidates</h1>
<p>The curves shown here contain only measured points at 0–200 h, each divided by its own maximum in that window. Shape names come from a separate stage rule / source review after anonymous DTW clustering. They are candidates, not labels discovered by DTW or independently validated truth. The anonymous cluster number is displayed for comparison.</p>
<div class="controls"><label>Shape<select id="shape"><option value="">All shapes</option><option>bridge</option><option>hill</option><option>valley</option><option>slope</option></select></label>
<label>Evidence<select id="status"><option value="">All evidence statuses</option><option>stage_candidate</option><option>insufficient_evidence</option><option>pattern_conflict</option></select></label>
<label>Curve ID<input id="search" placeholder="F00001"></label></div>
<p id="count"></p><main><div id="list"></div><section><div id="details">Select a curve.</div><svg id="chart" viewBox="0 0 760 450" aria-label="0 to 200 hour normalized PCE chart"></svg></section></main>
<p>Download: <a href="four_shape_candidates.csv">candidate index</a> · <a href="four_shape_points.csv">all measured points</a> · <a href="bridge_candidates.csv">Bridge</a> · <a href="hill_candidates.csv">Hill</a> · <a href="valley_candidates.csv">Valley</a> · <a href="slope_candidates.csv">Slope</a>.</p>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id), palette={bridge:'#24734f',hill:'#998200',valley:'#cd7440',slope:'#446fbb'};
let current=null;
function filtered(){return DATA.filter(r=>(!$('shape').value||r[1]==$('shape').value)&&(!$('status').value||r[2]==$('status').value)&&r[0].toLowerCase().includes($('search').value.trim().toLowerCase()))}
function show(r){current=r;$('details').textContent=`${r[0]} · ${r[1]} candidate · ${r[2]} · anonymous DTW cluster ${r[3]||'unavailable'} · ${r[4].length} original points`;
 const pts=r[4], lo=Math.min(0.75,...pts.map(p=>p[1]))-0.02, hi=Math.max(1.02,...pts.map(p=>p[1]))+0.02;
 const X=x=>55+670*x/200,Y=y=>390-335*(y-lo)/(hi-lo);
 const path=pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(2)+' '+Y(p[1]).toFixed(2)).join(' ');
 const dots=pts.map(p=>`<circle cx="${X(p[0]).toFixed(2)}" cy="${Y(p[1]).toFixed(2)}" r="2.4" fill="${palette[r[1]]}"/>`).join('');
 $('chart').innerHTML=`<line x1="55" y1="390" x2="725" y2="390" stroke="#777"/><line x1="55" y1="55" x2="55" y2="390" stroke="#777"/><path d="${path}" fill="none" stroke="${palette[r[1]]}" stroke-width="2.2"/>${dots}<text x="345" y="430">Hour (h), 0–200</text><text transform="translate(20 285) rotate(-90)">Normalized PCE</text><text x="50" y="408">0</text><text x="692" y="408">200</text><text x="12" y="67">${hi.toFixed(2)}</text><text x="12" y="390">${lo.toFixed(2)}</text>`;
 document.querySelectorAll('#list button').forEach(b=>b.classList.toggle('active',b.dataset.id==r[0]));}
function render(){const rows=filtered(), list=$('list');$('count').textContent=`${rows.length} curves match. Showing the first ${Math.min(rows.length,300)} in the list; search by ID for any curve.`;list.replaceChildren();
 rows.slice(0,300).forEach(r=>{const b=document.createElement('button');b.dataset.id=r[0];b.textContent=`${r[0]} · ${r[1]} · ${r[2]}`;b.onclick=()=>show(r);list.append(b)});
 if(rows.length)show(rows.find(r=>current&&r[0]==current[0])||rows[0]);else{$('details').textContent='No matching curves.';$('chart').replaceChildren();}}
 ['shape','status','search'].forEach(id=>$(id).addEventListener('input',render));render();</script></html>"""
    (RESULTS / "four_shape_viewer.html").write_text(page.replace("__DATA__", payload), encoding="utf-8")
    summary = dict(four_shape_numeric_candidates=len(candidates), counts=dict(counts),
                   measured_points=len(point_rows), model_eligible=sum(r["dtw_model_status"] == "model_eligible" for r in candidates),
                   shape_origin="posthoc_stage_rule_and_prior_source_review",
                   dtw_clusters_remain_anonymous=True)
    (RESULTS / "four_shape_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
