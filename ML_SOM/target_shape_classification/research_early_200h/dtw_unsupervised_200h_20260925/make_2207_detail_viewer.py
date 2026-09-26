"""Build a zoomable view of all 2207 strict numeric four-shape candidates.

Candidate names are post hoc, unvalidated annotations and never DTW inputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRICT = HERE.parent / "evaluation/strict_numeric_coverage_20260925/strict_200h_class_index.csv"
INTEGRATED = HERE.parent / "final_integrated_20260925"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    strict = read_csv(STRICT)
    index = {r["curve_id"]: r for r in read_csv(RESULTS / "curve_index.csv")}
    assignments = {r["curve_id"]: r for r in read_csv(RESULTS / "anonymous_assignments.csv")}
    assert len(strict) == 2207
    rows = []
    for record in strict:
        curve_id = record["curve_id"]
        model = index[curve_id]
        assignment = assignments[curve_id]
        source = (INTEGRATED / record["curve_csv"]).resolve()
        assert source.is_relative_to(INTEGRATED.resolve())
        points = [p for p in read_csv(source) if p["region"] == "strict_200h"]
        assert len({float(p["hour"]) for p in points}) == int(record["numeric_strict_points"])
        rows.append([
            curve_id, record["strict_200h_candidate_class"], record["evidence_status"],
            model["model_status"], assignment["cluster_fused_k8"],
            "../../final_integrated_20260925/" + record["curve_csv"],
            [[round(float(p["hour"]), 8), round(float(p["normalized_pce"]), 8), p["source_row"]]
             for p in points],
        ])
    assert Counter(r[1] for r in rows) == {"bridge": 87, "hill": 176, "slope": 1868, "valley": 76}
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    output = RESULTS / "strict_numeric_2207_detail_viewer.html"
    output.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {output} ({len(rows)} curves; {sum(bool(r[4]) for r in rows)} DTW eligible)")


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2207 条 0–200 h 数值曲线细节</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#20343b;background:#f6f8f8}
body{max-width:1280px;margin:24px auto;padding:0 16px}h1{font-size:26px;margin-bottom:6px}p{line-height:1.5}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;background:white;padding:16px;border:1px solid #d9e2e4;border-radius:8px}
label{display:grid;gap:4px;font-size:13px}input,select,button{font:inherit}input,select{padding:6px;border:1px solid #aebec3;border-radius:4px}input[type=number]{width:90px}input[type=search]{width:125px}
button{padding:6px 11px;background:#e8f1f3;border:1px solid #adc7cf;border-radius:4px;cursor:pointer}button:hover{background:#d4e9ee}
.presets{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:14px}
.panel{background:white;border:1px solid #d9e2e4;border-radius:8px;padding:12px}svg{display:block;width:100%;height:auto;min-height:420px}
#list{max-height:500px;overflow:auto}#list button{display:block;width:100%;text-align:left;margin:3px 0;background:white}#list button.active{background:#d6e9ed;border-color:#4b8090}
.small{font-size:13px;color:#4c6269}@media(max-width:850px){.layout{grid-template-columns:1fr}#list{max-height:190px}}
</style>
<h1>2207 条数值候选：0–200 h 局部细节</h1>
<p>这里的 Bridge / Hill / Valley / Slope 是旧规则给出的候选名称，未经独立人工真值验证。选择曲线后可放大时间和纵轴；圆点是原始观测，悬停查看数值及源 CSV 行号。空缺超过 40 h 的区间不连线。</p>
<div class="controls">
<label>候选形态<select id="shape"><option value="">全部</option><option>bridge</option><option>hill</option><option>valley</option><option>slope</option></select></label>
<label>DTW 状态<select id="dtw"><option value="">全部</option><option value="eligible">已进入 DTW（2065）</option><option value="excluded">未进入 DTW（142）</option></select></label>
<label>曲线编号<input id="search" type="search" placeholder="如 F00001" list="ids"></label><datalist id="ids"></datalist>
<label>小时起点<input id="xmin" type="number" min="0" max="200" step="1" value="0"></label><label>小时终点<input id="xmax" type="number" min="0" max="200" step="1" value="200"></label>
<label>PCE 下限<input id="ymin" type="number" step="0.001" value="0"></label><label>PCE 上限<input id="ymax" type="number" step="0.001" value="1.07"></label>
<label>纵轴<select id="ymode"><option value="auto">自动放大可见点</option><option value="manual">手动范围</option></select></label>
</div>
<div class="presets"><button data-range="0,200">全程</button><button data-range="0,50">0–50 h</button><button data-range="50,100">50–100 h</button><button data-range="100,150">100–150 h</button><button data-range="150,200">150–200 h</button><button id="reset">重置纵轴</button></div>
<div class="layout"><div class="panel"><svg id="chart" viewBox="0 0 1000 550" role="img" aria-label="所选曲线的原始观测点"></svg><p id="status" class="small"></p><p class="small">纵轴归一化分母是该曲线 0–200 h 原始点中的最大 PCE；曲线段只是连接观测点，未显示为 DTW 建模所用的 2 h 插值网格。</p></div>
<div class="panel"><strong>曲线列表</strong><p id="count" class="small"></p><div id="list"></div><p id="details" class="small"></p></div></div>
<script>
const DATA=__DATA__, byId=new Map(DATA.map(r=>[r[0],r])), $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let current=null;
function el(name,attrs={},parent=$('chart')){const e=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));parent.append(e);return e}
function label(x,y,s,attrs={}){const e=el('text',{x,y,fill:'#50636a','font-size':13,...attrs});e.textContent=s}
function render(){if(!current)return;const x0=Number($('xmin').value),x1=Number($('xmax').value);let y0=Number($('ymin').value),y1=Number($('ymax').value);
 if(!Number.isFinite(x0)||!Number.isFinite(x1)||x0<0||x1>200||x0>=x1||$('ymode').value==='manual'&&(!Number.isFinite(y0)||!Number.isFinite(y1)||y0>=y1)){$('status').textContent='坐标范围无效，请检查上下限。';return}
 const pts=current[6],visible=pts.filter(p=>p[0]>=x0&&p[0]<=x1);
 if($('ymode').value==='auto'){if(visible.length){const lo=Math.min(...visible.map(p=>p[1])),hi=Math.max(...visible.map(p=>p[1])),pad=Math.max(0.005,(hi-lo)*0.12);y0=Math.max(0,lo-pad);y1=hi+pad}else{y0=0;y1=1.07}$('ymin').value=y0.toFixed(4);$('ymax').value=y1.toFixed(4)}
 const svg=$('chart');svg.replaceChildren();const l=72,r=975,t=25,b=475,X=x=>l+(x-x0)/(x1-x0)*(r-l),Y=y=>b-(y-y0)/(y1-y0)*(b-t);
 const defs=el('defs'),clip=el('clipPath',{id:'plotClip2207'},defs);el('rect',{x:l,y:t,width:r-l,height:b-t},clip);
 for(let i=0;i<=5;i++){const x=l+(r-l)*i/5,y=b-(b-t)*i/5;el('line',{x1:x,y1:t,x2:x,y2:b,stroke:'#e4eaec'});el('line',{x1:l,y1:y,x2:r,y2:y,stroke:'#e4eaec'});
 label(x,b+23,(x0+(x1-x0)*i/5).toFixed(x1-x0<20?1:0),{'text-anchor':'middle'});label(l-10,y+4,(y0+(y1-y0)*i/5).toFixed(y1-y0<0.1?3:2),{'text-anchor':'end'})}
 el('rect',{x:l,y:t,width:r-l,height:b-t,fill:'none',stroke:'#70858c'});label((l+r)/2,535,'Hour (h)',{'text-anchor':'middle','font-size':16});label(20,250,'Normalized PCE',{transform:'rotate(-90 20 250)','text-anchor':'middle','font-size':16});
 const plot=el('g',{'clip-path':'url(#plotClip2207)'}),colors={bridge:'#287f69',hill:'#a18026',valley:'#be7049',slope:'#416eab'},color=colors[current[1]];
 let segments=[],part=[];for(const p of pts){if(part.length&&p[0]-part[part.length-1][0]>40){segments.push(part);part=[]}part.push(p)}if(part.length)segments.push(part);
 for(const seg of segments)el('path',{d:seg.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(2)+' '+Y(p[1]).toFixed(2)).join(' '),fill:'none',stroke:color,'stroke-width':2.3},plot);
 for(const p of visible)if(p[1]>=y0&&p[1]<=y1){const dot=el('circle',{cx:X(p[0]),cy:Y(p[1]),r:4,fill:color},plot);el('title',{},dot).textContent=`${current[0]} · ${p[0]} h · PCE ${p[1]} · 源 CSV 行 ${p[2]}`}
 $('status').textContent=`${current[0]} · ${current[1]} 候选 · ${pts.length} 个 0–200 h 原始点；当前显示 ${x0}–${x1} h、PCE ${y0.toFixed(4)}–${y1.toFixed(4)}。`;
 $('details').innerHTML=`DTW：${current[4]?`匿名群 ${current[4]}`:'未参与（'+current[3]+'）'}<br>证据状态：${current[2]}<br><a href="${current[5]}">查看逐点 CSV</a>`;
 document.querySelectorAll('#list button').forEach(b=>b.classList.toggle('active',b.dataset.id===current[0]));
}
function filtered(){const q=$('search').value.trim().toUpperCase(),shape=$('shape').value,dtw=$('dtw').value;return DATA.filter(r=>(!shape||r[1]===shape)&&(!dtw||(dtw==='eligible')===Boolean(r[4]))&&(!q||r[0].includes(q)))}
function fillList(){const rows=filtered(),list=$('list');$('count').textContent=`匹配 ${rows.length} / 2207 条`;list.replaceChildren();for(const row of rows){const b=document.createElement('button');b.dataset.id=row[0];b.textContent=`${row[0]} · ${row[1]} · ${row[4]?'群 '+row[4]:'未聚类'}`;b.onclick=()=>{current=row;render()};list.append(b)}
 if(rows.length){current=rows.find(r=>current&&r[0]===current[0])||rows[0];render()}else{current=null;$('chart').replaceChildren();$('status').textContent='没有匹配的曲线。';$('details').textContent=''}}
for(const r of DATA){const o=document.createElement('option');o.value=r[0];$('ids').append(o)}
for(const id of ['shape','dtw','search'])$(id).addEventListener('input',fillList);
for(const id of ['xmin','xmax','ymin','ymax','ymode'])$(id).addEventListener('change',render);
for(const b of document.querySelectorAll('[data-range]'))b.onclick=()=>{const [a,z]=b.dataset.range.split(',');$('xmin').value=a;$('xmax').value=z;render()};
$('reset').onclick=()=>{$('ymode').value='auto';render()};fillList();
</script></html>'''


if __name__ == "__main__":
    main()
