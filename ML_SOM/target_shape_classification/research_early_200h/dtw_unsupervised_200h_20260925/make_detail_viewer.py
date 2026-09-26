"""Build a standalone, zoomable viewer from saved original 0–200 h points.

This is visualization only: it does not modify DTW inputs or assignments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    assignments = {row["curve_id"]: row for row in read_csv(RESULTS / "anonymous_assignments.csv")}
    representatives = {
        row["cluster"]: row["representative_ids"].split(";")
        for row in read_csv(RESULTS / "cluster_representatives.csv")
    }
    curves = []
    for row in read_csv(RESULTS / "curve_index.csv"):
        assigned = assignments.get(row["curve_id"])
        if not assigned or assigned["cluster_fused_k8"] == "":
            continue
        points = read_csv(RESULTS / row["normalized_csv"])
        curves.append([
            row["curve_id"], int(assigned["cluster_fused_k8"]),
            [[round(float(p["hour"]), 8), round(float(p["normalized_pce"]), 8), p["source_row"]]
             for p in points],
        ])
    payload = json.dumps({"curves": curves, "representatives": representatives},
                         separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    page = PAGE.replace("__DATA__", payload)
    output = RESULTS / "anonymous_cluster_detail_viewer.html"
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output} ({len(curves)} eligible curves)")


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0–200 h DTW 匿名群：局部细节</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#20343b;background:#f6f8f8}
body{max-width:1320px;margin:24px auto;padding:0 16px}h1{font-size:26px;margin-bottom:6px}
p{line-height:1.5}.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;background:#fff;padding:16px;border:1px solid #d9e2e4;border-radius:8px}
label{display:grid;gap:4px;font-size:13px}input,select,button{font:inherit}input,select{padding:6px;border:1px solid #aebec3;border-radius:4px}
input[type=number]{width:88px}input[type=search]{width:110px}button{padding:6px 11px;background:#e8f1f3;border:1px solid #adc7cf;border-radius:4px;cursor:pointer}
button:hover{background:#d4e9ee}.presets{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:14px}.panel{background:white;border:1px solid #d9e2e4;border-radius:8px;padding:12px}
svg{display:block;width:100%;height:auto;min-height:440px;touch-action:none}#list{max-height:490px;overflow:auto}#list button{display:block;width:100%;text-align:left;margin:3px 0;background:white}
#list button.active{background:#d6e9ed;border-color:#4b8090}.small{font-size:13px;color:#4c6269}#status{min-height:22px}
@media(max-width:850px){.layout{grid-template-columns:1fr}#list{max-height:180px}}
</style>
<h1>DTW 匿名群：0–200 h 局部细节</h1>
<p>选择匿名群或输入曲线编号；调整小时与 PCE 范围以放大局部。圆点是实际原始观测，悬停可看小时、归一化 PCE 和源 CSV 行号。超过 40 h 的观测空隙不连线。</p>
<div class="controls">
<label>匿名群<select id="cluster"></select></label>
<label>曲线编号<input id="search" type="search" placeholder="如 F00001" list="curveIds"></label><datalist id="curveIds"></datalist>
<label>小时起点<input id="xmin" type="number" min="0" max="200" step="1" value="0"></label>
<label>小时终点<input id="xmax" type="number" min="0" max="200" step="1" value="200"></label>
<label>PCE 下限<input id="ymin" type="number" step="0.01" value="0"></label>
<label>PCE 上限<input id="ymax" type="number" step="0.01" value="1.07"></label>
<label><span>纵轴</span><select id="ymode"><option value="auto">按可见点自动缩放</option><option value="manual">使用上述范围</option></select></label>
<label><span>背景</span><select id="background"><option value="none">仅代表曲线</option><option value="all">该群所有曲线</option></select></label>
</div>
<div class="presets"><button data-range="0,200">全程</button><button data-range="0,50">0–50 h</button><button data-range="50,100">50–100 h</button><button data-range="100,150">100–150 h</button><button data-range="150,200">150–200 h</button><button id="reset">重置纵轴</button></div>
<div class="layout"><div class="panel"><svg id="chart" viewBox="0 0 1000 550" role="img" aria-label="原始观测点和匿名群代表曲线"></svg><p id="status" class="small"></p><p class="small">代表线由原始点连接；连线仅帮助阅读。聚类使用的 2 h 网格没有画成观测点。群号无形态语义。</p></div>
<div class="panel"><strong>该群曲线</strong><p id="count" class="small"></p><div id="list"></div><p id="selected" class="small"></p></div></div>
<script>
const DATA=__DATA__, curves=DATA.curves, reps=DATA.representatives;
const $=id=>document.getElementById(id), svgNS='http://www.w3.org/2000/svg';
const byId=new Map(curves.map(c=>[c[0],c]));
let selected=null;
function el(name,attrs={},parent=$('chart')){const node=document.createElementNS(svgNS,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));parent.append(node);return node}
function text(x,y,value,attrs={}){const node=el('text',{x,y,fill:'#50636a','font-size':13,...attrs});node.textContent=value;return node}
function range(){const x0=Number($('xmin').value),x1=Number($('xmax').value),y0=Number($('ymin').value),y1=Number($('ymax').value);
 if(!Number.isFinite(x0)||!Number.isFinite(x1)||x0<0||x1>200||x0>=x1)return null;
 if($('ymode').value==='manual'&&(!Number.isFinite(y0)||!Number.isFinite(y1)||y0>=y1))return null;
 return {x0,x1,y0,y1}}
function draw(){const r=range();if(!r){$('status').textContent='坐标范围无效：请检查起点、终点和上下限。';return}
 const cluster=Number($('cluster').value),group=curves.filter(c=>c[1]===cluster),repIds=reps[String(cluster)]||[];
 if(!selected||selected[1]!==cluster)selected=byId.get(repIds[0])||group[0];
 const focus=[...new Set([...repIds,selected[0]])].map(id=>byId.get(id)).filter(Boolean);
 if($('ymode').value==='auto'){
  const values=focus.flatMap(c=>c[2].filter(p=>p[0]>=r.x0&&p[0]<=r.x1).map(p=>p[1]));
  if(values.length){let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max(0.015,(hi-lo)*0.12);r.y0=Math.max(0,lo-pad);r.y1=hi+pad}
  else {r.y0=0;r.y1=1.07}
  $('ymin').value=r.y0.toFixed(3);$('ymax').value=r.y1.toFixed(3);
 }
 const svg=$('chart');svg.replaceChildren();const left=72,right=975,top=25,bottom=475;
 const X=x=>left+(x-r.x0)/(r.x1-r.x0)*(right-left),Y=y=>bottom-(y-r.y0)/(r.y1-r.y0)*(bottom-top);
 const defs=el('defs'),clip=el('clipPath',{id:'plotClip'},defs);el('rect',{x:left,y:top,width:right-left,height:bottom-top},clip);
 for(let i=0;i<=5;i++){
  const x=left+(right-left)*i/5,y=bottom-(bottom-top)*i/5;
  el('line',{x1:x,y1:top,x2:x,y2:bottom,stroke:'#e5ebed'});el('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#e5ebed'});
  text(x,bottom+23,(r.x0+(r.x1-r.x0)*i/5).toFixed(r.x1-r.x0<20?1:0),{'text-anchor':'middle'});
  text(left-10,y+4,(r.y0+(r.y1-r.y0)*i/5).toFixed(r.y1-r.y0<0.1?3:2),{'text-anchor':'end'});
 }
 el('rect',{x:left,y:top,width:right-left,height:bottom-top,fill:'none',stroke:'#70858c'});
 text((left+right)/2,535,'Hour (h)',{'text-anchor':'middle','font-size':16});text(20,250,'Normalized PCE',{transform:'rotate(-90 20 250)','text-anchor':'middle','font-size':16});
 const plot=el('g',{'clip-path':'url(#plotClip)'});
 function trace(c,color,width,opacity,dots){let paths=[],segment=[];for(const p of c[2]){
   if(segment.length&&p[0]-segment[segment.length-1][0]>40){paths.push(segment);segment=[]}segment.push(p)
  }if(segment.length)paths.push(segment);
  for(const s of paths){const d=s.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(2)+' '+Y(p[1]).toFixed(2)).join(' ');el('path',{d,fill:'none',stroke:color,'stroke-width':width,'stroke-opacity':opacity},plot)}
  if(dots)for(const p of c[2])if(p[0]>=r.x0&&p[0]<=r.x1&&p[1]>=r.y0&&p[1]<=r.y1){const dot=el('circle',{cx:X(p[0]),cy:Y(p[1]),r:c===selected?4:2.8,fill:color,opacity},plot);el('title',{},dot).textContent=`${c[0]} · ${p[0]} h · PCE ${p[1]} · 源 CSV 行 ${p[2]}`}
 }
 if($('background').value==='all')for(const c of group)trace(c,'#7893a0',0.75,0.13,false);
 const palette=['#ce6841','#3379aa','#6b9250','#9a69a6','#b18a2f'];
 focus.forEach((c,i)=>{if(c!==selected)trace(c,palette[repIds.indexOf(c[0])]||palette[i%5],2,0.8,true)});
 trace(selected,'#163b47',3.2,1,true);
 $('status').textContent=`匿名群 ${cluster}：${group.length} 条曲线；当前 ${selected[0]}，${selected[2].length} 个原始点。显示 ${r.x0}–${r.x1} h、PCE ${r.y0.toFixed(3)}–${r.y1.toFixed(3)}。`;
 $('selected').innerHTML=`当前：<strong>${selected[0]}</strong> · <a href="points/${selected[0]}.csv">查看原始点 CSV</a>`;
 document.querySelectorAll('#list button').forEach(b=>b.classList.toggle('active',b.dataset.id===selected[0]));
}
function fillList(){const cluster=Number($('cluster').value),group=curves.filter(c=>c[1]===cluster),query=$('search').value.trim().toUpperCase(),ids=reps[String(cluster)]||[];
 const shown=group.filter(c=>!query||c[0].includes(query));$('count').textContent=`${group.length} 条；匹配 ${shown.length} 条。前 5 条带 ★ 的是群内代表。`;
 const list=$('list');list.replaceChildren();for(const c of shown){const b=document.createElement('button');b.dataset.id=c[0];b.textContent=`${ids.includes(c[0])?'★ ':''}${c[0]} · ${c[2].length} 点`;b.onclick=()=>{selected=c;draw()};list.append(b)}draw()}
for(let i=0;i<8;i++){const option=document.createElement('option');option.value=i;option.textContent=`群 ${i}`;$('cluster').append(option)}
for(const c of curves){const option=document.createElement('option');option.value=c[0];$('curveIds').append(option)}
$('cluster').onchange=()=>{selected=null;fillList()};$('search').oninput=()=>{const exact=byId.get($('search').value.trim().toUpperCase());if(exact){$('cluster').value=exact[1];selected=exact}fillList()};
for(const id of ['xmin','xmax','ymin','ymax','ymode','background'])$(id).addEventListener('change',draw);
for(const button of document.querySelectorAll('[data-range]'))button.onclick=()=>{const [a,b]=button.dataset.range.split(',');$('xmin').value=a;$('xmax').value=b;draw()};
$('reset').onclick=()=>{$('ymode').value='auto';draw()};
fillList();
</script></html>'''


if __name__ == "__main__":
    main()
