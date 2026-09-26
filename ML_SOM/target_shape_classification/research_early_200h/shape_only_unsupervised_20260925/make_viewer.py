"""Visualize frozen anonymous clusters; no clustering or label input here."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import HERE, OUT, POINTS_ROOT, observations, read_rows


def zoom_pages(rows: list[dict], raw: np.ndarray) -> None:
    for k in (16, 32, 64):
        rep_rows = read_rows(OUT / f"representatives_k{k}.csv")
        ids = {r["curve_id"]: i for i, r in enumerate(rows)}
        labels = np.array([int(r[f"anonymous_k{k}"]) for r in rows])
        x = np.linspace(0, 1, raw.shape[1])
        for page in range((k + 15) // 16):
            fig, axes = plt.subplots(4, 4, figsize=(17, 15), sharex=True)
            for ax, rep in zip(axes.flat, rep_rows[page * 16:(page + 1) * 16]):
                cluster = int(rep["cluster"])
                members = np.flatnonzero(labels == cluster)
                selected = [ids[cid] for cid in rep["representative_ids"].split(";")]
                selected_y = raw[selected]
                low, high = float(np.min(selected_y)), float(np.max(selected_y))
                pad = max(0.005, (high - low) * 0.08)
                for i in members:
                    ax.plot(x, raw[i], color="#547080", lw=0.6, alpha=0.12)
                for i in selected:
                    ax.plot(x, raw[i], lw=1.5, alpha=0.85, label=rows[i]["curve_id"])
                ax.set_ylim(max(0, low - pad), high + pad)
                ax.set_title(f"Anonymous {cluster} · n={len(members)} · y zoom", fontsize=10)
                ax.grid(alpha=0.15)
                ax.legend(fontsize=5.5, loc="lower left", ncol=2)
            for ax in axes[-1]: ax.set_xlabel("Relative observation rank")
            for ax in axes[:, 0]: ax.set_ylabel("Normalized PCE")
            fig.suptitle(f"Shape-only anonymous DTW · k={k} · locally zoomed y-axis · page {page+1}", fontsize=15)
            fig.tight_layout()
            fig.savefig(OUT / f"anonymous_k{k}_zoom_page{page+1}.png", dpi=170)
            plt.close(fig)


def main() -> None:
    rows = read_rows(OUT / "anonymous_assignments.csv")
    raw = np.load(OUT / "shape_inputs.npz")["raw_rank"]
    assert len(rows) == len(raw)
    zoom_pages(rows, raw)
    data = []
    for row in rows:
        values = observations(row)
        data.append([row["curve_id"], int(row["anonymous_k16"]), int(row["anonymous_k32"]),
                     int(row["anonymous_k64"]), row["evidence_tier"],
                     row["normalized_csv"], [round(float(y), 8) for y in values]])
    reps = {str(k): {r["cluster"]: r["representative_ids"].split(";")
                     for r in read_rows(OUT / f"representatives_k{k}.csv")}
            for k in (16, 32, 64)}
    payload = json.dumps({"curves": data, "representatives": reps},
                         separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    output = OUT / "anonymous_shape_detail_viewer.html"
    output.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {output} and locally zoomed pages for 16/32/64 clusters")


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>无监督形状 DTW：匿名细簇</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#20343b;background:#f6f8f8}
body{max-width:1320px;margin:24px auto;padding:0 16px}h1{font-size:26px;margin-bottom:6px}p{line-height:1.5}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;background:#fff;padding:16px;border:1px solid #d9e2e4;border-radius:8px}
label{display:grid;gap:4px;font-size:13px}input,select,button{font:inherit}input,select{padding:6px;border:1px solid #aebec3;border-radius:4px}
input[type=number]{width:90px}input[type=search]{width:110px}button{padding:6px 11px;background:#e8f1f3;border:1px solid #adc7cf;border-radius:4px;cursor:pointer}button:hover{background:#d4e9ee}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:14px;margin-top:12px}.panel{background:white;border:1px solid #d9e2e4;border-radius:8px;padding:12px}
svg{display:block;width:100%;height:auto;min-height:440px}#list{max-height:520px;overflow:auto}#list button{display:block;width:100%;text-align:left;margin:3px 0;background:white}#list button.active{background:#d6e9ed;border-color:#4b8090}
.small{font-size:13px;color:#4c6269}@media(max-width:850px){.layout{grid-template-columns:1fr}#list{max-height:190px}}
</style>
<h1>无监督形状 DTW：匿名细簇</h1>
<p>横轴是曲线内部的相对观测顺序，聚类不使用绝对小时。点击簇内曲线查看实测点；纵轴可自动放大到约 1% 的变化。背景线是同簇其他曲线；★ 是该簇的 5 条图内代表。群号没有四类语义。</p>
<div class="controls">
<label>细分程度<select id="k"><option>16</option><option>32</option><option selected>64</option></select></label>
<label>匿名簇<select id="cluster"></select></label>
<label>曲线编号<input id="search" type="search" placeholder="F00001" list="ids"></label><datalist id="ids"></datalist>
<label>纵轴下限<input id="ymin" type="number" step="0.001" value="0"></label><label>纵轴上限<input id="ymax" type="number" step="0.001" value="1.07"></label>
<label>纵轴模式<select id="ymode"><option value="auto">按当前曲线自动缩放</option><option value="manual">手动范围</option></select></label>
<label>背景曲线<select id="background"><option value="all">显示</option><option value="none">隐藏</option></select></label>
</div>
<div class="layout"><div class="panel"><svg id="chart" viewBox="0 0 1000 550" role="img" aria-label="匿名簇曲线，横轴为相对观测顺序"></svg><p id="status" class="small"></p><p class="small">同一小时的重复观测在模型中取中位数；其余原始点按顺序保留。64 点线性重采样仅供 DTW 计算，图中的圆点是聚合后的原始观测。<a id="rawlink">查看完整逐点 CSV</a></p></div>
<div class="panel"><strong>簇内曲线</strong><p id="count" class="small"></p><div id="list"></div></div></div>
<script>
const DATA=__DATA__, curves=DATA.curves,reps=DATA.representatives,byId=new Map(curves.map(c=>[c[0],c]));
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';let selected=null;
function el(name,attrs={},parent=$('chart')){const x=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,String(v));parent.append(x);return x}
function txt(x,y,s,attrs={}){const t=el('text',{x,y,fill:'#50636a','font-size':13,...attrs});t.textContent=s}
function key(){return 'anonymous_k'+$('k').value}function clusterOf(c){return c[Number($('k').value)===16?1:Number($('k').value)===32?2:3]}
function draw(){if(!selected)return;const y0in=Number($('ymin').value),y1in=Number($('ymax').value);let y0=y0in,y1=y1in;
 if($('ymode').value==='manual'&&(!Number.isFinite(y0)||!Number.isFinite(y1)||y0>=y1)){$('status').textContent='纵轴范围无效。';return}
 const vals=selected[6],cluster=Number($('cluster').value),group=curves.filter(c=>clusterOf(c)===cluster);
 if($('ymode').value==='auto'){const lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max(0.005,(hi-lo)*0.1);y0=Math.max(0,lo-pad);y1=hi+pad;$('ymin').value=y0.toFixed(4);$('ymax').value=y1.toFixed(4)}
 const svg=$('chart');svg.replaceChildren();const l=72,r=975,t=25,b=475,X=u=>l+u*(r-l),Y=y=>b-(y-y0)/(y1-y0)*(b-t);
 const defs=el('defs'),clip=el('clipPath',{id:'shapeClip'},defs);el('rect',{x:l,y:t,width:r-l,height:b-t},clip);
 for(let i=0;i<=5;i++){const x=X(i/5),y=b-(b-t)*i/5;el('line',{x1:x,y1:t,x2:x,y2:b,stroke:'#e4eaec'});el('line',{x1:l,y1:y,x2:r,y2:y,stroke:'#e4eaec'});txt(x,b+23,(i/5).toFixed(1),{'text-anchor':'middle'});txt(l-10,y+4,(y0+(y1-y0)*i/5).toFixed(y1-y0<0.1?3:2),{'text-anchor':'end'})}
 el('rect',{x:l,y:t,width:r-l,height:b-t,fill:'none',stroke:'#70858c'});txt((l+r)/2,535,'Relative observation rank',{'text-anchor':'middle','font-size':16});txt(20,250,'Normalized PCE',{transform:'rotate(-90 20 250)','text-anchor':'middle','font-size':16});
 const plot=el('g',{'clip-path':'url(#shapeClip)'});
 function trace(c,color,width,opacity,dots){const a=c[6],n=a.length;el('path',{d:a.map((y,i)=>(i?'L':'M')+X(i/(n-1)).toFixed(2)+' '+Y(y).toFixed(2)).join(' '),fill:'none',stroke:color,'stroke-width':width,'stroke-opacity':opacity},plot);
  if(dots)for(let i=0;i<n;i++){const dot=el('circle',{cx:X(i/(n-1)),cy:Y(a[i]),r:4,fill:color},plot);el('title',{},dot).textContent=`${c[0]} · 观测序号 ${i+1}/${n} · PCE ${a[i]}`}}
 if($('background').value==='all')for(const c of group)trace(c,'#648193',0.8,0.16,false);
 const repIds=reps[$('k').value][String(cluster)]||[],palette=['#cc6e42','#3379aa','#6b9250','#9a69a6','#b18a2f'];
 repIds.forEach((id,i)=>{const c=byId.get(id);if(c&&c!==selected)trace(c,palette[i],1.8,0.9,false)});
 trace(selected,'#173d48',3.2,1,true);$('rawlink').href='../../dtw_unsupervised_200h_20260925/results/'+selected[5];
 $('status').textContent=`${selected[0]} · k=${$('k').value} 匿名簇 ${cluster} · ${group.length} 条曲线 · ${vals.length} 个不同观测位置 · ${selected[4]} · PCE ${y0.toFixed(4)}–${y1.toFixed(4)}`;
 document.querySelectorAll('#list button').forEach(b=>b.classList.toggle('active',b.dataset.id===selected[0]));
}
function options(){const k=Number($('k').value),s=$('cluster');s.replaceChildren();for(let i=0;i<k;i++){const o=document.createElement('option');o.value=i;o.textContent=`簇 ${i}`;s.append(o)}}
function fillList(){const cluster=Number($('cluster').value),q=$('search').value.trim().toUpperCase(),group=curves.filter(c=>clusterOf(c)===cluster),shown=group.filter(c=>!q||c[0].includes(q));
 $('count').textContent=`该簇 ${group.length} 条；匹配 ${shown.length} 条。`;const list=$('list');list.replaceChildren();const repIds=reps[$('k').value][String(cluster)]||[];
 for(const c of shown){const b=document.createElement('button');b.dataset.id=c[0];b.textContent=`${repIds.includes(c[0])?'★ ':''}${c[0]} · ${c[6].length} 点${c[4]==='sparse_4_to_7'?' · 稀疏':''}`;b.onclick=()=>{selected=c;draw()};list.append(b)}
 if(group.length){selected=shown.find(c=>selected&&c[0]===selected[0])||byId.get(repIds[0])||group[0];draw()}}
for(const c of curves){const o=document.createElement('option');o.value=c[0];$('ids').append(o)}
$('k').onchange=()=>{const old=selected;options();if(old)$('cluster').value=clusterOf(old);fillList()};$('cluster').onchange=()=>{selected=null;fillList()};
$('search').oninput=()=>{const c=byId.get($('search').value.trim().toUpperCase());if(c){$('cluster').value=clusterOf(c);selected=c}fillList()};
for(const id of ['ymin','ymax','ymode','background'])$(id).addEventListener('change',draw);options();fillList();
</script></html>'''


if __name__ == "__main__":
    main()
