"""Build a standalone HTML viewer for the frozen anonymous fit."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from run import CONFIG, HERE, OUT, original_observations, read_rows


def main() -> None:
    rows = read_rows(OUT / "anonymous_assignments.csv")
    source = {r["curve_id"]: r for r in read_rows((HERE / CONFIG["input_index"]).resolve())}
    with np.load(OUT / "model.npz") as arrays:
        rank_values = arrays["raw_rank"]
        assert arrays["curve_id"].tolist() == [r["curve_id"] for r in rows]
    curves = []
    for row, values in zip(rows, rank_values):
        measured = original_observations(source[row["curve_id"]])
        curves.append(dict(id=row["curve_id"], cluster=int(row["anonymous_cluster"]),
                           n=int(row["distinct_observations"]),
                           rough=float(row["total_variation_to_range"]),
                           rank=np.round(values, 6).tolist(),
                           measured=np.round(measured, 6).tolist()))
    payload = json.dumps(curves, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>四个匿名形状簇 · 逐曲线查看</title>
<style>
body{font:15px/1.55 system-ui,-apple-system,sans-serif;color:#173042;background:#f5f7f7;margin:0}
main{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 6px}
p{margin:6px 0 18px}.card{background:#fff;border:1px solid #dbe4e5;border-radius:12px;padding:16px;margin:14px 0}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}label{display:grid;gap:3px;font-weight:600}
select,input,button{font:inherit;padding:7px 9px;border:1px solid #b7c7c9;border-radius:6px;background:#fff;color:#173042}
input[type=number]{width:90px}.chart{width:100%;height:590px;display:block}.small{font-size:13px;color:#52676e}
code{background:#eef2f2;padding:1px 4px;border-radius:3px}a{color:#176c83}
</style></head><body><main>
<h1>四个匿名形状簇</h1>
<p>横轴为相对观测序位（0–1），绝对 hour 未参与聚类。灰线是该簇的全部曲线；蓝线与圆点分别是选中曲线的等间距插值和原始实测点。簇号不等于 Bridge/Hill/Slope/Valley 标签。</p>
<div class="card controls">
<label>匿名簇<select id="cluster"></select></label>
<label>曲线 ID<input id="curve" list="ids" placeholder="例如 F00001"><datalist id="ids"></datalist></label>
<label>纵轴<select id="scale"><option value="cluster">簇内自动放大</option><option value="selected">选中曲线放大</option><option value="full">统一 0–1</option><option value="manual">手动范围</option></select></label>
<label>最小值<input id="ymin" type="number" step="0.001" value="0.9"></label>
<label>最大值<input id="ymax" type="number" step="0.001" value="1.01"></label>
</div>
<div class="card"><canvas class="chart" id="chart"></canvas><div id="info" class="small"></div></div>
<p class="small">原始点以归一化 PCE 显示；浅灰线可能在纵轴范围外被裁切。上一轮质量筛选暂缓 255 条，本轮有监督输入整理另暂缓 55 条；<a href="../full_input_decisions.csv">完整清单</a>保存原因，<a href="anonymous_assignments.csv">CSV</a>保存本轮匿名归属。小幅约 1% 的起伏可用“选中曲线放大”检查；它是否超出测量误差，需要原图或重复观测核查。</p>
</main><script>
const data=__PAYLOAD__;
const byId=new Map(data.map(d=>[d.id,d]));
const cluster=document.getElementById('cluster'), curve=document.getElementById('curve'),scale=document.getElementById('scale');
const ymin=document.getElementById('ymin'),ymax=document.getElementById('ymax'),chart=document.getElementById('chart');
const counts=[0,0,0,0];data.forEach(d=>counts[d.cluster]++);
for(let i=0;i<4;i++){let o=document.createElement('option');o.value=i;o.textContent=`簇 ${i} · ${counts[i]} 条`;cluster.append(o)}
document.getElementById('ids').innerHTML=data.map(d=>`<option value="${d.id}">`).join('');
function bounds(list,selected){let mode=scale.value;
 if(mode==='full')return [0,1.04];
 if(mode==='manual'){let a=Number(ymin.value),b=Number(ymax.value);return b>a?[a,b]:[0,1.04]}
 let arr=(mode==='selected'&&selected?selected.measured:list.flatMap(d=>d.rank)).filter(Number.isFinite).sort((a,b)=>a-b);
 let lo,hi;if(mode==='selected'&&selected){lo=arr[0];hi=arr[arr.length-1]}
 else{lo=arr[Math.floor(.03*(arr.length-1))];hi=arr[Math.floor(.97*(arr.length-1))]}
 let pad=Math.max(.003,(hi-lo)*.08);return [lo-pad,hi+pad]
}
function draw(){let chosen=byId.get(curve.value.trim().toUpperCase());if(chosen&&String(chosen.cluster)!==cluster.value)cluster.value=chosen.cluster;
 let list=data.filter(d=>String(d.cluster)===cluster.value),b=bounds(list,chosen);
 let rect=chart.getBoundingClientRect(),dpi=window.devicePixelRatio||1;chart.width=Math.round(rect.width*dpi);chart.height=Math.round(rect.height*dpi);
 let ctx=chart.getContext('2d');ctx.setTransform(dpi,0,0,dpi,0,0);let W=rect.width,H=rect.height,L=76,R=18,T=18,B=55;
 let px=x=>L+x*(W-L-R),py=y=>T+(b[1]-y)/(b[1]-b[0])*(H-T-B);
 ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.font='12px system-ui';ctx.lineWidth=1;
 for(let k=0;k<=5;k++){let v=b[0]+(b[1]-b[0])*k/5,y=py(v);ctx.strokeStyle='#e6ecec';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();ctx.fillStyle='#52676e';ctx.fillText(v.toFixed(3),9,y+4)}
 for(let k=0;k<=5;k++){let x=px(k/5);ctx.strokeStyle='#e6ecec';ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,H-B);ctx.stroke();ctx.fillStyle='#52676e';ctx.fillText((k/5).toFixed(1),x-9,H-B+20)}
 ctx.save();ctx.beginPath();ctx.rect(L,T,W-L-R,H-T-B);ctx.clip();
 ctx.strokeStyle='rgba(86,114,126,.09)';ctx.lineWidth=.7;
 for(let d of list){ctx.beginPath();d.rank.forEach((v,i)=>{let x=px(i/63),y=py(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
 if(chosen){ctx.strokeStyle='#007da5';ctx.lineWidth=2.5;ctx.beginPath();chosen.rank.forEach((v,i)=>{let x=px(i/63),y=py(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
 ctx.fillStyle='#d45c35';chosen.measured.forEach((v,i)=>{ctx.beginPath();ctx.arc(px(i/Math.max(1,chosen.measured.length-1)),py(v),3.5,0,2*Math.PI);ctx.fill()})}
 ctx.restore();ctx.fillStyle='#173042';ctx.fillText('相对观测序位',W/2-38,H-10);
 document.getElementById('info').textContent=`簇 ${cluster.value}：${list.length} 条，${list.filter(d=>d.n<8).length} 条只有 4–7 个原始时刻。`+(chosen?` 选中 ${chosen.id}：${chosen.n} 个原始时刻；原始 PCE 范围 ${Math.min(...chosen.measured).toFixed(4)}–${Math.max(...chosen.measured).toFixed(4)}。`:' 输入曲线 ID 可查看真实测点。');
}
cluster.onchange=()=>{curve.value='';draw()};curve.oninput=draw;scale.onchange=draw;ymin.oninput=draw;ymax.oninput=draw;window.onresize=draw;
const start=new URLSearchParams(location.search).get('id');if(start&&byId.has(start.toUpperCase())){curve.value=start.toUpperCase();scale.value='selected'}
draw();
</script></body></html>'''
    (OUT / "anonymous_cluster_detail_viewer.html").write_text(
        template.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"viewer curves={len(curves)}, cluster sizes={dict(Counter(d['cluster'] for d in curves))}")


if __name__ == "__main__":
    main()
