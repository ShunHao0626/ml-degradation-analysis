#!/usr/bin/env python3
"""Run the frozen original-CSV early-window unsupervised baseline and delivery."""
from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from early_data import HERE, CLASSES, digest, features, freeze, prepare, write_rows
from interpret import rule_suggestion
from masked_som import MaskedSOM


def csv_points(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hour", "normalized_pce", "point_source", "source_row", "region"])
        for hour, value, row in points:
            if 0 <= hour <= 250:
                writer.writerow([f"{hour:.12g}", f"{value:.12g}", "original_source_csv", row,
                                 "strict_200h" if hour <= 200 else "boundary_unassessed"])


def plot_curve(path, item, title, status, boundary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.7), dpi=110)
    t, y = item["t"], item["y"]
    ax.plot(t, y, color="#236381", lw=1, alpha=.7)
    raw = [(h, v) for h, v, _ in item["points"] if 0 <= h <= 200]
    ax.scatter([p[0] for p in raw], [p[1] for p in raw],
               s=9 if len(raw) < 300 else 3, color="#236381", label="Source CSV points")
    if boundary and item["boundary"]:
        ax.axvspan(200, 250, color="#ecb569", alpha=.2, label="Boundary review")
        bt = [p[0] for p in item["boundary"]]
        by = [p[1] for p in item["boundary"]]
        ax.plot([t[-1]] + bt, [y[-1]] + by, color="#b87830", ls="--", lw=1)
        ax.scatter(bt, by, color="#b87830", s=25)
    ax.axvline(200, color="#9c3d44", ls="--", lw=1, label="200 h")
    ax.set_xlim(0, 250 if boundary else 205)
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Normalized PCE")
    ax.set_title(title, fontsize=10)
    ax.text(.02, .03, status.replace("_", " "), transform=ax.transAxes,
            fontsize=8, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gallery(path, records):
    payload = []
    for r in records:
        payload.append({k: r.get(k, "") for k in ("curve_id", "candidate_class", "evidence_status",
                         "source_group", "source_csv", "source_image", "curve_csv", "curve_png", "boundary_png",
                         "som_bmu", "model_score", "rule_candidate_class", "review_reason")})
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    body = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Early PCE candidates</title>
<style>body{font:14px system-ui;margin:24px;color:#24333c}input,select{padding:7px;margin:4px}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}tr:hover{background:#f2f6f7}a{color:#146b88}#chart{max-width:850px;width:100%;border:1px solid #ddd}small{color:#596a73}</style>
<h1>0–200 h PCE 四类候选图库</h1><p>全部曲线为模型辅助候选，历史原图复核仍需按本次 CSV 与归一化版本重新确认。橙色边界点不参与严格模型。</p>
<select id="cls"><option value="">全部类别</option><option>bridge</option><option>hill</option><option>slope</option><option>valley</option></select>
<select id="status"><option value="">全部状态</option></select><input id="search" placeholder="搜索 ID、DOI、来源路径"><span id="count"></span>
<div><img id="chart"><div id="links"></div></div><table><thead><tr><th>ID</th><th>候选</th><th>状态</th><th>DOI</th><th>SOM 节点</th><th>核图原因</th></tr></thead><tbody id="body"></tbody></table>
<script>const rows=__DATA__;const status=document.querySelector('#status'),cls=document.querySelector('#cls'),search=document.querySelector('#search'),body=document.querySelector('#body');
for(const x of [...new Set(rows.map(r=>r.evidence_status))].sort()){let o=document.createElement('option');o.value=x;o.textContent=x;status.append(o)}
function select(r){document.querySelector('#chart').src=r.curve_png||'';let links=document.querySelector('#links');links.textContent='';for(const [label,url] of [['CSV',r.curve_csv],['PNG',r.curve_png],['边界图',r.boundary_png],['原始CSV',r.source_csv?'../../../original_curves_2250/'+r.source_csv:''],['论文原图',r.source_image?'../../../data_final/'+r.source_image:'']])if(url){let a=document.createElement('a');a.href=url;a.textContent=label+' ';links.append(a)}}
function render(){body.textContent='';let term=search.value.toLowerCase();let shown=rows.filter(r=>(!cls.value||r.candidate_class===cls.value)&&(!status.value||r.evidence_status===status.value)&&(!term||(r.curve_id+r.source_group+r.source_csv).toLowerCase().includes(term)));document.querySelector('#count').textContent=` ${shown.length} / ${rows.length}`;for(const r of shown){let tr=document.createElement('tr');for(const k of ['curve_id','candidate_class','evidence_status','source_group','som_bmu','review_reason']){let td=document.createElement('td');td.textContent=r[k];tr.append(td)}tr.onclick=()=>select(r);body.append(tr)}}
for(const x of [cls,status,search])x.addEventListener('input',render);render();</script>'''.replace("__DATA__", data)
    path.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--skip-plots", action="store_true", help="Only for development; delivery is incomplete")
    parser.add_argument("--reuse-plots", action="store_true", help="Reuse existing charts after input hash audit")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["config_sha256"] = digest(args.config)
    prior_summary_path = HERE / "runs/summary.json"
    if not prior_summary_path.exists():
        prior_summary_path = HERE / "runs/initial_8x8/summary.json"
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8")) if prior_summary_path.exists() else None
    frozen, canonical, split, snapshot = freeze(config, HERE)
    if args.reuse_plots:
        keys = ("input_manifest_sha256", "canonical_manifest_sha256", "axis_overrides_sha256")
        plot_keys = ("source_root", "image_root", "window_h", "boundary_limit_h", "boundary_max_points")
        if (not prior_summary or
                any(prior_summary["snapshot"][key] != snapshot[key] for key in keys) or
                any(prior_summary["config"][key] != config[key] for key in plot_keys)):
            raise ValueError("plot inputs changed; rerun without --reuse-plots")
    source_root = Path(config["source_root"])
    (HERE / "runs").mkdir(exist_ok=True)
    all_ids = sorted(canonical)
    observations, errors, matrices = {}, {}, []
    feature_blocks = []
    for fid in all_ids:
        item, reason = prepare(canonical[fid], source_root, config)
        if reason:
            errors[fid] = reason
            continue
        observations[fid] = item
        feature_blocks.append(features(item, config))
        matrices.append(fid)
    sizes = [len(block) for block in feature_blocks[0]]
    raw = np.asarray([np.concatenate(block) for block in feature_blocks])
    mask = np.isfinite(raw)
    if config.get("detail_mode") == "extrema_only":
        start = sizes[0] + sizes[1]
        for k in range(20):
            mask[:, start + 4 * k + 2:start + 4 * k + 4] = False
    elif config.get("detail_mode", "full") != "full":
        raise ValueError("unknown detail mode")
    data = np.where(mask, raw, 0.)
    np.savez_compressed(HERE / "manifests/strict_features.npz", data=data, mask=mask,
                        audit_raw=raw, curve_ids=np.array(matrices), group_sizes=np.array(sizes))
    positions = {fid: i for i, fid in enumerate(matrices)}
    train = np.array([i for i, fid in enumerate(matrices) if split[canonical[fid]["source_group"]] == "train"])
    validation = np.array([i for i, fid in enumerate(matrices) if split[canonical[fid]["source_group"]] == "validation"])
    test = np.array([i for i, fid in enumerate(matrices) if split[canonical[fid]["source_group"]] == "test"])
    weights = [config["group_weights"][key] for key in ("trajectory", "velocity", "detail")]
    results = []
    for variant in ("E1", "E2"):
        active_sizes = sizes[:1] if variant == "E1" else sizes
        width = sum(active_sizes)
        active_weights = [1.] if variant == "E1" else weights
        active = mask[:, :width].any(axis=1)
        fit_indices = train[active[train]]
        validation_indices = validation[active[validation]]
        for seed in config["seeds"]:
            model = MaskedSOM(config["map_rows"], config["map_columns"], active_sizes,
                              active_weights, seed)
            trace = model.fit(data[fit_indices, :width], mask[fit_indices, :width], config["epochs"])
            metrics = {name: model.metrics(data[idx, :width], mask[idx, :width])
                       for name, idx in (("train", fit_indices), ("validation", validation_indices))}
            run_id = f"{variant}_seed{seed}"
            np.savez_compressed(HERE / "runs" / f"{run_id}.npz", prototypes=model.prototypes,
                                supported=model.supported, coordinates=model.coordinates,
                                feature_sizes=np.array(active_sizes), weights=np.array(active_weights),
                                seed=seed, train_ids=np.array(matrices)[train])
            record = dict(run_id=run_id, variant=variant, seed=seed, metrics=metrics, trace=trace,
                          model_sha256=digest(HERE / "runs" / f"{run_id}.npz"))
            results.append(record)
            print(run_id, json.dumps(metrics["validation"]), flush=True)
    # An entirely unlabeled backbone choice: validation QE with topology as a tie-breaker.
    e2 = [r for r in results if r["variant"] == "E2"]
    best_q = min(r["metrics"]["validation"]["quantization_error"] for r in e2)
    preferred = min((r for r in e2 if r["metrics"]["validation"]["quantization_error"] <= 1.05 * best_q),
                    key=lambda r: (r["metrics"]["validation"]["topology_error"],
                                   r["metrics"]["validation"]["quantization_error"]))
    run_id = preferred["run_id"]
    saved = np.load(HERE / "runs" / f"{run_id}.npz")
    model = MaskedSOM(config["map_rows"], config["map_columns"], sizes, weights, preferred["seed"])
    model.prototypes = saved["prototypes"]
    model.supported = saved["supported"]
    bmu, q, second = model.assign(data, mask)
    selected_test_metrics = model.metrics(data[test], mask[test])
    train_q = q[train]
    # Post-hoc names use only nearest REAL train members; no semantic input reached fit().
    node_rows = []
    node_names = {}
    for node in range(len(model.prototypes)):
        candidates = [i for i in train if bmu[i] == node]
        if not candidates:
            candidates = list(train)
        representative = min(candidates, key=lambda i: model.distance_one(data[i], mask[i])[node])
        fid = matrices[representative]
        item = observations[fid]
        name, basis = rule_suggestion(item["t"], item["y"])
        node_names[node] = name
        node_rows.append(dict(node=node, representative_curve_id=fid, source_group=canonical[fid]["source_group"],
                              train_occupancy=int(np.sum(bmu[train] == node)), posthoc_class=name,
                              interpretation_basis=basis, human_confirmed=0))
    write_rows(HERE / "runs" / f"{run_id}_node_interpretation.csv", node_rows, list(node_rows[0]))
    alias_map = defaultdict(list)
    review_status_by_canon = defaultdict(set)
    revised_by_canon = defaultdict(set)
    for row in frozen:
        alias_map[row["canonical_id"]].append(row["file_id"])
        if row["historical_review_status"]:
            review_status_by_canon[row["canonical_id"]].add(row["historical_review_status"])
        if row["historical_analysis_version"] != row["source_csv"]:
            revised_by_canon[row["canonical_id"]].add(row["historical_analysis_version"])
    records, unresolved, queue = [], [], []
    for fid in all_ids:
        row = canonical[fid]
        item = observations.get(fid)
        aliases = ";".join(alias_map[fid])
        base = dict(curve_id=fid, file_ids=aliases, source_csv=row["source_csv"],
                    source_sha256=row["source_sha256"], source_image=row["source_image"],
                    image_sha256=row["image_sha256"], source_group=row["source_group"],
                    split=split[row["source_group"]], candidate_class="", final_class="",
                    evidence_status="data_issue", decision_basis="", observation_end_used_h="",
                    strict_200h_class="", boundary_confirmed_class="", som_bmu="",
                    som_posthoc_class="", semantic_predicted_class="", rule_candidate_class="",
                    model_score="", window_points="", unique_window_times="", max_gap_h="",
                    normalization_divisor="", boundary_points="", review_reason="",
                    curve_csv="", curve_png="", boundary_png="")
        if item is None:
            base["review_reason"] = errors[fid]
            unresolved.append(base)
            records.append(base)
            continue
        i = positions[fid]
        som_name = node_names[int(bmu[i])]
        rule_name, rule_basis = rule_suggestion(item["t"], item["y"])
        # Rule-assisted delivery is explicitly distinct from the SOM-only A baseline.
        # This keeps rare candidate shapes visible for human review without calling
        # a rule-generated label an unsupervised discovery or a reference truth.
        name = rule_name
        low_support = item["unique_window_times"] < 5 or item["window_end_h"] < 100 or item["max_gap_h"] > 40
        rare_score = q[i] > np.quantile(train_q, .9)
        disagreements = som_name != rule_name
        reasons = []
        if low_support: reasons.append("limited_observation_support")
        if rare_score: reasons.append("high_prototype_distance")
        if disagreements: reasons.append("som_rule_disagreement")
        if rule_basis == "weak_four_way_fallback": reasons.append("weak_four_way_fallback")
        if item["boundary"]: reasons.append("boundary_points_available_unassessed")
        if revised_by_canon[fid]: reasons.append("accepted_redigitization_exists_original_primary")
        if "ambiguous" in review_status_by_canon[fid]:
            reasons.append("historical_ambiguous_review")
        status = "insufficient_evidence" if low_support else "candidate_review"
        if "verified" in review_status_by_canon[fid]:
            status = "legacy_image_review_needs_revalidation"
            reasons.append("historical_image_review_not_new_reference")
        relative = f"{name}/{fid}"
        base.update(candidate_class=name, strict_200h_class=name, som_posthoc_class=som_name,
                    evidence_status=status, decision_basis="rule_assisted_candidate_after_frozen_anonymous_som",
                    observation_end_used_h=item["window_end_h"], som_bmu=int(bmu[i]),
                    rule_candidate_class=rule_name, model_score=round(float(q[i]), 8),
                    window_points=item["window_points"], unique_window_times=item["unique_window_times"],
                    max_gap_h=item["max_gap_h"], normalization_divisor=item["divisor"],
                    boundary_points=len(item["boundary"]), review_reason=";".join(reasons),
                    curve_csv=relative + ".csv", curve_png=relative + ".png",
                    boundary_png=relative + "_boundary.png" if item["boundary"] else "")
        delivery = HERE / "delivery"
        csv_points(delivery / base["curve_csv"], item["points"])
        if not args.skip_plots:
            if not args.reuse_plots or not (delivery / base["curve_png"]).is_file():
                plot_curve(delivery / base["curve_png"], item, f"{fid} | {name.title()} candidate", status)
            if item["boundary"]:
                if not args.reuse_plots or not (delivery / base["boundary_png"]).is_file():
                    plot_curve(delivery / base["boundary_png"], item,
                               f"{fid} | boundary evidence (not classified)", status, boundary=True)
        if reasons or status != "candidate_review":
            priority = 3 * int(disagreements) + 2 * int(low_support) + 2 * int(rare_score) + len(reasons)
            queue.append(dict(curve_id=fid, source_group=row["source_group"], split=base["split"],
                              candidate_class=name, evidence_status=status, priority=priority,
                              review_reason=base["review_reason"], source_image=row["source_image"],
                              source_csv=row["source_csv"], curve_png=base["curve_png"]))
        records.append(base)
    delivery = HERE / "delivery"
    write_rows(delivery / "class_index.csv", [r for r in records if r["candidate_class"]], list(records[0]))
    write_rows(delivery / "unresolved.csv", unresolved, list(records[0]))
    file_rows = []
    by_id = {r["curve_id"]: r for r in records}
    for r in frozen:
        combined = dict(by_id[r["canonical_id"]])
        combined["canonical_source_csv"] = combined["source_csv"]
        combined["canonical_source_sha256"] = combined["source_sha256"]
        combined.update(r)  # retain EACH alias's own source path and hash
        file_rows.append(combined)
    write_rows(delivery / "all_files.csv", file_rows, list(file_rows[0]))
    queue.sort(key=lambda r: (-r["priority"], r["curve_id"]))
    write_rows(HERE / "reviews/review_queue.csv", queue,
               list(queue[0]) if queue else ["curve_id", "review_reason"])
    gallery(delivery / "all_curves_index.html", records)
    counts = Counter(r["candidate_class"] for r in records if r["candidate_class"])
    summary = dict(snapshot=snapshot, config=config, environment=dict(python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in
                              ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")}),
                   eligible=len(matrices), unresolved=len(unresolved), classes=counts,
                   split_curves={s: sum(split[canonical[f]["source_group"]] == s for f in all_ids)
                                 for s in ("train", "validation", "test")},
                   runs=results, selected_run=run_id, selected_model_sha256=preferred["model_sha256"],
                   selected_test_unlabeled_metrics=selected_test_metrics,
                   label_status="No independent blinded labels; no four-class accuracy claimed.")
    (HERE / "runs/summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# 初期 PCE 匿名 SOM 基线实际运行报告

协议：`{config['protocol_version']}`；原始 CSV 根目录：`{source_root}`。模型文件：`runs/{run_id}.npz`，SHA-256 `{preferred['model_sha256']}`。

源文件 {len(frozen)}；规范曲线 {len(all_ids)}；0–200 h 有效建模 {len(matrices)}；未解析 {len(unresolved)}。四类模型辅助候选：Bridge {counts['bridge']}、Hill {counts['hill']}、Slope {counts['slope']}、Valley {counts['valley']}。来源分组及哈希见 `manifests/`。

E1、E2 均为单张匿名掩码 SOM；使用 {config['seeds']} 共 {len(config['seeds'])} 个预设种子、{config['epochs']} 轮。E1 对共同网格完全无支持的曲线只保留在总清单，不纳入其训练和量化误差。修订 E2 保留每 10 h 原始极值，累计正负变化保存在审计特征但因明显采点密度混杂而不送入主地图；完整细节版另存消融。模型选择仅依据验证来源的无标签指标：在量化误差距最优不超过 5% 的种子中取拓扑误差最低，选中 `{run_id}`。验证量化误差 {preferred['metrics']['validation']['quantization_error']:.5f}；拓扑误差 {preferred['metrics']['validation']['topology_error']:.5f}。地图后验命名来自训练来源真实代表曲线的启发式建议，尚无独立专家确认。

四类实体目录按**逐曲线真实点阶段规则建议**组织，是规则辅助候选，保留 `som_posthoc_class` 以显示 SOM 单独的后验解释。采用规则辅助目录的原因是 SOM 节点解释主要落在 Slope，尚不能可靠覆盖稀有形态。`rule_candidate_class` 与 `som_posthoc_class` 分列；两者不一致者进入核图队列。这些规则候选不得解释为无监督 SOM 自动发现的四类，亦不得计入人工确认。

`delivery/` 中每个有效规范曲线都有真实源点 CSV 和初期图；边界图仅用于复核。`class_index.csv` 记录候选及证据状态，`all_files.csv` 保留全部 2250 个源文件别名。`reviews/review_queue.csv` 是具体核图队列。

实体核验见 `verification.json`；多种子与历史差异见 `../evaluation/structure_report.md`，地图尺寸敏感性见 `../evaluation/map_size_sweep.json`，原始/已接受补采版本对照见 `../manifests/accepted_redigitization_comparison.csv`。中性双阶段盲判表在 `../reviews/`。

地图尺寸和训练轮数的无标签对照、完整细节版的采样混杂消融、DOI 来源重采样以及 U-matrix 和真实训练代表曲线，分别记录于 `../evaluation/map_size_sweep.json`、`../evaluation/epoch_sweep.json`、`../evaluation/detail_ablation.csv`、`../evaluation/source_resampling.json`、`../evaluation/som_map.png` 和 `../evaluation/node_atlas.html`。这些都是结构诊断，不是四类正确率。

## 解释限制和剩余工作

历史 36 条确认仅核验过原图哈希，缺少本协议的源 CSV、小时转换和归一化锁定；标为 `legacy_image_review_needs_revalidation`。尚未完成两位盲判、路线 B 的 E3 分类器、E4 原始特征对照、E5 人工边界确认和独立四类精度评价。所有四类名称均为候选，不能当作已验证类别或论文成绩。全库旧探索使本次锁定测试仅具内部前瞻含义；强外部结论需要新 DOI。

归一化分母是指定原始 CSV 在 0–200 h 的观测最大值，已保存于索引；个别孤立峰的原图真实性尚待人工审查，故这也是需复核的分析值。约 1% 波动被保留为原始点和细节审计，不被当作显著性或自动确认依据。
"""
    (delivery / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "unresolved": len(unresolved), "selected": run_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
