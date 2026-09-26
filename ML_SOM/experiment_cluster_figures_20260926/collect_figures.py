"""Copy existing cluster-result figures into a traceable, read-only collection."""

from __future__ import annotations

import csv
import hashlib
import html
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
OUT = Path(__file__).resolve().parent
RECORDS: list[dict[str, str]] = []


def source_path(source: str | Path) -> Path:
    source = Path(source)
    if source.is_absolute():
        return source
    # The shared thesis and derivative projects have one maintained copy at
    # the repository root; the shape research remains under ML_SOM/.
    base = REPO_ROOT if source.parts[0] in {"thesis", "derivative_clustering"} else ROOT
    return base / source


def add(source: str | Path, group: str, method: str, dataset: str) -> None:
    src = source_path(source)
    if not src.is_file():
        raise FileNotFoundError(src)
    dest = OUT / "figures" / group / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    src_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == src_hash
    RECORDS.append(
        {
            "group": group,
            "method": method,
            "dataset": dataset,
            "figure": str(dest.relative_to(OUT)),
            "original_path": str(src.relative_to(REPO_ROOT)),
            "sha256": src_hash,
        }
    )


def add_glob(directory: str, pattern: str, group: str, method: str, dataset: str) -> None:
    for path in sorted(source_path(directory).glob(pattern)):
        if path.suffix.lower() in {".png", ".svg"}:
            add(path, group, method, dataset)


THESIS = "Hartono 2245 条 PCE 老化曲线"
PCE = "本项目 PCE 曲线"
SYNTH = "模拟或半合成曲线"

# Early PvkSOM output and a separate archived run: their images differ by hash.
for branch in ("result", "20230816_run_revision_excN2"):
    base = f"thesis/{branch}"
    group = f"01_thesis_som/{branch}"
    for name in (
        "som_clusters.png", "som_cluster_0.png", "som_cluster_1.png",
        "som_cluster_2.png", "som_cluster_3.png", "som_cluster_counts.png",
        "som_umatrix.png", "som_distance_map.png", "som_vs_kmeans_comparison.png",
    ):
        add(f"{base}/{name}", group, "SOM 2×2；附 KMeans 对照及 U-Matrix", THESIS)

# Controlled SOM preprocessing comparison. Literature reference image is excluded.
prep = "thesis/som_preprocessing_comparison"
for branch, method in (
    ("test1_raw", "发布数值直接输入 SOM；自然 K=4"),
    ("test2_resample_normalize", "重采样审计＋MaxAbs 归一化＋SOM；自然 K=5"),
    ("test3_resample_normalize_smooth", "MaxAbs＋Savitzky–Golay 平滑＋SOM；自然 K=5"),
):
    group = f"02_preprocessing/{branch}"
    for path in sorted((source_path(prep) / branch).glob("*.png")):
        if path.name == "initial_max_pce_distributions.png":
            continue
        add(path, group, method, THESIS)
add(
    f"{prep}/comparison/04_three_pipeline_comparison.png",
    "02_preprocessing/comparison", "三种输入固定 K=4 的 SOM 形状对照", THESIS,
)

# Target-PCE SOM variants. These two figures exist for every run.
variant_methods = {
    "som_baseline": "10×10 密度均衡 SOM 基线",
    "som_baseline_current_review": "同数据、当前审阅集的未加早期阶段均衡 SOM 基线",
    "som_coupled_sampling": "早期阶段均衡 SOM；Gumbel 抽样耦合诊断",
    "som_density0": "密度权重幂次 0 的 SOM 消融",
    "som_density1": "密度权重幂次 1 的 SOM 消融",
    "som_density3": "密度权重幂次 3 的 SOM 消融",
    "som_derivative1": "导数通道权重 1 的 SOM 消融",
    "som_drawup_weight1": "上升特征权重 1 的 SOM 消融",
    "som_medoid": "SOM 节点用 medoid 事后解释",
    "som_primary": "早期阶段均衡的主 SOM；10×10 节点",
    "som_side12": "12×12 网格 SOM 消融",
}
for branch, method in variant_methods.items():
    for name in ("four_som_motifs.png", "som_verified_representatives.png"):
        add(
            f"target_shape_classification/results/{branch}/{name}",
            f"03_target_som/{branch}", method, PCE,
        )

# Research SOM maps and anonymous prototype group analyses.
for path, group, method in (
    ("evaluation/som_map.png", "04_research_som/frozen_e2", "冻结 E2 的 10×10 SOM 节点图"),
    ("evaluation/full_detail_initial/som_map.png", "04_research_som/full_detail_initial", "早期细节输入 SOM 节点图"),
    ("evaluation/som_rule_alignment_20260925/node_rule_alignment.png", "04_research_som/rule_alignment", "SOM 节点与旧规则候选的事后对应"),
):
    add(f"target_shape_classification/research_early_200h/{path}", group, method, PCE)
add_glob(
    "target_shape_classification/research_early_200h/evaluation/unsupervised_groups_20260925",
    "*.png", "04_research_som/prototype_groups",
    "冻结 SOM 原型的平均连接层次聚类；k=2/4 对照", PCE,
)

# Later PCE shape-clustering experiments; clearly separated from SOM.
research = "target_shape_classification/research_early_200h"
add_glob(
    f"{research}/dtw_unsupervised_200h_20260925/results", "*.png",
    "05_pce_other_clustering/hour_aware_dtw", "0–200 h 幅值＋斜率双通道受限 DTW 图聚类", PCE,
)
add_glob(
    f"{research}/shape_only_unsupervised_20260925/results", "anonymous_k*.png",
    "05_pce_other_clustering/shape_dtw_k16_32_64", "相对观测序位的形状 DTW；16/32/64 匿名细簇", PCE,
)
for branch, method in (
    ("shape_only_four_group_exploration_20260925", "形状编码＋密度加权 KMeans，四簇"),
    ("shape_only_input_curation_20260925", "无标签质量筛选＋固定形状 KMeans，四簇"),
    ("supervised_input_frozen_unsupervised_20260926", "旧候选引导输入筛选＋冻结的无监督 KMeans，四簇"),
):
    add_glob(
        f"{research}/{branch}/results", "anonymous_four_groups*",
        f"05_pce_other_clustering/{branch}", method, PCE,
    )

# Derivative-clustering experiments elsewhere in this workspace.
for dataset in ("ECG5000", "Plane", "Trace"):
    add_glob(
        f"derivative_clustering/outputs/{dataset}", "*_dendrogram.png",
        f"06_derivative_comparisons/ucr_{dataset}",
        "欧氏／DTW／趋势特征／PFD-DTW 层次聚类树对照", f"UCR {dataset} 子集",
    )
add_glob(
    "derivative_clustering/eg_dtw_derivative_experiment/outputs", "*.png",
    "06_derivative_comparisons/eg_dtw_derivative",
    "幅值 DTW 与多阶导数 DTW 融合消融", SYNTH,
)
add_glob(
    "derivative_clustering/functional_validation_20260920/outputs", "comparison.png",
    "06_derivative_comparisons/functional_validation",
    "SOM、导数及 DTW 的固定功能对照", SYNTH,
)
for sub in ("existing_0", "independent_31001", "mixed_31001", "same_family_31001", "sparse_31001", "truncated_31001"):
    add_glob(
        f"derivative_clustering/functional_validation_20260920/outputs/{sub}",
        "curves_and_clusters.png", f"06_derivative_comparisons/functional_validation/{sub}",
        "SOM 与导数聚类功能验证；含采样压力情景", SYNTH,
    )
for sub in ("02_injection", "03_derivative_graph"):
    add_glob(
        f"derivative_clustering/accepted_pfd_experiment/outputs/{sub}", "*.png",
        f"06_derivative_comparisons/accepted_pfd/{sub}",
        "普通 DTW、PFD-DTW 与有序导数图谱聚类的恢复对照", "真实采样背景上的半合成曲线",
    )

# Final physical-hour gallery, plus archived solver-sensitive first pass.
hour = "derivative_clustering/hour_aware_analysis"
for branch in ("outputs", "initial_arpack_outputs"):
    for sub in ("existing_400", "independent_31001", "same_family_31001", "clock_replicas_42001", "clock_replicas_42002", "clock_replicas_42003"):
        folder = source_path(hour) / branch / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.png")):
            add(
                path, f"07_hour_aware/{branch}/{sub}",
                "相对进程形态分组＋原始小时响应时间子群", SYNTH,
            )
        for path in sorted(folder.glob("*.svg")):
            add(
                path, f"07_hour_aware/{branch}/{sub}",
                "相对进程形态分组＋原始小时响应时间子群", SYNTH,
            )

with (OUT / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(RECORDS[0]))
    writer.writeheader()
    writer.writerows(RECORDS)

parts = [
    "<!doctype html><html lang='zh'><meta charset='utf-8'>",
    "<meta name='viewport' content='width=device-width,initial-scale=1'>",
    "<title>ML_SOM 聚类实验图集</title>",
    "<style>body{font:15px system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#17212b;background:#f7f9fc}",
    "h1{margin-bottom:5px}h2{border-top:1px solid #ccd5df;padding-top:20px;margin-top:35px}",
    ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}",
    "figure{background:white;border:1px solid #dde4ec;border-radius:8px;margin:0;padding:10px}",
    "img{width:100%;height:210px;object-fit:contain;background:white}figcaption{overflow-wrap:anywhere;font-size:13px}",
    "a{color:#075aa6}</style><body>",
    f"<h1>ML_SOM 聚类实验图集</h1><p>{len(RECORDS)} 张原图副本。按文件夹区分 SOM 与其他聚类方法；方法见 <a href='README.md'>README</a>。</p>",
]
current = None
for row in RECORDS:
    if row["group"] != current:
        if current is not None:
            parts.append("</div>")
        current = row["group"]
        parts.append(f"<h2>{html.escape(current)}</h2><p>{html.escape(row['method'])}</p><div class='grid'>")
    name = html.escape(Path(row["figure"]).name)
    href = html.escape(row["figure"], quote=True)
    parts.append(f"<figure><a href='{href}'><img loading='lazy' src='{href}' alt='{name}'></a><figcaption><a href='{href}'>{name}</a></figcaption></figure>")
if current is not None:
    parts.append("</div>")
parts.append("</body></html>")
(OUT / "index.html").write_text("\n".join(parts), encoding="utf-8")
print(f"Copied {len(RECORDS)} figures into {OUT / 'figures'}")
