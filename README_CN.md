# ML degradation analysis

[English](README.md) · 中文

> **一个项目，两个仓库。** 本仓库是早期衰减曲线聚类研究的完整工作区。它的独立复现包——固定 1,842 条曲线的四簇结果、核验脚本和离线浏览页面——位于 [perovskite-early-degradation-clusters](https://github.com/ShunHao0626/perovskite-early-degradation-clusters)。复现主要结果请从该仓库开始；数据集构建、方法对照与溯源记录见本仓库。

对钙钛矿太阳能电池的 PCE–时间曲线进行整理、无监督聚类与形态分析。本仓库按**当前分析、方法复现、历史实验**组织；曲线原始数据、论文全文和逐曲线图片留在本地，不随 Git 发布。

## 从哪里开始

| 任务 | 入口 | 内容 |
|---|---|---|
| 当前 0–200 h 四形态分析 | [`ML_SOM/target_shape_classification/README.md`](ML_SOM/target_shape_classification/README.md) | 输入审计、SOM、形态判读、来源复核和测试 |
| 固定 1,842 条曲线的四簇复现 | [独立仓库 perovskite-early-degradation-clusters](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) | 固定输入、密度加权 KMeans、核验与完整结果；此处不再维护副本 |
| 2,246 条去重曲线的构建 | [`ML_SOM/canonical_2246_full_dataset_20260925/README.md`](ML_SOM/canonical_2246_full_dataset_20260925/README.md) | 构建、来源映射与完整性验证；生成的数据留在本地 |
| PFD–DTW 方法复现 | [`derivative_clustering/README.md`](derivative_clustering/README.md) | 独立的导数特征、DTW、层次聚类及单元测试 |
| Hartono 等人的 PvkSOM 复现 | [`thesis/README.md`](thesis/README.md) | 上游 PvkSOM 工作流的改编副本、运行脚本，以及在 Hartono 自有数据集上的三种预处理对比 |
| 其他导数 DTW 实验 | [`DTW/result_derivative_dtw/code/REPRODUCE.md`](DTW/result_derivative_dtw/code/REPRODUCE.md) | 早期独立实验和结果导出 |
| 历史 SOM、HDBSCAN 和参数探索 | [`by-products/README_CN.md`](by-products/README_CN.md) | 按研究问题索引既有实验，不参与当前入口 |

`environment.yml` 是早期 PvkSOM 实验环境。当前子项目优先使用各自的 `requirements.txt`、配置和 README。脚本通常从其子项目目录运行；需要先放入相应的本地数据，再按该目录说明执行。

当前 0–200 h 分析的模块依赖和数据流另见 [代码导航](ML_SOM/target_shape_classification/CODE_MAP_CN.md)。

## 项目中出现的曲线数量

各子项目文档中出现了多个曲线数量，它们对应不同的处理阶段，并非结果相互矛盾：

| 数量 | 含义 |
|---:|---|
| 2,250 | 从文献数字化得到的源文件（`ML_SOM/original_curves_2250/`） |
| 2,246 | 去除 2,250 个文件中 4 个重复别名后的规范曲线 |
| 2,216 | 整合交付中带四形态记录的规范曲线：2,205 条来自原自动规则，2 条为论文核轴后的自动规则，9 条仅来自原图人工判读 |
| 2,207 | 2,216 条中有数值数据的部分（2,205 + 2），用于严格的 0–200 h 数值分析 |
| 2,205 | 适用原数值阶段规则的曲线，用于稳健性审计 |
| 2,208 | 独立 0–200 h 重新分类（`classify_200h.py`）中，2,250 个源文件里获得四形态候选的文件数 |
| 2,152 | 纯形状 K-means 实验的输入池；较早一轮无标签质量门槛保留其中 1,897 条 |
| 1,842 | 四簇复现的固定输入，在借助暂定形状候选的情况下从 2,152 条中筛出 |
| 2,245 | `thesis/` 使用的 Hartono 等人自有高通量老化数据集，与文献曲线集无关 |

## 两个仓库的边界

本仓库保留上游曲线库构建、0–200 h SOM 与形态研究、导数/DTW 对照、PvkSOM 复现，以及可追溯的历史探索。固定 **1,842 条输入**上的四簇密度加权 KMeans 复现，以 [早期衰减四簇仓库](https://github.com/ShunHao0626/perovskite-early-degradation-clusters) 为唯一公开入口。它的输入筛选来自本仓库的早期探索，但两仓库承担的工作阶段不同；历史实验保留作方法溯源。

## 研究边界

项目主要检验 Bridge、Hill、Slope、Valley 四种目标曲线形态在无标签聚类中是否自然分开。历史 SOM、HDBSCAN 与 DTW 实验表明，真实文献曲线的簇更常按衰减幅度和速度分开。目标形态可作为逐曲线判读线索，但不能把簇号当作已验证的四类标签。固定 1,842 条输入的四簇实验使用过旧的候选筛选，因此属于探索性分析；其方法是密度加权 KMeans，不是 SOM。具体限制见独立仓库的 README。

## 数据与发布范围

Git 中仅保留代码、配置、方法文档、必要的汇总表与汇总图。以下内容保存在工作机或由使用者从原始来源获取：

- `ML_SOM/data_final/`、`ML_SOM/original_curves_2250/`、`ML_SOM/canonical_2246_full_dataset_20260925/{dataset,derived}/` 等原始或派生曲线；
- 固定 1,842 条输入及复现结果由独立仓库维护；本仓库不再跟踪该复现包；
- `by-products/` 下历史实验使用的曲线、矩阵、模型和论文全文；
- 本地文献库、原图、PDF、逐曲线图库及压缩包。

仓库中的索引、配置和结果可能包含 DOI、图号、曲线 ID 或原工作机路径，用于溯源，不代表相应原始文件已公开。完整复现需要按子项目 README 准备数据。

原 SI 中独有的文稿、图表和逐曲线绘图数值暂存于本地 `ML_SOM/si_1842_supplementary/`，不加入本 Git 仓库；其中的历史簇编号应与独立仓库的展示编号区分。

## 来源

PvkSOM 方法及参考数据：Hartono 等，*Nature Communications* 14, 4869 (2023)，[DOI: 10.1038/s41467-023-40585-3](https://doi.org/10.1038/s41467-023-40585-3)。`thesis/LICENSE` 为相应代码许可。文献数字化曲线应回查并引用各原始论文。
