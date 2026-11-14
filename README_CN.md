# 钙钛矿太阳能电池早期衰减曲线：四类形状复现

[English](README.md) · [详细方法](docs/WORKFLOW_CN.md)

> **同一项目的一部分。** 本仓库是早期衰减曲线聚类研究的独立复现包。完整的研究工作区——数据集构建（2,250 个源文件 → 2,246 条规范曲线）、更广泛的 0–200 h SOM／形态分析、PFD–DTW 导数聚类，以及 Hartono PvkSOM 复现——位于 [ml-degradation-analysis](https://github.com/ShunHao0626/ml-degradation-analysis)。本仓库使用的固定 1,842 条输入是该规范曲线集的子集。

本仓库整理了论文中提取的 **1842 条钙钛矿太阳能电池稳定性曲线**，探索 **0–200 小时**内的早期变化。现有结果分为四个探索性簇，展示时分别命名为 **Slope（Cluster 1）、Bridge（Cluster 2）、Valley（Cluster 3）、Hill（Cluster 4）**。

四簇规模极不均衡：Slope 包含 **1842 条中的 1710 条（93%）**，Bridge、Valley、Hill 三簇合计只有 132 条（44／21／67）。

## 核心结果：四簇曲线局部放大图

<p align="center"><a href="results/figures/Supplementary_Fig_S1_zoom.png"><img src="results/figures/Supplementary_Fig_S1_zoom.png" alt="四簇曲线及其中位趋势的局部放大图" width="440"></a></p>

浅色线为单条曲线，黑线为仅供展示的平滑中位趋势。各面板的纵轴分别放大；横轴为**相对观测序位**，不是实际经过的小时数。可[打开矢量图](results/figures/Supplementary_Fig_S1_zoom.svg)，或[浏览单条曲线](results/dashboard.html)。

> **方法说明：**每条曲线使用预先在相对序位轴上生成的 64 点形状（`data/shape_inputs.npz`），先平移使起点为 0，再除以自身幅度（最大值减最小值，下限 0.01），并做高斯平滑（σ = 1.5 个点）；随后在 −1 到 1 的序位轴上拟合 5 次切比雪夫多项式，去掉常数项后的 5 个系数（第 1–5 阶）即形状特征。特征经 `RobustScaler`（第 10–90 百分位范围）缩放后，用 K-means 聚为四簇；样本权重正比于到第 31 个近邻的距离（限制在 0.2–10），因此特征空间中较稀疏区域的曲线权重更高。**代码并未实现 SOM（自组织映射）**。拟合过程不使用类别名称，但输入曲线此前曾借助暂定形状候选筛选，特征与模型方案的选择也受目标形状影响，因此四簇不能视作经独立验证的真实类别。

## 目录

| 目录 | 内容 |
| --- | --- |
| [`code/`](code/) | 聚类、核验、绘图、浏览页面生成脚本，以及参数和依赖 |
| [`data/`](data/) | 固定输入索引、0–200 小时观测点、64 点形状输入、原始 CSV，以及从原论文裁出的来源图 |
| [`results/`](results/) | 四簇结果、图片、按簇复制的来源文件及离线[浏览页面](results/dashboard.html) |
| [`docs/`](docs/) | 详细流程与方法边界 |

## 一步步复现

使用 **Python 3.9**。在仓库根目录执行以下命令（macOS/Linux）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python code/run.py
.venv/bin/python code/verify.py --check-results
```

运行会写入 `results/` 并替换其中生成的簇文件夹。预期四簇数量为 **1710／44／21／67**。直接打开 `results/dashboard.html` 可浏览全部曲线和来源图片。参数见 [`code/config.json`](code/config.json)，完整分析步骤见 [`docs/WORKFLOW_CN.md`](docs/WORKFLOW_CN.md)。

**复现范围：**本仓库从已提取、已固定的 1842 条曲线开始，不重新对论文图片进行数字化，也不重新从更大候选池筛选；这些步骤见 [ml-degradation-analysis](https://github.com/ShunHao0626/ml-degradation-analysis)。聚类中的形状轴是观测序位，不是实际经过的小时数。

**来源图：**`data/raw_csv/` 及 `results/cluster_*/` 下按簇复制的文件中，包含每条曲线数字化所依据的论文原图裁图，便于逐条对照来源。这些图的版权归原出版方所有，每个文件可通过其所在文件夹名中的文章 DOI 识别。
