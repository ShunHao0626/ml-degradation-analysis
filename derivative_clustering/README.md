# PFD-DTW-HC 论文复现

本目录复现 Kang、Wu、Yu 在 *Information Sciences* 678 (2024) 120939 中提出的
“基于多项式拟合和多阶趋势特征的时间序列聚类”。核心代码严格对应论文的处理链：

1. 使用二阶差分惩罚做 HP 滤波，默认 `lambda=1000`；
2. 对滤波后的序列做全局多项式最小二乘拟合；
3. 在每个时间点提取一至三阶导数，形成 DTS/PFD 特征；
4. 使用加权欧氏距离作为 DTW 的局部代价；
5. 由 PFD-DTW 距离矩阵执行层次聚类。

## 文件

- `pfd_dtw.py`：论文算法、传统 DTW 基线和层次聚类。
- `ucr_data.py`：读取 UCR/UEA `.ts` 数据。
- `reproduce_experiments.py`：ECG5000、Trace、Plane 子集实验及论文中的四种基线。
- `download_ucr.py`：从 UCR/UEA 官方镜像下载三个数据集。
- `demo_synthetic.py`：无需外部数据的最小示例。
- `tests/`：公式、距离与聚类的单元测试。
- `文章复现说明.md`：文章详读、公式映射、歧义与复现边界。

## 快速运行

```bash
cd derivative_clustering
python3 demo_synthetic.py
python3 -m pytest -q
python3 reproduce_experiments.py --dataset Plane
```

若 `data/` 不存在，可联网下载：

```bash
python3 download_ucr.py
```

实验结果写入 `outputs/<dataset>/`，包括选中曲线、各方法树状图、距离矩阵和
`results.json`。脚本同时报告 ARI 与经过标签最优匹配后的聚类准确率。

## 关键参数

```bash
python3 reproduce_experiments.py \
  --dataset Trace \
  --smoothing 1000 \
  --max-degree 20 \
  --weights 3 2 1 \
  --linkage single \
  --seed 2024
```

论文没有公布最高多项式阶数、三个导数权重、层次聚类 linkage 和随机样本索引。
因此上述默认值是明确记录的工程假设，不是作者参数。默认搜索上限为 20；可通过命令行
改动并做敏感性分析。
固定某一多项式阶数时使用 `--degree K`。

## 计算量

未约束 DTW 的单对复杂度为 `O(n*m)`，全数据两两计算为 `O(N^2*n*m)`。论文的图
7--12 只使用 6、8、9 条序列，本脚本默认复现这些“小子集”实验规模。若直接对数千条
ECG5000 序列运行，需要并行化、窗口约束或编译型 DTW 实现；这些加速并不属于论文方法。
