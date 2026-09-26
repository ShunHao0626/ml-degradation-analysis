# 0–200 h 原始曲线的匿名 DTW 聚类

本目录是独立实验，不读取旧分类作为训练输入。`run.py` 从指定的
`original_curves_2250` 读取规范曲线，核对源 CSV 哈希，统一小时轴，只保留
`0 ≤ hour ≤ 200` 的原始点。每曲线用该窗口内有效原始点的最大 PCE 归一化。
所有 2250 个原文件、2246 条去重规范曲线及不能建模的原因均入索引；有效
原始点逐条写入 `results/points/`，不外推短曲线。

模型视图使用 2 h 小时网格，只在真实点间隔不超过 40 h 时插值；插值点不算
原始证据。局部 10 h 线性斜率乘以 20 h 与 PCE 幅值构成双通道。幅值、导数、
两者联合的匿名近邻并集确定待算 DTW 对；每对仅在双方都受原始观测支持的
小时上比较，限制最大错位 20 h，并惩罚仅有短暂共同覆盖的配对。
主分组固定为融合 DTW 相似图的 8 个匿名群；4 和 12 群只是无标签诊断。
保留幅值 DTW、导数 DTW 的 8 群对照。组数 8 是预设的细分群容量，**不是**
宣称自然存在 8 种 PCE 形态。少数形态可对应多个小群；是否存在四种目标
形态必须在聚类完成后检查真实代表和独立盲判。

`results/posthoc_rule_comparison.csv` 和 `posthoc_contingency.csv` 在匿名结果
落盘后才读取旧的自动规则候选，仅供探索性对照。它们不是真值、准确率或
选择模型的依据。当前图式中的 200 h 后走势未参与本实验输入与距离。

实际数量、匿名群外观及限制见 [RESULTS.md](RESULTS.md)。少数转折形态的
`results/event_review_queue.csv` 是聚类后独立生成的核图队列，不改变聚类。

细节观察：打开 [匿名群局部查看页](results/anonymous_cluster_detail_viewer.html)，
可选 8 个群、搜索曲线编号，直接输入横轴小时范围和纵轴 PCE 范围，或用
50 h 分段按钮放大；纵轴可按当前可见的代表曲线与所选曲线自动缩放。
圆点对应原始观测，悬停显示源 CSV 行号。查看页收录全部 2065 条可建模曲线，
由 `python3 make_detail_viewer.py` 从已保存的逐点数据重新生成；它只改变显示尺度，
不改变 2 h 模型网格、DTW 距离或匿名分组。

若按[严格数值候选的 2207 条口径](../evaluation/strict_numeric_coverage_20260925/README.md)
逐条查看，可打开 [2207 条细节查看页](results/strict_numeric_2207_detail_viewer.html)。
它读取整合图库的 `classes/*/*.csv` 中 0–200 h 的真实点；其中 2065 条有
既有 DTW 匿名群号，142 条显示未进入 DTW 的原因。四类名称只是事后自动规则
候选，**没有作为 DTW 输入**。运行 `python3 make_2207_detail_viewer.py` 可重建此页。

运行：

```bash
cd /Users/shunhao/Desktop/ML_SOM/target_shape_classification/research_early_200h/dtw_unsupervised_200h_20260925
python3 run.py
python3 verify.py
```

本次运行的 Python 包版本见 `requirements.txt`；运行还需要 C++17 编译器。

主要文件：`config.json`、`run.py`、`masked_dtw.cpp`、`results/curve_index.csv`、
`results/model_inputs.npz`、`results/dtw_distances.npz`、
`results/anonymous_assignments.csv`、`results/cluster_representatives.csv`、
`results/anonymous_cluster_overview.png`、`results/summary.json`。
