# 无标签、无绝对小时坐标的形状 DTW 细分实验

本实验从已有源 CSV 校验和归一化后的 **0–200 h 原始 PCE 观测点**出发；
`run.py` 只读取前一实验的 `curve_index.csv` 和 `results/points/*.csv`，
不读取任何四类名称、旧规则候选或旧 DTW 分群。小时只用于确定观测先后顺序
以及合并同一小时的重复测量；具体小时数不进入距离。纳入至少 4 个不同观测
时刻的 2152 条曲线，原始点不足的曲线保持未分配。

每条曲线按观测顺序线性重采样到 64 个**相对序位**，保留归一化 PCE 变化量，
另计算短尺度和长尺度的带符号变化通道。短尺度软尺度 0.0075、长尺度软尺度
0.015，使约 1% 的持续起伏在距离中可见；它们不是“1% 即真实事件”的判定阈值。
DTW 允许相对序位上最多 50% 的错位，不要求两个转折发生在相同小时。
对所有 2,314,476 对曲线计算精确 DTW，再以 36 近邻构造相似图，谱嵌入后
层次聚类为 16、32、64 个匿名细簇。簇数是预设观察尺度，**不是自然类别数**。

先看 [交互式细节页](results/anonymous_shape_detail_viewer.html)：可切换 16/32/64
簇、搜索 ID、自动或手动放大纵轴。64 簇的 [全尺度图 1](results/anonymous_k64_page1.png)、
[2](results/anonymous_k64_page2.png)、[3](results/anonymous_k64_page3.png)、
[4](results/anonymous_k64_page4.png) 与 [纵轴局部放大图 1](results/anonymous_k64_zoom_page1.png)、
[2](results/anonymous_k64_zoom_page2.png)、[3](results/anonymous_k64_zoom_page3.png)、
[4](results/anonymous_k64_zoom_page4.png) 方便总览。各群成员及 5 条代表见
`results/anonymous_assignments.csv`、`results/representatives_k64.csv`。

匿名分群落盘后，**单独**运行 `posthoc_audit.py`，才与旧四类自动候选比较；
该对照不是训练、调参或独立真值。具体结果与稳定性限制见 [RESULTS.md](RESULTS.md)。

运行：

```bash
cd /Users/shunhao/Desktop/ML_SOM/target_shape_classification/research_early_200h/shape_only_unsupervised_20260925
python3 run.py
python3 make_viewer.py
python3 verify.py
python3 sensitivity.py
python3 posthoc_audit.py
```

需要 `numpy`、`scipy`、`scikit-learn`、`matplotlib` 及 C++17 编译器。
`sensitivity.py` 只用无标签噪声扰动评估簇稳定性；其输出不能证明真实误差大小。
