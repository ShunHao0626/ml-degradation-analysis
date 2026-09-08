# 初始值 2% + 50/100/200 h 卡点补充实验

该实验将老师提出的规则操作化，并保持全过程无监督、无平滑。四种 IFO 名称和人工标签没有进入筛选、特征、SOM 或类别数选择。

## 操作逻辑

1. 基础样本必须有可信小时轴、非恒定非负输出、时长<100000 h，并实际覆盖200 h，共126条。
2. 以第一条原始观测值 `y0` 为基准，计算 `r(t)=(y(t)-y0)/abs(y0)`。不使用平滑。
3. 活动门槛分别测试1%、2%、3%；只要任一原始点达到门槛就保留。2%主实验保留120/126条。
4. 在50、100、200 h读取原始分段折线值，并记录0–50、50–100、100–200、200 h后的最大上升和最大下降。短暂峰谷不会因只看卡点而消失。
5. 对事件特征应用连续 deadband：门槛内为0，超过门槛的部分保留符号和幅度。原始MaxAbs折线与事件块各自缩放到中位成对距离为1，再等权拼接。该权重在训练前固定。

## 类别数结果

| 方法 | n | k* | silhouette | seed ARI mean | seed elbows | range elbows 8/10/12/16 | piecewise | sizes |
|---|---:|---:|---:|---:|---|---|---:|---|
| raw_maxabs_same_p02 | 120 | 4 | 0.491 | 0.781 | [4, 3, 4, 5, 3] | [3, 3, 3, 4] | 3 | [5, 38, 21, 56] |
| threshold_p01 | 126 | 5 | 0.381 | 0.727 | [3, 6, 5, 2, 5] | [2, 2, 5, 5] | 3 | [7, 56, 31, 25, 7] |
| threshold_p02 | 120 | 5 | 0.452 | 0.867 | [5, 4, 5, 5, 6] | [2, 3, 3, 5] | 2 | [6, 64, 21, 23, 6] |
| threshold_p03 | 117 | 5 | 0.454 | 0.772 | [5, 5, 3, 5, 2] | [2, 2, 3, 5] | 2 | [5, 63, 22, 21, 6] |

主结论必须结合 `figures/01_qe_threshold_sensitivity.png` 和 `figures/03_p02_adjacent_k_review.png` 判断；类别数没有按目标示意图预设。

**按预先规定的k=1–16 chord规则，2%主实验选择k=5。** 5个seed平均ARI=0.867、最低ARI=0.764；3%也选择k=5，2%与3%共同117条曲线的最终划分ARI=0.951。k=5的五组规模为6/64/21/23/6，中位特征依次表现为早期上升、轻微衰减、早期快速衰减、延迟或后期衰减、强衰减。相邻k图显示这些组的中位轨迹并非完全重叠。

类别数仍有范围依赖：2%的不同k上限结果为[2, 3, 3, 5]，逐seed为[5, 4, 5, 5, 6]，分段elbow为2。因此k=5是本策略按既定主规则得到的解，而不是已经证明自然界只有5类；尤其两个n=6小组需要更多曲线验证。

k=4对照的5次训练平均ARI=0.839、最低ARI=0.699，用于观察从4到5时发生的拆分。

事后核对显示：Hill-like C004、Slope-like C182、Valley-like C059分别落在k=5的第1、2、5组；Bridge-like C096只有约101 h，因无法提供200 h卡点而未进入本实验。该核对发生在训练完成后，不影响模型。

## 与直接关闭平滑的区别

`raw_maxabs_same_p02` 使用完全相同的 120 条曲线和SOM参数，但没有加入阈值事件特征；它选择k=4。2%策略选择k=5。两者共同样本的ARI保存在 `experiments/model_comparison_ari.csv`。QE值不能跨不同表示直接比较，因此这里只比较各自elbow、稳定性、簇形状和ARI。

## 坐标与限制

50/100/200 h按从第一条观测开始的真实elapsed hour计算。短于200 h的曲线不进入本次主实验，也不外推。整体曲线块仍使用每条完整观测窗口的相对进程u，因此绝对卡点信息只由事件特征块提供。第一点可能存在数字化误差，后续可把初始短窗中位数作为基线敏感性，但本次严格按‘初始值’使用第一点。

## 文件

- `figures/00_activity_threshold_retention.png`：1/2/3%门槛与样本变化。
- `figures/01_qe_threshold_sensitivity.png`：类别数判断。
- `figures/02_main_p02_clusters.png`：2%主模型在真实0–200 h与完整相对进程下的簇。
- `figures/03_p02_adjacent_k_review.png`：相邻k重叠检查。
- `figures/04_p02_k4_shape_substructure.png`：比最终解少一个单元的k=4对照。
- `data/H200_checkpoint_features_and_assignments.csv`：126条基础曲线的全部事件特征与归属。
- `data/p02_cluster_checkpoint_summary.csv`：五组在50/100/200 h和终点的中位变化。
- `data/posthoc_example_audit.csv`：四条既有解释样本的训练后核对。
- `data/p02_manifest.csv`、`data/p02_selected_curves/`：2%规则保留的120条曲线。
- `experiments/qe_sweeps.csv`：全部320次长训练。

## 复现

```bash
cd /Users/shunhao/Desktop/ML/curve_discovery_unsupervised_20260906_01
/usr/bin/python3 code/run_initial_threshold_checkpoints.py
```
