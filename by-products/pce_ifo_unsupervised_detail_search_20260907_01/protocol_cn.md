# 冻结分析协议

本文件在查看本轮聚类的 IFO 形态名称之前建立。

## 允许的数据预处理

- 根据坐标轴元数据把 minute/hour/day/week/month/year 转换成小时。
- 从每条曲线的第一个有效时间点开始计算 elapsed time。
- 合并完全相同的时间戳（y 取均值），删除非有限行。
- 只保留可由 y 轴元数据或记录名识别为 PCE/efficiency 的真实轨迹；排除 delta/change/loss 等派生面板。
- 用预先写入 `config.json` 的点数、时间覆盖和最大空档规则建立 coverage/detail/dense 三个子集。
- 在原始相邻观测点之间做分段线性插值，仅用于建立公共输入坐标，不作外推。

## 禁止事项

- smoothing、滤波、样条、LOWESS、Savitzky–Golay、移动平均或人为删掉局部起伏。
- 因为曲线不像目标四类而删样本。
- 固定 K=4、按 IFO 名称评分、挑选恰好产生四类的随机种子。
- 在模型冻结前使用 Bridge/Hill/Slope/Valley 模板、标签或形态阈值。

## 两阶段搜索

1. K-means 只作快速、无标签的数据表示筛选。每个 window × cohort × representation 独立寻找 inertia elbow，并计算 silhouette、子样本重拟合 ARI、最小簇占比及样本保留率。150/200 h 作为敏感性结果，不具备主结论资格。
2. 为避免单一幅值表示垄断筛选，每种预注册 representation 取评分最高的一个至少 300 h 的 window/cohort 进入完整 SOM 搜索，共四个候选。搜索 sigma、learning rate、随机/顺序训练和 K；每个组合使用多个随机种子。每个设置的 K 只由 QE 几何 elbow 确定。

最终 SOM 以 silhouette、跨种子 ARI、topographic error、中心分离、最小簇支持和样本保留率的无标签秩聚合选出。四种 IFO 名称在该选择写入 frozen selection 后才计算。

## 结论规则

- 若冻结模型自然产生四个稳定节点，且事后分别呈现四种 IFO 形态，可称为“本轮无监督流程支持四个 IFO-like 簇”。
- 若 K 不是 4，或四种形态未分别形成稳定节点，必须保持实际结果。
- 可报告数据中存在四类个体候选，但不得把个体候选当作四个无监督簇。
