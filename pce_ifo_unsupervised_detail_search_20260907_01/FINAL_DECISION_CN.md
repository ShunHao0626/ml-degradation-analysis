# 最终判断

本轮已经尽可能扩大了不使用标签的搜索范围，并全程关闭 smoothing。

## 无监督主结果

- 自动选择 K=4，而不是预设4类。
- 数据：300 h coverage 子集，1590条曲线。
- 表征：MaxAbs level。
- SOM：sigma=0.5、learning rate=0.03、random order、30,000次更新。
- 10个随机种子的平均两两 ARI=0.9578；20次样本 bootstrap ARI 中位数=0.9182。
- 冻结后四个真实 medoid 全部为 Slope-like，只是衰减强度不同。

因此，**主结果不支持“Bridge、Hill、Slope、Valley 是四个自然稳定簇”**。

## 扩展搜索是否找到目标四簇

- 主网格：4032次 SOM，192个 elbow 设置；没有 exact-four，最多覆盖 Bridge/Hill/Slope 三种。
- 全窗口扩展：72个数据变体 × 6组代表设置 × K=2–8 × 3个种子，共9072次 SOM；审计432个 elbow 模型。
- exact-four 模型数仍为0；14个模型覆盖3/4种形态。
- 最稳定的三形态候选使用300 h、z-level+d1，K=4，seed ARI=1.0，但缺少 Valley。
- Valley-like 节点会在部分1000 h形状模型中出现，但没有与 Bridge、Hill、Slope 同时成为四个独立节点。

## 可以与不可以宣称的内容

可以宣称：原始数据中能找到 Bridge/Hill/Slope/Valley 四种外观的个体曲线；形状敏感表征能把其中最多三种分成独立节点。

不能宣称：当前数据和无监督判据已得到四个稳定 IFO 类。若从搜索结果中只挑“最像四类”的运行作为最终模型，会把目标形态用于模型选择，不再是全过程标签盲。
