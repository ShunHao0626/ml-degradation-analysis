# 曲线形态文献证据核查

## 核心判断

大批量检索证明文献中确实存在构成目标四种形态的若干局部动力学，但没有找到把
`IFO-Bridge / IFO-Hill / IFO-Slope / IFO-Valley` 作为一组正式定义的论文或方法。

真正的方法来源 Hartono 等人的 [Nature Communications 论文](https://doi.org/10.1038/s41467-023-40585-3)
只分析前 150 h，类别为 initial gain、slow/medium/fast exponential decay。它既没有
`IFO` 命名，也不能在严格 150 h 窗口内核验目标图所标出的 “After 200h” 行为。

## 图级证据

- [Cross-linking PSC，2018](https://doi.org/10.1038/s41467-018-06204-2)：Fig. 5 提供典型
  IFO-Slope 近似——早期 burn-in 后进入慢衰减/近平台；这是单篇实验曲线，不是分类体系。
- [HUBLA 动态钝化，2024](https://doi.org/10.1038/s41586-024-07705-5)：Fig. 4 的处理组存在
  很弱的早期上升后慢降，可视为 Bridge-like 个例，但没有快速上升，也不是一个无监督类别。
- [PHASET，2025](https://doi.org/10.1038/s41467-025-63176-w)：恢复来自 light soaking 后
  dark rest；连续 1-sun PCE 曲线仍是单调衰减。这种条件切换不能带回当前训练。
- [TiO2 自愈器件，2023](https://doi.org/10.1038/s41598-023-33473-9)：表中 PCE 从第 1 天
  11.08% 降至第 2 天 7.17%，随后恢复至第 90 天 11.27%，是 Valley-like 的稀疏离散证据；
  它不是统一条件下的连续 PCE-hour 曲线，且没有恢复后的慢衰减段。
- [UV degradation and recovery，2016](https://doi.org/10.1038/srep38150)：PCE 在 UV 下下降，
  换成 1-sun light soaking 后恢复，明确依赖外部照明阶段切换。
- [Photoinduced degradation dynamics，2018](https://doi.org/10.1021/acsaem.7b00256) 与
  [light-activated self-healing，2016](https://doi.org/10.1038/ncomms11574)：均证明恢复动力学，
  但依赖 illumination/dark 循环或暗态休息。
- [ISOS 共识，2020](https://doi.org/10.1038/s41560-019-0529-5) 正式总结了两种相反动力学：
  光致 PCE 上升后暗态下降，以及光致退化后暗态恢复；它没有提出 IFO 四分类。

## 对当前项目的约束

这些文献可以作为“某种形态在物理上可能出现”的证据，却不能作为额外变量、目标模板或
分类标签加入当前 SOM。尤其是 UV↔1-sun、light↔dark、喷水/干燥和昼夜循环都属于明确的
外部干预或协议变化，与用户要求的固定输入无监督聚类不同。

因此文献阶段没有推翻本地结论：严格 Hartono 方法在当前数据中仍只保留 K=4 的衰减强度/
近稳定主类，未自然形成四个目标 IFO 类别。
