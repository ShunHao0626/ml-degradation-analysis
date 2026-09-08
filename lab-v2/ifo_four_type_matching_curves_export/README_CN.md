# 四类严格匹配曲线导出

本目录从 `/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted/samples_test` 的218条曲线中，导出了符合四种目标拓扑严格条件的曲线。

## 数量

- IFO-Bridge：3 条
- IFO-Hill：3 条
- IFO-Slope：51 条
- IFO-Valley：4 条
- 合计：61 条
- 其中置信度 high 58 条，low_review 3 条；61 条均通过严格拓扑条件。

## 目录

- `csv/IFO-*/`：原始曲线 CSV 副本。
- `png/IFO-*/`：与 CSV 同名、同类别的原始 PNG 副本。
- `export_manifest.csv`：逐条来源、导出路径、形态证据和 SHA-256。
- `stable_rule_matches.csv`：规则扰动下仍稳定的 58 条，不代表形态振幅一定很强。
- `low_stability_review_queue.csv`：规则扰动稳定性较低、建议复核的 3 条。
- `class_summary.csv`：分类数量汇总。
- `selection_basis.json`：筛选来源和边界。

## 方法边界

仅导出 `strict_topology_match=True` 的曲线，统一来自750 h严格形态筛选。`low_review` 表示该曲线在规则扰动下稳定性较低，虽然通过严格条件，仍建议人工复核图片。这里的“匹配”首先指上升、下降、恢复等方向顺序符合规则；部分曲线变化振幅较弱，不能理解为与概念示意图等强度复现。它是依据目标拓扑定义进行的严格筛选，不应表述为纯无监督 SOM 自然发现。原始数据没有被移动或修改。
