# IFO 四种曲线形态识别报告

## 最终结果

完整审计了原始 `lab-v2/som_references/accepted` 下的 218 条 CSV；其中 155 条能确认是时间—PCE/效率/功率曲线。最终模型通过联合评分自动选择 750h 时间窗口，共有 87 条真实曲线具备足够时长，没有向训练数据加入任何合成样本。

结果同时提供两层标签：`final_class` 是便于与普通四聚类比较的四向标签；`strict_class` 只接受逐条满足目标拓扑的曲线，否则标为 `unresolved_other_shape`。严格结果共 61 条，未强行贴入四类的曲线共 26 条。

- IFO-Bridge: 四向标签 3 条；严格拓扑命中 3 条；规则集稳定度中位数 1.000
- IFO-Hill: 四向标签 3 条；严格拓扑命中 3 条；规则集稳定度中位数 1.000
- IFO-Slope: 四向标签 77 条；严格拓扑命中 51 条；规则集稳定度中位数 1.000
- IFO-Valley: 四向标签 4 条；严格拓扑命中 4 条；规则集稳定度中位数 1.000

四类中心的硬性拓扑检查全部通过：Bridge 早期上升后晚期衰减；Hill 早期上升且峰后明显下降；Slope 早期下降且恢复有限；Valley 先下降、随后恢复，并在恢复峰之后再次进入慢衰减。

## 为什么没有直接采用普通 SOM

先完成了论文参数搜索与扩展 SOM 搜索。最佳候选：window=150h、sigma=0.5、learning_rate=0.1、iterations=50000、5-seed mean ARI=1.000。它是否通过“四个独立目标原型”验收：`False`。

数据天然严重不均衡，普通四聚类倾向于把数量最多的单调衰减继续拆成多个强/中/弱衰减簇，并把其中一个误命名为 Valley。按照用户要求，在 SOM 无法同时满足四种拓扑硬条件时，最终改用“200h 相位边界 + 峰谷次序 + 上升/衰减/恢复幅度”的形态引导分类器。阈值不是手选单点：共保存了全部网格候选，并用拓扑匹配、目标原型相关性、轮廓度、覆盖率、最小类规模和均衡度联合选择。

## 其他模型对照

- feature_kmeans: silhouette=0.315, ARI vs final=0.129
- feature_ward: silhouette=0.283, ARI vs final=0.121
- feature_gmm: silhouette=0.156, ARI vs final=0.100

这些无监督模型用于对照，不作为四类名称的真值。最终标签由可解释的物理形态顺序决定。

## 质量控制

- 自动验证：`True`。
- 低规则稳定度：3 条；包括拓扑未命中在内，人工复核队列共 29 条，已写入 `07_validation/manual_review_queue.csv`。
- 所有 218 条曲线均在 `06_final_four_classes/all_218_curves_final_status.csv` 中有最终状态；未分类者保留明确的轴/质量或时长不足原因。
- 由于输入是论文图数字化曲线而非统一实验条件的原始 MPPT 数据，本报告只解释曲线形态，不把类别差异直接解释成材料或器件机理。

## 复现

```bash
cd /Users/shunhao/Desktop/ML/lab-v2/ifo_four_shape_discovery
MPLCONFIGDIR=/tmp/mplconfig_ifo_four_shape conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```
