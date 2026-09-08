# IFO 四种曲线形态识别报告

## 最终结果

完整审计了原始 `lab/data_all` 下的 2151 条 CSV；其中 2134 条能确认是时间—PCE/效率/功率曲线。时间单位小时、天、周、月、年统一转换成小时。最终模型通过联合评分自动选择 500h 时间窗口，共有 1229 条真实曲线具备足够时长，没有向训练数据加入任何合成样本。

结果同时提供两层标签：`final_class` 是便于与普通四聚类比较的四向标签；`strict_class` 只接受逐条满足目标拓扑的曲线，否则标为 `unresolved_other_shape`。严格结果共 1003 条，未强行贴入四类的曲线共 226 条。

- IFO-Bridge: 四向标签 56 条；严格拓扑命中 56 条；规则集稳定度中位数 1.000
- IFO-Hill: 四向标签 39 条；严格拓扑命中 39 条；规则集稳定度中位数 1.000
- IFO-Slope: 四向标签 1118 条；严格拓扑命中 892 条；规则集稳定度中位数 1.000
- IFO-Valley: 四向标签 16 条；严格拓扑命中 16 条；规则集稳定度中位数 1.000

四类中心的硬性拓扑检查全部通过：Bridge 早期上升后晚期衰减；Hill 早期上升且峰后明显下降；Slope 早期下降且恢复有限；Valley 先下降、随后恢复，并在恢复峰之后再次进入慢衰减。

## 为什么没有直接采用普通 SOM

先完成了论文参数搜索与扩展 SOM 搜索。最佳候选：window=300h、sigma=0.2、learning_rate=0.5、iterations=100000、5-seed mean ARI=0.998。它是否通过“四个独立目标原型”验收：`False`。

数据天然严重不均衡，普通四聚类倾向于把数量最多的单调衰减继续拆成多个强/中/弱衰减簇，并把其中一个误命名为 Valley。按照用户要求，在 SOM 无法同时满足四种拓扑硬条件时，最终改用“200h 相位边界 + 峰谷次序 + 上升/衰减/恢复幅度”的形态引导分类器。阈值不是手选单点：共保存了全部网格候选，并用拓扑匹配、目标原型相关性、轮廓度、覆盖率、最小类规模和均衡度联合选择。

## 其他模型对照

- feature_kmeans: silhouette=0.351, ARI vs final=0.161
- feature_ward: silhouette=0.259, ARI vs final=0.129
- feature_gmm: silhouette=0.151, ARI vs final=0.318

这些无监督模型用于对照，不作为四类名称的真值。最终标签由可解释的物理形态顺序决定。

## 质量控制

- 自动验证：`True`。
- 低规则稳定度：30 条；包括拓扑未命中在内，人工复核队列共 253 条，已写入 `07_validation/manual_review_queue.csv`。
- 所有 2151 条曲线均在 `06_final_four_classes/all_input_curves_final_status.csv` 中有最终状态；未分类者保留明确的轴/质量或时长不足原因。
- 由于输入是论文图数字化曲线而非统一实验条件的原始 MPPT 数据，本报告只解释曲线形态，不把类别差异直接解释成材料或器件机理。

## 复现

```bash
cd /Users/shunhao/Desktop/ML/lab-v2/ifo_four_shape_data_all_discovery
MPLCONFIGDIR=/tmp/mplconfig_ifo_four_shape_data_all conda run -n PvkSOM python run_pipeline.py
conda run -n PvkSOM python verify_outputs.py
```
