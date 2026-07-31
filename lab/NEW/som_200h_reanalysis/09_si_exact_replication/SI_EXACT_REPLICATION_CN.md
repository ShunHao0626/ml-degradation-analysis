# SI中全部SOM类别数量的复现

## SI实际使用的类别数量

- Supplementary Fig. 1：n=16，作为初始曲线形状概览。
- Supplementary Fig. 6：n=2–10，用于Quantisation Error elbow扫描。
- Main Fig. 4：n=4，最终主分类。
- Supplementary Figs. 15–17：n=2、5、6，用于展示分类不足和分类重叠。

因此，类别数层面需要运行的是n=2–10以及额外的n=16。二者现已全部完成。

## 新增n=16结果

- 主数据集：QE=0.7450，平均种子ARI=0.6781，最小类别占比=0.28%。
- 文献一致性数据集：QE=0.6966，平均种子ARI=0.7297，最小类别占比=0.22%。

n=16用于展示数据多样性，不用于替代最终n=4结论。随着类别增加，部分节点样本很少且质心相互接近，正是SI随后进行elbow分析和类别合并判断的原因。

## 文件

- `SI_FIGURE_TO_LOCAL_RESULT_MAP.csv`：SI图号与本地结果一一对应。
- `{dataset}/si_class_count_overview.png`：n=16、2、4、5、6和QE扫描总览。
- `{dataset}/som_metrics_n2_to_n10_plus_n16.csv`：2–10及16类指标。
- `{dataset}/n_XX/classes/class_XX/members.csv`：每一类具体曲线。
- 每个`raw_curves/`目录按原始相对路径保存可浏览曲线硬链接。
