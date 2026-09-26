# 四类目标 PCE 曲线分类

公开仓库保留分析代码、方法说明与测试。下文引用的 `data_final/`、`review_inputs/` 和 `results/` 是本地输入及运行产物，不随 Git 分发；运行前请按实际工作目录准备数据。以下命令以 `ML_SOM/` 为当前目录。

## 独立的 0–200 h 重新分类

按新的时间窗口要求运行：

```bash
python3 target_shape_classification/classify_200h.py
```

结果位于 [`results/som_200h/`](results/som_200h/README.md)：[离线交互页面](results/som_200h/all_curves_index.html) 展示全部原始文件，[全文件 CSV](results/som_200h/all_files_200h.csv) 分列 SOM 神经元、SOM 候选、原图复核和最终类别。SOM 只使用实际 0–200 h 的 PCE 点，按该窗口内最大值归一化；谷底在 150–200 h 且 200–250 h 出现至少 1% 恢复者另列边界复核候选，原图证实后可归为 Valley。归一化后至少 1% 的时间波动保留。少于两个窗口点、非 PCE 或小时轴未确认的文件明确列为 unresolved。当前 2,208/2,250 个原始文件获四类候选，42 个未分类（1.87%）；[当前复核队列](results/som_200h/review_priority_200h.csv)尚有 1,320 条曲线、1,094 张不同原图。已逐图重判 46 条曲线、44 张原图，其中 36 条确认、9 条仍有歧义、1 条采点不符；各类确认至少 5 条且覆盖至少 5 个 DOI 组。记录见 [`review_inputs/source_reviews_200h.csv`](review_inputs/source_reviews_200h.csv)。下文为之前的全程分析方法与结果，两套产物彼此独立。

执行入口（先运行审计与形态对照，再运行 SOM 主方法）：

```bash
python3 target_shape_classification/run.py --data-root data_final --output target_shape_classification/results --stage all
python3 target_shape_classification/audit_duplicate_time_spread.py
python3 target_shape_classification/audit_review_workload.py
python3 target_shape_classification/unsupervised_som.py --root data_final --output target_shape_classification/results/som_primary --early-stage-balance-power 2
python3 target_shape_classification/hill_stage_retrieval.py
python3 target_shape_classification/classification_decision_view.py
python3 target_shape_classification/delivery_coverage.py
python3 target_shape_classification/hill_heldout_diagnostic.py
python3 -m unittest discover -s target_shape_classification -p 'test_*.py' -v
```

程序保留全部原始 CSV，不改动 `data_final`。`input_manifest.csv` 和 `point_mapping.csv` 分别追溯每个文件、每个原始数值点；相同时间的点在分析视图中取中位数，各原始纵值仍保留在逐点映射。`duplicate_time_spread_audit.csv` 另行列出同时间戳纵值的归一化差异：86 条 PCE 曲线有重复时间，其中 77 条至少有一组差异达到 1%；这类差异不是时间上的峰谷。归一化按各分析版本的最大观测 PCE 计算。形态模型使用未平滑的唯一时间点及 PCHIP 区间内重采样，记录显著度达到 0.01 的局部峰谷；轻度平滑仅用于噪声估计。最终图库使用真实小时，不外推。

`classification_all_files.csv` 是此前形态模板方法的全文件最近分组，`classification_unique_curves.csv` 是去重后的形态模型结果。`evidence_status=supported` 仅在 `source_reviews.csv` 中有源图和关键阶段核对记录时授予。`supported_members.csv` 含核验成员与核验理由，`templates.csv` 只取这些真实曲线。`galleries/all_curves.html` 可离线交互浏览所有规范曲线；时间或 PCE 身份未确认的文件仍保留结果行，但不假标小时或 PCE 图。

`candidate_review_queue.csv` 每类列出最多 20 条优先候选，`source_review_sheets/` 把原图和 CSV 曲线并排，供人工核查系列对应。`source_review_tasks.csv` 记录形态不完整或类别接近的复核任务；只有点数、采样缺口、单位、物理量或文件问题进入 `digitization_tasks.csv`。源图与 CSV 串线、漏段可在人工复核后另增补采任务。这类并排图**不是像素标定叠加图**。

补采前在原图上记录 `pixel_x,pixel_y,provenance`，其中 `provenance` 仅为 `source_marker` 或 `digitized_trace`。标定 JSON 的 `x`、`y` 各需两个刻度的 `pixel_1,value_1,pixel_2,value_2` 及 `scale`（`linear` 或 `log10`），并填入当前 `source_image_sha256`。运行：

```bash
python3 target_shape_classification/redigitize.py \
  --data-root data_final --output target_shape_classification/results \
  --source-csv 'data_all/.../accepted/curve.csv' \
  --points points.csv --calibration calibration.json --status draft
```

核查 `source_overlays/` 的采点叠加图后，可改用 `--status accepted` 写入版本化补采数据，再运行主入口。没有原图可见信息时保持任务待处理，不由插值制造实验点。

`verification.json` 的 `targets_met` 只表示四类可信数量和来源数达到门槛。`overall_acceptance_met` 同时检查全量最近标签、单位、物理量与补采闭环。`parameter_search.csv` 记录此前 24 组形态模板参数在当时 20 个复核来源上的分组留出一致性，`development_sample.csv` 记录按来源、时长与点密度抽取的 300 条开发曲线。该对照的复核曲线来自本轮模型初筛，因此一致性不是独立泛化精度。形态对照产物中的 `som_assignments.csv` 与 `som_crosstab.csv` 是早期匿名 SOM 簇；主方法结果在 `results/som_primary/` 下。

## SOM 主方法

主入口 `unsupervised_som.py` 将 SOM 作为四类 PCE 曲线的无标签发现方法；`run.py` 提供输入审计、原图复核、版本化补采和形态对照。当前 **2,221** 条可比较 PCE 规范曲线全部参与 SOM 拟合，结果回填到 **2,250** 个原始文件。早期上升分支按输入曲线自身的完整阶段稀有程度增加训练抽样机会，幂次 2、倍率上限 15；其他分支维持原抽样。源图审阅类别不用于训练。不同时间的源点归一化后约 **1%** 的波动不删点、不剪平；唯一时间点的分箱极值和显著度至少 0.01 的峰谷数进入 SOM 特征，当前 535 条曲线检出此类峰谷。同一时间戳的多个纵值逐点保留并单独审计。

用户已明确本数据中的 η 为 PCE%，`normalized performance` 为归一化 PCE。逐图、逐系列复核使 25 条先前保守排除的 PCE 曲线恢复参与分析；同一图中明确是 FF、Jsc、Voc 的系列仍不作为 PCE。名称带 FF/Jsc/Voc/EQE 的 `final_data` 图也已按实际纵轴及所选系列追加九条原图哈希审计规则；固定 1 Sun 下的归一化 Pmax 与归一化 PCE 具有相同的相对轨迹。两条 MPP 功率密度曲线缺少可核验的 PCE 换算依据；一条 Arrhenius 等效时间曲线不具有实际经过小时。共 36 条轴与物理量复核规则及原图哈希在 [`review_inputs/source_axis_overrides.csv`](review_inputs/source_axis_overrides.csv)。实际曲线图用 `Time (h)` 和 `Normalized PCE`；四条月轴曲线按平均公历月 730.485 h 近似换算并逐条标注。

主 SOM 的五个种子 41–45 先满足全量无标签拓扑误差距最小值 ≤0.005，再以四类完整阶段曲线的无标签平均覆盖率选代表；当前为种子 **41**。神经元在训练后才用阶段规则解释，源图审阅类别和模板不进入拟合。`som_assignments.csv` 保留四选一最近已命名神经元候选、可为 `unresolved` 的最佳匹配神经元类别及逐种子投票。SOM 最近标签是检索线索，不自动成为可信成员。

[全文件交付表](results/som_primary/som_delivery_all_files.csv) 与 [覆盖率核算](results/som_primary/som_delivery_coverage.json) 检查用户要求的**最终舍弃 ≤15%**：四类最近匹配或来源确认 2,196 个文件，待核查扩展组 28 个，排除 26 个。排除占原始文件和去重曲线均约 **1.16%**。逐行 `assignment_basis` 区分源图确认、SOM 与完整阶段一致、仅最近 SOM 相似；最后一层不是已核验目标成员。`corrected_series` 列标出经原图核对发现的系列错名。全部文件、原始点和排除原因仍可追溯。

45 条原图审阅中，40 条四类阶段明确。当前主 SOM 的审阅吻合为 Bridge 5/5、Hill 19/21、Slope 6/6、Valley 5/8；排除 40 个已审阅 DOI 组重新训练的来源留出诊断为 5/5、8/21、6/6、5/8。以同一留出组重拟合的[未均衡基线](results/som_baseline_current_review/README_CN.md)分别为 Hill 0/21 与 3/21；[耦合抽样诊断](results/som_coupled_sampling/README_CN.md)的来源留出 Hill 为 1/21。审阅例参与过特征探索或由 SOM 检索发现，这些比例**不是独立泛化精度**；稀有形态的跨来源稳定性仍有限。四类严格源图和形态双重核验成员为 Bridge 5、Hill 19、Slope 5、Valley 6。完整 Hill 阶段检索有 27 条线索，5 条未获当前代表 SOM 的 Hill 候选；[多种子支持清单](results/som_primary/hill_som_corroborated.csv)列出其中 23 条得到至少两个种子支持的曲线及其独立的源图审阅状态。

[原图审阅工作量](results/source_review_workload.json)按曲线与去重原图分别统计：当前形态复核任务中，尚未首次审阅 **1,332 条曲线、1,090 张图**；优先候选队列为 **74 条曲线、70 张图**。若全部 2,221 条可比 PCE 都逐条核图，尚未审阅 2,176 条曲线、1,670 张图。生成任务表中的 1,337 条还包含 5 条已看图但阶段未确认的曲线。

12 条原图标定补采使用独立版本，原 CSV 未改。完整方法、敏感性配置、输入版本反事实和当前限制见 [运行报告](results/som_primary/REPORT_CN.md)。`results/verification.json` 的 `overall_acceptance_met=false` 表示严格的全量来源与形态验收尚未完成；15% 舍弃上限已在 SOM 交付表中单独核算并满足。
