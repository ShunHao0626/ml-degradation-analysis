# PCE 曲线形态发现（严格文献参数）

本项目是独立的新目录，输入只读取：

- `../thesis` 中的 Hartono 等人 PvkSOM 方法、论文与补充材料；
- `../lab-v2/som_references/accepted/samples_test` 中的数字化曲线。

执行顺序：

1. `python3 run_search.py`：审计数据、复现固定预处理、运行 27 个论文参数候选；
2. 根据 `04_k_selection/` 的肘部和匿名中心图冻结 K；
3. 冻结后才核查 IFO-Bridge/Hill/Slope/Valley；
4. 若四类不能作为自然类别出现，进入 `07_literature/` 文献检索阶段。

约束见 `00_scope/METHOD_BOUNDARY_CN.md`。本项目不修改原始数据、`thesis` 或任何旧实验目录。

当前执行已完成。优先阅读 `REPORT_CN.md`、`04_k_selection/K_SELECTION_DECISION_CN.md`、
`06_ifo_posthoc/IFO_ASSESSMENT_CN.md` 与 `07_literature/CURATED_EVIDENCE_CN.md`。
