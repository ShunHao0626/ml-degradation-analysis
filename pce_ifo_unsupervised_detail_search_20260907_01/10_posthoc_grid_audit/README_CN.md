# 冻结后全 SOM 网格形态审计

该审计只在主模型与 K 已写入 `05_frozen_selection/frozen_selection.json` 后运行。每个被审计的分类本身仍是无标签 SOM，但这里按 IFO 名称寻找候选属于**事后目标导向检查**，不能替代全过程标签盲的主选择。

- 审计设置数：192。
- 同时覆盖 Bridge/Hill/Slope/Valley 的 K=4 设置数：0。
- 任一设置最多覆盖目标形态数：3/4。
- 最佳形态覆盖候选：`w0300_coverage_z_level_d1` / `sigma1.2_lr0.5_sequential` / K=4。
- 该候选的搜索期 seed ARI=1.0000，silhouette=0.1983，事后名称为 `Bridge-like;Hill-like;Slope-like`。

如果 exact-four 数为0，则本轮广泛搜索仍没有找到可被诚实描述为四个目标簇的配置。如果大于0，候选图只能证明“存在某些无监督运行产生四种外观”，还需结合其稳定性与主选择排名判断是否可信。
