# 批量文献检索报告

检索覆盖 2010-01-01 至 2026-09-03。Crossref 与 OpenAlex 各运行 8 组检索式，每组抓取
相关度排序前 200 条。原始带重复记录 2829 条，按 DOI（无 DOI 时按规范化题名）去重后
1953 条；其中高优先级 PCE/效率—时间或稳定性曲线候选 142 条，包含恢复、
可逆、light-soaking、burn-in 或昼夜行为关键词的重点候选 115 条。

## 重要边界

此阶段只进行元数据、摘要和开放全文链接抓取。关键词命中不等于图中真的存在所需拓扑；
必须继续检查全文图与补充材料。付费墙内容不绕过访问控制，仅保存 DOI/出版社页面。

## 每个查询的抓取状态

- Crossref / `burn_in_nonmonotonic`：返回 200 条，服务报告总量 2010614。
- Crossref / `curve_clustering`：返回 200 条，服务报告总量 3692508。
- Crossref / `light_soaking_recovery`：返回 200 条，服务报告总量 2588377。
- Crossref / `operational_stability`：返回 200 条，服务报告总量 2166610。
- Crossref / `outdoor_diurnal`：返回 200 条，服务报告总量 2011553。
- Crossref / `pce_time_curve`：返回 200 条，服务报告总量 2782685。
- Crossref / `reversible_degradation`：返回 200 条，服务报告总量 2187011。
- Crossref / `som_degradation`：返回 200 条，服务报告总量 1012632。
- OpenAlex / `burn_in_nonmonotonic`：返回 8 条，服务报告总量 8。
- OpenAlex / `curve_clustering`：返回 200 条，服务报告总量 3950。
- OpenAlex / `light_soaking_recovery`：返回 200 条，服务报告总量 707。
- OpenAlex / `operational_stability`：返回 200 条，服务报告总量 1088。
- OpenAlex / `outdoor_diurnal`：返回 21 条，服务报告总量 21。
- OpenAlex / `pce_time_curve`：返回 200 条，服务报告总量 2479。
- OpenAlex / `reversible_degradation`：返回 200 条，服务报告总量 3019。
- OpenAlex / `som_degradation`：返回 200 条，服务报告总量 444。

## 输出

- `all_records_deduplicated.csv`：完整去重目录；
- `high_priority_curve_papers.csv`：高优先级曲线文献；
- `nonmonotonic_recovery_priority.csv`：Bridge/Hill/Valley 更相关的非单调候选；
- `open_fulltext_download_queue.csv`：可公开下载的全文队列；
- `retrieval_manifest.json`：检索与计数审计。
