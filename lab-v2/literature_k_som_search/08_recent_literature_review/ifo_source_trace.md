# IFO-Bridge / IFO-Hill / IFO-Slope / IFO-Valley 出处追溯

检索日期：2026-09-02（Asia/Shanghai）  
结论状态：**未找到可核验的正式论文原始出处**。

## 结论

截至检索日，公开学术全文索引、预印本库、DOI 元数据库、常规网页及代码/数据仓库中，均没有找到把 `IFO-Bridge`、`IFO-Hill`、`IFO-Slope`、`IFO-Valley` 作为一组使用的论文、图注或方法定义。三个词在 OpenAlex 全文检索中为零命中；`IFO-Slope` 的唯一命中是经济学论文中相邻出现的 “ifo” 与 “slope”，不是光伏术语。其他数据库中的少数命中也均为同类误匹配。

因此，目前最稳妥的判断是：

1. 这四个名称**不是已被证实的正式文献分类术语**；`IFO` 的全称也没有从任何一手来源得到定义。
2. 用户给出的图更符合一张根据若干常见 PCE–时间形态重新绘制的**二次示意图或自定义目标标签图**，而不是 Hartono 等人 2023 年论文中的原图。
3. 这不等同于证明图一定由 AI 生成，也不能排除未公开、未索引的幻灯片或内部稿件；现有证据只能支持“**暂无可验证的文献出处，不应冒充文献命名**”。

在后续分析中，应把这四个词写成“用户提供的事后形态别名”，不能把它们当成论文规定的类别、训练标签或选择聚类数的依据。

## 可复核的精确检索记录

常规网页检索同时尝试了带连字符、去连字符/空格、大小写不敏感及与 `perovskite`、`solar cell`、`PCE`、`hour` 联合的写法；数据库 API 表中的计数则只对应表内明确列出的标准连字符精确式，避免把限流或松散分词结果误报成零命中。

| 来源 | 精确查询 | 返回情况 | 人工核对结果 |
|---|---|---:|---|
| OpenAlex 全文 | `"IFO-Bridge"` | 0 | 无 |
| OpenAlex 全文 | `"IFO-Hill"` | 0 | 无 |
| OpenAlex 全文 | `"IFO-Slope"` | 1 | 排除：2015 年经济学预印本 *Value Creation Drivers in Large Leveraged Buyouts*，DOI `10.2139/ssrn.2621525`；是 “ifo” 与 “slope” 的版面/词项误匹配，与太阳能电池无关 |
| OpenAlex 全文 | `"IFO-Valley"` | 0 | 无 |
| arXiv API | 上述四个精确词分别检索 | 0 / 0 / 0 / 0 | 无 |
| Europe PMC | Bridge / Hill / Slope / Valley | 0 / 0 / 1 / 0 | 唯一 Slope 命中是 2016 年 BODIPY–蛋白相互作用论文，DOI `10.1039/c6cp00420b`，与光伏无关 |
| DataCite | Bridge / Hill / Slope / Valley | 2 / 0 / 0 / 0 | 两个 Bridge 结果是同一 Zenodo 哲学/数学预印本的版本，正文短语为 “ISUN-IFO bridge”，不是 `IFO-Bridge` 类别，也与光伏无关 |
| 常规网页检索 | 四个词逐一精确检索；四词 OR；`site:github.com`、`site:gitlab.com`、`site:zenodo.org`、`site:figshare.com` | 无相关命中 | 搜索结果仅有德国 ifo 研究所、IFO 燃油、DVD IFO 文件、地名等误匹配 |
| OpenAlex 全文 | `"rapid decay" "power recovery" perovskite` | 0 | 图中 Valley 的关键措辞没有联合命中 |
| OpenAlex 全文 | `"0-200h" "After 200h" perovskite` | 0 | 图中的阶段边界措辞没有联合命中 |
| 常规网页检索 | `"IFO-Bridge" "rapid increase"`、`"IFO-Hill" "rapid decay"`、`"IFO-Valley" "power recovery"`、`"IFO-Slope" "slow decay"` | 0 个相关结果 | 唯一 IFO/decay 结果来自油藏泡沫 Injection Fall-Off 测试，已排除 |

可直接复核的 API 查询示例：

- [OpenAlex: IFO-Bridge](https://api.openalex.org/works?search=%22IFO-Bridge%22&per-page=25)
- [OpenAlex: IFO-Hill](https://api.openalex.org/works?search=%22IFO-Hill%22&per-page=25)
- [OpenAlex: IFO-Slope](https://api.openalex.org/works?search=%22IFO-Slope%22&per-page=25)
- [OpenAlex: IFO-Valley](https://api.openalex.org/works?search=%22IFO-Valley%22&per-page=25)
- [Europe PMC: IFO-Bridge](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22IFO-Bridge%22&format=json)
- [arXiv API: IFO-Bridge](https://export.arxiv.org/api/query?search_query=all:%22IFO-Bridge%22&start=0&max_results=5)

Crossref 的 `query.bibliographic` 不支持这里所需的严格短语语义，会把 `IFO` 和 `Bridge` 分词后返回数万条无关记录，因此没有把 Crossref 的松散结果数当作证据。Semantic Scholar API 在本次调用中返回 HTTP 429，亦没有伪装成“零结果”。

## 最接近的一手论文逐项排除

### 1. Hartono 等，Nature Communications 2023

- 题名：*Stability follows efficiency based on the analysis of a large perovskite solar cells ageing dataset*
- DOI：[10.1038/s41467-023-40585-3](https://doi.org/10.1038/s41467-023-40585-3)
- 一手全文：[PMC 开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10423264/)
- 真正的 Figure 4：[出版社存档图像](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/5b41/10423264/f762a8acd650/41467_2023_40585_Fig4_HTML.jpg)
- 方法上下文：2,245 条 MPPT 老化曲线，只分析前 150 h；SOM 的四个最终中心为 `initial gain`、`slow exponential decay`、`medium exponential decay`、`fast-exponential decay`；`K=4` 来自量化误差 elbow 与中心重叠的人工判断。
- 排除理由：全文和图注都没有 `IFO`、Bridge、Hill 或 Valley；论文窗口止于 150 h，不可能原生定义用户图中的 `0~200h / After 200h` 阶段；Figure 4 也没有 Hill 或 Valley 的峰谷恢复拓扑。

这是本地 thesis 方法的真正来源，也是四分类视觉布局最接近的**方法学前身**，但不是用户图四个名称的出处。

### 2. 2025 年 Materials Today Energy 老化曲线预测论文

- 题名：*Predicting and analyzing stability in perovskite solar cells: Insights from machine learning models and SHAP analysis*
- DOI：[10.1016/j.mtener.2024.101769](https://doi.org/10.1016/j.mtener.2024.101769)
- 一手页面：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2468606924002818)
- 公开 Figure 8：[Elsevier 原图](https://ars.els-cdn.com/content/image/1-s2.0-S2468606924002818-gr8_lrg.jpg)
- 方法上下文：汇总大量 T80/T90 与老化曲线，以 XGBoost、神经网络和 SHAP 做寿命/曲线预测；Figure 8 展示八组 experimental/predicted normalized-PCE 曲线。
- 排除理由：八组曲线不是四类无监督形态；公开的全部 8 张主文图均无 IFO 四词或用户图；该文引用 Hartono 工作，但没有提出这组新分类。

### 3. “power recovery” 相关论文

- Jeong 等，*Pulsatile therapy for perovskite solar cells*，Joule 2022，DOI：[10.1016/j.joule.2022.04.007](https://doi.org/10.1016/j.joule.2022.04.007)。论文中的性能恢复来自短反向偏压脉冲，是主动干预，不是四类自然老化形态，也没有 IFO 命名。
- *Diurnal Changes and Machine Learning Analysis of Perovskite Modules Based on Two Years of Outdoor Monitoring*，ACS Energy Letters 2024，DOI：[10.1021/acsenergylett.4c01943](https://doi.org/10.1021/acsenergylett.4c01943)。论文讨论日间衰减与夜间恢复，但没有 Bridge/Hill/Slope/Valley 四分类。
- *Consensus statement for stability assessment and reporting for perovskite photovoltaics based on ISOS procedures*，Nature Energy 2020，DOI：[10.1038/s41560-019-0529-5](https://doi.org/10.1038/s41560-019-0529-5)。论文讨论 burn-in 和多样的 PCE–time 形状，但没有这四个术语，也没有 200 h 四相拓扑分类。

这些论文能解释“上升、burn-in、衰减、恢复”等**单独物理现象**，却不能作为用户图联合分类体系的出处。

## 为什么更像二次示意图，而非正式论文原图

以下是来源判断的辅助证据，不单独作为定论：

- 纵轴只写通用的 `Output`，不是正文语境中的 `PCE`、`normalized PCE` 或 `PMPP`；
- 四个面板没有图号、子图字母、单位刻度、样本量、图注或参考文献信息；
- 使用 `0~200h` 与 `After 200h` 的演示式色带，而 thesis 对应论文明确只分析 0–150 h；
- Bridge/Hill/Slope/Valley 是形状比喻词，未在任何检索到的一手论文中被定义；
- 四宫格布局和色彩可看作对 Hartono Figure 4 的二次视觉借用，但曲线拓扑和时间边界已经被重新设计。

所以可以说“**最可能是二次/自定义示意图**”，但不能在没有原始文件元数据或来源链接时断言具体作者或生成工具。

## 对当前 SOM 任务的约束

1. 不能写成“文献提出了 IFO 四分类”；应写成“冻结聚类结果后，与用户给定的四种目标形态作事后比较”。
2. 聚类数仍按 Hartono 论文的 QE elbow + 中心可区分性决定，不因这张图强制为 4。
3. `0~200h / After 200h` 不是 Hartono 方法参数。若严格复现论文，窗口仍是 150 h；将窗口延长到 200 h 或更长属于方法外推。
4. “power recovery” 文献多涉及暗态休息、昼夜循环或电偏压脉冲，不能偷偷作为额外变量或干预加入无监督训练。

## 可证伪条件

只要出现下列任一证据，本结论应立即重审：

- 一篇带 DOI/正式仓储链接的论文或补充材料，原文/图注明确同时定义四个 IFO 名称；
- 用户图的原始高分辨率文件带有作者、期刊、图号、嵌入元数据或可反向匹配的来源页面；
- 作者公开幻灯片、机构仓储或代码仓库中存在同一图，并说明 `IFO` 的全称与分类方法。

在这些证据出现之前，报告中不得为该图编造 DOI、年份、图号或 `IFO` 全称。
