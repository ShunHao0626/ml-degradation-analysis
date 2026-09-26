# 原文参数溯源

本实验只改变输入数据：使用人工合成且带已知真值的 1,000 条曲线。训练与预处理参数复刻原文。

| 项目 | 本实验值 | 本地原文/作者代码证据 |
|---|---:|---|
| 分析窗口 | 150 h | `thesis/paper/work.md`：只分析前 150 h |
| 重采样 | 10 min | `thesis/paper/work.md` Methods；作者 notebook |
| 插值 | Akima | `thesis/paper/work.md` Methods；作者 notebook |
| 归一化 | 每条曲线 MaxAbs | `thesis/paper/work.md` Eq. 2 |
| 平滑 | Savitzky–Golay，window=71，order=2 | 正文给 71；作者 notebook 给 `savgol_filter(..., 71, 2)` |
| SOM 实现 | MiniSom 2.2.9 | `thesis/environment.yml` |
| SOM 网格 | 2×2 | 作者 notebook：`som_x=2`, `som_y=2` |
| sigma | 0.5 | 正文 Fig. 4 / Methods；作者 notebook |
| learning rate | 0.1 | 正文 Fig. 4 / Methods；作者 notebook |
| 初始化 | `random_weights_init(data)` | 作者 notebook |
| 训练 | 顺序 `train(data, 50000)` | 作者 notebook；MiniSom 默认 `random_order=False` |
| 随机种子 | 未指定 / `None` | 原文及作者 notebook 均未给 seed；主实验保留该默认值 |

本地证据文件：

- `/Users/shunhao/Desktop/ML/thesis/paper/work.md`
- `/Users/shunhao/Desktop/ML/thesis/paper/Supplementary.md`
- `/Users/shunhao/Desktop/ML/thesis/20230227_degradation_analysis_revision_10_cleaned.ipynb`
- `/Users/shunhao/Desktop/ML/thesis/environment.yml`

## 无监督边界

`true_class` 只用于生成数据和训练后的节点命名/评价。传入 MiniSom 的对象只有 901 维预处理曲线矩阵，训练阶段不读取标签。四个 SOM 节点训练完成后，才用 Hungarian 一一映射将节点命名为 Bridge/Hill/Slope/Valley。
