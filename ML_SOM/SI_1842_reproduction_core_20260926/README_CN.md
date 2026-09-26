# 1842 条曲线四簇分析：精简复现包

公开仓库只包含本目录的代码、配置和说明。下文列出的 `data/` 固定输入保存在本地，运行前需要单独放入本目录；这些曲线文件不会随 Git 分发。

本目录保留**固定的 1842 条输入曲线**、其原始 CSV、聚类代码和环境信息。初始包不含任何拟合结果、图表、论文原图或 replot 图片。

## 最核心的内容

| 路径 | 用途 |
| --- | --- |
| `data/input_index.csv` | 1842 条输入的固定 ID 顺序、来源与归一化点文件位置 |
| `data/points/` | 每条曲线的早期归一化观测点；用于质量筛查 |
| `data/shape_inputs.npz` | 同序的 64 点序位形状输入；模型直接读取 `raw_rank` |
| `data/raw_csv/`、`data/curve_index.csv` | 1842 份原始提取 CSV、来源及 SHA-256 对照；用于追溯，模型不直接读取 |
| `config.json`、`run.py` | 固定参数及四簇拟合代码 |
| `render_si_zoom.py` | 依据本次拟合结果生成 zoom in 补充图 S1 |
| `verify.py` | 输入完整性和可选输出核验 |
| `requirements.txt` | 复现环境中的 Python 库版本 |

## 运行

建议使用 Python 3.9 和隔离环境；先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

在**本目录**执行：

```bash
python3 verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run.py
python3 verify.py --check-results
```

`run.py` 默认在本目录**旁边**新建 `SI_1842_reproduction_outputs/`，写入逐曲线簇号、数值模型、代表曲线表、图和摘要；其中 `figures/Supplementary_Fig_S1_zoom.png` 和 `.svg` 就是之前的 zoom in 图。该图使用本次拟合的曲线和簇号重新生成：簇 0 的中位趋势线仅为展示使用 `sigma=3.0` 平滑，其余簇使用 `sigma=1.2`，各面板分别放大纵轴。这些绘图设置不改变聚类。核心目录始终不接收结果文件。输出位置可在 `config.json` 的 `output_dir` 中修改。预期纳入 1842 条，匿名簇大小为 `1710/44/21/67`。

交付给别人时，如需同时附上已生成的 zoom in 图，把核心目录与旁边的 `SI_1842_reproduction_outputs/` 一起交付；只交付核心目录，对方也可以运行上述命令自行生成。

## 范围与限制

本包重跑的是**已经固定的 1842 条曲线**，不是从论文图片重新数字化、从更大的候选池重新筛选曲线的端到端流程。原始 CSV 保留原横轴单位；模型实际读取预先整理的早期归一化点与 64 点序位输入，原始 CSV 不能直接替代它们。输入筛选曾使用旧的暂定形状候选，因此四个匿名簇是探索性分析，不能作为独立验证的真实类别或分类准确率。原始交付中的噪声敏感性分析及结果表不属于此精简包。
