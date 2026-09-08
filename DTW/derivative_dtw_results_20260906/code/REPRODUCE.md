# 复现与文件格式

工作目录可以任意；脚本根据自身路径定位输出目录以及旁边的 `samples_test/`。原始数据不写入。依赖版本记录在 `runtime_environment.json`，无需 DTW 专用库；精确 DTW 核心使用 C++ 加速，Python 实现独立核验。

在 macOS 上：

```sh
cd /Users/shunhao/Desktop/ML/method2/derivative_dtw_results_20260906
clang++ -O3 -std=c++11 -dynamiclib code/dtw_core.cpp -o code/libdtw.dylib
python3 -u code/run_analysis.py
python3 -u code/finalize.py
python3 -u code/check_final_k.py
python3 code/write_report.py
python3 code/verify_results.py
```

在 Linux 上编译时改用 `-shared -fPIC`，并相应修改 Python 加载库的文件名。依赖：numpy、pandas、scipy、scikit-learn、matplotlib；准确运行版本见环境 JSON。运行会覆盖本结果目录内的计算产物；如需保留该次结果，先复制整个结果文件夹。

- `aligned/*_matrices.npz`：`ids`、`time_h`、`y_normalized`；每一行一条真实不等长序列，尾部 NaN 仅用于矩阵存储，不是插值点。
- `aligned/*_normalized_long.csv`：每个窗口/归一化的完整长表。
- `features/*_derivatives.csv`：每条曲线各相邻观测区间的右端时刻、真实 dt、有限差分斜率。
- `distance_matrices/*.npz`：`distance` 为 NxN 矩阵，`ids` 指定行列顺序。
- `clustering_results/labels_all_K2_to_K10.npz`：键为 Mxxxxxx 候选 ID；标签从 0 起。用 `internal_metrics_all.csv` 映射候选参数及窗口，再用相应距离文件 `ids` 映射样本。
- 面向阅读的 `final_labels.csv`、`final_parameters_labels_K2_to_K10.csv` 标签从 1 起。所有 Cxxx 样本 ID 用 `preprocessing_audit.csv` 连接原始路径。
- `stability/*_consensus.npz`：`consensus`、`numerator`、`denominator`、`ids`；缺乏共同观测的对子保留 NaN。
- `state.pkl` 只是本次执行生成的可信本地中间状态，供后处理继续使用；完整原始/数值结果均另有 CSV/NPZ，不依赖读取 pickle 才能审阅。

主分析没有平滑。公共网格、三点斜率、绝对时间窗口、重复时间戳回纳等仅见冻结后敏感性结果。`PROTOCOL.md` 解释在执行前已写入主代码的筛选与评分规则。最终人工形态描述位于报告生成脚本中，原始数据更改后必须重审这些描述和报告中的固定数字，不能把报告生成器当作任意数据集的自动结论程序。
