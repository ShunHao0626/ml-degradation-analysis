# EG 数据集 DTW × 导数融合实验

该实验直接读取 `../eg_based_synthetic_dataset`，比较以下固定消融：

- 原始幅值 DTW；
- 一、二阶导数 DTW；
- 幅值与一、二阶导数共享路径的多通道 DTW；
- 幅值 DTW 与导数 DTW 的分距离融合；
- 加入三阶导数后的对应方案。

内部相对进程坐标只用于形态比较；原始文件中的 `time_h` 和 `duration_h` 不会被覆盖。
生成标签、形态质检和异常点标记不参与距离计算或聚类。

运行：

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/pycache \
python3 eg_dtw_derivative_experiment/run_experiment.py --jobs 4
```

若距离矩阵已经存在，只重跑聚类和报告：

```bash
python3 eg_dtw_derivative_experiment/run_experiment.py --reuse-distances
```

测试：

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
python3 -m pytest -q eg_dtw_derivative_experiment/test_experiment.py
```

结果写入 `outputs/`，其中 `REPORT_CN.md` 是中文结论，`metrics.csv` 是全数据方法比较，
`duration_regime_metrics.csv` 是四个实际时长层内的敏感性检查。
`knn_sensitivity.csv` 比较 5、10、15、20、30 个近邻，避免结论只依赖单一图参数。
