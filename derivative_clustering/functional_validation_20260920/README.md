# 四种 IFO 形态的 SOM / 导数功能验证

先读 `REPORT_CN.md`；运行前固定的实验设置见 `PROTOCOL.md`。

## 已完成的验证

现有 400 条曲线 + 15 个模拟/压力批次，共 16 批、4,360 个观测实例。
同一潜在曲线会出现在完整、稀疏、截断和混合情景中，不能视作 4,360 个独立样本。
主对照的 k=4 是已知四类的功能测试假设；混合背景另有 k=8 诊断。
所有新批次参数在运行前固定；无根据结果重新调参或筛去形态不合格样本。

结果包含负结果：本轮直接拼接导数的 SOM 低于幅值 SOM；论文式高阶多项式
加 single linkage 经常退化；背景形态会使强制四簇的目标召回率和精确率下降。

## 环境

Python 3.9+，numpy、scipy、pandas、scikit-learn、matplotlib、MiniSom、pytest，
以及 clang++。本机实际版本在 `outputs/manifest.json`。首次运行将自动编译
`dtw.cpp` 为本机 `_dtw.so`；这只是加速，递推与原 Python 实现一致。

## 完整复跑

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 functional_validation_20260920/validate.py

python3 -m pytest -q functional_validation_20260920/test_validation.py \
  eg_dtw_derivative_experiment/test_experiment.py tests

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 functional_validation_20260920/validate.py \
  --input-curves eg_based_synthetic_dataset/curves \
  --output functional_validation_20260920/cli_smoke

python3 functional_validation_20260920/verify_outputs.py
```

完整复跑会更新本目录 `outputs`。若想保留本次结果，使用 `--output 新目录`。
`verify_outputs.py` 核验本次固定默认 16 批结果，包含原输入哈希、矩阵性质、
214 行 ARI 重算、汇总均值和无标签 CLI 一致性核对。它不是未来真实数据效果验证器。

## 对新的 CSV 目录做匿名聚类

```bash
cd /Users/shunhao/Desktop/ML/derivative_clustering
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 functional_validation_20260920/validate.py \
  --input-curves /实际的曲线CSV目录 \
  --output /实际的结果目录
```

一文件一曲线；横坐标必须是已核对单位的 `time_h`，纵坐标为 `y_relative`、
`normalized_pce`、`y` 三者之一。CSV 目录需要至少 11 条曲线，每条至少 5 个
不同的时间点。相同时间的响应取中位数；非有限值或全零响应报错。
入口仅读取横纵坐标列，不读取外部标签文件、不根据目录类名命名簇。

输出 `anonymous_clusters.csv`、`quality.csv`、`distances.npz`，各方法给出匿名簇号。
这个入口仍固定四簇，适合功能测试；不要把它直接当成真实混合数据的最终四类检索器。
`quality.csv` 中稀疏警告仅供检查，不应解释成自动拒识或分类置信度。

## 复用函数

`validate.fit_curves(list_of_dataframes)` 返回各方法的匿名簇编号、距离矩阵、
平滑后的形态通道、观测质量信息、图连通分量数和多项式选阶。
没有 `target_class` 或其他标签参数。归一化、通道缩放和聚类都是在当前批次
无标签拟合；这不是一个固定训练模型的 `.predict()` 接口。

完整结论、论文方法差别、错误解释风险和下一版数据的实验步骤见 `REPORT_CN.md`。
