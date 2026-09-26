# 固定输入与软件环境

## Python环境

- Python：3.9.6
- 依赖版本：见 `requirements-lock.txt`

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r environment/requirements-lock.txt
```

## 固定输入

为避免在结果包中重复存放约104 MB的完整预处理目录，输入数据仍保留在项目统一数据目录中：

```text
../03_preprocessed/main_high_quality_min10
```

相对于本文件所在的11结果目录，其绝对路径为：

```text
/Users/shunhao/Desktop/ML/lab/NEW/som_200h_reanalysis/03_preprocessed/main_high_quality_min10
```

复现分类所需的最小输入为：

| 文件 | SHA-256 |
| --- | --- |
| `X_smoothed.npy` | `8c9b473c1b3a56ffcda2d0b1464bcaf3f969a3cbead871f6cb9103cad8dd0096` |
| `time_grid.csv` | `5395426af31831ac13ab63ceafc1d52831333fb8c8aa59818dbab5058f2cebbd` |
| `curve_metadata.csv` | `c6d076940a5f0a4084c33ab6adc36b991ee4308754c7f4f4a66a9a0bb4d3d6c0` |
| `preprocessing_config.json` | `386d296d6db17d9c93b81ebb4055f23c448607992ad9e608ad5b26159ee7ba58` |

完整参数扫描脚本在绘制原始点对照图时，还会依据 `curve_metadata.csv` 中的来源路径读取原始曲线；如果元数据缺少时间单位列，则从以下文件恢复：

```text
../01_data_selection/main_selected_curves.csv
```

`reproduce_selected_36_model.py` 只复现最终分类、权重与指标，不需要重新读取原始曲线文件。

## 输入一致性

复现脚本会在训练前检查：

- `X_smoothed.npy` 的SHA-256；
- 输入形状必须为 `1442 × 1201`；
- 时间网格和元数据行列数必须一致；
- 按固定规则重新计算的RTF-core数量必须为56。

任何一项不同都会停止运行，从而避免在输入已经变化的情况下误称为“复现”。
