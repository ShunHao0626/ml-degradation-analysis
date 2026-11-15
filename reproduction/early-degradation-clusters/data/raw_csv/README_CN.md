# 原始曲线与论文图片

本目录按 DOI 存放 1204 个来源文件夹。每个文件夹包含已提取曲线的原始 CSV、对应论文图 PNG 和提取后的 `_replot.png` 图片。CSV 保留原横轴单位；模型读取的是上一级 `shape_inputs.npz` 中已对齐的 64 点形状输入，而不是直接读取这些原始 CSV。

本目录的 `curve_index.csv` 为展示页面提供曲线 ID、四簇编号、DOI、原始来源路径、本目录中的 CSV/论文图/replot 路径，以及三种文件的 SHA-256。这里的簇编号来自已拟合结果，**不是独立验证的真实类别**。同一张论文图可对应多条曲线，因此图片可能被多行索引共用。

在仓库根目录运行 `python3 code/verify.py` 可核验固定输入的 ID、原始 CSV 哈希和形状矩阵；`python3 code/verify.py --check-results` 还会核验生成的四簇结果和来源副本。
