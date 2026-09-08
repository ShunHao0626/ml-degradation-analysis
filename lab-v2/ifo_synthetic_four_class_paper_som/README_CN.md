# ifo_synthetic_four_class_paper_som

这是四类 IFO 拓扑的受控合成数据与论文参数 SOM 可恢复性实验。先阅读 `REPORT_CN.md`，复现执行 `python3 run_pipeline.py`，结果核验执行 `python3 verify_outputs.py`。

注意：合成真值从不进入 SOM 训练，只在训练结束后用于节点命名和评价。
