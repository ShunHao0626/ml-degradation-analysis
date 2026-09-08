# 方法与决策记录

## 数据边界

唯一输入根目录：`/Users/shunhao/Desktop/ML/lab-v2/som_references/accepted`。程序递归发现 218 条 CSV。`samples_test_png` 仅含图片，不作为数值输入。循环轴、轴单位无法确认、纵轴不能确认为 PCE/效率/功率以及明显损坏的时间尺度均排除；每条记录保留在审计表中。

## 第一阶段：论文 SOM

保持论文预处理：10min 网格、Akima、逐曲线 MaxAbs、Savitzky–Golay(71,2)、2×2 SOM、MiniSom 2.2.9。先搜索论文正文/补充材料出现的 sigma={0.3,0.5} 与 learning_rate={0.1,0.3}，再扩展 sigma、learning rate、iterations 和窗口。固定 seed=0 仅用于公平比较候选，并对前 8 名用 5 个 seed 做稳定性评估。

四原型硬门槛结果：`False`。详情见 `03_paper_som_search/paper_som_assessment.json`。

## 第二阶段：形态引导分类

最终选择窗口：750h。前 200h 被定义为快速变化相位，200h 后用于慢衰减判断。每条曲线提取：早期增益、峰后下降、早期下降、谷后恢复及其相对振幅、峰/谷/恢复的时间顺序、早晚斜率与负斜率比例。

- Bridge：200h 内存在显著上升，之后慢衰减，峰后下降不满足 Hill 的强衰减条件；
- Hill：200h 内显著上升，且峰后下降强、末值低于初值，后段斜率为负；
- Slope：未出现可靠上升或谷后恢复，以快速下降后慢衰减为主；
- Valley：200h 内先出现谷值，之后经过最小相位间隔出现显著恢复；恢复峰不能位于窗口末端，且恢复峰之后必须再次衰减。

程序保留 `final_class` 四向兼容标签，同时生成更保守的 `strict_class`。Slope 只有在逐条满足早期下降、恢复有限、初始增益有限且后段不再上升时才进入严格集合；其余曲线写为 `unresolved_other_shape`，防止默认剩余类污染 Slope。

最终参数：

```json
{
  "rise_gain_abs": 0.002,
  "rise_gain_ratio": 0.2,
  "hill_rapid_drop_ratio": 0.2,
  "valley_drop_ratio": 0.22,
  "valley_recovery_ratio": 0.35,
  "valley_recovery_abs": 0.006,
  "valley_post_recovery_drop_ratio": 0.2,
  "minimum_phase_gap": 0.06,
  "latest_recovery_fraction": 0.85
}
```

合成四原型只用于候选评分和命名校验，不参与模型拟合，也没有被加入真实数据矩阵。
