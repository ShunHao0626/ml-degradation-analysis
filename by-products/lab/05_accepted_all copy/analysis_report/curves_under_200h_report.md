# 未超过 200 小时曲线汇总报告

生成时间: 2026-07-28（数据来自 `x_time_day`、`x_time_h`、`x_time_min`、`x_time_week_month_year` 四个目录）

## 一、扫描统计

| 指标 | 数量 |
|------|------|
| 总曲线数（CSV 数据序列数） | 2151 |
| 总 (DOI, Figure) 数（去重） | 1372 |
| 唯一 DOI 总数 | 1346 |
| **最大时间 < 200 h 的曲线数** | **293** |
| **最大时间 ≥ 200 h 的曲线数** | **1858** |
| **未达标的唯一 DOI 数** | **213** |
| **未达标的唯一 (DOI, Figure) 数** | **215** |

## 二、按目录划分

| 目录 | < 200 h 曲线数 | 总曲线数 | < 200 h 唯一 DOI 数 | < 200 h 唯一 (DOI, Figure) 对数 |
|------|---------------|---------|------------------|----------------------------|
| `x_time_day` | 40 | 663 | 33 | 33 |
| `x_time_h` | 251 | 1480 | 179 | 181 |
| `x_time_min` | 2 | 2 | 1 | 1 |
| `x_time_week_month_year` | 0 | 6 | 0 | 0 |

> 备注：`x_time_week_month_year` 中存放的曲线虽然在数据中以"小数值"出现（≤ 5），但根据 `validation_result.json` 中的单位（months / weeks）换算后均**已超过** 200 小时，因此未列入下方"未达标"清单。

## 三、按 (DOI, Figure) 汇总的最大时间未超过 200 h 的曲线（按最大时间升序）

| DOI | Figure | 涉及子图 | 单位 | 最大时间 | 源目录 |
|-----|--------|----------|------|----------|--------|
| 10.1002/adma.201705596 | Fig5 | _D | hours | 44.63 h | x_time_h |
| 10.1016/j.solmat.2019.110284 | Fig3 | _B | days | 45.35 h | x_time_day |
| 10.1039/c5ee03255e | Fig6 | _D | hours | 46.60 h | x_time_h |
| 10.1002/solr.201900167 | Fig4 | _D | hours | 46.84 h | x_time_h |
| 10.1002/aenm.201602120 | Fig4 | _A | hours | 47.72 h | x_time_h |
| 10.1016/j.nanoen.2022.107818 | Fig5 | _E | hours | 49.22 h | x_time_h |
| 10.1038/ncomms10228 | Fig5 | _A | hours | 54.39 h | x_time_h |
| 10.1002/adfm.201807850 | Fig7 | _F | hours | 58.44 h | x_time_h |
| 10.1039/d4ee02427c | Fig4 | _H | hours | 60.95 h | x_time_h |
| 10.1039/d0ta12286f | Fig5 | _A, _B | hours | 71.36 h | x_time_h |
| 10.1039/d0ta01860k | Fig5 | _B | hours | 74.13 h | x_time_h |
| 10.1016/j.xcrp.2020.100224 | Fig5 | _B, _D | hours | 74.49 h | x_time_h |
| 10.1002/adfm.202002964 | Fig5 | _C | hours | 78.31 h | x_time_h |
| 10.1021/jacs.6b10227 | Fig3 | _D | hours | 79.31 h | x_time_h |
| 10.1021/acs.chemmater.5b03137 | Fig3 | _B | days | 81.53 h | x_time_day |
| 10.1002/aenm.202301218 | Fig3 | _E | hours | 88.75 h | x_time_h |
| 10.1016/j.nanoen.2015.10.010 | Fig7 | _D | hours | 88.88 h | x_time_h |
| 10.1002/smll.201907283 | Fig6 | _D | hours | 94.68 h | x_time_h |
| 10.1039/d0ra09294k | Fig5 | _C | days | 95.14 h | x_time_day |
| 10.1016/j.nanoen.2021.105751 | Fig4 | _B | hours | 95.25 h | x_time_h |
| 10.1021/acsaem.0c01796 | Fig10 | Fig10 | hours | 95.28 h | x_time_h |
| 10.1039/d3ee03687a | Fig5 | _B | hours | 99.00 h | x_time_h |
| 10.1002/eem2.12089 | Fig5 | _B | hours | 99.52 h | x_time_h |
| 10.1021/acsenergylett.7b00375 | Fig4 | _C | hours | 100.01 h | x_time_h |
| 10.1002/pssr.202000062 | Fig4 | _B | hours | 100.06 h | x_time_h |
| 10.1016/j.orgel.2018.09.019 | Fig4 | _A | hours | 100.39 h | x_time_h |
| 10.1002/smll.201904399 | Fig6 | _C | hours | 101.57 h | x_time_h |
| 10.1016/j.nanoen.2016.09.041 | Fig2 | _A | hours | 103.16 h | x_time_h |
| 10.1016/j.jpowsour.2021.230534 | Fig4 | _C | hours | 104.67 h | x_time_h |
| 10.1021/acsami.0c02751 | Fig9 | _A | hours | 104.81 h | x_time_h |
| 10.1016/j.nanoen.2021.106157 | Fig3 | _G | hours | 105.25 h | x_time_h |
| 10.1016/j.cej.2021.130685 | Fig5 | _F | hours | 105.80 h | x_time_h |
| 10.1021/acsami.9b11229 | Fig5 | _B, _C, _D | hours | 106.97 h | x_time_h |
| 10.3390/coatings11060627 | Fig3 | _A | hours | 107.31 h | x_time_h |
| 10.1002/aenm.201700226 | Fig4 | _A | days | 107.51 h | x_time_day |
| 10.1002/advs.201901241 | Fig5 | _C, _D | hours | 108.97 h | x_time_h |
| 10.1002/adfm.202101163 | Fig7 | _B | hours | 109.10 h | x_time_h |
| 10.1039/d4ee02427c | Fig3 | _M | hours | 109.29 h | x_time_h |
| 10.1002/solr.201700245 | Fig6 | Fig6 | days | 110.75 h | x_time_day |
| 10.1016/j.nanoen.2014.12.022 | Fig4 | _A | hours | 115.58 h | x_time_h |
| 10.1039/c9ta08314f | Fig5 | _D | hours | 117.13 h | x_time_h |
| 10.1016/j.nanoen.2019.103964 | Fig4 | _B | hours | 117.34 h | x_time_h |
| 10.1016/j.solener.2016.10.005 | Fig7 | Fig7 | days | 117.42 h | x_time_day |
| 10.1016/j.orgel.2019.03.036 | Fig5 | _C, _D | hours | 117.49 h | x_time_h |
| 10.1016/j.jechem.2021.08.021 | Fig5 | _G | hours | 117.70 h | x_time_h |
| 10.1002/aenm.201803287 | Fig6 | _B | hours | 117.96 h | x_time_h |
| 10.1039/c5ta09080f | Fig6 | _C | hours | 118.05 h | x_time_h |
| 10.1002/adem.202000990 | Fig5 | _C | hours | 118.07 h | x_time_h |
| 10.1016/j.electacta.2018.07.028 | Fig2 | _A | days | 118.20 h | x_time_day |
| 10.1126/science.aam6620 | Fig4 | _A | hours | 118.32 h | x_time_h |
| 10.1002/aenm.201802774 | Fig3 | _C | hours | 118.38 h | x_time_h |
| 10.1007/s40820-022-00995-2 | Fig5 | _F | hours | 118.53 h | x_time_h |
| 10.1021/acsenergylett.1c01446 | Fig5 | Fig5 | hours | 118.58 h | x_time_h |
| 10.1039/d1na00172h | Fig6 | _B | hours | 118.67 h | x_time_h |
| 10.1016/j.jpowsour.2021.229781 | Fig4 | _B | hours | 118.74 h | x_time_h |
| 10.1002/solr.201900319 | Fig4 | _F | hours | 118.99 h | x_time_h |
| 10.1021/jacs.0c13069 | Fig4 | _C | hours | 119.33 h | x_time_h |
| 10.1016/j.solmat.2017.12.015 | Fig8 | _D | hours | 119.75 h | x_time_h |
| 10.1021/acsenergylett.8b00085 | Fig2 | _F | days | 119.83 h | x_time_day |
| 10.1021/acs.langmuir.8b01650 | Fig6 | Fig6 | hours | 119.84 h | x_time_h |
| 10.1021/acsaem.0c00584 | Fig3 | _D | hours | 119.96 h | x_time_h |
| 10.1002/solr.201800290 | Fig7 | _A | hours | 119.98 h | x_time_h |
| 10.1002/solr.202000164 | Fig4 | _C | hours | 120.05 h | x_time_h |
| 10.1016/j.apsusc.2017.02.057 | Fig4 | _D | day | 120.08 h | x_time_day |
| 10.1038/s41467-018-05760-x | Fig6 | _B | hours | 120.12 h | x_time_h |
| 10.1002/advs.201903250 | Fig5 | _D, _E | hours | 120.17 h | x_time_h |
| 10.1002/adfm.202002230 | Fig6 | _B | hours | 120.70 h | x_time_h |
| 10.1016/j.flatc.2021.100254 | Fig10 | Fig10 | hours | 120.91 h | x_time_h |
| 10.1016/j.nanoen.2019.104244 | Fig5 | _B | hours | 124.72 h | x_time_h |
| 10.1007/s10853-020-04965-0 | Fig8 | _B | hours | 125.84 h | x_time_h |
| 10.1039/c7ta09716f | Fig7 | _B | hours | 128.48 h | x_time_h |
| 10.1039/d5ee01110h | Fig1 | _E | hours | 133.98 h | x_time_h |
| 10.1039/c7ee02272g | Fig2 | _E | hours | 135.14 h | x_time_h |
| 10.1039/d4ra08786k | Fig7 | _B | hours | 135.25 h | x_time_h |
| 10.1021/acsami.0c11247 | Fig5 | _A | hours | 136.38 h | x_time_h |
| 10.1039/d4ee03001j | Fig5 | _F | hours | 137.86 h | x_time_h |
| 10.1021/acsami.0c00142 | Fig4 | _F | hours | 138.10 h | x_time_h |
| 10.1038/s41598-020-79348-1 | Fig4 | _A | hours | 138.13 h | x_time_h |
| 10.1021/jacs.7b07949 | Fig4 | _B | hours | 138.72 h | x_time_h |
| 10.1021/acs.chemmater.0c00995 | Fig6 | _D | hours | 140.06 h | x_time_h |
| 10.1002/aenm.201903654 | Fig5 | _E | hours | 141.44 h | x_time_h |
| 10.1016/j.jpowsour.2020.229345 | Fig8 | _D | hours | 142.00 h | x_time_h |
| 10.1039/d1ta05699a | Fig5 | _E | hours | 142.85 h | x_time_h |
| 10.1002/smll.202101380 | Fig5 | _H | hours | 142.87 h | x_time_h |
| 10.1002/admi.201500799 | Fig7 | Fig7 | hours | 143.04 h | x_time_h |
| 10.1002/adma.201907623 | Fig5 | _C | hours | 143.15 h | x_time_h |
| 10.1002/adfm.201503559 | Fig6 | Fig6 | hours | 143.36 h | x_time_h |
| 10.1016/j.orgel.2018.02.026 | Fig8 | _A | hours | 143.79 h | x_time_h |
| 10.1021/acsaem.9b01652 | Fig6 | _B | hours | 147.18 h | x_time_h |
| 10.1039/c9tc01741k | Fig7 | _B | hours | 147.24 h | x_time_h |
| 10.1039/d0sc06354a | Fig6 | _D | hours | 147.55 h | x_time_h |
| 10.1021/acsenergylett.0c00596 | Fig4 | _C | hours | 147.83 h | x_time_h |
| 10.1016/j.solener.2021.01.030 | Fig6 | _A | hours | 148.06 h | x_time_h |
| 10.1039/d0se00528b | Fig6 | _A | hours | 148.30 h | x_time_h |
| 10.1039/d3ta07851e | Fig3 | _E | hours | 148.31 h | x_time_h |
| 10.1039/d3ta01935g | Fig5 | _F | hours | 148.80 h | x_time_h |
| 10.1039/d1na00173f | Fig5 | _F | hours | 148.84 h | x_time_h |
| 10.1016/j.jechem.2019.10.006 | Fig4 | _F | hours | 148.90 h | x_time_h |
| 10.1039/C9TA03131F | Fig5 | _D | hours | 149.12 h | x_time_h |
| 10.1016/j.solmat.2018.07.009 | Fig5 | _F | days | 149.17 h | x_time_day |
| 10.1016/j.jallcom.2021.159658 | Fig23 | Fig23 | hours | 149.67 h | x_time_h |
| 10.1002/smtd.201900652 | Fig5 | _E | hours | 151.49 h | x_time_h |
| 10.1016/j.xcrp.2021.100395 | Fig6 | _F | hours | 152.52 h | x_time_h |
| 10.1039/c9ta09260a | Fig4 | _I | hours | 152.60 h | x_time_h |
| 10.1002/solr.202100320 | Fig7 | _F | hours | 156.82 h | x_time_h |
| 10.1002/aenm.201902239 | Fig5 | _C | hours | 157.20 h | x_time_h |
| 10.1016/j.jmst.2021.05.084 | Fig5 | _B | hours | 158.15 h | x_time_h |
| 10.1021/acsaem.0c00811 | Fig5 | _B | hours | 159.37 h | x_time_h |
| 10.1016/j.jmat.2021.08.007 | Fig6 | _A | hours | 160.23 h | x_time_h |
| 10.1021/acs.chemmater.1c00662 | Fig4 | _B | hours | 162.06 h | x_time_h |
| 10.1002/adma.201903448 | Fig3 | _G | hours | 163.46 h | x_time_h |
| 10.1039/c7ta09543k | Fig5 | _D | days | 164.11 h | x_time_day |
| 10.1039/C9SC01697J | Fig6 | _E | hours | 164.27 h | x_time_h |
| 10.1021/acsenergylett.7b00508 | Fig4 | _F | hours | 164.68 h | x_time_h |
| 10.1039/c7nr03507a | Fig5 | _A | days | 164.75 h | x_time_day |
| 10.1002/adma.201907396 | Fig5 | _E | hours | 164.96 h | x_time_h |
| 10.1016/j.nanoen.2020.105462 | Fig5 | _B, _C | hours | 165.08 h | x_time_h |
| 10.1002/advs.201700025 | Fig7 | Fig7 | days | 165.08 h | x_time_day |
| 10.1016/j.cplett.2021.138496 | Fig5 | _B | days | 165.25 h | x_time_day |
| 10.1016/j.jallcom.2021.161455 | Fig5 | _D | days | 165.35 h | x_time_day |
| 10.1021/acs.jpcc.0c02628 | Fig6 | _B | days | 165.54 h | x_time_day |
| 10.1016/j.solmat.2016.12.025 | Fig3 | _C | days | 165.57 h | x_time_day |
| 10.1002/adma.202003965 | Fig6 | _B | hours | 166.15 h | x_time_h |
| 10.1039/c6ta10588b | Fig6 | Fig6 | days | 166.16 h | x_time_day |
| 10.1016/j.jmrt.2021.12.012 | Fig14 | _D | hours | 166.28 h | x_time_h |
| 10.1109/jphotov.2020.2987390 | Fig4 | _D | hours | 166.31 h | x_time_h |
| 10.1016/j.electacta.2016.03.002 | Fig7 | _A | hours | 166.38 h | x_time_h |
| 10.1021/acsami.8b12675 | Fig6 | Fig6 | hours | 166.56 h | x_time_h |
| 10.1126/science.aad1015 | Fig4 | _C | days | 166.71 h | x_time_day |
| 10.3390/pr10081477 | Fig6 | Fig6 | days | 166.71 h | x_time_day |
| 10.1016/j.jallcom.2020.157396 | Fig6 | _B | days | 166.79 h | x_time_day |
| 10.1002/adma.202003296 | Fig5 | _B | hours | 166.91 h | x_time_h |
| 10.1016/j.solmat.2017.07.022 | Fig7 | Fig7 | days | 167.03 h | x_time_day |
| 10.1021/acsami.0c05527 | Fig7 | _C | days | 167.09 h | x_time_day |
| 10.1016/j.mssp.2018.10.034 | Fig12 | Fig12 | days | 167.23 h | x_time_day |
| 10.1039/d0ta06033j | Fig4 | _F | hours | 167.27 h | x_time_h |
| 10.1016/j.joule.2018.11.026 | Fig6 | _B, _C | hours | 167.40 h | x_time_h |
| 10.1016/j.orgel.2021.106244 | Fig7 | Fig7 | hours | 167.45 h | x_time_h |
| 10.1039/d0dt03201h | Fig6 | _G | days | 167.69 h | x_time_day |
| 10.1021/acsaem.8b00602 | Fig8 | Fig8 | hours | 167.71 h | x_time_h |
| 10.1016/j.carbon.2019.10.101 | Fig4 | _A | days | 167.79 h | x_time_day |
| 10.1016/j.apsusc.2019.143552 | Fig10 | Fig10 | hours | 167.91 h | x_time_h |
| 10.1016/j.apsusc.2018.09.080 | Fig16 | Fig16 | hours | 167.96 h | x_time_h |
| 10.1016/j.heliyon.2024.e25352 | Fig6 | Fig6 | hours | 168.00 h | x_time_h |
| 10.1016/j.electacta.2018.04.055 | Fig4 | _D | days | 168.24 h | x_time_day |
| 10.1002/adfm.201600746 | Fig6 | _A | hours | 168.48 h | x_time_h |
| 10.1021/acsami.0c17141 | Fig6 | _B | hours | 169.55 h | x_time_h |
| 10.1016/j.solmat.2015.10.008 | Fig6 | _D | hours | 170.38 h | x_time_h |
| 10.1021/acsami.7b06681 | Fig6 | _D | hours | 172.42 h | x_time_h |
| 10.1016/j.joule.2020.12.003 | Fig6 | _C | hours | 173.58 h | x_time_h |
| 10.1002/admi.202000105 | Fig5 | _D | hours | 174.11 h | x_time_h |
| 10.1021/acsenergylett.0c01207 | Fig5 | _A | minutes | 175.40 h | x_time_min |
| 10.1039/c9ta13922b | Fig4 | _H | hours | 175.41 h | x_time_h |
| 10.1039/c6ta06152d | Fig5 | _A | hours | 176.41 h | x_time_h |
| 10.1002/aenm.202301218 | Fig5 | _F | hours | 176.51 h | x_time_h |
| 10.1016/j.nanoen.2017.11.008 | Fig3 | _F | hours | 176.56 h | x_time_h |
| 10.1039/d4su00641k | Fig1 | Fig1 | hours | 177.25 h | x_time_h |
| 10.1002/adma.201905766 | Fig4 | _E | hours | 177.99 h | x_time_h |
| 10.1039/d4ee05699j | Fig4 | _I | hours | 184.83 h | x_time_h |
| 10.1021/acsami.6b15563 | Fig5 | _F | days | 186.40 h | x_time_day |
| 10.1126/sciadv.aav8925 | Fig5 | _B | hours | 186.82 h | x_time_h |
| 10.1039/c9ta05802h | Fig5 | _A | hours | 187.18 h | x_time_h |
| 10.1016/j.solener.2021.06.076 | Fig9 | _A | hours | 187.20 h | x_time_h |
| 10.1016/j.jallcom.2020.157784 | Fig7 | _H | hours | 187.31 h | x_time_h |
| 10.1016/j.electacta.2017.07.040 | Fig8 | Fig8 | days | 188.22 h | x_time_day |
| 10.1038/s41586-019-1036-3 | Fig4 | _A | hours | 188.27 h | x_time_h |
| 10.1016/j.orgel.2019.03.033 | Fig7 | _C | hours | 188.40 h | x_time_h |
| 10.1002/admi.201800499 | Fig6 | _E | days | 188.50 h | x_time_day |
| 10.1016/j.nanoen.2021.106445 | Fig5 | _C | hours | 188.78 h | x_time_h |
| 10.1039/c8nh00163d | Fig4 | _A | days | 188.82 h | x_time_day |
| 10.1016/j.nanoen.2016.08.055 | Fig9 | _A | days | 189.58 h | x_time_day |
| 10.3390/coatings9110766 | Fig8 | _D | hours | 189.76 h | x_time_h |
| 10.1016/j.electacta.2016.09.138 | Fig6 | _D | hours | 189.88 h | x_time_h |
| 10.1016/j.jpowsour.2022.231243 | Fig6 | _B | days | 190.12 h | x_time_day |
| 10.1007/s40843-019-1174-3 | Fig7 | _D | hours | 190.51 h | x_time_h |
| 10.1016/j.solmat.2019.04.007 | Fig6 | _B | hours | 190.71 h | x_time_h |
| 10.1039/c8ta11977e | Fig8 | _A | hours | 190.72 h | x_time_h |
| 10.1038/s41598-017-00866-6 | Fig1 | _A, _B | hours | 191.07 h | x_time_h |
| 10.1016/j.orgel.2019.105384 | Fig5 | Fig5 | hours | 191.27 h | x_time_h |
| 10.1039/c8ta06557h | Fig6 | _E, _F | hours | 191.35 h | x_time_h |
| 10.1002/aenm.201601156 | Fig4 | _A | hours | 191.62 h | x_time_h |
| 10.1039/c9ta05744g | Fig6 | _B | hours | 191.92 h | x_time_h |
| 10.1039/c7tc00882a | Fig7 | Fig7 | hours | 192.31 h | x_time_h |
| 10.1039/c6ee00612d | Fig2 | _E | hours | 192.86 h | x_time_h |
| 10.1002/solr.202000621 | Fig6 | _B | hours | 193.09 h | x_time_h |
| 10.1002/celc.201701054 | Fig7 | _B | hours | 193.35 h | x_time_h |
| 10.1002/smtd.201900476 | Fig5 | _D | hours | 193.37 h | x_time_h |
| 10.1002/adma.202205769 | Fig4 | _B | hours | 193.84 h | x_time_h |
| 10.1021/jacs.7b13229 | Fig3 | _D | hours | 194.03 h | x_time_h |
| 10.1016/j.nanoen.2020.104929 | Fig5 | _D | hours | 194.05 h | x_time_h |
| 10.1039/c9ta05422g | Fig9 | _F | hours | 195.28 h | x_time_h |
| 10.1002/adma.201701221 | Fig5 | _B, _C | hours | 195.69 h | x_time_h |
| 10.1016/j.jechem.2021.05.042 | Fig6 | _F | hours | 195.78 h | x_time_h |
| 10.1016/j.nanoen.2020.105181 | Fig6 | _B | hours | 196.38 h | x_time_h |
| 10.1016/j.xcrp.2021.100450 | Fig5 | _F | hours | 196.53 h | x_time_h |
| 10.1039/c8ta02121j | Fig6 | _C | hours | 196.90 h | x_time_h |
| 10.1016/j.solmat.2019.110383 | Fig3 | Fig3 | hours | 197.20 h | x_time_h |
| 10.1021/acsami.9b22627 | Fig9 | _C | hours | 197.20 h | x_time_h |
| 10.1039/c7ta04851c | Fig5 | _D | hours | 197.27 h | x_time_h |
| 10.3390/cryst12091194 | Fig5 | _D | hours | 197.48 h | x_time_h |
| 10.1039/c9tc06578d | Fig5 | _B | hours | 197.71 h | x_time_h |
| 10.1021/acsenergylett.1c00443 | Fig3 | _I | hours | 197.92 h | x_time_h |
| 10.1039/d0ta11916d | Fig5 | _C | hours | 198.22 h | x_time_h |
| 10.1039/d0ta05676f | Fig7 | _C | hours | 198.23 h | x_time_h |
| 10.1002/aenm.201900243 | Fig6 | _C | hours | 198.29 h | x_time_h |
| 10.1016/j.nanoen.2018.09.037 | Fig6 | _D | hours | 198.37 h | x_time_h |
| 10.1021/acsnano.6b01904 | Fig8 | _B | hours | 198.50 h | x_time_h |
| 10.1016/j.solener.2020.01.048 | Fig4 | _D | hours | 198.63 h | x_time_h |
| 10.1002/admi.202001144 | Fig3 | _B | hours | 198.70 h | x_time_h |
| 10.1016/j.solmat.2019.110335 | Fig1 | _E, _F | hours | 198.83 h | x_time_h |
| 10.1016/j.jpcs.2020.109792 | Fig5 | _C | hours | 198.96 h | x_time_h |
| 10.1002/solr.202100899 | Fig5 | _B, _C, _D | hours | 198.97 h | x_time_h |
| 10.1002/advs.201903047 | Fig4 | _D | hours | 198.99 h | x_time_h |
| 10.1016/j.jpowsour.2020.228818 | Fig6 | _B | hours | 199.48 h | x_time_h |
| 10.1016/j.solmat.2019.110297 | Fig6 | _F | hours | 199.50 h | x_time_h |

## 四、按目录分子表

### x_time_day

| DOI | Figure | 子图 | 最大时间 (h) | 文件夹 | CSV |
|-----|--------|------|-------------|--------|-----|
| 10.1016/j.solmat.2019.110284 | Fig3 | _B | 45.35 | `Fig3_B_structure` | `10.1016_j.solmat.2019.110284__Fig3__B__structure__Temperature_30_oC__Series_2.csv` |
| 10.1021/acs.chemmater.5b03137 | Fig3 | _B | 81.53 | `Fig3_B_profile` | `10.1021_acs.chemmater.5b03137__Fig3__B__profile__B__Series_2.csv` |
| 10.1039/d0ra09294k | Fig5 | _C | 94.64 | `Fig5_C_stability_performance_curve` | `10.1039_d0ra09294k__Fig5__C__stability_performance_curve__Cs0.15__Series_2.csv` |
| 10.1039/d0ra09294k | Fig5 | _C | 94.89 | `Fig5_C_stability_performance_curve` | `10.1039_d0ra09294k__Fig5__C__stability_performance_curve__Cs0.25__Series_3.csv` |
| 10.1039/d0ra09294k | Fig5 | _C | 95.14 | `Fig5_C_stability_performance_curve` | `10.1039_d0ra09294k__Fig5__C__stability_performance_curve__Cs0.05__Series_1.csv` |
| 10.1002/aenm.201700226 | Fig4 | _A | 107.51 | `Fig4_A_stability_performance_curve_pce` | `10.1002_aenm.201700226__Fig4__A__stability_performance_curve_pce__★1-TiO2__Series_3.csv` |
| 10.1002/solr.201700245 | Fig6 | - | 110.75 | `Fig6_whole_stability_performance_curve_pce` | `10.1002_solr.201700245__Fig6__whole__stability_performance_curve_pce__without_CrOX_layer__Series_2.csv` |
| 10.1016/j.solener.2016.10.005 | Fig7 | - | 117.42 | `Fig7_whole_stability` | `10.1016_j.solener.2016.10.005__Fig7__whole__stability__PEDOT_PSS__Series_2.csv` |
| 10.1016/j.electacta.2018.07.028 | Fig2 | _A | 117.98 | `Fig2_A_spectrum_2` | `10.1016_j.electacta.2018.07.028__Fig2__A__spectrum_2__Ref.ETL__Series_2.csv` |
| 10.1016/j.electacta.2018.07.028 | Fig2 | _A | 118.20 | `Fig2_A_spectrum_2` | `10.1016_j.electacta.2018.07.028__Fig2__A__spectrum_2__m-TiO2__Series_3.csv` |
| 10.1016/j.mssp.2018.10.034 | Fig12 | - | 119.63 | `Fig12_whole_stability` | `10.1016_j.mssp.2018.10.034__Fig12__whole__stability__ZnOnanorods__Series_2.csv` |
| 10.1016/j.mssp.2018.10.034 | Fig12 | - | 167.23 | `Fig12_whole_stability` | `10.1016_j.mssp.2018.10.034__Fig12__whole__stability__ZnO-TiO2nanorods__Series_1.csv` |
| 10.1021/acsenergylett.8b00085 | Fig2 | _F | 119.83 | `Fig2_F_stability_performance_curve_image` | `10.1021_acsenergylett.8b00085__Fig2__F__stability_performance_curve_image__FASnl3__Series_2.csv` |
| 10.1016/j.apsusc.2017.02.057 | Fig4 | _D | 120.08 | `Fig4_D_stability_performance_curve_pce` | `10.1016_j.apsusc.2017.02.057__Fig4__D__stability_performance_curve_pce__c-Pbl,based__Series_2.csv` |
| 10.1016/j.solmat.2018.07.009 | Fig5 | _F | 149.17 | `Fig5_F_stability_performance_curve_pce` | `10.1016_j.solmat.2018.07.009__Fig5__F__stability_performance_curve_pce__CH3NH3PbI3cell__Series_1.csv` |
| 10.1039/c7ta09543k | Fig5 | _D | 164.11 | `Fig5_D_stability_performance_curve` | `10.1039_c7ta09543k__Fig5__D__stability_performance_curve__TiO2__Series_1.csv` |
| 10.1039/c7nr03507a | Fig5 | _A | 164.75 | `Fig5_A_stability_performance_curve_pce` | `10.1039_c7nr03507a__Fig5__A__stability_performance_curve_pce__Control__Series_1.csv` |
| 10.1002/advs.201700025 | Fig7 | - | 165.08 | `Fig7_whole_stability` | `10.1002_advs.201700025__Fig7__whole__stability__w_oBDTS-2DPP__Series_2.csv` |
| 10.1016/j.cplett.2021.138496 | Fig5 | _B | 165.25 | `Fig5_B_stability_performance_curve` | `10.1016_j.cplett.2021.138496__Fig5__B__stability_performance_curve__c-TiO₂_m-TiO₂_based__Series_2.csv` |
| 10.1016/j.jallcom.2021.161455 | Fig5 | _D | 165.35 | `Fig5_D_stability_performance_curve` | `10.1016_j.jallcom.2021.161455__Fig5__D__stability_performance_curve__Device_A__Series_2.csv` |
| 10.1021/acs.jpcc.0c02628 | Fig6 | _B | 165.54 | `Fig6_B_stability_performance_curve_pce` | `10.1021_acs.jpcc.0c02628__Fig6__B__stability_performance_curve_pce__SnO₂__Series_2.csv` |
| 10.1016/j.solmat.2016.12.025 | Fig3 | _C | 165.57 | `Fig3_C_stability_performance_curve_pce` | `10.1016_j.solmat.2016.12.025__Fig3__C__stability_performance_curve_pce__By_normal_Pbl2__Series_2.csv` |
| 10.1126/science.aad1015 | Fig4 | _C | 165.80 | `Fig4_C_jv_curve_stability_performance_jsc` | `10.1126_science.aad1015__Fig4__C__jv_curve_stability_performance_jsc__LiF__Series_2.csv` |
| 10.1126/science.aad1015 | Fig4 | _C | 166.71 | `Fig4_C_jv_curve_stability_performance_jsc` | `10.1126_science.aad1015__Fig4__C__jv_curve_stability_performance_jsc__Ca__Series_3.csv` |
| 10.1039/c6ta10588b | Fig6 | - | 165.91 | `Fig6_whole_stability` | `10.1039_c6ta10588b__Fig6__whole__stability__ITO_PEDOT_PSS__Series_2.csv` |
| 10.1039/c6ta10588b | Fig6 | - | 166.16 | `Fig6_whole_stability` | `10.1039_c6ta10588b__Fig6__whole__stability__FTO_TiO2__Series_1.csv` |
| 10.1016/j.solmat.2017.07.022 | Fig7 | - | 166.05 | `Fig7_whole_stability` | `10.1016_j.solmat.2017.07.022__Fig7__whole__stability__PDDA-perovskite__Series_2.csv` |
| 10.1016/j.solmat.2017.07.022 | Fig7 | - | 167.03 | `Fig7_whole_stability` | `10.1016_j.solmat.2017.07.022__Fig7__whole__stability__Control__Series_1.csv` |
| 10.3390/pr10081477 | Fig6 | - | 166.71 | `Fig6_whole_figure` | `10.3390_pr10081477__Fig6__whole__figure__without_TOPO__Series_2.csv` |
| 10.1016/j.jallcom.2020.157396 | Fig6 | _B | 166.79 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.jallcom.2020.157396__Fig6__B__stability_performance_curve_pce__No_additive__Series_2.csv` |
| 10.1021/acsami.0c05527 | Fig7 | _C | 167.09 | `Fig7_C_stability_performance_curve` | `10.1021_acsami.0c05527__Fig7__C__stability_performance_curve__Sn-based__Series_2.csv` |
| 10.1039/d0dt03201h | Fig6 | _G | 167.69 | `Fig6_G_stability_performance_curve_pce` | `10.1039_d0dt03201h__Fig6__G__stability_performance_curve_pce__Without_BiWg-Zn__Series_1.csv` |
| 10.1016/j.carbon.2019.10.101 | Fig4 | _A | 167.79 | `Fig4_A_stability_performance_curve_pce` | `10.1016_j.carbon.2019.10.101__Fig4__A__stability_performance_curve_pce__CuSCN_AuPSC__Series_1.csv` |
| 10.1016/j.electacta.2018.04.055 | Fig4 | _D | 168.24 | `Fig4_D_stability_performance_curve_pce` | `10.1016_j.electacta.2018.04.055__Fig4__D__stability_performance_curve_pce__PEDOT_PSS__Series_1.csv` |
| 10.1021/acsami.6b15563 | Fig5 | _F | 186.40 | `Fig5_F_stability_performance_curve_pce` | `10.1021_acsami.6b15563__Fig5__F__stability_performance_curve_pce__CA__Series_2.csv` |
| 10.1016/j.electacta.2017.07.040 | Fig8 | - | 188.22 | `Fig8_whole_stability_performance_curve_pce` | `10.1016_j.electacta.2017.07.040__Fig8__whole__stability_performance_curve_pce__MAPbI_3__Series_3.csv` |
| 10.1002/admi.201800499 | Fig6 | _E | 188.50 | `Fig6_E_stability_performance_curve_pce` | `10.1002_admi.201800499__Fig6__E__stability_performance_curve_pce__Without_CF_film__Series_2.csv` |
| 10.1039/c8nh00163d | Fig4 | _A | 188.82 | `Fig4_A_stability_performance_curve_pce` | `10.1039_c8nh00163d__Fig4__A__stability_performance_curve_pce__Pristine__Series_1.csv` |
| 10.1016/j.nanoen.2016.08.055 | Fig9 | _A | 189.58 | `Fig9_A_stability_performance_curve_pce` | `10.1016_j.nanoen.2016.08.055__Fig9__A__stability_performance_curve_pce__Pristine__Series_1.csv` |
| 10.1016/j.jpowsour.2022.231243 | Fig6 | _B | 190.12 | `Fig6_B_jv_curve_stability_performance_pce` | `10.1016_j.jpowsour.2022.231243__Fig6__B__jv_curve_stability_performance_pce__W_O_M-P_chloride__Series_2.csv` |


### x_time_h

| DOI | Figure | 子图 | 最大时间 (h) | 文件夹 | CSV |
|-----|--------|------|-------------|--------|-----|
| 10.1039/d1na00173f | Fig5 | _F | 19.07 | `Fig5_F_mppt_stability_performance_curve_spectrum` | `10.1039_d1na00173f__Fig5__F__mppt_stability_performance_curve_spectrum__Pristine_biased_at_0.85_V__Series_2.csv` |
| 10.1039/d1na00173f | Fig5 | _F | 148.84 | `Fig5_F_mppt_stability_performance_curve_spectrum` | `10.1039_d1na00173f__Fig5__F__mppt_stability_performance_curve_spectrum__ZIF-8@FAI_biased_at_0.85_V__Series_1.csv` |
| 10.1021/jacs.0c13069 | Fig4 | _C | 38.64 | `Fig4_C_stability_performance_curve_pce` | `10.1021_jacs.0c13069__Fig4__C__stability_performance_curve_pce__CsSnI₃__Series_2.csv` |
| 10.1021/jacs.0c13069 | Fig4 | _C | 119.33 | `Fig4_C_stability_performance_curve_pce` | `10.1021_jacs.0c13069__Fig4__C__stability_performance_curve_pce__CsSnI₃-MBAA__Series_1.csv` |
| 10.1016/j.xcrp.2020.100224 | Fig5 | _D | 43.57 | `Fig5_D_stability_performance_curve` | `10.1016_j.xcrp.2020.100224__Fig5__D__stability_performance_curve__control__Series_2.csv` |
| 10.1016/j.xcrp.2020.100224 | Fig5 | _B | 74.49 | `Fig5_B_stability` | `10.1016_j.xcrp.2020.100224__Fig5__B__stability__control__Series_1.csv` |
| 10.1002/adma.201705596 | Fig5 | _D | 44.63 | `Fig5_D_stability_performance_curve` | `10.1002_adma.201705596__Fig5__D__stability_performance_curve__(CsFAMA)Pb(BrI)₃__Series_2.csv` |
| 10.1039/c5ee03255e | Fig6 | _D | 46.60 | `Fig6_D_stability_performance_curve` | `10.1039_c5ee03255e__Fig6__D__stability_performance_curve__FAPbI₃__Series_2.csv` |
| 10.1002/solr.201900167 | Fig4 | _D | 46.84 | `Fig4_D_stability_performance_curve` | `10.1002_solr.201900167__Fig4__D__stability_performance_curve__M&C-TiO₂__Series_2.csv` |
| 10.1002/aenm.201602120 | Fig4 | _A | 47.72 | `Fig4_A_pce` | `10.1002_aenm.201602120__Fig4__A__pce__PCBM__Series_2.csv` |
| 10.1016/j.nanoen.2022.107818 | Fig5 | _E | 49.22 | `Fig5_E_mppt_stability_performance_curve_pce` | `10.1016_j.nanoen.2022.107818__Fig5__E__mppt_stability_performance_curve_pce__Control__Series_3.csv` |
| 10.1038/ncomms10228 | Fig5 | _A | 54.39 | `Fig5_A_stability_performance_curve_pce` | `10.1038_ncomms10228__Fig5__A__stability_performance_curve_pce__0_mg_ml⁻¹__Series_3.csv` |
| 10.1002/adfm.201807850 | Fig7 | _F | 58.44 | `Fig7_F_stability_performance_curve_pce` | `10.1002_adfm.201807850__Fig7__F__stability_performance_curve_pce__w_o_NbF₅__Series_2.csv` |
| 10.1039/d4ee02427c | Fig4 | _H | 60.95 | `Fig4_H_mppt_stability_performance_curve_pce_eqe` | `10.1039_d4ee02427c__Fig4__H__mppt_stability_performance_curve_pce_eqe__Control__Series_2.csv` |
| 10.1039/d4ee02427c | Fig3 | _M | 109.29 | `Fig3_M_mppt_stability_performance_curve_density_of_states` | `10.1039_d4ee02427c__Fig3__M__mppt_stability_performance_curve_density_of_states__Control__Series_2.csv` |
| 10.1039/d0ta12286f | Fig5 | _A | 70.88 | `Fig5_A_stability_performance_curve_pce` | `10.1039_d0ta12286f__Fig5__A__stability_performance_curve_pce__FA₀.₈Cs₀.₂Pb(I₀.₆Br₀.₄)₃-based_tandem__Series_2.csv` |
| 10.1039/d0ta12286f | Fig5 | _B | 71.36 | `Fig5_B_stability_performance_curve_pce` | `10.1039_d0ta12286f__Fig5__B__stability_performance_curve_pce__FA0.8Cs0.2Pb(1o.6Br0.4)3-based_tandem__Series_2.csv` |
| 10.1002/adma.201907623 | Fig5 | _C | 72.46 | `Fig5_C_stability_performance_curve_pce` | `10.1002_adma.201907623__Fig5__C__stability_performance_curve_pce__0%_PHCl__Series_2.csv` |
| 10.1002/adma.201907623 | Fig5 | _C | 143.15 | `Fig5_C_stability_performance_curve_pce` | `10.1002_adma.201907623__Fig5__C__stability_performance_curve_pce__5%_PHCl__Series_1.csv` |
| 10.1039/d0ta01860k | Fig5 | _B | 74.13 | `Fig5_B_mppt_stability_performance_curve_pce` | `10.1039_d0ta01860k__Fig5__B__mppt_stability_performance_curve_pce__doped_Spiro__Series_2.csv` |
| 10.1002/adfm.202002964 | Fig5 | _C | 78.31 | `Fig5_C_mppt_stability_performance_curve_profile` | `10.1002_adfm.202002964__Fig5__C__mppt_stability_performance_curve_profile__o-Control__Series_1.csv` |
| 10.1021/jacs.6b10227 | Fig3 | _D | 79.31 | `Fig3_D_stability_performance_curve_pce_photovoltaic_parameter` | `10.1021_jacs.6b10227__Fig3__D__stability_performance_curve_pce_photovoltaic_parameter__MAPbI₃_carbon__Series_1.csv` |
| 10.1002/advs.201901241 | Fig5 | _D | 84.24 | `Fig5_D_stability_performance_curve_pce_plot` | `10.1002_advs.201901241__Fig5__D__stability_performance_curve_pce_plot__Control__Series_2.csv` |
| 10.1002/advs.201901241 | Fig5 | _C | 100.19 | `Fig5_C_stability_performance_curve_pce_plot` | `10.1002_advs.201901241__Fig5__C__stability_performance_curve_pce_plot__Control__Series_1.csv` |
| 10.1002/advs.201901241 | Fig5 | _C | 100.57 | `Fig5_C_stability_performance_curve_pce_plot` | `10.1002_advs.201901241__Fig5__C__stability_performance_curve_pce_plot__2%DIFA__Series_2.csv` |
| 10.1002/advs.201901241 | Fig5 | _D | 108.97 | `Fig5_D_stability_performance_curve_pce_plot` | `10.1002_advs.201901241__Fig5__D__stability_performance_curve_pce_plot__2%DIFA__Series_1.csv` |
| 10.1002/aenm.202301218 | Fig3 | _E | 88.75 | `Fig3_E_mppt_stability_performance_curve_pce_eqe` | `10.1002_aenm.202301218__Fig3__E__mppt_stability_performance_curve_pce_eqe__Control__Series_2.csv` |
| 10.1002/aenm.202301218 | Fig5 | _F | 176.51 | `Fig5_F_mppt_stability_performance_curve_pce` | `10.1002_aenm.202301218__Fig5__F__mppt_stability_performance_curve_pce__Control__Series_2.csv` |
| 10.1016/j.nanoen.2015.10.010 | Fig7 | _D | 88.88 | `Fig7_D_stability_performance_curve_pce` | `10.1016_j.nanoen.2015.10.010__Fig7__D__stability_performance_curve_pce__PbCl₂__Series_3.csv` |
| 10.1002/smll.201907283 | Fig6 | _D | 94.68 | `Fig6_D_ff` | `10.1002_smll.201907283__Fig6__D__ff__WO__Series_2.csv` |
| 10.1016/j.nanoen.2021.105751 | Fig4 | _B | 95.25 | `Fig4_B_stability_performance_curve_pce_pattern` | `10.1016_j.nanoen.2021.105751__Fig4__B__stability_performance_curve_pce_pattern__Li-TFSI_t-BP__Series_2.csv` |
| 10.1021/acsaem.0c01796 | Fig10 | - | 95.28 | `Fig10_whole_stability_performance_curve_pce` | `10.1021_acsaem.0c01796__Fig10__whole__stability_performance_curve_pce__spiro-OMeTAD__Series_2.csv` |
| 10.1002/advs.201903250 | Fig5 | _E | 97.12 | `Fig5_E_tga_stability_performance_curve_pce` | `10.1002_advs.201903250__Fig5__E__tga_stability_performance_curve_pce__wTGA__Series_2.csv` |
| 10.1002/advs.201903250 | Fig5 | _D | 97.44 | `Fig5_D_tga_stability_performance_curve_pce` | `10.1002_advs.201903250__Fig5__D__tga_stability_performance_curve_pce__wTGA__Series_3.csv` |
| 10.1002/advs.201903250 | Fig5 | _E | 99.69 | `Fig5_E_tga_stability_performance_curve_pce` | `10.1002_advs.201903250__Fig5__E__tga_stability_performance_curve_pce__Control__Series_1.csv` |
| 10.1002/advs.201903250 | Fig5 | _D | 120.17 | `Fig5_D_tga_stability_performance_curve_pce` | `10.1002_advs.201903250__Fig5__D__tga_stability_performance_curve_pce__Control__Series_2.csv` |
| 10.1039/d3ee03687a | Fig5 | _B | 99.00 | `Fig5_B_stability_performance_curve_pce` | `10.1039_d3ee03687a__Fig5__B__stability_performance_curve_pce__AS__Series_1.csv` |
| 10.1016/j.orgel.2018.09.019 | Fig4 | _A | 99.50 | `Fig4_A_stability_performance_curve_pce` | `10.1016_j.orgel.2018.09.019__Fig4__A__stability_performance_curve_pce__TiO₂_(UV)__Series_3.csv` |
| 10.1016/j.orgel.2018.09.019 | Fig4 | _A | 100.39 | `Fig4_A_stability_performance_curve_pce` | `10.1016_j.orgel.2018.09.019__Fig4__A__stability_performance_curve_pce__TiO₂_SrO_(UV)__Series_2.csv` |
| 10.1002/eem2.12089 | Fig5 | _B | 99.52 | `Fig5_B_stability_performance_curve` | `10.1002_eem2.12089__Fig5__B__stability_performance_curve__Ag__Series_2.csv` |
| 10.1021/acsenergylett.7b00375 | Fig4 | _C | 100.01 | `Fig4_C_stability_performance_curve_eqe` | `10.1021_acsenergylett.7b00375__Fig4__C__stability_performance_curve_eqe__uncoated__Series_1.csv` |
| 10.1002/pssr.202000062 | Fig4 | _B | 100.06 | `Fig4_B_stability_performance_curve_pce` | `10.1002_pssr.202000062__Fig4__B__stability_performance_curve_pce__TiO2__Series_2.csv` |
| 10.1002/smll.201904399 | Fig6 | _C | 101.57 | `Fig6_C_stability_performance_curve_pce` | `10.1002_smll.201904399__Fig6__C__stability_performance_curve_pce__P3HT_Li+TBP__Series_2.csv` |
| 10.1016/j.nanoen.2016.09.041 | Fig2 | _A | 102.59 | `Fig2_A_pce` | `10.1016_j.nanoen.2016.09.041__Fig2__A__pce__Unsealed__Series_3.csv` |
| 10.1016/j.nanoen.2016.09.041 | Fig2 | _A | 103.16 | `Fig2_A_pce` | `10.1016_j.nanoen.2016.09.041__Fig2__A__pce__SP-D__Series_2.csv` |
| 10.1016/j.jpowsour.2021.230534 | Fig4 | _C | 104.38 | `Fig4_C_stability` | `10.1016_j.jpowsour.2021.230534__Fig4__C__stability__Y6_treated__Series_2.csv` |
| 10.1016/j.jpowsour.2021.230534 | Fig4 | _C | 104.67 | `Fig4_C_stability` | `10.1016_j.jpowsour.2021.230534__Fig4__C__stability__Crotrol_devices__Series_1.csv` |
| 10.1021/acsami.0c02751 | Fig9 | _A | 104.50 | `Fig9_A_stability_performance_curve_pce` | `10.1021_acsami.0c02751__Fig9__A__stability_performance_curve_pce__CJ-04__Series_1.csv` |
| 10.1021/acsami.0c02751 | Fig9 | _A | 104.65 | `Fig9_A_stability_performance_curve_pce` | `10.1021_acsami.0c02751__Fig9__A__stability_performance_curve_pce__CJ-03__Series_2.csv` |
| 10.1021/acsami.0c02751 | Fig9 | _A | 104.81 | `Fig9_A_stability_performance_curve_pce` | `10.1021_acsami.0c02751__Fig9__A__stability_performance_curve_pce__spiro-OMeTAD__Series_3.csv` |
| 10.1016/j.nanoen.2021.106157 | Fig3 | _G | 105.25 | `Fig3_G_stability_performance_curve_pce` | `10.1016_j.nanoen.2021.106157__Fig3__G__stability_performance_curve_pce__0%FeCl2__Series_2.csv` |
| 10.1016/j.cej.2021.130685 | Fig5 | _F | 105.80 | `Fig5_F_stability_performance_curve_pce` | `10.1016_j.cej.2021.130685__Fig5__F__stability_performance_curve_pce__Control__Series_2.csv` |
| 10.1021/acsami.9b11229 | Fig5 | _B | 106.01 | `Fig5_B_stability_performance_curve` | `10.1021_acsami.9b11229__Fig5__B__stability_performance_curve__TiO₂__Series_2.csv` |
| 10.1021/acsami.9b11229 | Fig5 | _C | 106.74 | `Fig5_C_stability_performance_curve` | `10.1021_acsami.9b11229__Fig5__C__stability_performance_curve__TiO₂__Series_2.csv` |
| 10.1021/acsami.9b11229 | Fig5 | _D | 106.97 | `Fig5_D_stability_performance_curve` | `10.1021_acsami.9b11229__Fig5__D__stability_performance_curve__TiO₂__Series_1.csv` |
| 10.1002/solr.202100899 | Fig5 | _C | 106.87 | `Fig5_C_stability_performance_curve_pce` | `10.1002_solr.202100899__Fig5__C__stability_performance_curve_pce__w_o_BPB__Series_2.csv` |
| 10.1002/solr.202100899 | Fig5 | _B | 198.95 | `Fig5_B_stability_performance_curve_pce` | `10.1002_solr.202100899__Fig5__B__stability_performance_curve_pce__w_oBPB__Series_2.csv` |
| 10.1002/solr.202100899 | Fig5 | _D | 198.97 | `Fig5_D_stability_performance_curve_pce` | `10.1002_solr.202100899__Fig5__D__stability_performance_curve_pce__w_oBPB__Series_1.csv` |
| 10.3390/coatings11060627 | Fig3 | _A | 107.31 | `Fig3_A_stability_performance_curve` | `10.3390_coatings11060627__Fig3__A__stability_performance_curve__PEDOT_PSS__Series_1.csv` |
| 10.1002/adfm.202101163 | Fig7 | _B | 109.10 | `Fig7_B_stability_performance_curve_pce` | `10.1002_adfm.202101163__Fig7__B__stability_performance_curve_pce__FAPbl3__Series_1.csv` |
| 10.1016/j.nanoen.2014.12.022 | Fig4 | _A | 115.58 | `Fig4_A_photovoltaic_parameter` | `10.1016_j.nanoen.2014.12.022__Fig4__A__photovoltaic_parameter__PEDOT_PSS__Series_2.csv` |
| 10.1016/j.jpowsour.2021.229781 | Fig4 | _B | 116.87 | `Fig4_B_stability_performance_curve_pce` | `10.1016_j.jpowsour.2021.229781__Fig4__B__stability_performance_curve_pce__FAPb(SCN)₂I__Series_2.csv` |
| 10.1016/j.jpowsour.2021.229781 | Fig4 | _B | 118.74 | `Fig4_B_stability_performance_curve_pce` | `10.1016_j.jpowsour.2021.229781__Fig4__B__stability_performance_curve_pce__FA₀.₉₈K₀.₀₂Pb(SCN)₂I__Series_1.csv` |
| 10.1039/c9ta08314f | Fig5 | _D | 117.13 | `Fig5_D_stability_performance_curve_pce` | `10.1039_c9ta08314f__Fig5__D__stability_performance_curve_pce__Pristine__Series_2.csv` |
| 10.1016/j.nanoen.2019.103964 | Fig4 | _B | 117.34 | `Fig4_B_stability_performance_curve_pce` | `10.1016_j.nanoen.2019.103964__Fig4__B__stability_performance_curve_pce__FTO_SnO,_AI,O,_MAPbI,CI_Spiro-IAu-2__Series_2.csv` |
| 10.1016/j.orgel.2019.03.036 | Fig5 | _C | 117.42 | `Fig5_C_stability_performance_curve_pce` | `10.1016_j.orgel.2019.03.036__Fig5__C__stability_performance_curve_pce__Spiro-OMeTAD_Agl(3%)__Series_1.csv` |
| 10.1016/j.orgel.2019.03.036 | Fig5 | _D | 117.49 | `Fig5_D_stability_performance_curve_pce` | `10.1016_j.orgel.2019.03.036__Fig5__D__stability_performance_curve_pce__o—Spiro-OMeTAD_AgI(3%)__Series_1.csv` |
| 10.1021/acsaem.0c00584 | Fig3 | _D | 117.57 | `Fig3_D_stability_performance_curve_pce` | `10.1021_acsaem.0c00584__Fig3__D__stability_performance_curve_pce__75mg_ml×5LFAIQDs__Series_4.csv` |
| 10.1021/acsaem.0c00584 | Fig3 | _D | 118.67 | `Fig3_D_stability_performance_curve_pce` | `10.1021_acsaem.0c00584__Fig3__D__stability_performance_curve_pce__75mg_ml×5L_Control_QDs__Series_3.csv` |
| 10.1021/acsaem.0c00584 | Fig3 | _D | 119.77 | `Fig3_D_stability_performance_curve_pce` | `10.1021_acsaem.0c00584__Fig3__D__stability_performance_curve_pce__145mg_ml×3LFAIQDs__Series_2.csv` |
| 10.1021/acsaem.0c00584 | Fig3 | _D | 119.96 | `Fig3_D_stability_performance_curve_pce` | `10.1021_acsaem.0c00584__Fig3__D__stability_performance_curve_pce__145mg_ml×3L_Control_QDs__Series_1.csv` |
| 10.1016/j.jechem.2021.08.021 | Fig5 | _G | 117.70 | `Fig5_G_stability_performance_curve_pce_voc` | `10.1016_j.jechem.2021.08.021__Fig5__G__stability_performance_curve_pce_voc__w_o_GABr__Series_2.csv` |
| 10.1002/aenm.201803287 | Fig6 | _B | 117.96 | `Fig6_B_stability_performance_curve_pce` | `10.1002_aenm.201803287__Fig6__B__stability_performance_curve_pce__o-Doped_Spiro-OMeTAD__Series_1.csv` |
| 10.1039/c5ta09080f | Fig6 | _C | 118.05 | `Fig6_C_stability_performance_curve_pce_profile` | `10.1039_c5ta09080f__Fig6__C__stability_performance_curve_pce_profile__Operating_stability__Series_2.csv` |
| 10.1016/j.jmat.2021.08.007 | Fig6 | _A | 118.06 | `Fig6_A_stability_performance_curve` | `10.1016_j.jmat.2021.08.007__Fig6__A__stability_performance_curve__pristine__Series_1.csv` |
| 10.1016/j.jmat.2021.08.007 | Fig6 | _A | 160.23 | `Fig6_A_stability_performance_curve` | `10.1016_j.jmat.2021.08.007__Fig6__A__stability_performance_curve__1.0_wt%_CNDs__Series_2.csv` |
| 10.1002/adem.202000990 | Fig5 | _C | 118.07 | `Fig5_C_jv_curve_stability_performance_pce` | `10.1002_adem.202000990__Fig5__C__jv_curve_stability_performance_pce__Pero@Pbl2-70°C__Series_3.csv` |
| 10.1002/adma.201905766 | Fig4 | _E | 118.17 | `Fig4_E_mppt_stability_performance_curve_pce` | `10.1002_adma.201905766__Fig4__E__mppt_stability_performance_curve_pce__In₂O₃_SnO₂__Series_1.csv` |
| 10.1002/adma.201905766 | Fig4 | _E | 118.37 | `Fig4_E_mppt_stability_performance_curve_pce` | `10.1002_adma.201905766__Fig4__E__mppt_stability_performance_curve_pce__SnO₂__Series_2.csv` |
| 10.1002/adma.201905766 | Fig4 | _E | 177.99 | `Fig4_E_stability_performance_curve_pce` | `10.1002_adma.201905766__Fig4__E__stability_performance_curve_pce__In₂O₃_SnO₂__Series_1.csv` |
| 10.1002/adma.201905766 | Fig4 | _E | 177.99 | `Fig4_E_stability_performance_curve_pce` | `10.1002_adma.201905766__Fig4__E__stability_performance_curve_pce__SnO₂__Series_2.csv` |
| 10.1126/science.aam6620 | Fig4 | _A | 118.32 | `Fig4_A_stability_performance_curve_pce_2` | `10.1126_science.aam6620__Fig4__A__stability_performance_curve_pce_2__TiO₂_PSC__Series_2.csv` |
| 10.1002/aenm.201802774 | Fig3 | _C | 118.38 | `Fig3_C_stability_performance_curve_pce` | `10.1002_aenm.201802774__Fig3__C__stability_performance_curve_pce__w_oGDR__Series_1.csv` |
| 10.1039/d1na00172h | Fig6 | _B | 118.50 | `Fig6_B_stability_performance_curve_pce` | `10.1039_d1na00172h__Fig6__B__stability_performance_curve_pce__PSC-Ref__Series_1.csv` |
| 10.1039/d1na00172h | Fig6 | _B | 118.67 | `Fig6_B_stability_performance_curve_pce` | `10.1039_d1na00172h__Fig6__B__stability_performance_curve_pce__PSC-2__Series_2.csv` |
| 10.1007/s40820-022-00995-2 | Fig5 | _F | 118.53 | `Fig5_F_sem_image_stability_performance_curve_pce` | `10.1007_s40820-022-00995-2__Fig5__F__sem_image_stability_performance_curve_pce__Filtered_OSC__Series_1.csv` |
| 10.1007/s40820-022-00995-2 | Fig5 | _F | 118.53 | `Fig5_F_sem_image_stability_performance_curve_pce` | `10.1007_s40820-022-00995-2__Fig5__F__sem_image_stability_performance_curve_pce__OSC__Series_2.csv` |
| 10.1021/acsenergylett.1c01446 | Fig5 | - | 118.58 | `Fig5_whole_stability_performance_curve_pce` | `10.1021_acsenergylett.1c01446__Fig5__whole__stability_performance_curve_pce__Control__Series_1.csv` |
| 10.1038/s41467-018-05760-x | Fig6 | _B | 118.85 | `Fig6_B_jv_curve_stability_performance_pce` | `10.1038_s41467-018-05760-x__Fig6__B__jv_curve_stability_performance_pce__E-SnO2__Series_2.csv` |
| 10.1038/s41467-018-05760-x | Fig6 | _B | 120.12 | `Fig6_B_jv_curve_stability_performance_pce` | `10.1038_s41467-018-05760-x__Fig6__B__jv_curve_stability_performance_pce__SnO2__Series_1.csv` |
| 10.1002/solr.201900319 | Fig4 | _F | 118.99 | `Fig4_F_stability_performance_curve_eqe` | `10.1002_solr.201900319__Fig4__F__stability_performance_curve_eqe__FB-OMeTPA__Series_1.csv` |
| 10.1016/j.solmat.2017.12.015 | Fig8 | _D | 119.75 | `Fig8_D_stability_performance_curve_pce` | `10.1016_j.solmat.2017.12.015__Fig8__D__stability_performance_curve_pce__Reference__Series_2.csv` |
| 10.1021/acs.langmuir.8b01650 | Fig6 | - | 119.84 | `Fig6_whole_stability` | `10.1021_acs.langmuir.8b01650__Fig6__whole__stability__Control__Series_1.csv` |
| 10.1002/solr.201800290 | Fig7 | _A | 119.98 | `Fig7_A_stability_performance_curve` | `10.1002_solr.201800290__Fig7__A__stability_performance_curve__W_O_TFEACI__Series_2.csv` |
| 10.1002/solr.202000164 | Fig4 | _C | 120.05 | `Fig4_C_stability_performance_curve_pce` | `10.1002_solr.202000164__Fig4__C__stability_performance_curve_pce__CsPbI₂Br__Series_1.csv` |
| 10.1002/adfm.202002230 | Fig6 | _B | 120.70 | `Fig6_B_stability_performance_curve_mapping` | `10.1002_adfm.202002230__Fig6__B__stability_performance_curve_mapping__0MABr__Series_2.csv` |
| 10.1016/j.flatc.2021.100254 | Fig10 | - | 120.91 | `Fig10_whole_stability_performance_curve` | `10.1016_j.flatc.2021.100254__Fig10__whole__stability_performance_curve__0.08_mg_mL__Series_2.csv` |
| 10.1016/j.nanoen.2019.104244 | Fig5 | _B | 124.72 | `Fig5_B_stability_performance_curve_pce` | `10.1016_j.nanoen.2019.104244__Fig5__B__stability_performance_curve_pce__PHJ__Series_2.csv` |
| 10.1007/s10853-020-04965-0 | Fig8 | _B | 125.84 | `Fig8_B_stability_performance_curve_jsc` | `10.1007_s10853-020-04965-0__Fig8__B__stability_performance_curve_jsc__No_passivation__Series_2.csv` |
| 10.1039/c7ta09716f | Fig7 | _B | 128.48 | `Fig7_B_stability_performance_curve_pce` | `10.1039_c7ta09716f__Fig7__B__stability_performance_curve_pce__PEDOT_PSS__Series_1.csv` |
| 10.1039/d5ee01110h | Fig1 | _E | 133.98 | `Fig1_E_stability_performance_curve` | `10.1039_d5ee01110h__Fig1__E__stability_performance_curve__Control__Series_2.csv` |
| 10.1039/c7ee02272g | Fig2 | _E | 135.14 | `Fig2_E_sem_image_stability_performance_curve_pce_2` | `10.1039_c7ee02272g__Fig2__E__sem_image_stability_performance_curve_pce_2__P-MAPbl₃__Series_2.csv` |
| 10.1039/d4ra08786k | Fig7 | _B | 135.25 | `Fig7_B_stability_performance_curve_pce` | `10.1039_d4ra08786k__Fig7__B__stability_performance_curve_pce__Untreated_Initial_PCE_=_16.51%__Series_2.csv` |
| 10.1021/acsami.0c11247 | Fig5 | _A | 136.38 | `Fig5_A_stability_performance_curve_pce` | `10.1021_acsami.0c11247__Fig5__A__stability_performance_curve_pce__PEDOT_PSS__Series_2.csv` |
| 10.1002/admi.201500799 | Fig7 | - | 137.04 | `Fig7_whole_stability_performance_curve_pce` | `10.1002_admi.201500799__Fig7__whole__stability_performance_curve_pce__CrOx__Series_2.csv` |
| 10.1002/admi.201500799 | Fig7 | - | 143.04 | `Fig7_whole_stability_performance_curve_pce` | `10.1002_admi.201500799__Fig7__whole__stability_performance_curve_pce__Cu_CrOx__Series_1.csv` |
| 10.1039/d4ee03001j | Fig5 | _F | 137.86 | `Fig5_F_mppt_stability_performance_curve_jsc` | `10.1039_d4ee03001j__Fig5__F__mppt_stability_performance_curve_jsc__Control__Series_2.csv` |
| 10.1021/acsami.0c00142 | Fig4 | _F | 138.10 | `Fig4_F_stability` | `10.1021_acsami.0c00142__Fig4__F__stability__NbOₓ__Series_1.csv` |
| 10.1038/s41598-020-79348-1 | Fig4 | _A | 138.13 | `Fig4_A_stability_performance_curve_ff` | `10.1038_s41598-020-79348-1__Fig4__A__stability_performance_curve_ff__P__Series_3.csv` |
| 10.1021/jacs.7b07949 | Fig4 | _B | 138.72 | `Fig4_B_stability_performance_curve_ff_pce` | `10.1021_jacs.7b07949__Fig4__B__stability_performance_curve_ff_pce__MAPbI3__Series_2.csv` |
| 10.1021/acs.chemmater.0c00995 | Fig6 | _D | 140.06 | `Fig6_D_xrd_pattern_stability_performance_curve_pce_spectrum` | `10.1021_acs.chemmater.0c00995__Fig6__D__xrd_pattern_stability_performance_curve_pce_spectrum__MAPI_PSC__Series_1.csv` |
| 10.1002/adfm.201503559 | Fig6 | - | 141.43 | `Fig6_whole_figure` | `10.1002_adfm.201503559__Fig6__whole__figure__DMF__Series_2.csv` |
| 10.1002/adfm.201503559 | Fig6 | - | 143.36 | `Fig6_whole_figure` | `10.1002_adfm.201503559__Fig6__whole__figure__DMF_+_2%_H₂O__Series_1.csv` |
| 10.1002/aenm.201903654 | Fig5 | _E | 141.44 | `Fig5_E_mppt_stability_performance_curve_pce` | `10.1002_aenm.201903654__Fig5__E__mppt_stability_performance_curve_pce__3D__Series_2.csv` |
| 10.1016/j.jpowsour.2020.229345 | Fig8 | _D | 142.00 | `Fig8_D_stability_performance_curve_jsc` | `10.1016_j.jpowsour.2020.229345__Fig8__D__stability_performance_curve_jsc__Bare__Series_1.csv` |
| 10.1016/j.jpowsour.2020.229345 | Fig8 | _D | 142.00 | `Fig8_D_stability_performance_curve_jsc` | `10.1016_j.jpowsour.2020.229345__Fig8__D__stability_performance_curve_jsc__Perovskite_PEG__Series_2.csv` |
| 10.1016/j.orgel.2018.02.026 | Fig8 | _A | 142.18 | `Fig8_A_stability_performance_curve_pce` | `10.1016_j.orgel.2018.02.026__Fig8__A__stability_performance_curve_pce__DCB__Series_1.csv` |
| 10.1016/j.orgel.2018.02.026 | Fig8 | _A | 142.36 | `Fig8_A_stability_performance_curve_pce` | `10.1016_j.orgel.2018.02.026__Fig8__A__stability_performance_curve_pce__CB__Series_3.csv` |
| 10.1016/j.orgel.2018.02.026 | Fig8 | _A | 143.79 | `Fig8_A_stability_performance_curve_pce` | `10.1016_j.orgel.2018.02.026__Fig8__A__stability_performance_curve_pce__CF__Series_2.csv` |
| 10.1039/d1ta05699a | Fig5 | _E | 142.64 | `Fig5_E_stability_performance_curve` | `10.1039_d1ta05699a__Fig5__E__stability_performance_curve__Control__Series_2.csv` |
| 10.1039/d1ta05699a | Fig5 | _E | 142.64 | `Fig5_E_stability_performance_curve` | `10.1039_d1ta05699a__Fig5__E__stability_performance_curve__n-DAbr__Series_1.csv` |
| 10.1039/d1ta05699a | Fig5 | _E | 142.85 | `Fig5_E_stability_performance_curve` | `10.1039_d1ta05699a__Fig5__E__stability_performance_curve__n-Babr__Series_4.csv` |
| 10.1002/smll.202101380 | Fig5 | _H | 142.87 | `Fig5_H_stability_performance_curve_pce` | `10.1002_smll.202101380__Fig5__H__stability_performance_curve_pce__CsPbo.5Sn0.45l2Br__Series_1.csv` |
| 10.1002/solr.202000621 | Fig6 | _B | 145.49 | `Fig6_B_stability_performance_curve_pce` | `10.1002_solr.202000621__Fig6__B__stability_performance_curve_pce__MAPblz-BP__Series_1.csv` |
| 10.1002/solr.202000621 | Fig6 | _B | 193.09 | `Fig6_B_stability_performance_curve_pce` | `10.1002_solr.202000621__Fig6__B__stability_performance_curve_pce__MAPblz-BA__Series_2.csv` |
| 10.1021/acsaem.9b01652 | Fig6 | _B | 147.18 | `Fig6_B_rate` | `10.1021_acsaem.9b01652__Fig6__B__rate__SnO₂__Series_1.csv` |
| 10.1039/c9tc01741k | Fig7 | _B | 147.24 | `Fig7_B_stability_performance_curve_pce` | `10.1039_c9tc01741k__Fig7__B__stability_performance_curve_pce__PC₆₁BM__Series_1.csv` |
| 10.1039/d3ta01935g | Fig5 | _F | 147.55 | `Fig5_F_stability_performance_curve_pce` | `10.1039_d3ta01935g__Fig5__F__stability_performance_curve_pce__Pure_F-PEA__Series_3.csv` |
| 10.1039/d3ta01935g | Fig5 | _F | 148.49 | `Fig5_F_stability_performance_curve_pce` | `10.1039_d3ta01935g__Fig5__F__stability_performance_curve_pce__iso-BA_F-PEA__Series_2.csv` |
| 10.1039/d3ta01935g | Fig5 | _F | 148.80 | `Fig5_F_stability_performance_curve_pce` | `10.1039_d3ta01935g__Fig5__F__stability_performance_curve_pce__Pure_iso-BA__Series_1.csv` |
| 10.1039/d0sc06354a | Fig6 | _D | 147.55 | `Fig6_D_stability_performance_curve` | `10.1039_d0sc06354a__Fig6__D__stability_performance_curve__Control__Series_1.csv` |
| 10.1039/d0se00528b | Fig6 | _A | 147.71 | `Fig6_A_stability_performance_curve_pce` | `10.1039_d0se00528b__Fig6__A__stability_performance_curve_pce__Perovskite_devices__Series_1.csv` |
| 10.1039/d0se00528b | Fig6 | _A | 148.30 | `Fig6_A_stability_performance_curve_pce` | `10.1039_d0se00528b__Fig6__A__stability_performance_curve_pce__MPIB-Perovskitedevices__Series_2.csv` |
| 10.1021/acsenergylett.0c00596 | Fig4 | _C | 147.83 | `Fig4_C_mppt_stability_performance_curve_pce` | `10.1021_acsenergylett.0c00596__Fig4__C__mppt_stability_performance_curve_pce__Control_L-PSC__Series_1.csv` |
| 10.1016/j.solener.2021.01.030 | Fig6 | _A | 148.06 | `Fig6_A_stability_performance_curve_pce` | `10.1016_j.solener.2021.01.030__Fig6__A__stability_performance_curve_pce__without_TM__Series_2.csv` |
| 10.1039/d3ta07851e | Fig3 | _E | 148.31 | `Fig3_E_stability_performance_curve_pce` | `10.1039_d3ta07851e__Fig3__E__stability_performance_curve_pce__Spiro-OMeTAD__Series_2.csv` |
| 10.1039/C9TA03131F | Fig5 | _D | 148.86 | `Fig5_D_stability_performance_curve_pce` | `10.1039_C9TA03131F__Fig5__D__stability_performance_curve_pce__Without_QDs__Series_1.csv` |
| 10.1039/C9TA03131F | Fig5 | _D | 149.12 | `Fig5_D_stability_performance_curve_pce` | `10.1039_C9TA03131F__Fig5__D__stability_performance_curve_pce__With_QDs__Series_2.csv` |
| 10.1016/j.jechem.2019.10.006 | Fig4 | _F | 148.90 | `Fig4_F_stability_performance_curve_voc` | `10.1016_j.jechem.2019.10.006__Fig4__F__stability_performance_curve_voc__MAPbI₃__Series_1.csv` |
| 10.1016/j.jallcom.2021.159658 | Fig23 | - | 149.33 | `Fig23_whole_stability_performance_curve_pce` | `10.1016_j.jallcom.2021.159658__Fig23__whole__stability_performance_curve_pce__ZnO_ETL__Series_1.csv` |
| 10.1016/j.jallcom.2021.159658 | Fig23 | - | 149.67 | `Fig23_whole_stability_performance_curve_pce` | `10.1016_j.jallcom.2021.159658__Fig23__whole__stability_performance_curve_pce__ZnO_Ag_(5%)_ETL__Series_2.csv` |
| 10.1002/smtd.201900652 | Fig5 | _E | 151.49 | `Fig5_E_stability_performance_curve_pce` | `10.1002_smtd.201900652__Fig5__E__stability_performance_curve_pce__150℃__Series_1.csv` |
| 10.1016/j.xcrp.2021.100395 | Fig6 | _F | 152.52 | `Fig6_F_stability_performance_curve_pce` | `10.1016_j.xcrp.2021.100395__Fig6__F__stability_performance_curve_pce__中RH0%__Series_2.csv` |
| 10.1039/c9ta09260a | Fig4 | _I | 152.60 | `Fig4_I_stability_performance_curve_pce` | `10.1039_c9ta09260a__Fig4__I__stability_performance_curve_pce__PCBM__Series_1.csv` |
| 10.1002/solr.202100320 | Fig7 | _F | 156.82 | `Fig7_F_stability_performance_curve_pce_pattern` | `10.1002_solr.202100320__Fig7__F__stability_performance_curve_pce_pattern__With_4F-PHCl__Series_2.csv` |
| 10.1002/aenm.201902239 | Fig5 | _C | 157.20 | `Fig5_C_stability_performance_curve_pce` | `10.1002_aenm.201902239__Fig5__C__stability_performance_curve_pce__control__Series_1.csv` |
| 10.1016/j.jmst.2021.05.084 | Fig5 | _B | 158.15 | `Fig5_B_stability_performance_curve_pce` | `10.1016_j.jmst.2021.05.084__Fig5__B__stability_performance_curve_pce__0wt%__Series_2.csv` |
| 10.1021/acsaem.0c00811 | Fig5 | _B | 159.37 | `Fig5_B_curve` | `10.1021_acsaem.0c00811__Fig5__B__curve__spiro-MeOTAD__Series_2.csv` |
| 10.1021/acs.chemmater.1c00662 | Fig4 | _B | 162.06 | `Fig4_B_stability_performance_curve_pce` | `10.1021_acs.chemmater.1c00662__Fig4__B__stability_performance_curve_pce__PCBM_ZnO__Series_2.csv` |
| 10.1002/adma.201903448 | Fig3 | _G | 163.46 | `Fig3_G_stability_performance_curve_pce` | `10.1002_adma.201903448__Fig3__G__stability_performance_curve_pce__CsPbI₃__Series_2.csv` |
| 10.1039/C9SC01697J | Fig6 | _E | 164.27 | `Fig6_E_sem_image_stability_performance_curve_photovoltaic_parameter` | `10.1039_C9SC01697J__Fig6__E__sem_image_stability_performance_curve_photovoltaic_parameter__Doped_Spiro-OMeTAD__Series_1.csv` |
| 10.1021/acsenergylett.7b00508 | Fig4 | _F | 164.68 | `Fig4_F_stability_performance_curve_pce` | `10.1021_acsenergylett.7b00508__Fig4__F__stability_performance_curve_pce__α-CsPbI3__Series_1.csv` |
| 10.1016/j.nanoen.2020.105462 | Fig5 | _B | 164.92 | `Fig5_B_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.105462__Fig5__B__stability_performance_curve_pce__SMe-TATPyr_Spiro-OMeTAD__Series_2.csv` |
| 10.1016/j.nanoen.2020.105462 | Fig5 | _B | 164.92 | `Fig5_B_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.105462__Fig5__B__stability_performance_curve_pce__→Spiro-OMeTAD__Series_1.csv` |
| 10.1016/j.nanoen.2020.105462 | Fig5 | _C | 165.08 | `Fig5_C_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.105462__Fig5__C__stability_performance_curve_pce__SMe-TATPyr_Spiro-OMeTAD__Series_1.csv` |
| 10.1016/j.nanoen.2020.105462 | Fig5 | _C | 165.08 | `Fig5_C_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.105462__Fig5__C__stability_performance_curve_pce__Spiro-OMeTAD__Series_2.csv` |
| 10.1002/adma.201907396 | Fig5 | _E | 164.96 | `Fig5_E_mppt_stability_performance_curve_pattern` | `10.1002_adma.201907396__Fig5__E__mppt_stability_performance_curve_pattern__Control__Series_1.csv` |
| 10.1016/j.joule.2018.11.026 | Fig6 | _B | 165.09 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.joule.2018.11.026__Fig6__B__stability_performance_curve_pce__MAPbl₃__Series_3.csv` |
| 10.1016/j.joule.2018.11.026 | Fig6 | _C | 166.63 | `Fig6_C_stability_performance_curve_pce` | `10.1016_j.joule.2018.11.026__Fig6__C__stability_performance_curve_pce__0-(PA)2(MA)₃Pb₂13-_MAPb13__Series_2.csv` |
| 10.1016/j.joule.2018.11.026 | Fig6 | _C | 166.63 | `Fig6_C_stability_performance_curve_pce` | `10.1016_j.joule.2018.11.026__Fig6__C__stability_performance_curve_pce__MAPb13__Series_3.csv` |
| 10.1016/j.joule.2018.11.026 | Fig6 | _B | 167.40 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.joule.2018.11.026__Fig6__B__stability_performance_curve_pce__(PA)2(MA)₃Pb413__Series_2.csv` |
| 10.1016/j.jmrt.2021.12.012 | Fig14 | _D | 165.30 | `Fig14_D_stability_performance_curve_pce` | `10.1016_j.jmrt.2021.12.012__Fig14__D__stability_performance_curve_pce__ZnPor_rGO_CuO_bio-nanocomposite__Series_1.csv` |
| 10.1016/j.jmrt.2021.12.012 | Fig14 | _D | 165.63 | `Fig14_D_stability_performance_curve_pce` | `10.1016_j.jmrt.2021.12.012__Fig14__D__stability_performance_curve_pce__HTM_free__Series_3.csv` |
| 10.1016/j.jmrt.2021.12.012 | Fig14 | _D | 166.28 | `Fig14_D_stability_performance_curve_pce` | `10.1016_j.jmrt.2021.12.012__Fig14__D__stability_performance_curve_pce__Spiro-OMeTAD__Series_2.csv` |
| 10.1016/j.electacta.2016.03.002 | Fig7 | _A | 165.41 | `Fig7_A_stability_performance_curve` | `10.1016_j.electacta.2016.03.002__Fig7__A__stability_performance_curve__FS__Series_1.csv` |
| 10.1016/j.electacta.2016.03.002 | Fig7 | _A | 165.65 | `Fig7_A_stability_performance_curve` | `10.1016_j.electacta.2016.03.002__Fig7__A__stability_performance_curve__LS__Series_2.csv` |
| 10.1016/j.electacta.2016.03.002 | Fig7 | _A | 166.38 | `Fig7_A_stability_performance_curve` | `10.1016_j.electacta.2016.03.002__Fig7__A__stability_performance_curve__S__Series_3.csv` |
| 10.1002/adma.202003965 | Fig6 | _B | 166.15 | `Fig6_B_chart` | `10.1002_adma.202003965__Fig6__B__chart__Eth-PsCs__Series_2.csv` |
| 10.1109/jphotov.2020.2987390 | Fig4 | _D | 166.31 | `Fig4_D_stability_performance_curve_pce` | `10.1109_jphotov.2020.2987390__Fig4__D__stability_performance_curve_pce__MoO₃__Series_2.csv` |
| 10.1016/j.apsusc.2018.09.080 | Fig16 | - | 166.32 | `Fig16_whole_stability_performance_curve_pce` | `10.1016_j.apsusc.2018.09.080__Fig16__whole__stability_performance_curve_pce__ZnO_NRs__Series_1.csv` |
| 10.1016/j.apsusc.2018.09.080 | Fig16 | - | 167.96 | `Fig16_whole_stability_performance_curve_pce` | `10.1016_j.apsusc.2018.09.080__Fig16__whole__stability_performance_curve_pce__→ZnO@TiO2_NRs__Series_2.csv` |
| 10.1021/acsami.8b12675 | Fig6 | - | 166.56 | `Fig6_whole_stability` | `10.1021_acsami.8b12675__Fig6__whole__stability__pcBM__Series_1.csv` |
| 10.1002/adma.202003296 | Fig5 | _B | 166.91 | `Fig5_B_stability_performance_curve_pce` | `10.1002_adma.202003296__Fig5__B__stability_performance_curve_pce__Control__Series_2.csv` |
| 10.1039/d0ta06033j | Fig4 | _F | 167.27 | `Fig4_F_stability_performance_curve_pattern` | `10.1039_d0ta06033j__Fig4__F__stability_performance_curve_pattern__0%CTAC__Series_2.csv` |
| 10.1016/j.orgel.2021.106244 | Fig7 | - | 167.45 | `Fig7_whole_stability_performance_curve_pce` | `10.1016_j.orgel.2021.106244__Fig7__whole__stability_performance_curve_pce__Spiro__Series_1.csv` |
| 10.1021/acsaem.8b00602 | Fig8 | - | 167.71 | `Fig8_whole_stability` | `10.1021_acsaem.8b00602__Fig8__whole__stability__TiO2__Series_1.csv` |
| 10.1016/j.apsusc.2019.143552 | Fig10 | - | 167.91 | `Fig10_whole_figure` | `10.1016_j.apsusc.2019.143552__Fig10__whole__figure__ZnO__Series_1.csv` |
| 10.1016/j.heliyon.2024.e25352 | Fig6 | - | 168.00 | `Fig6_whole_stability_performance_curve_pce` | `10.1016_j.heliyon.2024.e25352__Fig6__whole__stability_performance_curve_pce__CsI-0__Series_2.csv` |
| 10.1002/adfm.201600746 | Fig6 | _A | 168.48 | `Fig6_A_stability_performance_curve_pce` | `10.1002_adfm.201600746__Fig6__A__stability_performance_curve_pce__PEDOT_PSS__Series_1.csv` |
| 10.1021/acsami.0c17141 | Fig6 | _B | 169.55 | `Fig6_B_stability_performance_curve_pce` | `10.1021_acsami.0c17141__Fig6__B__stability_performance_curve_pce__reference_Ag_electrode__Series_4.csv` |
| 10.1016/j.solmat.2015.10.008 | Fig6 | _D | 170.07 | `Fig6_D_ff` | `10.1016_j.solmat.2015.10.008__Fig6__D__ff__Spiro-OMeTAD__Series_1.csv` |
| 10.1016/j.solmat.2015.10.008 | Fig6 | _D | 170.38 | `Fig6_D_ff` | `10.1016_j.solmat.2015.10.008__Fig6__D__ff__O-PDPP3T__Series_2.csv` |
| 10.1021/acsami.7b06681 | Fig6 | _D | 172.42 | `Fig6_D_stability_performance_curve_pce` | `10.1021_acsami.7b06681__Fig6__D__stability_performance_curve_pce__Bphen__Series_2.csv` |
| 10.1016/j.joule.2020.12.003 | Fig6 | _C | 173.58 | `Fig6_C_mppt_stability_performance_curve_pce` | `10.1016_j.joule.2020.12.003__Fig6__C__mppt_stability_performance_curve_pce__o-w_Spiro__Series_2.csv` |
| 10.1002/admi.202000105 | Fig5 | _D | 174.11 | `Fig5_D_stability_performance_curve_pce` | `10.1002_admi.202000105__Fig5__D__stability_performance_curve_pce__Reference__Series_2.csv` |
| 10.1039/c9ta13922b | Fig4 | _H | 175.41 | `Fig4_H_stability_performance_curve_pce` | `10.1039_c9ta13922b__Fig4__H__stability_performance_curve_pce__w_ot-BCA__Series_2.csv` |
| 10.1039/c6ta06152d | Fig5 | _A | 175.95 | `Fig5_A_stability_performance_curve_pce` | `10.1039_c6ta06152d__Fig5__A__stability_performance_curve_pce__CHNHPbICI_C-PCBSD__Series_1.csv` |
| 10.1039/c6ta06152d | Fig5 | _A | 176.41 | `Fig5_A_stability_performance_curve_pce` | `10.1039_c6ta06152d__Fig5__A__stability_performance_curve_pce__CHNHPbICI__Series_2.csv` |
| 10.1039/d4su00641k | Fig1 | - | 176.11 | `Fig1_whole_stability_performance_curve` | `10.1039_d4su00641k__Fig1__whole__stability_performance_curve__Pure_perovskite__Series_2.csv` |
| 10.1039/d4su00641k | Fig1 | - | 177.25 | `Fig1_whole_stability_performance_curve` | `10.1039_d4su00641k__Fig1__whole__stability_performance_curve__Mixed_perovskite__Series_1.csv` |
| 10.1016/j.nanoen.2017.11.008 | Fig3 | _F | 176.56 | `Fig3_F_stability_performance_curve_pce` | `10.1016_j.nanoen.2017.11.008__Fig3__F__stability_performance_curve_pce__口—TiO2__Series_2.csv` |
| 10.1039/d4ee05699j | Fig4 | _I | 184.83 | `Fig4_I_stability_performance_curve_pce` | `10.1039_d4ee05699j__Fig4__I__stability_performance_curve_pce__Control-PSMs__Series_2.csv` |
| 10.1126/sciadv.aav8925 | Fig5 | _B | 186.82 | `Fig5_B_stability_performance_curve_pce` | `10.1126_sciadv.aav8925__Fig5__B__stability_performance_curve_pce__MAPbI3__Series_2.csv` |
| 10.1039/c9ta05802h | Fig5 | _A | 187.18 | `Fig5_A_stability_performance_curve_jsc` | `10.1039_c9ta05802h__Fig5__A__stability_performance_curve_jsc__C-ZnO__Series_2.csv` |
| 10.1016/j.solener.2021.06.076 | Fig9 | _A | 187.20 | `Fig9_A_stability_performance_curve_pce` | `10.1016_j.solener.2021.06.076__Fig9__A__stability_performance_curve_pce__PC₀NH₀__Series_1.csv` |
| 10.1016/j.jallcom.2020.157784 | Fig7 | _H | 187.31 | `Fig7_H_stability_performance_curve_pce` | `10.1016_j.jallcom.2020.157784__Fig7__H__stability_performance_curve_pce__With_0.05wt%CA-CQDs__Series_3.csv` |
| 10.1016/j.jallcom.2020.157784 | Fig7 | _H | 187.31 | `Fig7_H_stability_performance_curve_pce` | `10.1016_j.jallcom.2020.157784__Fig7__H__stability_performance_curve_pce__Without__Series_2.csv` |
| 10.1038/s41598-017-00866-6 | Fig1 | _B | 187.64 | `Fig1_B_xrd_pattern_stability_performance_curve` | `10.1038_s41598-017-00866-6__Fig1__B__xrd_pattern_stability_performance_curve__under_85°C_20-25%_RH__Series_1.csv` |
| 10.1038/s41598-017-00866-6 | Fig1 | _A | 191.07 | `Fig1_A_stability_performance_curve_pce` | `10.1038_s41598-017-00866-6__Fig1__A__stability_performance_curve_pce__under_25°C_20-25%_RH__Series_1.csv` |
| 10.1038/s41586-019-1036-3 | Fig4 | _A | 188.27 | `Fig4_A_mppt_stability_performance_curve` | `10.1038_s41586-019-1036-3__Fig4__A__mppt_stability_performance_curve__Control__Series_2.csv` |
| 10.1016/j.orgel.2019.03.033 | Fig7 | _C | 188.40 | `Fig7_C_stability_performance_curve_pce` | `10.1016_j.orgel.2019.03.033__Fig7__C__stability_performance_curve_pce__reference__Series_2.csv` |
| 10.1039/c8ta06557h | Fig6 | _F | 188.68 | `Fig6_F_stability_performance_curve_pce` | `10.1039_c8ta06557h__Fig6__F__stability_performance_curve_pce__FAPbI₃__Series_2.csv` |
| 10.1039/c8ta06557h | Fig6 | _E | 191.35 | `Fig6_E_stability_performance_curve_pce` | `10.1039_c8ta06557h__Fig6__E__stability_performance_curve_pce__FAPbI₃__Series_2.csv` |
| 10.1016/j.nanoen.2021.106445 | Fig5 | _C | 188.78 | `Fig5_C_stability_performance_curve_pce` | `10.1016_j.nanoen.2021.106445__Fig5__C__stability_performance_curve_pce__Control__Series_2.csv` |
| 10.3390/coatings9110766 | Fig8 | _D | 189.76 | `Fig8_D_stability_performance_curve_pce` | `10.3390_coatings9110766__Fig8__D__stability_performance_curve_pce__100%_toluene__Series_1.csv` |
| 10.1016/j.electacta.2016.09.138 | Fig6 | _D | 189.88 | `Fig6_D_eis_plot_stability_performance_curve` | `10.1016_j.electacta.2016.09.138__Fig6__D__eis_plot_stability_performance_curve__PEDOT_PSS__Series_1.csv` |
| 10.1016/j.orgel.2019.105384 | Fig5 | - | 190.47 | `Fig5_whole_figure` | `10.1016_j.orgel.2019.105384__Fig5__whole__figure__MAPbl3__Series_2.csv` |
| 10.1016/j.orgel.2019.105384 | Fig5 | - | 191.27 | `Fig5_whole_figure` | `10.1016_j.orgel.2019.105384__Fig5__whole__figure__MAPbI3_PCBM__Series_1.csv` |
| 10.1007/s40843-019-1174-3 | Fig7 | _D | 190.51 | `Fig7_D_stability_performance_curve_pce` | `10.1007_s40843-019-1174-3__Fig7__D__stability_performance_curve_pce__3D__Series_2.csv` |
| 10.1016/j.solmat.2019.04.007 | Fig6 | _B | 190.71 | `Fig6_B_stability` | `10.1016_j.solmat.2019.04.007__Fig6__B__stability__PCBM__Series_2.csv` |
| 10.1039/c8ta11977e | Fig8 | _A | 190.72 | `Fig8_A_stability_performance_curve_pce` | `10.1039_c8ta11977e__Fig8__A__stability_performance_curve_pce__Control_device__Series_1.csv` |
| 10.1002/aenm.201601156 | Fig4 | _A | 191.62 | `Fig4_A_pce` | `10.1002_aenm.201601156__Fig4__A__pce__Spiro-OMeTAD__Series_1.csv` |
| 10.1039/c9ta05744g | Fig6 | _B | 191.92 | `Fig6_B_mppt_stability_performance_curve` | `10.1039_c9ta05744g__Fig6__B__mppt_stability_performance_curve__W_OMWCNTs__Series_2.csv` |
| 10.1039/c7tc00882a | Fig7 | - | 192.31 | `Fig7_whole_stability` | `10.1039_c7tc00882a__Fig7__whole__stability__w_o__Series_1.csv` |
| 10.1039/c6ee00612d | Fig2 | _E | 192.86 | `Fig2_E_mppt_stability_performance_curve_eqe` | `10.1039_c6ee00612d__Fig2__E__mppt_stability_performance_curve_eqe__Ag_W_o_CIL__Series_2.csv` |
| 10.1002/celc.201701054 | Fig7 | _B | 193.35 | `Fig7_B_stability_performance_curve` | `10.1002_celc.201701054__Fig7__B__stability_performance_curve__α-Fe203__Series_2.csv` |
| 10.1002/smtd.201900476 | Fig5 | _D | 193.37 | `Fig5_D_mppt_stability_performance_curve_pce_eqe` | `10.1002_smtd.201900476__Fig5__D__mppt_stability_performance_curve_pce_eqe__SnO₂__Series_2.csv` |
| 10.1002/adma.202205769 | Fig4 | _B | 193.84 | `Fig4_B_stability_performance_curve_pce` | `10.1002_adma.202205769__Fig4__B__stability_performance_curve_pce__PEDOT_PSS__Series_2.csv` |
| 10.1039/c9ta05422g | Fig9 | _F | 193.98 | `Fig9_F_stability_performance_curve` | `10.1039_c9ta05422g__Fig9__F__stability_performance_curve__Ti02_SnO2__Series_2.csv` |
| 10.1039/c9ta05422g | Fig9 | _F | 195.28 | `Fig9_F_stability_performance_curve` | `10.1039_c9ta05422g__Fig9__F__stability_performance_curve__Ti02__Series_1.csv` |
| 10.1021/jacs.7b13229 | Fig3 | _D | 194.03 | `Fig3_D_stability_performance_curve_pce_jsc` | `10.1021_jacs.7b13229__Fig3__D__stability_performance_curve_pce_jsc__Organic-inorganic_PSCs__Series_2.csv` |
| 10.1016/j.nanoen.2020.104929 | Fig5 | _D | 194.05 | `Fig5_D_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.104929__Fig5__D__stability_performance_curve_pce__MAPbI₃__Series_2.csv` |
| 10.1002/adma.201701221 | Fig5 | _B | 195.53 | `Fig5_B_stability_performance_curve_pce` | `10.1002_adma.201701221__Fig5__B__stability_performance_curve_pce__spiro__Series_1.csv` |
| 10.1002/adma.201701221 | Fig5 | _C | 195.69 | `Fig5_C_stability_performance_curve_pce` | `10.1002_adma.201701221__Fig5__C__stability_performance_curve_pce__w_o__Series_3.csv` |
| 10.1002/admi.202001144 | Fig3 | _B | 195.73 | `Fig3_B_stability_performance_curve_pce` | `10.1002_admi.202001144__Fig3__B__stability_performance_curve_pce__C₆₀-BCT@Au_NPs__Series_1.csv` |
| 10.1002/admi.202001144 | Fig3 | _B | 196.10 | `Fig3_B_stability_performance_curve_pce` | `10.1002_admi.202001144__Fig3__B__stability_performance_curve_pce__c-TiO₂__Series_3.csv` |
| 10.1002/admi.202001144 | Fig3 | _B | 198.70 | `Fig3_B_stability_performance_curve_pce` | `10.1002_admi.202001144__Fig3__B__stability_performance_curve_pce__PCBM__Series_2.csv` |
| 10.1016/j.jechem.2021.05.042 | Fig6 | _F | 195.78 | `Fig6_F_stability_performance_curve` | `10.1016_j.jechem.2021.05.042__Fig6__F__stability_performance_curve__Control__Series_3.csv` |
| 10.1016/j.nanoen.2020.105181 | Fig6 | _B | 196.38 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.nanoen.2020.105181__Fig6__B__stability_performance_curve_pce__S6__Series_3.csv` |
| 10.1016/j.jpowsour.2020.228818 | Fig6 | _B | 196.43 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.jpowsour.2020.228818__Fig6__B__stability_performance_curve_pce__MAPb0.99TI0.013__Series_2.csv` |
| 10.1016/j.jpowsour.2020.228818 | Fig6 | _B | 199.48 | `Fig6_B_stability_performance_curve_pce` | `10.1016_j.jpowsour.2020.228818__Fig6__B__stability_performance_curve_pce__MAPbl3__Series_1.csv` |
| 10.1016/j.xcrp.2021.100450 | Fig5 | _F | 196.53 | `Fig5_F_stability_performance_curve_pce_image` | `10.1016_j.xcrp.2021.100450__Fig5__F__stability_performance_curve_pce_image__Control__Series_2.csv` |
| 10.1039/c8ta02121j | Fig6 | _C | 196.90 | `Fig6_C_stability_performance_curve_pce` | `10.1039_c8ta02121j__Fig6__C__stability_performance_curve_pce__Without_CH₃CONH₂__Series_1.csv` |
| 10.1016/j.solmat.2019.110335 | Fig1 | _F | 196.91 | `Fig1_F_schematic_diagram_stability_performance_curve_pce` | `10.1016_j.solmat.2019.110335__Fig1__F__schematic_diagram_stability_performance_curve_pce__with_PC60BM__Series_1.csv` |
| 10.1016/j.solmat.2019.110335 | Fig1 | _E | 198.83 | `Fig1_E_schematic_diagram_pce` | `10.1016_j.solmat.2019.110335__Fig1__E__schematic_diagram_pce__withoutPC6oBM__Series_1.csv` |
| 10.1016/j.solmat.2019.110383 | Fig3 | - | 197.20 | `Fig3_whole_figure` | `10.1016_j.solmat.2019.110383__Fig3__whole__figure__Series_1__Series_1.csv` |
| 10.1021/acsami.9b22627 | Fig9 | _C | 197.20 | `Fig9_C_mppt_stability_performance_curve` | `10.1021_acsami.9b22627__Fig9__C__mppt_stability_performance_curve__MAPbI_C1__Series_1.csv` |
| 10.1002/advs.201903047 | Fig4 | _D | 197.27 | `Fig4_D_stability_performance_curve_pce` | `10.1002_advs.201903047__Fig4__D__stability_performance_curve_pce__Current_method__Series_2.csv` |
| 10.1002/advs.201903047 | Fig4 | _D | 198.99 | `Fig4_D_stability_performance_curve_pce` | `10.1002_advs.201903047__Fig4__D__stability_performance_curve_pce__One_step__Series_1.csv` |
| 10.1039/c7ta04851c | Fig5 | _D | 197.27 | `Fig5_D_stability_performance_curve_pce` | `10.1039_c7ta04851c__Fig5__D__stability_performance_curve_pce__O-PC61BM__Series_1.csv` |
| 10.3390/cryst12091194 | Fig5 | _D | 197.48 | `Fig5_D_stability_performance_curve_pce` | `10.3390_cryst12091194__Fig5__D__stability_performance_curve_pce__0%_FAHCOO__Series_2.csv` |
| 10.1039/c9tc06578d | Fig5 | _B | 197.71 | `Fig5_B_stability_performance_curve_pce` | `10.1039_c9tc06578d__Fig5__B__stability_performance_curve_pce__Control__Series_1.csv` |
| 10.1021/acsenergylett.1c00443 | Fig3 | _I | 197.92 | `Fig3_I_stability_performance_curve` | `10.1021_acsenergylett.1c00443__Fig3__I__stability_performance_curve__SnO₂__Series_1.csv` |
| 10.1039/d0ta11916d | Fig5 | _C | 198.22 | `Fig5_C_stability_performance_curve` | `10.1039_d0ta11916d__Fig5__C__stability_performance_curve__Raw_film__Series_2.csv` |
| 10.1039/d0ta05676f | Fig7 | _C | 198.23 | `Fig7_C_stability_performance_curve_pattern` | `10.1039_d0ta05676f__Fig7__C__stability_performance_curve_pattern__W_O_FACI__Series_2.csv` |
| 10.1002/aenm.201900243 | Fig6 | _C | 198.29 | `Fig6_C_stability_performance_curve` | `10.1002_aenm.201900243__Fig6__C__stability_performance_curve__wlo__Series_2.csv` |
| 10.1016/j.nanoen.2018.09.037 | Fig6 | _D | 198.37 | `Fig6_D_stability_performance_curve` | `10.1016_j.nanoen.2018.09.037__Fig6__D__stability_performance_curve__TiO₂_based_PSCs__Series_2.csv` |
| 10.1021/acsnano.6b01904 | Fig8 | _B | 198.50 | `Fig8_B_stability_performance_curve_pce_image` | `10.1021_acsnano.6b01904__Fig8__B__stability_performance_curve_pce_image__PEDOT_PSS__Series_1.csv` |
| 10.1016/j.solener.2020.01.048 | Fig4 | _D | 198.63 | `Fig4_D_stability_performance_curve_eqe` | `10.1016_j.solener.2020.01.048__Fig4__D__stability_performance_curve_eqe__Uncoated_PsCs__Series_2.csv` |
| 10.1016/j.jpcs.2020.109792 | Fig5 | _C | 198.96 | `Fig5_C_stability_performance_curve_pce` | `10.1016_j.jpcs.2020.109792__Fig5__C__stability_performance_curve_pce__ACN_0%__Series_1.csv` |
| 10.1016/j.solmat.2019.110297 | Fig6 | _F | 199.50 | `Fig6_F_stability_performance_curve_pce` | `10.1016_j.solmat.2019.110297__Fig6__F__stability_performance_curve_pce__control_device__Series_3.csv` |


### x_time_min

| DOI | Figure | 子图 | 最大时间 (h) | 文件夹 | CSV |
|-----|--------|------|-------------|--------|-----|
| 10.1021/acsenergylett.0c01207 | Fig5 | _A | 174.40 | `Fig5_A_pce` | `10.1021_acsenergylett.0c01207__Fig5__A__pce__α_δ-FA0.33MA0.33Cs0.33PbI3__Series_1.csv` |
| 10.1021/acsenergylett.0c01207 | Fig5 | _A | 175.40 | `Fig5_A_pce` | `10.1021_acsenergylett.0c01207__Fig5__A__pce__α-FA0.76MA0.15Cs0.09PbI3__Series_2.csv` |

---

## 附:文件清单

- `analysis_report/curves_under_200h_detailed.csv` — 每条曲线一行（含 panel）
- `analysis_report/curves_under_200h_doi_summary.csv` — 每个 (DOI, Figure) 一行

*扫描脚本：`analysis_report/find_under_200h.py`*