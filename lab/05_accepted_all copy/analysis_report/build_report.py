#!/usr/bin/env python3
"""Generate a human-readable, per-directory report listing under-200h curves.

Outputs:
  analysis_report/curves_under_200h_detailed.csv  -- per-series rows
  analysis_report/curves_under_200h_doi_summary.csv -- per (DOI, figure+panel) row
  analysis_report/curves_under_200h_report.md  -- full Markdown report
"""

import csv
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict

BASE_DIR = Path("/Users/shunhao/Desktop/ML/lab/05_accepted_all copy")

# Reuse logic from the scanner by importing its core functions
sys.path.insert(0, str(BASE_DIR / "analysis_report"))
from find_under_200h import (
    get_replot_or_validation_unit,
    read_x_values_from_csv,
    doi_from_dirname,
    figure_from_folder,
)


def is_panel(figure_label: str) -> bool:
    """Return True if figure_label has a single letter suffix (Fig5_D)."""
    return bool(re.match(r"^Fig\d+[a-z]$", figure_label, re.IGNORECASE))


def short_figure(figure_label: str) -> str:
    """Drop panel letter from a figure label. 'Fig5_D' -> 'Fig5', 'Fig23' -> 'Fig23'."""
    parts = figure_label.split("_", 1)
    return parts[0] if parts else figure_label


def main():
    out_dir = BASE_DIR / "analysis_report"
    out_dir.mkdir(exist_ok=True)

    all_curves = []
    for top in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        top_path = BASE_DIR / top
        if not top_path.exists():
            continue
        for doi_dir in sorted(top_path.iterdir()):
            if not doi_dir.is_dir():
                continue
            doi = doi_from_dirname(doi_dir.name)
            for fig_dir in doi_dir.iterdir():
                if not fig_dir.is_dir() or fig_dir.name in ("image",):
                    continue
                fig_label = figure_from_folder(fig_dir.name)
                h_factor, unit_label = get_replot_or_validation_unit(fig_dir)
                accepted_dir = fig_dir / "accepted"
                csv_files = sorted(accepted_dir.glob("*.csv")) if accepted_dir.is_dir() else []
                if not csv_files:
                    csv_files = sorted(fig_dir.glob("*.csv"))
                for csv_file in csv_files:
                    xs = read_x_values_from_csv(csv_file)
                    if not xs:
                        continue
                    max_x = max(xs)
                    max_x_h = max_x * h_factor
                    all_curves.append({
                        "top_dir": top,
                        "doi": doi,
                        "doi_dir": doi_dir.name,
                        "figure_folder": fig_dir.name,
                        "figure_label": fig_label,
                        "figure_short": short_figure(fig_label),
                        "panel_letter": fig_label[len(short_figure(fig_label)):],
                        "csv_file": csv_file.relative_to(BASE_DIR).as_posix(),
                        "unit_label": unit_label or "?",
                        "max_x_raw": max_x,
                        "max_x_hours": max_x_h,
                        "n_points": len(xs),
                    })

    under200 = [c for c in all_curves if c["max_x_hours"] < 200.0]
    over200 = [c for c in all_curves if c["max_x_hours"] >= 200.0]
    under200.sort(key=lambda r: r["max_x_hours"])

    # ------ per-series CSV (already similar, include all info)
    with open(out_dir / "curves_under_200h_detailed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["top_dir", "doi", "figure_label", "figure_short", "panel_letter",
                    "figure_folder", "unit_label", "max_x_raw", "max_x_hours",
                    "n_points", "csv_file"])
        for r in under200:
            w.writerow([r["top_dir"], r["doi"], r["figure_label"], r["figure_short"],
                        r["panel_letter"], r["figure_folder"], r["unit_label"],
                        round(r["max_x_raw"], 3), round(r["max_x_hours"], 3),
                        r["n_points"], r["csv_file"]])

    # ------ DOI summary: per (DOI, figure_short) -> min/max hours, panels, source dir
    doi_summary = defaultdict(lambda: {
        "hours": [],
        "panels": set(),
        "max_panel_hours": {},
        "top_dirs": set(),
        "unit_labels": set(),
    })
    for r in under200:
        k = (r["doi"], r["figure_short"])
        doi_summary[k]["hours"].append(r["max_x_hours"])
        doi_summary[k]["panels"].add(r["panel_letter"] if r["panel_letter"] else r["figure_short"])
        label_key = r["figure_label"]
        if label_key not in doi_summary[k]["max_panel_hours"] or r["max_x_hours"] > doi_summary[k]["max_panel_hours"][label_key]:
            doi_summary[k]["max_panel_hours"][label_key] = r["max_x_hours"]
        doi_summary[k]["top_dirs"].add(r["top_dir"])
        doi_summary[k]["unit_labels"].add(r["unit_label"])

    summary_rows = []
    for (doi, fig_short), v in doi_summary.items():
        panels = sorted(v["panels"], key=lambda p: (len(p), p))
        summary_rows.append({
            "doi": doi,
            "figure": fig_short,
            "panels": ", ".join(panels),
            "min_hours": round(min(v["hours"]), 2),
            "max_hours": round(max(v["hours"]), 2),
            "top_dirs": ", ".join(sorted(v["top_dirs"])),
            "unit": ", ".join(sorted(v["unit_labels"])) or "?",
        })
    summary_rows.sort(key=lambda r: r["max_hours"])

    with open(out_dir / "curves_under_200h_doi_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doi", "figure", "panels", "min_hours", "max_hours", "source_dirs", "unit"])
        for r in summary_rows:
            w.writerow([r["doi"], r["figure"], r["panels"], r["min_hours"], r["max_hours"], r["top_dirs"], r["unit"]])

    # ------ Markdown report
    by_top = defaultdict(lambda: [0, 0])
    for c in all_curves:
        by_top[c["top_dir"]][1] += 1
    for c in under200:
        by_top[c["top_dir"]][0] += 1

    unique_dois = len({c["doi"] for c in under200})
    unique_pairs = len({(c["doi"], c["figure_short"]) for c in under200})

    lines = []
    lines.append("# 未超过 200 小时曲线汇总报告")
    lines.append("")
    lines.append(f"生成时间: 2026-07-28（数据来自 `x_time_day`、`x_time_h`、`x_time_min`、`x_time_week_month_year` 四个目录）")
    lines.append("")
    lines.append("## 一、扫描统计")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| 总曲线数（CSV 数据序列数） | {len(all_curves)} |")
    lines.append(f"| 总 (DOI, Figure) 数（去重） | {len({(c['doi'], c['figure_short']) for c in all_curves})} |")
    lines.append(f"| 唯一 DOI 总数 | {len({c['doi'] for c in all_curves})} |")
    lines.append(f"| **最大时间 < 200 h 的曲线数** | **{len(under200)}** |")
    lines.append(f"| **最大时间 ≥ 200 h 的曲线数** | **{len(over200)}** |")
    lines.append(f"| **未达标的唯一 DOI 数** | **{unique_dois}** |")
    lines.append(f"| **未达标的唯一 (DOI, Figure) 数** | **{unique_pairs}** |")
    lines.append("")
    lines.append("## 二、按目录划分")
    lines.append("")
    lines.append("| 目录 | < 200 h 曲线数 | 总曲线数 | < 200 h 唯一 DOI 数 | < 200 h 唯一 (DOI, Figure) 对数 |")
    lines.append("|------|---------------|---------|------------------|----------------------------|")
    for top in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        u = sum(1 for c in under200 if c["top_dir"] == top)
        t = sum(1 for c in all_curves if c["top_dir"] == top)
        u_dois = len({c["doi"] for c in under200 if c["top_dir"] == top})
        u_pairs = len({(c["doi"], c["figure_short"]) for c in under200 if c["top_dir"] == top})
        lines.append(f"| `{top}` | {u} | {t} | {u_dois} | {u_pairs} |")
    lines.append("")
    lines.append("> 备注：`x_time_week_month_year` 中存放的曲线虽然在数据中以\"小数值\"出现（≤ 5），但根据 `validation_result.json` 中的单位（months / weeks）换算后均**已超过** 200 小时，因此未列入下方\"未达标\"清单。")
    lines.append("")
    lines.append("## 三、按 (DOI, Figure) 汇总的最大时间未超过 200 h 的曲线（按最大时间升序）")
    lines.append("")
    lines.append("| DOI | Figure | 涉及子图 | 单位 | 最大时间 | 源目录 |")
    lines.append("|-----|--------|----------|------|----------|--------|")
    for r in summary_rows:
        lines.append(f"| {r['doi']} | {r['figure']} | {r['panels']} | {r['unit']} | {r['max_hours']:.2f} h | {r['top_dirs']} |")
    lines.append("")

    lines.append("## 四、按目录分子表")
    for top in ["x_time_day", "x_time_h", "x_time_min", "x_time_week_month_year"]:
        rows = [c for c in under200 if c["top_dir"] == top]
        if not rows:
            continue
        lines.append("")
        lines.append(f"### {top}")
        lines.append("")
        # Sub-group by DOI for readability
        doi_to_rows = defaultdict(list)
        for r in rows:
            doi_to_rows[r["doi"]].append(r)
        ordered = sorted(doi_to_rows.keys(), key=lambda d: min(rr["max_x_hours"] for rr in doi_to_rows[d]))
        lines.append("| DOI | Figure | 子图 | 最大时间 (h) | 文件夹 | CSV |")
        lines.append("|-----|--------|------|-------------|--------|-----|")
        for d in ordered:
            rrs = sorted(doi_to_rows[d], key=lambda r: r["max_x_hours"])
            for r in rrs:
                lines.append(f"| {r['doi']} | {r['figure_short']} | {r['panel_letter'] or '-'} | {r['max_x_hours']:.2f} | `{r['figure_folder']}` | `{Path(r['csv_file']).name}` |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 附:文件清单")
    lines.append("")
    lines.append("- `analysis_report/curves_under_200h_detailed.csv` — 每条曲线一行（含 panel）")
    lines.append("- `analysis_report/curves_under_200h_doi_summary.csv` — 每个 (DOI, Figure) 一行")
    lines.append("")
    lines.append("*扫描脚本：`analysis_report/find_under_200h.py`*")

    rpt = out_dir / "curves_under_200h_report.md"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {rpt}")
    print(f"Wrote {out_dir / 'curves_under_200h_doi_summary.csv'}")
    print(f"Wrote {out_dir / 'curves_under_200h_detailed.csv'}")
    print(f"\nTotals: under200={len(under200)}, over200={len(over200)}, unique DOIs<200h={unique_dois}, unique (DOI,Figure)={unique_pairs}")


if __name__ == "__main__":
    main()
