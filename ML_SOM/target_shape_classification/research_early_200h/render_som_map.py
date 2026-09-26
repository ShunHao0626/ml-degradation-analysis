#!/usr/bin/env python3
"""U-matrix, occupancy, and real train-member atlas for the frozen SOM."""
from __future__ import annotations

import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from early_data import HERE, rows
from masked_som import MaskedSOM


def main():
    summary = json.loads((HERE / "runs/summary.json").read_text())
    saved = np.load(HERE / "runs" / f"{summary['selected_run']}.npz")
    cfg = summary["config"]
    model = MaskedSOM(cfg["map_rows"], cfg["map_columns"], list(saved["feature_sizes"]),
                      list(saved["weights"]), int(saved["seed"]))
    model.prototypes, model.supported = saved["prototypes"], saved["supported"]
    stored = np.load(HERE / "manifests/strict_features.npz")
    ids = [str(x) for x in stored["curve_ids"]]
    split = {r["curve_id"]: r["split"] for r in rows(HERE / "delivery/class_index.csv")}
    train = [i for i, fid in enumerate(ids) if split[fid] == "train"]
    validation = [i for i, fid in enumerate(ids) if split[fid] == "validation"]
    bmu_train = model.assign(stored["data"][train], stored["mask"][train])[0]
    bmu_val = model.assign(stored["data"][validation], stored["mask"][validation])[0]
    train_counts = np.bincount(bmu_train, minlength=len(model.prototypes)).reshape(model.rows, model.columns)
    val_counts = np.bincount(bmu_val, minlength=len(model.prototypes)).reshape(model.rows, model.columns)
    matrix = np.zeros((model.rows, model.columns))
    for r in range(model.rows):
        for c in range(model.columns):
            here = r * model.columns + c
            neighbors = [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                         if 0 <= rr < model.rows and 0 <= cc < model.columns]
            values = []
            for rr, cc in neighbors:
                other = rr * model.columns + cc
                total = 0.
                for weight, sl in zip(model.weights, model.slices):
                    m = model.supported[sl]
                    diff = model.prototypes[here, sl][m] - model.prototypes[other, sl][m]
                    total += weight * np.mean(diff ** 2)
                values.append(np.sqrt(total / sum(model.weights)))
            matrix[r, c] = np.mean(values)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    for ax, values, title in zip(axes, (matrix, train_counts, val_counts),
                                 ("U-matrix (neighbor distance)", "Train occupancy", "Validation occupancy")):
        image = ax.imshow(values, cmap="viridis", origin="upper")
        fig.colorbar(image, ax=ax, shrink=.75)
        ax.set_title(title)
        ax.set_xlabel("SOM column")
        ax.set_ylabel("SOM row")
    fig.tight_layout()
    output = HERE / "evaluation"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "som_map.png")
    plt.close(fig)
    index = {r["curve_id"]: r for r in rows(HERE / "delivery/class_index.csv")}
    nodes = rows(HERE / "runs" / f"{summary['selected_run']}_node_interpretation.csv")
    content = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Anonymous SOM atlas</title>",
               "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}img{max-width:800px;width:100%}</style>",
               "<h1>匿名 SOM 地图与真实训练代表曲线</h1>",
               "<p>节点后验名称来自规则建议，尚未经过独立专家确认。单个代表也不能证明节点所有成员同类。</p>",
               "<img src='som_map.png'><table><tr><th>Node</th><th>Train</th><th>Validation</th><th>Real train member</th><th>Posthoc suggestion</th><th>Basis</th></tr>"]
    for row in nodes:
        node = int(row["node"])
        fid = row["representative_curve_id"]
        curve = index[fid]
        url = "../delivery/" + curve["curve_png"]
        content.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in
                      (node, int(train_counts.flat[node]), int(val_counts.flat[node]))) +
                      f"<td><a href='{html.escape(url, quote=True)}'>{html.escape(fid)}</a></td>" +
                      "".join(f"<td>{html.escape(row[k])}</td>" for k in ("posthoc_class", "interpretation_basis")) + "</tr>")
    content.append("</table>")
    (output / "node_atlas.html").write_text("\n".join(content), encoding="utf-8")
    print(json.dumps({"nodes": len(nodes), "train_occupied": int(np.sum(train_counts > 0)),
                      "validation_occupied": int(np.sum(val_counts > 0))}))


if __name__ == "__main__":
    main()
