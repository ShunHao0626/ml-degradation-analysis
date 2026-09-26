#!/usr/bin/env python3
"""Convert manually marked source-image pixels into a traceable curve version."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import rows, sha, write_rows


def axis_value(pixel, spec):
    p1, p2 = float(spec["pixel_1"]), float(spec["pixel_2"])
    v1, v2 = float(spec["value_1"]), float(spec["value_2"])
    if p1 == p2:
        raise ValueError("two distinct pixel ticks are required")
    fraction = (np.asarray(pixel, dtype=float)-p1)/(p2-p1)
    if spec.get("scale", "linear") == "linear":
        return v1+fraction*(v2-v1)
    if spec["scale"] == "log10":
        if v1 <= 0 or v2 <= 0:
            raise ValueError("log10 tick values must be positive")
        return 10**(np.log10(v1)+fraction*(np.log10(v2)-np.log10(v1)))
    raise ValueError("scale must be linear or log10")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root",type=Path,default=Path("data_final"))
    ap.add_argument("--output",type=Path,default=Path("target_shape_classification/results"))
    ap.add_argument("--source-csv",required=True,help="path relative to data root")
    ap.add_argument("--points",type=Path,required=True,help="CSV: pixel_x,pixel_y,provenance")
    ap.add_argument("--calibration",type=Path,required=True,help="JSON with x/y two-tick calibration")
    ap.add_argument("--status",choices=["draft","accepted"],default="draft")
    args=ap.parse_args()
    root,out=args.data_root.resolve(),args.output.resolve()
    source=(root/args.source_csv).resolve()
    if not source.exists() or not source.is_relative_to(root):
        raise ValueError("source CSV must exist under data root")
    manifest={r["source_csv"]:r for r in rows(out/"input_manifest.csv")}
    record=manifest.get(args.source_csv)
    if not record:
        raise ValueError("source CSV absent from audited input manifest")
    if sha(source)!=record["source_sha256"]:
        raise ValueError("source CSV changed since audit")
    image=root/record["source_image"]
    if not image.exists() or sha(image)!=record["image_sha256"]:
        raise ValueError("source image absent or hash changed")
    config=json.loads(args.calibration.read_text(encoding="utf-8"))
    if config.get("source_image_sha256")!=record["image_sha256"]:
        raise ValueError("calibration image hash does not match source image")
    for axis in ("x","y"):
        for field in ("pixel_1","pixel_2","value_1","value_2"):
            if field not in config.get(axis,{}): raise ValueError(f"missing {axis}.{field}")
    points=rows(args.points)
    if len(points)<2: raise ValueError("at least two visible trace points are required")
    allowed={"source_marker","digitized_trace"}
    if any(p.get("provenance") not in allowed for p in points):
        raise ValueError("provenance must be source_marker or digitized_trace; interpolation is not source evidence")
    px=np.array([float(p["pixel_x"]) for p in points])
    py=np.array([float(p["pixel_y"]) for p in points])
    image_array=plt.imread(image)
    if np.any(px<0) or np.any(py<0) or np.any(px>=image_array.shape[1]) or np.any(py>=image_array.shape[0]):
        raise ValueError("pixel point outside source image")
    x=axis_value(px,config["x"]); y=axis_value(py,config["y"])
    folder=record["folder"]
    xcol,ycol=("x","y") if folder=="data_all" else ("time_h","normalized_pce")
    version_dir=out/"redigitized"/record["file_id"]
    version_dir.mkdir(parents=True,exist_ok=True)
    version_hash=sha(args.points)[:12]+sha(args.calibration)[:12]
    curve_path=version_dir/f"{version_hash}.csv"
    converted=[{xcol:float(a),ycol:float(b),"pixel_x":float(c),"pixel_y":float(d),
                "point_source":p["provenance"],"point_note":p.get("note","")}
               for a,b,c,d,p in zip(x,y,px,py,points)]
    write_rows(curve_path,converted,[xcol,ycol,"pixel_x","pixel_y","point_source","point_note"])
    calibration_path=version_dir/f"{version_hash}_calibration.json"
    calibration_path.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,6));ax.imshow(image_array)
    ax.scatter(px,py,s=25,facecolors="none",edgecolors="magenta",linewidths=1.2)
    ax.axis("off");fig.tight_layout()
    overlay=out/"source_overlays"/f"{record['file_id']}_{version_hash}.png"
    overlay.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(overlay,dpi=140);plt.close(fig)
    manifest_path=out/"digitization_manifest.csv"
    existing=[r for r in rows(manifest_path) if r.get("source_csv")!=args.source_csv]
    existing.append(dict(source_csv=args.source_csv,source_sha256=record["source_sha256"],
        source_image=record["source_image"],source_image_sha256=record["image_sha256"],
        digitized_csv=str(curve_path.relative_to(out)),digitized_sha256=sha(curve_path),
        status=args.status,calibration_json=str(calibration_path.relative_to(out)),
        overlay_png=str(overlay.relative_to(out)),point_provenance=";".join(sorted(set(p["provenance"] for p in points))),
        notes=config.get("notes","")))
    write_rows(manifest_path,existing,["source_csv","source_sha256","source_image","source_image_sha256",
        "digitized_csv","digitized_sha256","status","calibration_json","overlay_png","point_provenance","notes"])
    print(f"Saved {curve_path} ({args.status}); {len(points)} source-visible points")


if __name__=="__main__":
    main()
