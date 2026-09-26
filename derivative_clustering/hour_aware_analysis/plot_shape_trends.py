"""Reference-style morphology galleries, separate from physical-hour plots.

Uses existing assignments and the shape model's existing 48-point smoothed
level representation. The black line is the exact pointwise median of the
displayed, relative-progress curves, not a fit to an assumed class template.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hour_model import HERE, baseline


def relative_progress_levels(physical):
    levels=[]
    for ch in physical:
        obs=pd.DataFrame({"time_h":ch.time_h.to_numpy(),
                          "y_relative":ch.y_max_normalized.to_numpy()})
        u,channels,_,_=baseline.prior.prepare_curve(obs,baseline.CONFIG)
        levels.append(channels[:,0])
    return u,np.stack(levels)


def cluster_medians(levels,labels):
    return {int(shape):np.median(levels[labels==shape],axis=0)
            for shape in np.unique(labels)}


def draw_panel(ax,u,levels,median,shape):
    for y in levels:
        ax.plot(u,y,color="#8da6b5",alpha=.19,lw=1.05,zorder=1)
    ax.plot(u,median,color="black",lw=2.8,zorder=4)
    ax.set_title(f"Shape cluster {shape}: n={len(levels)}",fontsize=17,pad=11)
    ax.set_xlim(-.045,1.045)
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_xlabel("Normalized time progress (0-1)",fontsize=12)
    ax.set_ylabel("Normalized response",fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(color="#d8d8d8",alpha=.42,lw=.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def render(folder):
    folder=Path(folder)
    source_names=("hour_assignments.csv","physical_channels.csv","raw_observations.csv")
    before={name:baseline.sha(folder/name) for name in source_names}
    a=pd.read_csv(folder/"hour_assignments.csv")
    physical=pd.read_csv(folder/"physical_channels.csv")
    groups=dict(tuple(physical.groupby("curve_id",sort=False)))
    u,levels=relative_progress_levels([groups[c] for c in a.curve_id])
    labels=a.shape_cluster.to_numpy()
    medians=cluster_medians(levels,labels)
    shapes=sorted(medians)
    ymin=min(.1,float(levels.min())-.035)
    ymax=max(1.04,float(levels.max())+.025)
    rows=int(np.ceil(len(shapes)/2))
    fig,axes=plt.subplots(rows,2,figsize=(12,4.55*rows),squeeze=False)
    output_rows=[]
    for ax,shape in zip(axes.flat,shapes):
        selected=levels[labels==shape]
        draw_panel(ax,u,selected,medians[shape],shape)
        ax.set_ylim(ymin,ymax)
        output_rows.append(pd.DataFrame({"shape_cluster":shape,"normalized_time_progress":u,
                                        "median_response":medians[shape],"n_curves":len(selected)}))
        single,single_ax=plt.subplots(figsize=(6.4,4.8))
        draw_panel(single_ax,u,selected,medians[shape],shape)
        single_ax.set_ylim(ymin,ymax)
        single.tight_layout()
        single.savefig(folder/f"shape_cluster_{shape}_median_trend.png",dpi=200)
        plt.close(single)
    for ax in list(axes.flat)[len(shapes):]:
        ax.axis("off")
    fig.text(.5,.009,"Black: pointwise median of aligned shape curves. Horizontal axis is relative progress, not hours.",
             ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.032,1,1),h_pad=2,w_pad=2)
    fig.savefig(folder/"shape_groups_median_trends.png",dpi=200)
    fig.savefig(folder/"shape_groups_median_trends.svg")
    plt.close(fig)
    pd.concat(output_rows,ignore_index=True).to_csv(folder/"shape_median_trends.csv",index=False)
    np.savez_compressed(folder/"shape_aligned_levels.npz",normalized_time_progress=u,
                        levels=levels,shape_cluster=labels,curve_id=a.curve_id.to_numpy(dtype=str))
    assert before=={name:baseline.sha(folder/name) for name in source_names}
    method={"source_sha256":before,"source_assignments_and_observations_unchanged":True,
            "x_axis":"u=(time_h-start_h)/(end_h-start_h), NOT physical hours",
            "background":"existing shape-model level preprocessing: 48-point PCHIP + existing outlier handling and SG(7,3)",
            "black_line":"exact pointwise median of displayed preprocessed curves; no additional median smoothing",
            "clustering":"existing assignments unchanged; no refitting, relabeling, or class template",
            "interpretation":"morphology summary, not a lifetime or a physical-hour median",
            "paper":"visual style reference only; no claim to reproduce the paper's statistical method",
            "plot_code_sha256":baseline.sha(Path(__file__))}
    (folder/"shape_median_trends_method.json").write_text(json.dumps(method,indent=2),encoding="utf-8")
    print(f"Saved {len(shapes)} median trends: {folder/'shape_groups_median_trends.png'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,default=HERE/"outputs"/"existing_400")
    render(parser.parse_args().folder)
