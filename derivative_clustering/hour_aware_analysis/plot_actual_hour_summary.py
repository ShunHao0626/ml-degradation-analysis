"""Gray individual curves + red cluster summary, on unchanged physical hours."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hour_model import HERE, baseline


def hour_summaries(channels):
    grid=np.unique(np.concatenate([ch.time_h.to_numpy(float) for ch in channels]))
    values=np.full((len(channels),len(grid)),np.nan)
    for i,ch in enumerate(channels):
        t=ch.time_h.to_numpy(float)
        y=ch.y_max_normalized.to_numpy(float)
        inside=(grid>=t[0])&(grid<=t[-1])
        values[i,inside]=np.interp(grid[inside],t,y)
    return pd.DataFrame({"time_h":grid,"mean":np.nanmean(values,axis=0),
                         "median":np.nanmedian(values,axis=0),
                         "n_contributing":np.isfinite(values).sum(axis=0),
                         "n_group":len(channels)})


def render(folder):
    folder=Path(folder)
    source_names=("hour_assignments.csv","physical_channels.csv","raw_observations.csv")
    before={name:baseline.sha(folder/name) for name in source_names}
    assignments=pd.read_csv(folder/"hour_assignments.csv")
    physical=pd.read_csv(folder/"physical_channels.csv")
    by_id=dict(tuple(physical.groupby("curve_id",sort=False)))
    shapes=sorted(assignments.shape_cluster.unique())
    curves={int(s):[by_id[c] for c in assignments.loc[assignments.shape_cluster.eq(s),"curve_id"]] for s in shapes}
    summaries={s:hour_summaries(ch) for s,ch in curves.items()}
    rows=int(np.ceil(len(shapes)/2))
    xmin=float(physical.time_h.min()); xmax=float(physical.time_h.max())
    ymin=min(0.,float(physical.y_max_normalized.min())-.025)
    ymax=max(1.05,float(physical.y_max_normalized.max())+.025)
    for stat in ("mean","median"):
        fig,axes=plt.subplots(rows,2,figsize=(12,4.35*rows),sharex=True,sharey=True,squeeze=False)
        for ax,s in zip(axes.flat,shapes):
            chs=curves[s]; summary=summaries[s]
            for ch in chs:
                ax.plot(ch.time_h,ch.y_max_normalized,color="gray",alpha=.18,lw=.65,zorder=1)
            ax.plot(summary.time_h,summary[stat],color="red",lw=2.8,zorder=3,label=stat.capitalize())
            ax.set_title(f"Cluster {s}\nn={len(chs)}",fontsize=14,pad=10)
            ax.set_xlabel("Time (hours)",fontsize=14)
            ax.set_ylabel("Normalized PCE",fontsize=14)
            ax.set_xlim(xmin-.025*(xmax-xmin),xmax+.025*(xmax-xmin))
            ax.set_ylim(ymin-.02,ymax)
            ax.tick_params(labelsize=11,labelbottom=True)
            ax.legend(loc="best",fontsize=10,frameon=True)
            for spine in ax.spines.values(): spine.set_linewidth(1.1)
        for ax in list(axes.flat)[len(shapes):]: ax.axis("off")
        fig.suptitle(f"Actual-hour Clusters: Individual Series + Cluster {stat.capitalize()}",fontsize=18,y=.985)
        fig.text(.5,.009,"Pointwise summary at actual hours; no time normalization or extrapolation. Fewer curves contribute at later hours.",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.955),h_pad=1.8,w_pad=2)
        name=f"actual_hour_clusters_{stat}"
        fig.savefig(folder/(name+".png"),dpi=200)
        fig.savefig(folder/(name+".svg"))
        plt.close(fig)
        print(folder/(name+".png"))
    pd.concat([f.assign(shape_cluster=s) for s,f in summaries.items()],ignore_index=True).to_csv(folder/"actual_hour_cluster_summaries.csv",index=False)
    assert before=={name:baseline.sha(folder/name) for name in source_names}
    # Median agrees with the actual-hour median previously delivered.
    previous=folder/"shape_medians_actual_hours.csv"
    if previous.exists():
        old=pd.read_csv(previous)
        for s,f in summaries.items():
            p=old.loc[old.shape_cluster.eq(s)]
            np.testing.assert_allclose(f.time_h,p.time_h,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(f["median"],p.median_response,rtol=1e-12,atol=1e-12)
    audit={"source_sha256":before,"source_files_unchanged":True,
           "time_axis":"original time_h; linear hours, no normalized progress",
           "summary":"equal curve weights; linear interpolation within observed interval only; mean and median separately exported",
           "warning":"contributing curve set decreases as observation windows end; late summary is not a fixed-cohort trajectory",
           "clusters":{str(s):len(curves[s]) for s in shapes},
           "code_sha256":baseline.sha(Path(__file__))}
    (folder/"actual_hour_summary_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,default=HERE/"outputs"/"existing_400")
    render(parser.parse_args().folder)
