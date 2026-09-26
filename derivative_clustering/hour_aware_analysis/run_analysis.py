"""Run the hour-aware benchmark, or analyze a folder of individual curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from hour_model import (HERE, ROOT, HourConfig, baseline, fit_hour_hierarchy,
                        physical_description, save_result)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def pointwise_hour_median(channels):
    """Equal-weight pointwise median on actual hours, without extrapolation.

    Each curve contributes one linearly interpolated value only within its own
    observed interval. Save coverage because the contributing set changes with
    hour; this is not an aligned median shape or a typical kinetic trajectory.
    """
    if not channels:
        raise ValueError("Need at least one curve for a median")
    grid=np.unique(np.concatenate([ch.time_h.to_numpy(float) for ch in channels]))
    values=np.full((len(channels),len(grid)),np.nan)
    for i,ch in enumerate(channels):
        t=ch.time_h.to_numpy(float)
        y=ch.y_max_normalized.to_numpy(float)
        if np.any(np.diff(t)<=0) or not np.isfinite(t).all() or not np.isfinite(y).all():
            raise ValueError("Median inputs require finite values and increasing hours")
        mask=(grid>=t[0])&(grid<=t[-1])
        values[i,mask]=np.interp(grid[mask],t,y)
    count=np.isfinite(values).sum(axis=0)
    return pd.DataFrame({"time_h":grid,"median_response":np.nanmedian(values,axis=0),
                         "n_contributing":count,"n_group":len(channels),
                         "coverage_fraction":count/len(channels)})


def plot_hour_median(ax,median,color="black",label="Median"):
    x=median.time_h.to_numpy(); y=median.median_response.to_numpy()
    high=median.coverage_fraction.to_numpy()>=.5
    # Draw the complete median dashed, then overlay the supported part solid.
    ax.plot(x,y,color=color,lw=2.5,ls="--",zorder=5,
            label=label+" (<50% coverage)" if (~high).any() else "_nolegend_")
    ax.plot(x,np.where(high,y,np.nan),color=color,lw=2.8,zorder=6,
            label=label+" (>=50% coverage)")


def refresh_galleries(folder):
    """Replot existing assignments without refitting or altering clustering."""
    folder=Path(folder)
    a=pd.read_csv(folder/"hour_assignments.csv")
    physical=pd.read_csv(folder/"physical_channels.csv")
    groups=dict(tuple(physical.groupby("curve_id",sort=False)))
    result={"assignments":a,"physical":[groups[c] for c in a.curve_id]}
    galleries(result,folder)


def galleries(result, folder, shape_only=False):
    folder=Path(folder)
    a = result["assignments"]
    shapes = sorted(a.shape_cluster.unique())
    medians={}
    fig, axes = plt.subplots(len(shapes), 2, figsize=(12, 2.7*len(shapes)), squeeze=False)
    for row, shape in enumerate(shapes):
        ids = np.flatnonzero(a.shape_cluster.to_numpy() == shape)
        median=pointwise_hour_median([result["physical"][i] for i in ids])
        medians[shape]=median
        for col in range(2):
            ax = axes[row, col]
            for i in ids:
                ch = result["physical"][i]
                ax.plot(ch.time_h, ch.y_max_normalized, alpha=.28, lw=.65)
            if col:
                ax.set_xscale("symlog", linthresh=.1)
            ax.set(title=f"Shape {shape}, n={len(ids)} | {'symlog hours' if col else 'linear hours'}",
                   xlabel="Time (hour)", ylabel="Response / observed max")
            ax.grid(alpha=.2)
    fig.suptitle("Unsupervised shape groups; original hour coordinates")
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(folder/"shape_groups_actual_hours.png", dpi=150)
    plt.close(fig)
    if shape_only:
        return
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for index,shape in enumerate(shapes):
        median=medians[shape]
        rows.append(median.assign(shape_cluster=shape))
        for ax in axes:
            plot_hour_median(ax,median,color=f"C{index%10}",label=f"Shape {shape}")
    for col,ax in enumerate(axes):
        if col: ax.set_xscale("symlog",linthresh=.1)
        ax.set(xlabel="Time (hour)",ylabel="Median response / observed max",
               title="Symlog hour axis" if col else "Linear hour axis")
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    fig.suptitle("Four shape-group medians at actual hours" if len(shapes)==4 else "Shape-group medians at actual hours")
    fig.text(.5,.012,"Solid: >=50% of group observed; dashed: <50%. No time alignment or endpoint padding.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(folder/"shape_medians_actual_hours.png",dpi=170)
    plt.close(fig)
    pd.concat(rows,ignore_index=True).to_csv(folder/"shape_medians_actual_hours.csv",index=False)
    dump(folder/"median_plot_method.json",{
        "definition":"pointwise median at actual hours, per-curve observed-max normalized response",
        "interpolation":"linear within each observed interval only; no extrapolation or endpoint padding",
        "grid":"union of original observation hours within each shape group",
        "solid_minimum_coverage_fraction":.5,
        "warning":"contributing population changes with time; not an aligned shape median or representative lifetime",
        "source_sha256":{name:baseline.sha(folder/name) for name in ("hour_assignments.csv","physical_channels.csv") if (folder/name).exists()},
        "plot_code_sha256":baseline.sha(Path(__file__))})
    subgroup_ids=list(range(max(0,int(a.timescale_subgroup.max())+1)))+[-1]
    fig, axes = plt.subplots(len(shapes), len(subgroup_ids), figsize=(4.4*len(subgroup_ids), 2.7*len(shapes)), squeeze=False)
    for row, shape in enumerate(shapes):
        for col, subgroup in enumerate(subgroup_ids):
            ax = axes[row,col]
            ids = np.flatnonzero((a.shape_cluster.to_numpy()==shape)&(a.timescale_subgroup.to_numpy()==subgroup))
            if not len(ids):
                ax.axis("off"); continue
            for i in ids:
                ch = result["physical"][i]
                ax.plot(ch.time_h,ch.y_max_normalized,alpha=.35,lw=.7)
            title = f"Shape {shape} / {'unresolved' if subgroup<0 else 'response '+str(subgroup)}; n={len(ids)}"
            if subgroup>=0:
                title += f"\nMedian onset={a.iloc[ids[0]].subgroup_median_response_h:.3g} h"
            ax.set(title=title,xlabel="Time (hour)",ylabel="Normalized response")
            ax.grid(alpha=.2)
    fig.suptitle("Within-shape response-time subgroups; independent LINEAR hour axes")
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(folder/"response_subgroups_actual_hours.png",dpi=150)
    plt.close(fig)


def replicas(seed):
    bases, truth, _ = baseline.new_batch("independent",seed,per_class=12)
    curves, labels, meta = [], [], []
    for i,(base,target) in enumerate(zip(bases,truth)):
        base=base.copy()
        t=base.time_h.to_numpy()
        t=(t-t[0])/(t[-1]-t[0])*20
        for scale in (1,10,100):
            f=base.copy(); f["time_h"]=t*scale
            curves.append(f); labels.append(target)
            meta.append({"prototype":i,"clock_scale":scale,"seed":seed})
    order=np.random.default_rng(seed).permutation(len(curves))
    return [curves[i] for i in order],np.array(labels)[order],[meta[i] for i in order]


def load_prior(name):
    folder=ROOT/"functional_validation_20260920"/"outputs"/name
    raw=pd.read_csv(folder/"observations.csv")
    meta=json.loads((folder/"metadata_posthoc.json").read_text())
    groups=dict(tuple(raw.groupby("curve_id",sort=False)))
    return ([groups[m["curve_id"]][["time_h","y"]] for m in meta],
            np.array([m["target_posthoc"] for m in meta]),meta)


def evaluate(name,curves,truth,meta,output,diagnostics=False):
    folder=output/name; folder.mkdir(parents=True,exist_ok=True)
    print(f"Fitting {name}: {len(curves)} curves",flush=True)
    # Fitting receives observations ONLY. Truth and simulation parameters are
    # used below, after the model has returned its assignments.
    result=fit_hour_hierarchy(curves,time_diagnostics=diagnostics)
    save_result(result,curves,folder)
    galleries(result,folder)
    a=result["assignments"]
    resolved=a.excursion_10_status.eq("resolved").to_numpy()
    assigned=a.timescale_subgroup.ge(0).to_numpy()
    metric={"batch":name,"n":len(curves),"shape_ari":float(ari(truth,a.shape_cluster)),
            "resolved_count":int(resolved.sum()),"subgroup_assigned_count":int(assigned.sum()),
            "status_counts":a.excursion_10_status.value_counts().to_dict(),
            "subgroups_per_shape":{s:m.get("selected_k",0) for s,m in result["subgroup_models"].items()}}
    posthoc=[{"curve_id":a.iloc[i].curve_id,"target_posthoc":str(truth[i]),**m} for i,m in enumerate(meta)]
    dump(folder/"metadata_posthoc.json",posthoc)
    pd.crosstab(a.shape_cluster,pd.Series(truth,name="target_posthoc")).to_csv(folder/"shape_interpretation_posthoc.csv")
    if diagnostics:
        scales=np.array([m["clock_scale"] for m in meta])
        target_joint=np.array([f"{t}/{s}" for t,s in zip(truth,scales)])
        metric["joint_ari_on_assigned"]=float(ari(target_joint[assigned],a.joint_group[assigned])) if assigned.any() else None
        metric["scale_ari_within_true_shape_on_assigned"]={str(t):float(ari(scales[assigned&(truth==t)],a.timescale_subgroup[assigned&(truth==t)])) for t in np.unique(truth)}
        metric["single_partition_diagnostic"]={w:{"shape_ari":float(ari(truth,d["labels"])),"clock_scale_ari":float(ari(scales,d["labels"]))} for w,d in result["time_diagnostics"].items()}
        checks=[]
        for prototype in sorted({m["prototype"] for m in meta}):
            ids=sorted([i for i,m in enumerate(meta) if m["prototype"]==prototype],key=lambda i:meta[i]["clock_scale"])
            first=ids[0]
            same=len(set(a.iloc[ids].shape_cluster))==1
            for i in ids[1:]:
                scale=meta[i]["clock_scale"]
                ch0=result["physical"][first]; ch=result["physical"][i]
                finite=np.isfinite(a.iloc[first].excursion_10_elapsed_h)
                event_ok=not finite or np.isclose(a.iloc[i].excursion_10_elapsed_h,a.iloc[first].excursion_10_elapsed_h*scale,rtol=1e-8,atol=1e-8)
                checks.append({"prototype":prototype,"scale":scale,"same_shape":same,
                               "event_scales_correctly":bool(event_ok),
                               "first_derivative_scales_correctly":bool(np.allclose(ch.dy_per_h*scale,ch0.dy_per_h,rtol=1e-7,atol=1e-8)),
                               "second_derivative_scales_correctly":bool(np.allclose(ch.d2y_per_h2*scale**2,ch0.d2y_per_h2,rtol=1e-7,atol=1e-8))})
        pd.DataFrame(checks).to_csv(folder/"replica_invariance_checks.csv",index=False)
        metric["replica_checks_all_pass"]=bool(pd.DataFrame(checks).iloc[:,2:].all().all())
    # Round-trip checks ensure exported coordinates are the actual inputs.
    saved=pd.read_csv(folder/"raw_observations.csv")
    groups=dict(tuple(saved.groupby("curve_id")))
    raw_ok=True
    for i,f in enumerate(curves):
        col=next(c for c in ("y","y_relative","normalized_pce") if c in f)
        raw_ok &= np.allclose(groups[f"curve_{i:04d}"][["time_h","y"]],f[["time_h",col]],rtol=1e-12,atol=1e-12)
    metric["export_preserves_observations"]=bool(raw_ok)
    metric["saved_raw_sha256"]=baseline.sha(folder/"raw_observations.csv")
    dump(folder/"metrics.json",metric)
    print(json.dumps(metric),flush=True)
    return metric


def physical_example(output):
    folder=output/"physical_examples"; folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    records=[]
    for col,tau in enumerate((1.,10.,100.)):
        t=np.r_[0,np.geomspace(.001,10,200)]*tau
        f=pd.DataFrame({"time_h":t,"y":.6+.4*np.exp(-t/tau)})
        record,ch=physical_description(f)
        record.update(case="time_dilation",true_tau_h=tau)
        records.append(record)
        ch.to_csv(folder/f"exponential_tau_{tau:g}h.csv",index=False)
        event=record["excursion_10_elapsed_h"]
        axes[0,col].plot(t,f.y)
        axes[0,col].axvline(event,color="C3",ls="--")
        axes[0,col].set(title=f"Same decay shape: tau={tau:g} h\n0.10-change time = {event:.3g} h",xlabel="Time (hour)",ylabel="Response",ylim=(.58,1.02))
        axes[1,col].plot(t,ch.dy_per_h)
        axes[1,col].set(xlabel="Time (hour)",ylabel="d(normalized response)/dh")
    fig.suptitle("Same morphology, different physical rates; all axes in actual hours")
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(folder/"same_shape_different_hours.png",dpi=160); plt.close(fig)
    # Prefix and its late extension contain exactly identical early samples.
    prefix=np.r_[0,np.geomspace(.01,100,160)]
    for name,t in [("prefix",prefix),("extended",np.r_[prefix,np.geomspace(101,1000,60)])]:
        f=pd.DataFrame({"time_h":t,"y":.6+.4*np.exp(-t/10)})
        record,ch=physical_description(f); record.update(case=name,true_tau_h=10.)
        records.append(record); ch.to_csv(folder/f"{name}.csv",index=False)
    pd.DataFrame(records).to_csv(folder/"example_measurements.csv",index=False)
    assert np.isclose(records[-1]["excursion_10_elapsed_h"],records[-2]["excursion_10_elapsed_h"])
    dump(folder/"extension_check.json",{"same_process_response_time_unchanged":True,
        "durations_h":[100,1000],"response_hours":[r["excursion_10_elapsed_h"] for r in records[-2:]]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-curves",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"outputs")
    parser.add_argument("--shape-clusters",type=int,default=4)
    parser.add_argument("--refresh-galleries",type=Path,help="Replot saved assignments in this folder without refitting")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.refresh_galleries:
        refresh_galleries(args.refresh_galleries)
        print(f"Updated median galleries in {args.refresh_galleries}")
        return
    if args.input_curves:
        files=sorted(args.input_curves.rglob("*.csv"))
        curves=[pd.read_csv(p) for p in files]
        result=fit_hour_hierarchy(curves,HourConfig(shape_clusters=args.shape_clusters))
        save_result(result,curves,args.output); galleries(result,args.output)
        dump(args.output/"source_manifest.json",[{"curve_id":f"curve_{i:04d}","path":str(p.resolve()),"sha256":baseline.sha(p)} for i,p in enumerate(files)])
        print(f"Saved {len(curves)} curves to {args.output}")
        return
    metrics=[]
    for seed in (42001,42002,42003):
        metrics.append(evaluate(f"clock_replicas_{seed}",*replicas(seed),args.output,diagnostics=True))
    curves,truth,meta=baseline.existing_batch()
    metrics.append(evaluate("existing_400",curves,truth,meta,args.output))
    source_unchanged=all(baseline.sha(ROOT/m["source"])==m["sha256"] for m in meta)
    for name in ("same_family_31001","independent_31001"):
        metrics.append(evaluate(name,*load_prior(name),args.output))
    physical_example(args.output)
    dump(args.output/"metrics_summary.json",metrics)
    pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,(list,dict))} for m in metrics]).to_csv(args.output/"metrics_summary.csv",index=False)
    dump(args.output/"audit.json",{"original_400_source_hashes_unchanged":source_unchanged,
         "all_exported_hour_coordinates_preserved":all(m["export_preserves_observations"] for m in metrics),
         "all_clock_replica_invariances_pass":all(m.get("replica_checks_all_pass",True) for m in metrics),
         "code_hashes":{p.name:baseline.sha(p) for p in HERE.glob("*.py")},
         "protocol_sha256":baseline.sha(HERE/"PROTOCOL.md")})


if __name__=="__main__":
    main()
