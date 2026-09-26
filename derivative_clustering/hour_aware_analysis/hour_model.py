"""Separate observed morphology from measured physical response times.

The shape branch retains the validated relative-progress DTW representation.
Physical landmarks and derivatives are estimated independently on original
irregular hour samples. Actual hours determine the second clustering level.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh, qr, svd
from scipy.sparse.csgraph import connected_components, laplacian

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "functional_validation_20260920"))
import validate as baseline


@dataclass(frozen=True)
class HourConfig:
    shape_clusters: int = 4
    neighbors: int = 10
    local_points: int = 7
    excursion_levels: tuple = (.05, .10, .20)
    primary_excursion: float = .10
    maximum_bracket_ratio: float = .5
    minimum_subgroup: int = 8
    maximum_subgroups: int = 4
    mixture_seed: int = 42001


def stable_shape_partition(distance, clusters=4, neighbors=10):
    """Same normalized graph/QR objective, dense symmetric eigensolver.

    Avoid shift-invert ARPACK instability at multiple zero eigenvalues. Intended
    for the current hundreds-of-curves batches, not million-node graphs.
    No labels or response-time features enter this numerical operation.
    """
    graph=baseline.prior.distance_to_knn_affinity(distance,neighbors)
    n_components,_=connected_components(graph,directed=False)
    lap,degree=laplacian(graph,normed=True,return_diag=True)
    values,vectors=eigh(lap,subset_by_index=[0,clusters],driver="evr")
    embedding=vectors[:,:clusters]/degree[:,None]
    _,_,pivots=qr(embedding.T,pivoting=True)
    u,_,vh=svd(embedding[pivots[:clusters],:].T)
    # Explicit reductions avoid spurious floating-point status warnings from
    # this host's BLAS matmul; validate the computed arrays, never hide NaNs.
    rotation=np.einsum("ij,jk->ik",u,vh,optimize=False)
    rotated=np.einsum("ij,jk->ik",embedding,rotation,optimize=False)
    product=np.einsum("ij,jk->ik",lap,vectors,optimize=False)
    if not np.isfinite(rotated).all() or not np.isfinite(product).all():
        raise ValueError("Nonfinite spectral solution")
    labels=np.abs(rotated).argmax(axis=1)
    residual=float(np.max(np.abs(product-vectors*values)))
    info={"solver":"dense symmetric eigh + cluster_qr", "connected_components":int(n_components),
          "smallest_eigenvalues":values.tolist(),"eigen_residual_max":residual,
          "warning":"more connected components than requested clusters" if n_components>clusters else None}
    return labels,info


def clean_observations(frame):
    columns = [c for c in ("y", "y_relative", "normalized_pce") if c in frame]
    if "time_h" not in frame or len(columns) != 1:
        raise ValueError("Require time_h and one response column: y/y_relative/normalized_pce")
    f = frame[["time_h", columns[0]]].astype(float).rename(columns={columns[0]:"y"})
    if not np.isfinite(f.to_numpy()).all():
        raise ValueError("Non-finite time or response")
    f = f.groupby("time_h", as_index=False).median().sort_values("time_h")
    if len(f) < 5 or f.time_h.iloc[-1] <= f.time_h.iloc[0]:
        raise ValueError("Need at least five distinct times and positive duration")
    denominator = float(np.max(np.abs(f.y)))
    if denominator <= 1e-12:
        raise ValueError("All-zero response has no usable amplitude reference")
    return f, denominator


def physical_local_polynomial(t, y, points=7):
    """Quadratic local regression; conditioning is undone in physical derivatives.

    Local bandwidth comes from nearest actual timestamps, not total duration.
    There is no imputation beyond the observation interval.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 5 or np.any(np.diff(t) <= 0):
        raise ValueError("Need at least five strictly increasing times")
    result = np.empty((len(t), 3))
    for i, time in enumerate(t):
        nearest = np.argsort(np.abs(t-time), kind="stable")[:min(points, len(t))]
        offsets = t[nearest]-time
        bandwidth = float(np.max(np.abs(offsets))*1.05)
        z = offsets / bandwidth
        design = np.column_stack([np.ones(len(z)), z, z*z])
        base_weights = (1-np.abs(z)**3)**3
        weights = base_weights.copy()
        for iteration in range(3):
            root = np.sqrt(np.maximum(weights, 1e-10))
            coef = np.linalg.lstsq(design*root[:,None], y[nearest]*root, rcond=None)[0]
            residual = y[nearest]-design@coef
            noise = 1.4826*np.median(np.abs(residual-np.median(residual)))
            if iteration < 2 and noise > 1e-10:
                huber = np.minimum(1., 1.5*noise/np.maximum(np.abs(residual),1e-12))
                weights = base_weights*huber
        result[i] = [coef[0], coef[1]/bandwidth, 2*coef[2]/bandwidth**2]
    return result


def first_excursion(t, y, delta, maximum_bracket_ratio=.5):
    """First crossing sustained through the next sample (not a survival estimate).

    Censored means no sustained crossing in the observed reconstructed trace.
    Unobserved changes inside gaps cannot be excluded. Brackets are sampling
    support, NOT statistical confidence intervals.
    """
    elapsed = t-t[0]
    change = y-y[0]
    for i in range(1, len(t)-1):
        direction = float(np.sign(change[i]))
        if abs(change[i]) < delta or direction*change[i+1] < .75*delta:
            continue
        # Use the most recent pre-threshold point on this excursion.
        before = np.flatnonzero(direction*change[:i] < delta)
        if not len(before):
            continue
        j = int(before[-1])
        a, b = direction*change[j], direction*change[i]
        fraction = float(np.clip((delta-a)/max(b-a,1e-12),0,1))
        crossing = float(t[j]+fraction*(t[i]-t[j]))
        response = crossing-float(t[0])
        width = float(t[i]-t[j])
        resolved = response>0 and width <= maximum_bracket_ratio*response
        return {"status":"resolved" if resolved else "interval_uncertain",
                "direction":"increase" if direction>0 else "decrease",
                "time_h":crossing, "elapsed_h":response,
                "bracket_start_h":float(t[j]),"bracket_end_h":float(t[i]),
                "bracket_width_h":width,"censor_elapsed_h":np.nan}
    return {"status":"not_observed_censored", "direction":"unknown",
            "time_h":np.nan,"elapsed_h":np.nan,"bracket_start_h":np.nan,
            "bracket_end_h":np.nan,"bracket_width_h":np.nan,
            "censor_elapsed_h":float(elapsed[-1])}


def physical_description(frame, config=HourConfig()):
    f, denominator = clean_observations(frame)
    t=f.time_h.to_numpy(); y=f.y.to_numpy()/denominator
    physical=physical_local_polynomial(t,y,config.local_points)
    record={"start_h":float(t[0]),"end_h":float(t[-1]),
            "observed_duration_h":float(t[-1]-t[0]),"n_observations":len(t),
            "amplitude_reference_raw":denominator,
            "initial_rate_per_h":float(physical[0,1]),
            "final_rate_per_h":float(physical[-1,1]),
            "observed_peak_h":float(t[np.argmax(physical[:,0])]),
            "observed_trough_h":float(t[np.argmin(physical[:,0])])}
    for delta in config.excursion_levels:
        prefix=f"excursion_{int(round(delta*100)):02d}_"
        event=first_excursion(t,physical[:,0],delta,config.maximum_bracket_ratio)
        record.update({prefix+k:v for k,v in event.items()})
    channels=pd.DataFrame({"time_h":t,"y_raw":f.y.to_numpy(),
                           "y_max_normalized":y,"y_local_smooth":physical[:,0],
                           "dy_per_h":physical[:,1],"d2y_per_h2":physical[:,2],
                           "dy_raw_per_h":physical[:,1]*denominator,
                           "d2y_raw_per_h2":physical[:,2]*denominator})
    return record,channels


def kinetic_subgroups(shape_labels, descriptors, config=HourConfig()):
    """Unsupervised within-shape response-time groups; never uses end duration."""
    primary=f"excursion_{int(round(config.primary_excursion*100)):02d}_"
    labels=np.full(len(descriptors),-1,dtype=int)
    medians=np.full(len(descriptors),np.nan)
    models={}
    for shape in np.unique(shape_labels):
        mask=(shape_labels==shape)&(descriptors[primary+"status"].to_numpy()=="resolved")
        indices=np.flatnonzero(mask)
        values=descriptors.loc[mask,primary+"elapsed_h"].to_numpy(float)
        valid=np.isfinite(values)&(values>0)
        indices,values=indices[valid],values[valid]
        info={"n_total":int(np.sum(shape_labels==shape)),"n_resolved":len(values),
              "feature":"log10(first sustained 0.10 excursion elapsed hour / 1 h)",
              "selection":"BIC with minimum 8 observations per component", "candidates":[]}
        if len(values)<config.minimum_subgroup:
            info["status"]="insufficient_resolved_curves"
            models[str(shape)]=info
            continue
        x=np.log10(values).reshape(-1,1)
        max_k=min(config.maximum_subgroups,len(values)//config.minimum_subgroup)
        best=None
        for k in range(1,max_k+1):
            if k>1 and np.ptp(x)<.05:
                break
            gmm=GaussianMixture(n_components=k,n_init=5,reg_covar=1e-4,random_state=config.mixture_seed).fit(x)
            pred=gmm.predict(x)
            sizes=np.bincount(pred,minlength=k)
            allowed=bool(np.all(sizes>=config.minimum_subgroup))
            score=float(gmm.bic(x))
            info["candidates"].append({"k":k,"bic":score,"sizes":sizes.tolist(),"allowed":allowed})
            if allowed and (best is None or score<best[0]):
                best=(score,gmm,pred)
        _,gmm,pred=best
        order=np.argsort(gmm.means_.ravel())
        remap={int(old):int(new) for new,old in enumerate(order)}
        ordered=np.array([remap[int(c)] for c in pred])
        labels[indices]=ordered
        group_hours={str(c):float(np.median(values[ordered==c])) for c in range(len(order))}
        medians[indices]=[group_hours[str(c)] for c in ordered]
        info.update(status="fitted",selected_k=len(order),median_response_h=group_hours,
                    means_log10_h=gmm.means_.ravel().tolist(),
                    variances_log10_h=gmm.covariances_.ravel().tolist(),weights=gmm.weights_.tolist(),
                    original_to_ordered_label=remap)
        models[str(shape)]=info
    return labels,medians,models


def fit_hour_hierarchy(curves, config=HourConfig(), time_diagnostics=False):
    if len(curves)<max(11,config.shape_clusters+1):
        raise ValueError("Need at least 11 curves for the 10-neighbor shape graph")
    raw,scaled,_,quality=baseline.representations(curves)
    shape_distance=baseline.accelerated_dtw(scaled[:,:,:3],[.5,.375,.125])
    shape_labels,graph_info=stable_shape_partition(shape_distance,config.shape_clusters,config.neighbors)
    records,physical=[],[]
    for frame in curves:
        record,channels=physical_description(frame,config)
        records.append(record); physical.append(channels)
    descriptors=pd.DataFrame(records)
    kinetic,median_hours,models=kinetic_subgroups(shape_labels,descriptors,config)
    output=descriptors.copy()
    output.insert(0,"curve_id",[f"curve_{i:04d}" for i in range(len(curves))])
    output.insert(1,"shape_cluster",shape_labels)
    output.insert(2,"timescale_subgroup",kinetic)
    output.insert(3,"subgroup_median_response_h",median_hours)
    output.insert(4,"joint_group",[f"shape_{a}/response_{b}" if b>=0 else f"shape_{a}/response_unresolved" for a,b in zip(shape_labels,kinetic)])
    output["shape_max_relative_gap"]=[q["max_relative_gap"] for q in quality]
    diagnostics={}
    if time_diagnostics:
        # A shared 100-hour reference across ALL curves, not per-curve duration
        # normalization. Raw clock mismatch now has a nonzero matching cost.
        t_channel=np.stack([np.linspace(r["start_h"],r["end_h"],48)/100 for r in records])
        combined=np.concatenate([scaled[:,:,:3],t_channel[:,:,None]],axis=2)
        for weight in (.01,.1,1.):
            distance=baseline.accelerated_dtw(combined,[.5,.375,.125,weight])
            labels,info=stable_shape_partition(distance,config.shape_clusters,config.neighbors)
            diagnostics[str(weight)]={"distance":distance,"labels":labels,"graph_info":info}
    return {"assignments":output,"physical":physical,"shape_distance":shape_distance,
            "subgroup_models":models,"time_diagnostics":diagnostics,"config":asdict(config),"graph_info":graph_info}


def save_result(result, curves, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    result["assignments"].to_csv(output/"hour_assignments.csv",index=False)
    all_channels=[]; all_raw=[]
    for i,(channels,raw) in enumerate(zip(result["physical"],curves)):
        ch=channels.copy(); ch.insert(0,"curve_id",f"curve_{i:04d}"); all_channels.append(ch)
        field=next(c for c in ("y","y_relative","normalized_pce") if c in raw)
        r=raw[["time_h",field]].rename(columns={field:"y"}).copy()
        r.insert(0,"curve_id",f"curve_{i:04d}"); all_raw.append(r)
    pd.concat(all_channels).to_csv(output/"physical_channels.csv",index=False)
    pd.concat(all_raw).to_csv(output/"raw_observations.csv",index=False)
    np.savez_compressed(output/"shape_distance.npz",distance=result["shape_distance"])
    (output/"subgroup_models.json").write_text(json.dumps(result["subgroup_models"],indent=2),encoding="utf-8")
    (output/"config.json").write_text(json.dumps(result["config"],indent=2),encoding="utf-8")
    (output/"shape_graph_diagnostics.json").write_text(json.dumps(result["graph_info"],indent=2),encoding="utf-8")
    if result["time_diagnostics"]:
        assignments=result["assignments"][["curve_id","shape_cluster"]].copy()
        for w,item in result["time_diagnostics"].items(): assignments["hour_penalty_"+w]=item["labels"]
        assignments.to_csv(output/"single_partition_hour_diagnostics.csv",index=False)
