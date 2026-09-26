#!/usr/bin/env python3
"""Complex semi-synthetic injection/recovery experiment using hybrid PFD-DTW.

Synthetic classes are generated in observation space and inherit irregular
sampling and empirical residual structure from real curves.  Their labels are
used only after unsupervised clustering for recovery metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfd_dtw import hp_filter  # noqa: E402


CLASSES = ("bridge", "hill", "slope", "valley")
COLORS = {
    "bridge": "#2f6db0",
    "hill": "#d95f02",
    "slope": "#2a9d58",
    "valley": "#8e5bb7",
}
GRID = np.linspace(0.0, 200.0, 81)
HP_LAMBDA = 100.0
DTW_RADIUS = 8
PFD_WEIGHTS = np.asarray([0.60, 0.30, 0.10], dtype=float)
LEVEL_WEIGHT = 0.40
PFD_WEIGHT = 0.60


@dataclass(frozen=True)
class SyntheticCurve:
    curve_id: str
    target_class: str
    donor_curve_id: str
    seed: int
    time_h: np.ndarray
    y_observed: np.ndarray
    grid_values: np.ndarray
    parameters: dict[str, float | int | str]


def read_long_curves(path: Path, value_column: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["curve_id"], []).append(
                (float(row["time_h"]), float(row[value_column]))
            )
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for curve_id, pairs in grouped.items():
        pairs.sort()
        values = np.asarray(pairs, dtype=float)
        result[curve_id] = (values[:, 0], values[:, 1])
    return result


def load_real_curves(audit_dir: Path) -> tuple[list[str], np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    primary = read_long_curves(audit_dir / "primary_0_200h_long.csv", "y_relative")
    all_hour = read_long_curves(audit_dir / "integrated_hours_long.csv", "y_relative")
    ids = sorted(primary)
    values = np.vstack([primary[curve_id][1] for curve_id in ids])
    return ids, values, all_hour


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _latent_shape(target: str, time_h: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    u = np.asarray(time_h, dtype=float) / 200.0
    gamma = rng.uniform(0.85, 1.15)
    warped = np.power(np.maximum(u, 0), gamma)
    base_level = rng.normal(1.0, 0.018)
    background_loss = rng.uniform(0.0, 0.06) * np.power(warped, rng.uniform(0.7, 1.8))
    values = base_level - background_loss
    params: dict[str, float] = {"gamma": gamma, "base_level": base_level, "background_loss": float(background_loss[-1])}

    if target == "bridge":
        gain = rng.uniform(0.08, 0.18)
        gain_center = rng.uniform(0.03, 0.13)
        decline = rng.uniform(0.14, 0.34)
        decline_center = rng.uniform(0.68, 0.86)
        values += gain * sigmoid((warped - gain_center) / rng.uniform(0.018, 0.050))
        values -= decline * sigmoid((warped - decline_center) / rng.uniform(0.045, 0.11))
        params.update(gain=gain, gain_center=gain_center, decline=decline, decline_center=decline_center)
    elif target == "hill":
        height = rng.uniform(0.12, 0.27)
        center = rng.uniform(0.20, 0.40)
        width = rng.uniform(0.11, 0.21)
        late_loss = rng.uniform(0.12, 0.30)
        values += height * np.exp(-0.5 * ((warped - center) / width) ** 2)
        values -= late_loss * sigmoid((warped - rng.uniform(0.48, 0.67)) / rng.uniform(0.07, 0.15))
        params.update(height=height, center=center, width=width, late_loss=late_loss)
    elif target == "slope":
        fast_loss = rng.uniform(0.12, 0.34)
        slow_loss = rng.uniform(0.08, 0.26)
        tau = rng.uniform(0.035, 0.22)
        values -= fast_loss * (1.0 - np.exp(-warped / tau))
        values -= slow_loss * np.power(warped, rng.uniform(0.7, 1.6))
        # A weak nuisance recovery prevents every slope from being monotone.
        if rng.random() < 0.65:
            values += rng.uniform(0.008, 0.035) * np.exp(
                -0.5 * ((warped - rng.uniform(0.35, 0.82)) / rng.uniform(0.05, 0.15)) ** 2
            )
        params.update(fast_loss=fast_loss, slow_loss=slow_loss, tau=tau)
    elif target == "valley":
        drop = rng.uniform(0.16, 0.32)
        recovery = rng.uniform(0.16, 0.31)
        late_drop = rng.uniform(0.14, 0.28)
        c1 = rng.uniform(0.08, 0.25)
        c2 = rng.uniform(max(c1 + 0.12, 0.30), 0.58)
        c3 = rng.uniform(max(c2 + 0.18, 0.64), 0.92)
        values -= drop * sigmoid((warped - c1) / rng.uniform(0.025, 0.09))
        values += recovery * sigmoid((warped - c2) / rng.uniform(0.035, 0.11))
        values -= late_drop * sigmoid((warped - c3) / rng.uniform(0.045, 0.14))
        params.update(drop=drop, recovery=recovery, late_drop=late_drop, c1=c1, c2=c2, c3=c3)
    else:
        raise ValueError(target)

    # Shared nuisance structure: small extra peaks/valleys unrelated to the label.
    for nuisance_index in range(rng.integers(1, 4)):
        amplitude = rng.uniform(-0.030, 0.030)
        center = rng.uniform(0.12, 0.92)
        width = rng.uniform(0.018, 0.11)
        values += amplitude * np.exp(-0.5 * ((warped - center) / width) ** 2)
        params[f"nuisance_{nuisance_index}_amplitude"] = amplitude
    return values, params


def empirical_residual_library(real_values: np.ndarray) -> np.ndarray:
    residuals = []
    for curve in real_values:
        trend = hp_filter(curve, smoothing=HP_LAMBDA)
        residual = curve - trend
        scale = np.median(np.abs(residual - np.median(residual))) * 1.4826
        if scale > 1e-8:
            residuals.append(residual / scale)
    return np.asarray(residuals, dtype=float)


def generate_synthetic_pool(
    real_ids: list[str],
    real_values: np.ndarray,
    raw_hour_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    per_class: int,
    seed: int,
) -> list[SyntheticCurve]:
    rng = np.random.default_rng(seed)
    residuals = empirical_residual_library(real_values)
    donors = [curve_id for curve_id in real_ids if np.count_nonzero(raw_hour_curves[curve_id][0] <= 200) >= 4]
    generated: list[SyntheticCurve] = []

    for target in CLASSES:
        for index in range(per_class):
            donor = str(rng.choice(donors))
            donor_t = raw_hour_curves[donor][0]
            within = donor_t[donor_t <= 200.0]
            after = donor_t[donor_t > 200.0]
            sample_times = within.copy()
            if after.size:
                sample_times = np.append(sample_times, min(float(after[0]), 260.0))
            else:
                sample_times = np.append(sample_times, 200.0)
            sample_times = np.unique(np.clip(sample_times, 0.0, 260.0))
            if sample_times.size > 8:
                keep = rng.random(sample_times.size) > rng.uniform(0.0, 0.22)
                keep[0] = True
                keep[-1] = True
                sample_times = sample_times[keep]
            if sample_times.size < 4:
                sample_times = np.linspace(0.0, 200.0, 4)

            dense_t = np.linspace(0.0, max(200.0, float(sample_times[-1])), 121)
            latent, params = _latent_shape(target, dense_t, rng)
            residual = residuals[rng.integers(0, residuals.shape[0])]
            residual_on_dense = np.interp(
                dense_t,
                GRID,
                np.roll(residual, rng.integers(0, residual.size)),
                left=residual[0],
                right=residual[-1],
            )
            empirical_scale = rng.uniform(0.004, 0.020)
            latent += empirical_scale * residual_on_dense

            rho = rng.uniform(0.45, 0.88)
            ar = np.zeros(dense_t.size)
            innovations = rng.normal(0.0, rng.uniform(0.002, 0.009), dense_t.size)
            for point in range(1, dense_t.size):
                ar[point] = rho * ar[point - 1] + innovations[point]
            latent += ar

            # Narrow disturbance pulses mimic isolated experimental excursions.
            for _ in range(rng.integers(0, 3)):
                latent += rng.uniform(-0.040, 0.040) * np.exp(
                    -0.5 * ((dense_t - rng.uniform(15, 190)) / rng.uniform(1.5, 7.0)) ** 2
                )
            if rng.random() < 0.20:
                quantum = rng.uniform(0.006, 0.018)
                latent = np.round(latent / quantum) * quantum

            observed = np.interp(sample_times, dense_t, latent)
            observed += rng.normal(0.0, rng.uniform(0.001, 0.010), observed.size)
            reference = float(np.median(observed[: min(3, observed.size)]))
            observed = observed / reference
            observed = np.clip(observed, 0.05, 1.55)
            interpolator = PchipInterpolator(sample_times, observed, extrapolate=False)
            grid_values = np.asarray(interpolator(GRID), dtype=float)
            if not np.all(np.isfinite(grid_values)):
                raise RuntimeError("synthetic interpolation failed")

            curve_id = f"SYN_{target}_{seed}_{index:03d}"
            params.update(
                empirical_noise_scale=empirical_scale,
                ar_rho=rho,
                observed_points=int(sample_times.size),
            )
            generated.append(
                SyntheticCurve(curve_id, target, donor, seed, sample_times, observed, grid_values, params)
            )
    return generated


def fit_pfd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    smoothed = hp_filter(values, smoothing=HP_LAMBDA)
    best: tuple[float, int, Polynomial] | None = None
    for degree in range(3, 9):
        model = Polynomial.fit(GRID, smoothed, deg=degree)
        residual = smoothed - model(GRID)
        rss = max(float(np.dot(residual, residual)), 1e-12)
        bic = values.size * math.log(rss / values.size) + (degree + 1) * math.log(values.size)
        if best is None or bic < best[0]:
            best = (bic, degree, model)
    assert best is not None
    model = best[2]
    derivatives = np.column_stack([model.deriv(order)(GRID) for order in (1, 2, 3)])
    return smoothed, derivatives, best[1]


def transform_curves(values: np.ndarray, derivative_scale: np.ndarray | None = None):
    levels=[]; derivatives=[]; degrees=[]
    for curve in values:
        level, derivative, degree = fit_pfd(curve)
        levels.append(level); derivatives.append(derivative); degrees.append(degree)
    level_array=np.asarray(levels); derivative_array=np.asarray(derivatives)
    if derivative_scale is None:
        flat=derivative_array.reshape(-1,3)
        center=np.median(flat,axis=0)
        scale=1.4826*np.median(np.abs(flat-center),axis=0)
        scale=np.where(scale<1e-9,1.0,scale)
    else:
        scale=np.asarray(derivative_scale,dtype=float)
    return level_array, derivative_array/scale, np.asarray(degrees), scale


def dtw_distance(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    a=np.asarray(first,dtype=float); b=np.asarray(second,dtype=float)
    if a.ndim==1: a=a[:,None]
    if b.ndim==1: b=b[:,None]
    n,m=a.shape[0],b.shape[0]
    radius=max(DTW_RADIUS,abs(n-m))
    previous=np.full(m+1,np.inf); previous[0]=0.0
    for i in range(1,n+1):
        current=np.full(m+1,np.inf)
        for j in range(max(1,i-radius),min(m,i+radius)+1):
            delta=a[i-1]-b[j-1]
            cost=float(np.sqrt(np.dot(weights,delta*delta)))
            current[j]=cost+min(previous[j-1],previous[j],current[j-1])
        previous=current
    return float(previous[m]/(n+m))


def pairwise_matrix(items: np.ndarray, distance: Callable[[np.ndarray,np.ndarray],float]) -> np.ndarray:
    n=len(items); matrix=np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(i+1,n):
            matrix[i,j]=matrix[j,i]=distance(items[i],items[j])
    return matrix


def extend_matrix(
    real_items: np.ndarray,
    synthetic_items: np.ndarray,
    real_matrix: np.ndarray,
    distance: Callable[[np.ndarray,np.ndarray],float],
) -> np.ndarray:
    n_real=len(real_items); n_syn=len(synthetic_items); total=n_real+n_syn
    matrix=np.zeros((total,total),dtype=float); matrix[:n_real,:n_real]=real_matrix
    for i in range(n_syn):
        target=n_real+i
        for j in range(n_real):
            matrix[target,j]=matrix[j,target]=distance(synthetic_items[i],real_items[j])
        for j in range(i):
            other=n_real+j
            matrix[target,other]=matrix[other,target]=distance(synthetic_items[i],synthetic_items[j])
    return matrix


def cluster(matrix: np.ndarray, k: int=4) -> tuple[np.ndarray,np.ndarray,float]:
    tree=linkage(squareform(matrix,checks=True),method="average")
    labels=fcluster(tree,t=k,criterion="maxclust")-1
    score=float(silhouette_score(matrix,labels,metric="precomputed")) if len(np.unique(labels))>1 else float("nan")
    return labels.astype(int),tree,score


def selected_synthetic_indices(pool: list[SyntheticCurve], per_class: int) -> list[int]:
    result=[]
    for label in CLASSES:
        positions=[i for i,item in enumerate(pool) if item.target_class==label]
        result.extend(positions[:per_class])
    return sorted(result)


def summary_features(values: np.ndarray) -> np.ndarray:
    features=[]
    for curve in values:
        derivative=np.gradient(curve,GRID)
        second=np.gradient(derivative,GRID)
        features.append([
            curve[-1],np.min(curve),np.max(curve),np.trapz(curve,GRID)/200.0,
            np.std(curve),np.max(np.abs(derivative)),np.max(np.abs(second)),
            np.sum(np.diff(np.sign(derivative))!=0),
        ])
    return np.asarray(features,dtype=float)


def plot_preview(pool: list[SyntheticCurve], real_values: np.ndarray, output: Path) -> None:
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for axis,label in zip(axes.flat,CLASSES):
        for real in real_values[:25]: axis.plot(GRID,real,color="#b8b8b8",alpha=0.12,linewidth=0.7)
        examples=[item for item in pool if item.target_class==label][:8]
        for item in examples:
            axis.plot(item.time_h,item.y_observed,color=COLORS[label],alpha=0.32,linewidth=0.8)
            axis.plot(GRID,item.grid_values,color=COLORS[label],alpha=0.88,linewidth=1.2)
        axis.set_title(label.capitalize())
        axis.set_xlabel("time (h)"); axis.set_ylabel("relative performance")
        axis.grid(alpha=0.18)
    fig.suptitle("Complex semi-synthetic curves: irregular observations + reconstructed grid")
    fig.tight_layout()
    fig.savefig(output,dpi=180); plt.close(fig)


def write_synthetic(pool: list[SyntheticCurve], output: Path) -> None:
    with (output/"synthetic_raw_long.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=["curve_id","target_class","donor_curve_id","seed","time_h","y_relative"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for item in pool:
            for t,y in zip(item.time_h,item.y_observed):
                writer.writerow(dict(curve_id=item.curve_id,target_class=item.target_class,donor_curve_id=item.donor_curve_id,seed=item.seed,time_h=t,y_relative=y))
    with (output/"synthetic_parameters.jsonl").open("w",encoding="utf-8") as handle:
        for item in pool:
            handle.write(json.dumps(dict(curve_id=item.curve_id,target_class=item.target_class,donor_curve_id=item.donor_curve_id,seed=item.seed,parameters=item.parameters),ensure_ascii=False)+"\n")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir",type=Path,default=Path(__file__).resolve().parent/"outputs"/"01_hour_audit")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"outputs"/"02_injection")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2026,2027,2028])
    parser.add_argument("--ratios",type=float,nargs="+",default=[0.10,0.20,0.30,0.40])
    parser.add_argument("--preview-only",action="store_true")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)

    real_ids,real_values,raw_hour=load_real_curves(args.audit_dir)
    real_levels,real_pfd,real_degrees,derivative_scale=transform_curves(real_values)
    level_distance=lambda a,b:dtw_distance(a,b,np.ones(1))
    pfd_distance=lambda a,b:dtw_distance(a,b,PFD_WEIGHTS)

    max_ratio=max(args.ratios)
    max_total_syn=math.ceil(len(real_ids)*max_ratio/(1.0-max_ratio))
    max_per_class=math.ceil(max_total_syn/len(CLASSES))
    first_pool=generate_synthetic_pool(real_ids,real_values,raw_hour,per_class=max_per_class,seed=args.seeds[0])
    plot_preview(first_pool,real_values,args.output/"synthetic_preview.png")
    write_synthetic(first_pool,args.output)
    if args.preview_only:
        print(json.dumps({"real_curves":len(real_ids),"synthetic_preview_curves":len(first_pool)},indent=2)); return

    print(f"Computing real-real distance blocks for {len(real_ids)} curves ...",flush=True)
    real_level_matrix=pairwise_matrix(real_levels,level_distance)
    real_pfd_matrix=pairwise_matrix(real_pfd,pfd_distance)
    metrics=[]; representative=None

    for seed in args.seeds:
        print(f"Seed {seed}: generating {max_per_class} curves per class ...",flush=True)
        pool=generate_synthetic_pool(real_ids,real_values,raw_hour,per_class=max_per_class,seed=seed)
        syn_values=np.vstack([item.grid_values for item in pool])
        syn_levels,syn_pfd,syn_degrees,_=transform_curves(syn_values,derivative_scale=derivative_scale)
        print(f"Seed {seed}: extending level and PFD distance matrices ...",flush=True)
        full_level=extend_matrix(real_levels,syn_levels,real_level_matrix,level_distance)
        full_pfd=extend_matrix(real_pfd,syn_pfd,real_pfd_matrix,pfd_distance)
        level_scale=np.median(real_level_matrix[real_level_matrix>0])
        pfd_scale=np.median(real_pfd_matrix[real_pfd_matrix>0])
        full_hybrid=LEVEL_WEIGHT*(full_level/level_scale)+PFD_WEIGHT*(full_pfd/pfd_scale)

        for ratio in args.ratios:
            target_total=math.ceil(len(real_ids)*ratio/(1.0-ratio))
            per_class=max(1,round(target_total/len(CLASSES)))
            chosen_pool=selected_synthetic_indices(pool,per_class)
            selected=np.asarray(list(range(len(real_ids)))+[len(real_ids)+i for i in chosen_pool],dtype=int)
            true=np.asarray([pool[i].target_class for i in chosen_pool])
            for method,full in (("level_dtw",full_level),("pfd_dtw",full_pfd),("hybrid_pfd_dtw",full_hybrid)):
                sub=full[np.ix_(selected,selected)]
                labels,tree,silhouette=cluster(sub,k=4)
                synthetic_pred=labels[len(real_ids):]
                record=dict(seed=seed,requested_ratio=ratio,real_curves=len(real_ids),synthetic_curves=len(chosen_pool),actual_ratio=len(chosen_pool)/len(selected),method=method,synthetic_ari=adjusted_rand_score(true,synthetic_pred),synthetic_homogeneity=homogeneity_score(true,synthetic_pred),synthetic_completeness=completeness_score(true,synthetic_pred),full_silhouette=silhouette,clusters_found=len(np.unique(labels)))
                metrics.append(record)
                if seed==args.seeds[0] and abs(ratio-0.30)<1e-9 and method=="hybrid_pfd_dtw":
                    representative=(pool,chosen_pool,selected,true,labels,tree,sub,syn_degrees)

        domain_x=np.vstack([summary_features(real_values),summary_features(syn_values)])
        domain_y=np.asarray([0]*len(real_values)+[1]*len(syn_values))
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        auc=float(np.mean(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),domain_x,domain_y,cv=folds,scoring="roc_auc")))
        for record in metrics:
            if record["seed"]==seed: record["synthetic_vs_real_auc"]=auc

    fields=list(metrics[0])
    with (args.output/"injection_metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(metrics)

    if representative is not None:
        pool,chosen_pool,selected,true,labels,tree,sub,syn_degrees=representative
        ids=real_ids+[pool[i].curve_id for i in chosen_pool]
        targets=["real_unlabelled"]*len(real_ids)+true.tolist()
        with (args.output/"representative_assignments.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["curve_id","origin_or_target","cluster"]); writer.writeheader()
            for curve_id,target,label in zip(ids,targets,labels): writer.writerow(dict(curve_id=curve_id,origin_or_target=target,cluster=int(label)))
        np.savez_compressed(args.output/"representative_hybrid_distance.npz",distance=sub,linkage=tree)
        fig,axes=plt.subplots(1,2,figsize=(15,5.5))
        dendrogram(tree,no_labels=True,color_threshold=tree[-4,2] if tree.shape[0]>=4 else None,ax=axes[0])
        axes[0].set_title("Hybrid PFD-DTW dendrogram (30% injection)"); axes[0].set_ylabel("distance")
        for target in CLASSES:
            for i in chosen_pool:
                if pool[i].target_class==target: axes[1].plot(GRID,pool[i].grid_values,color=COLORS[target],alpha=.32,linewidth=.8)
        axes[1].set_title("Injected curves used in representative run"); axes[1].set_xlabel("time (h)"); axes[1].set_ylabel("relative performance")
        fig.tight_layout(); fig.savefig(args.output/"representative_result.png",dpi=180); plt.close(fig)

    # Aggregate recovery curves.
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),sharex=True)
    for method,color in (("level_dtw","#555555"),("pfd_dtw","#d95f02"),("hybrid_pfd_dtw","#2f6db0")):
        for metric,axis,title in (("synthetic_ari",axes[0],"Synthetic-label ARI"),("synthetic_homogeneity",axes[1],"Synthetic homogeneity"),("full_silhouette",axes[2],"Full-data silhouette")):
            means=[];stds=[]
            for ratio in args.ratios:
                values=[float(r[metric]) for r in metrics if r["method"]==method and r["requested_ratio"]==ratio]
                means.append(float(np.mean(values))); stds.append(float(np.std(values)))
            axis.errorbar(args.ratios,means,yerr=stds,marker="o",label=method,color=color,capsize=3)
            axis.set_title(title); axis.set_xlabel("requested synthetic share"); axis.grid(alpha=.22)
    axes[0].set_ylim(-.05,1.05); axes[1].set_ylim(-.05,1.05); axes[0].legend()
    fig.tight_layout(); fig.savefig(args.output/"injection_recovery.png",dpi=180); plt.close(fig)

    summary={"real_curves":len(real_ids),"seeds":args.seeds,"ratios":args.ratios,"max_synthetic_per_class":max_per_class,"hp_lambda":HP_LAMBDA,"dtw_radius":DTW_RADIUS,"pfd_weights":PFD_WEIGHTS.tolist(),"hybrid_weights":{"level":LEVEL_WEIGHT,"pfd":PFD_WEIGHT},"real_polynomial_degree_counts":{str(k):int(v) for k,v in zip(*np.unique(real_degrees,return_counts=True))}}
    (args.output/"experiment_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
