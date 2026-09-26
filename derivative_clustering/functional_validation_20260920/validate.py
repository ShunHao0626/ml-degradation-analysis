#!/usr/bin/env python3
"""Frozen, label-blind SOM / derivative functional benchmark; see PROTOCOL.md."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, silhouette_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import build_eg_based_dataset as generator
import pfd_dtw
from eg_dtw_derivative_experiment import run_experiment as prior

CLASSES = ("bridge", "hill", "slope", "valley")
SEEDS = (31001, 31002, 31003)
SOM_SEEDS = (11, 22, 33)
CONFIG = prior.Config()
LIB = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accelerated_dtw(channels, weights, radius=6, normalize=True):
    """Equivalent C++ computation; validated against two Python DTW versions."""
    global LIB
    binary = HERE / "_dtw.so"
    if LIB is None:
        source = HERE / "dtw.cpp"
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            subprocess.run(["clang++", "-O3", "-std=c++17", *flags,
                            str(source), "-o", str(binary)], check=True)
        LIB = ctypes.CDLL(str(binary))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        LIB.pairwise_dtw.argtypes = [arr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    arr, ctypes.c_int, ctypes.c_int, arr]
        LIB.pairwise_dtw.restype = None
    data = np.ascontiguousarray(channels, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if data.ndim != 3 or weights.shape != (data.shape[-1],):
        raise ValueError("expected [curve,time,channel] and one weight per channel")
    if not np.isfinite(data).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("nonfinite data or invalid weights")
    if radius < 0 or min(data.shape) < 1:
        raise ValueError("empty array or negative radius")
    output = np.zeros((len(data), len(data)), dtype=np.float64)
    LIB.pairwise_dtw(data, *data.shape, weights, radius, int(normalize), output)
    return output


def observed_frame(frame):
    """Only time and response are read. Metadata cannot enter the model."""
    choices = [x for x in ("y_relative", "normalized_pce", "y") if x in frame]
    if "time_h" not in frame or len(choices) != 1:
        raise ValueError("need time_h and exactly one of y_relative/normalized_pce/y")
    clean = frame[["time_h", choices[0]]].astype(float).rename(columns={choices[0]: "y_relative"})
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("time and response must be finite")
    clean = clean.groupby("time_h", as_index=False).median().sort_values("time_h")
    if len(clean) < 5 or clean.time_h.iloc[-1] <= clean.time_h.iloc[0]:
        raise ValueError("need five distinct times and positive duration")
    scale = float(np.max(np.abs(clean.y_relative)))
    if scale <= 1e-12:
        raise ValueError("zero response has no usable normalization")
    clean["y_relative"] /= scale
    return clean


def representations(curves):
    raw, interpolated, quality = [], [], []
    for frame in curves:
        obs = observed_frame(frame)
        u, channels, replacements, gap = prior.prepare_curve(obs, CONFIG)
        t = obs.time_h.to_numpy()
        raw.append(channels)
        interpolated.append(PchipInterpolator((t-t[0])/(t[-1]-t[0]), obs.y_relative)(u))
        quality.append({"n_observations": len(obs), "max_relative_gap": gap,
                        "outlier_replacements": replacements,
                        "sparse_warning": len(obs) < 12 or gap > .25,
                        "relative_range": float(np.ptp(obs.y_relative))})
    raw = np.asarray(raw)
    scaled, _, _ = prior.robust_channel_scale(raw)
    return raw, scaled, np.asarray(interpolated), quality


def fit_curves(curves, include_paper=True, som_seeds=SOM_SEEDS):
    """Transductive batch clustering. No labels, filenames or class counts input.

    The four-cluster count is a declared functional-test prior. Cluster IDs
    are anonymous, not semantic class names. Scaling is fit anew on this batch.
    """
    raw, ch, interp, quality = representations(curves)
    matrices = {}
    for name, cols, weights in [
        ("level", [0], [1.]), ("d1", [1], [1.]),
        ("d12", [1, 2], [.75, .25]),
        ("shared", [0, 1, 2], [.5, .375, .125]),
    ]:
        matrices[name] = accelerated_dtw(ch[:, :, cols], weights)
    matrices["fusion"] = .5 * prior.median_scale_distance(matrices["level"]) + .5 * prior.median_scale_distance(matrices["d12"])
    predictions = {}
    graph_info = {}
    for name, distance in matrices.items():
        graph = prior.distance_to_knn_affinity(distance, 10)
        graph_info[name] = int(connected_components(graph, directed=False)[0])
        predictions[(name+"_graph", 0)] = prior.graph_cluster_distance(distance, 4, 10)
    vectors = {
        "som_level": ch[:, :, 0],
        "som_level_d12": (ch[:, :, :3] * np.sqrt([.5, .375, .125])).reshape(len(ch), -1),
    }
    for name, values in vectors.items():
        for seed in som_seeds:
            som = MiniSom(2, 2, values.shape[1], sigma=.5, learning_rate=.1, random_seed=seed)
            som.random_weights_init(values)
            som.train_random(values, 10000)
            predictions[(name, seed)] = np.array([np.ravel_multi_index(som.winner(v), (2, 2)) for v in values])
    degrees = []
    if include_paper:
        transforms = [pfd_dtw.pfd_transform(y, smoothing=1000, max_degree=20) for y in interp]
        features = np.stack([v.features for v in transforms])
        degrees = [v.degree for v in transforms]
        matrices["pfd"] = accelerated_dtw(features, [3., 2., 1.], radius=48, normalize=False)
        for method in ("average", "single"):
            predictions[("pfd_"+method, 0)] = prior.cluster_distance(matrices["pfd"], 4, method)
    return {"predictions": predictions, "matrices": matrices, "raw": raw,
            "quality": quality, "graph_components": graph_info, "pfd_degrees": degrees}


def anchor_curve(target, rng, regime):
    """Independent shape family: randomized monotone anchor segments, not core_shape."""
    low, high, _ = generator.REGIMES[regime]
    duration = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    n = int(rng.integers(15, 81))
    early = np.geomspace(.001, .15, max(4, n//3))
    u = np.unique(np.r_[0., early, rng.uniform(.02, 1., n-len(early)-2), 1.])
    baseline = rng.uniform(.85, 1.1)
    if target == "bridge":
        rise = rng.uniform(.06, .16); plateau = rng.uniform(.32, .60)
        gain = rng.uniform(.20, .48); loss = rng.uniform(.10, .32)
        tx = [0, rise, plateau, 1]
        yy = [baseline, baseline+gain, baseline+gain-rng.uniform(0, .035), baseline+gain-loss]
    elif target == "hill":
        peak = rng.uniform(.08, .28); fall = peak+rng.uniform(.08, .18)
        gain = rng.uniform(.22, .50); tail = baseline-rng.uniform(.03, .14)
        tx = [0, peak, fall, 1]
        yy = [baseline, baseline+gain, tail, tail-rng.uniform(.06, .23)]
    elif target == "slope":
        burn = rng.uniform(.04, .17); loss = rng.uniform(.17, .35)
        tx = [0, burn, .5, 1]
        yy = [baseline, baseline-loss, baseline-loss-rng.uniform(.03, .12), baseline-loss-rng.uniform(.16, .38)]
    elif target == "valley":
        trough = rng.uniform(.05, .18); recover = rng.uniform(.35, .65)
        drop = rng.uniform(.24, .47); gain = rng.uniform(.25, .48)
        tx = [0, trough, recover, 1]
        yy = [baseline, baseline-drop, baseline-drop+gain, baseline-drop+gain-rng.uniform(.07, .17)]
    elif target == "other_flat":
        tx, yy = [0, 1], [baseline, baseline+rng.uniform(-.04, .04)]
    elif target == "other_increase":
        tx, yy = [0, .3, 1], [baseline, baseline+.15, baseline+rng.uniform(.3, .55)]
    elif target == "other_failure":
        when = rng.uniform(.45, .8)
        tx, yy = [0, when-.025, when+.025, 1], [baseline, baseline-.04, .1, .04]
    elif target == "other_multipeak":
        tx = [0, .15, .32, .5, .67, .82, 1]
        yy = np.array([1, 1.3, .8, 1.2, .7, 1.1, .65])*baseline
    else:
        raise ValueError(target)
    y = PchipInterpolator(tx, yy)(u)
    y += rng.uniform(.003, .02)*np.sin(2*np.pi*rng.uniform(1, 6)*u+rng.uniform(0, 6))
    noise_scale = rng.uniform(.004, .02)
    y += rng.normal(0, noise_scale, len(u))
    if rng.random() < .4:
        ix = int(rng.integers(1, len(u)-1))
        y[ix] += rng.choice([-1, 1])*rng.uniform(.04, .12)
    return pd.DataFrame({"time_h": u*duration, "y": np.maximum(y, .001)}), {
        "anchors_u": list(tx), "anchors_y": np.asarray(yy).tolist(),
        "noise_scale": noise_scale, "latent_duration_h": duration,
    }


def new_batch(family, seed, per_class=60):
    rng = np.random.default_rng(seed)
    curves, truth, metadata = [], [], []
    targets = list(CLASSES)
    if family == "mixed":
        targets += ["other_flat", "other_increase", "other_failure", "other_multipeak"]
    for ci, target in enumerate(targets):
        count = per_class if target in CLASSES else per_class//2
        regimes = generator.regime_sequence(count, rng)
        for index, regime in enumerate(regimes):
            curve_seed = seed*100000+ci*1000+index
            if family == "same_family":
                t, y, _, params = generator.make_curve(target, regime, curve_seed)
                frame = pd.DataFrame({"time_h": t, "y": y})
            else:
                frame, params = anchor_curve(target, np.random.default_rng(curve_seed), regime)
            if family == "sparse":
                frame = frame.iloc[np.linspace(0, len(frame)-1, 8).astype(int)].copy()
            elif family == "truncated":
                frame = frame.loc[frame.time_h <= .55*frame.time_h.max()].copy()
            curves.append(frame)
            truth.append(target)
            metadata.append({"seed": curve_seed, "duration_regime": regime, "parameters": params})
    order = rng.permutation(len(curves))
    return [curves[i] for i in order], np.array(truth)[order], [metadata[i] for i in order]


def existing_batch():
    base = ROOT / "eg_based_synthetic_dataset"
    meta = pd.read_csv(base / "curve_metadata.csv").set_index("curve_id")
    files = sorted((base / "curves").glob("*/*.csv"))
    files = [files[i] for i in np.random.default_rng(31000).permutation(len(files))]
    return ([pd.read_csv(p) for p in files],
            np.array([str(meta.loc[p.stem, "target_class"]) for p in files]),
            [{"source": str(p.relative_to(ROOT)), "sha256": sha(p),
              "duration_regime": str(meta.loc[p.stem, "duration_regime"])} for p in files])


def assess(truth, predicted, regimes):
    mask = np.isin(truth, CLASSES)
    ids = np.unique(predicted)
    counts = np.array([[np.sum((truth==c)&(predicted==k)) for k in ids] for c in CLASSES])
    r, c = linear_sum_assignment(-counts)
    recall, precision, mapping = [], [], {}
    for ri, ci in zip(r, c):
        recall.append(counts[ri,ci]/max(1, np.sum(truth==CLASSES[ri])))
        precision.append(counts[ri,ci]/max(1, np.sum(predicted==ids[ci])))
        mapping[CLASSES[ri]] = int(ids[ci])
    per_class = {cl: float(np.sum((truth==cl)&(predicted==mapping.get(cl,-1)))/max(1,np.sum(truth==cl))) for cl in CLASSES}
    return {"ari": float(adjusted_rand_score(truth[mask], predicted[mask])),
            "ari_all": float(adjusted_rand_score(truth, predicted)),
            "mapped_accuracy": float(counts[r,c].sum()/mask.sum()),
            "macro_recall": float(np.mean(list(per_class.values()))),
            "macro_precision_with_background": float(sum(precision)/4),
            "duration_ari": float(adjusted_rand_score(regimes[mask], predicted[mask])),
            "occupied_clusters": len(ids),
            **{"recall_"+k:v for k,v in per_class.items()}}, mapping


def save_batch(folder, curves, truth, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    rows, meta_rows = [], []
    for index, (frame, target, meta) in enumerate(zip(curves, truth, metadata)):
        original_y = next(c for c in ("y_relative", "normalized_pce", "y") if c in frame)
        raw_frame = frame[["time_h", original_y]].rename(columns={original_y:"y"}).copy()
        raw_frame.insert(0, "curve_id", f"curve_{index:04d}")
        rows.append(raw_frame)
        meta_rows.append({"curve_id": f"curve_{index:04d}", "target_posthoc": target, **meta})
    pd.concat(rows).to_csv(folder/"observations.csv", index=False)
    (folder/"metadata_posthoc.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")


def plot_batch(folder, curves, truth, result):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    for j, target in enumerate(CLASSES):
        for index in np.flatnonzero(truth==target)[:8]:
            f = observed_frame(curves[index]); t=f.time_h.to_numpy()
            axes[0,j].plot((t-t[0])/(t[-1]-t[0]), f.y_relative, "o-", ms=2, lw=.7, alpha=.5)
        axes[0,j].set_title("Generated: "+target, fontsize=11)
    pred = result["predictions"][("fusion_graph",0)]
    for j in range(4):
        subset = result["raw"][pred==j,:,0]
        for y in subset:
            axes[1,j].plot(np.linspace(0,1,len(y)), y, color="#56738c", alpha=.13, lw=.7)
        if len(subset):
            axes[1,j].plot(np.linspace(0,1,48), np.median(subset,axis=0), "k", lw=2)
        axes[1,j].set_title(f"Fusion cluster {j}: n={len(subset)}", fontsize=10)
    for ax in axes.ravel():
        ax.grid(alpha=.15)
    fig.supxlabel("Observed relative progress u (not physical hours)")
    fig.supylabel("Per-curve max-normalized response")
    fig.suptitle(folder.name+": raw observations above; anonymous inferred clusters below")
    fig.tight_layout()
    fig.savefig(folder/"curves_and_clusters.png", dpi=150)
    plt.close(fig)


def run_batch(family, seed, curves, truth, meta, output):
    folder = output / f"{family}_{seed}"
    save_batch(folder, curves, truth, meta)
    print(f"Fitting {family} seed={seed}, n={len(curves)}", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_curves(curves)
    regimes = np.array([m["duration_regime"] for m in meta])
    rows = []
    assignments = pd.DataFrame({"curve_id":[f"curve_{i:04d}" for i in range(len(curves))], "target_posthoc":truth})
    mappings = {}
    for (method, model_seed), predicted in result["predictions"].items():
        scores, mapping = assess(truth, predicted, regimes)
        rows.append({"family":family, "data_seed":seed, "method":method, "model_seed":model_seed, **scores})
        key = f"{method}_seed{model_seed}"
        assignments[key] = predicted
        mappings[key] = mapping
    k_rows=[]
    if seed in (0, SEEDS[0]):
        for name in ("level", "fusion"):
            for k in range(2,9):
                pred = prior.graph_cluster_distance(result["matrices"][name], k, 10)
                k_rows.append({"family":family, "data_seed":seed, "method":name, "k":k,
                               "silhouette":float(silhouette_score(result["matrices"][name],pred,metric="precomputed"))})
    if family=="mixed":
        for name in ("level", "fusion"):
            pred = prior.graph_cluster_distance(result["matrices"][name],8,10)
            scores, mapping = assess(truth,pred,regimes)
            rows.append({"family":family,"data_seed":seed,"method":name+"_graph_k8","model_seed":0,**scores})
            assignments[name+"_graph_k8"] = pred
            mappings[name+"_graph_k8"] = mapping
    assignments.to_csv(folder/"assignments.csv",index=False)
    pd.DataFrame(result["quality"]).to_csv(folder/"quality.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/"metrics.csv",index=False)
    np.savez_compressed(folder/"distances.npz", **result["matrices"])
    audit = {"graph_components":result["graph_components"], "pfd_degree_counts":
             {str(k):result["pfd_degrees"].count(k) for k in sorted(set(result["pfd_degrees"]))},
             "warnings":sorted(set(str(w.message) for w in caught)),
             "posthoc_mappings_not_for_prediction":mappings,
             "observations_sha256":sha(folder/"observations.csv")}
    (folder/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    if seed in (0, SEEDS[0]):
        plot_batch(folder, curves, truth, result)
    print(pd.DataFrame(rows).groupby("method").ari.mean().round(3).to_string(),flush=True)
    return rows, k_rows


def summarize(output, all_rows, all_k):
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(all_k).to_csv(output/"k_diagnostics.csv",index=False)
    # Average SOM initialization repeats within each generated batch first.
    batch = frame.groupby(["family","data_seed","method"],as_index=False).mean(numeric_only=True)
    batch.to_csv(output/"batch_means.csv",index=False)
    summary = batch.groupby(["family","method"]).agg(
        ari_mean=("ari","mean"), ari_min=("ari","min"), ari_max=("ari","max"),
        accuracy_mean=("mapped_accuracy","mean"), recall_mean=("macro_recall","mean"),
        precision_mean=("macro_precision_with_background","mean"),
        batches=("data_seed","count")).reset_index()
    summary.to_csv(output/"summary.csv",index=False)
    methods=["som_level","som_level_d12","pfd_average","level_graph","d1_graph","shared_graph","fusion_graph"]
    families=["existing","same_family","independent","sparse","truncated","mixed"]
    fig, ax=plt.subplots(figsize=(14,6))
    x=np.arange(len(families)); width=.105
    for i,method in enumerate(methods):
        sub=summary.loc[summary.method==method].set_index("family").reindex(families)
        vals=sub.ari_mean.to_numpy()
        err=np.vstack([vals-sub.ari_min.to_numpy(),sub.ari_max.to_numpy()-vals])
        ax.bar(x+(i-3)*width,vals,width,yerr=err,capsize=2,label=method)
    ax.set_xticks(x,families); ax.set_ylim(-.04,1.06); ax.set_ylabel("Target-only adjusted Rand index")
    ax.set_title("Frozen parameters; bars = batch means, whiskers = batch range\nSOM averaged over 3 model seeds within each batch; existing = development")
    ax.legend(ncol=3,fontsize=9,loc="lower left"); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(output/"comparison.png",dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"outputs")
    parser.add_argument("--per-class",type=int,default=60)
    parser.add_argument("--seeds",type=int,nargs="+",default=SEEDS)
    parser.add_argument("--input-curves",type=Path,help="Cluster CSV directory (recursive), no ground truth needed")
    args=parser.parse_args(); output=args.output; output.mkdir(parents=True,exist_ok=True)
    manifest={"protocol_sha256":sha(HERE/"PROTOCOL.md"),
              "sources":{str(p):sha(p) for p in [Path(__file__),HERE/"dtw.cpp",ROOT/"build_eg_based_dataset.py",ROOT/"pfd_dtw.py",ROOT/"eg_dtw_derivative_experiment/run_experiment.py"]},
              "platform":platform.platform(),"python":sys.version,
              "versions":{x:importlib.metadata.version(x) for x in ["numpy","scipy","pandas","scikit-learn","minisom"]},
              "seeds":list(args.seeds),"per_class":args.per_class,"som_seeds":list(SOM_SEEDS)}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if args.input_curves:
        files=sorted(args.input_curves.rglob("*.csv"))
        if len(files)<11:
            raise ValueError("10-neighbor graph needs at least 11 curves")
        result=fit_curves([pd.read_csv(p) for p in files])
        frame=pd.DataFrame({"source_file":[str(p) for p in files]})
        for (name,seed),pred in result["predictions"].items(): frame[f"{name}_{seed}"]=pred
        frame.to_csv(output/"anonymous_clusters.csv",index=False)
        pd.DataFrame(result["quality"]).to_csv(output/"quality.csv",index=False)
        np.savez_compressed(output/"distances.npz",**result["matrices"])
        print(output/"anonymous_clusters.csv"); return
    rows,ks=run_batch("existing",0,*existing_batch(),output)
    for family in ("same_family","independent","sparse","truncated","mixed"):
        for seed in args.seeds:
            rr,kk=run_batch(family,seed,*new_batch(family,seed,args.per_class),output)
            rows.extend(rr); ks.extend(kk)
            pd.DataFrame(rows).to_csv(output/"metrics_partial.csv",index=False)
    summarize(output,rows,ks)
    print("FINISHED",output,flush=True)


if __name__=="__main__":
    main()
