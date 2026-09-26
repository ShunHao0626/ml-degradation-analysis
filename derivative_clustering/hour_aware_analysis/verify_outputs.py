"""Independent artifact checks; run after run_analysis.py and its CLI smoke test."""
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from hour_model import HERE, ROOT, baseline, stable_shape_partition
from run_analysis import dump, load_prior


def main():
    output=HERE/"outputs"
    summary=json.loads((output/"metrics_summary.json").read_text())
    checks=[]
    for metric in summary:
        folder=output/metric["batch"]
        a=pd.read_csv(folder/"hour_assignments.csv")
        channels=pd.read_csv(folder/"physical_channels.csv")
        raw=pd.read_csv(folder/"raw_observations.csv")
        assert len(a)==metric["n"] and a.curve_id.is_unique
        assert baseline.sha(folder/"raw_observations.csv")==metric["saved_raw_sha256"]
        assert a.timescale_subgroup.ge(0).sum()==metric["subgroup_assigned_count"]
        assert a.loc[a.timescale_subgroup.ge(0),"excursion_10_status"].eq("resolved").all()
        for delta in (5,10,20):
            p=f"excursion_{delta:02d}_"
            measured=a[p+"status"].ne("not_observed_censored")
            rows=a.loc[measured]
            assert (rows[p+"time_h"]>=rows[p+"bracket_start_h"]-1e-10).all()
            assert (rows[p+"time_h"]<=rows[p+"bracket_end_h"]+1e-10).all()
            assert np.allclose(rows[p+"time_h"]-rows.start_h,rows[p+"elapsed_h"])
            assert a.loc[~measured,p+"elapsed_h"].isna().all()
        assert np.isfinite(channels[["time_h","dy_per_h","d2y_per_h2","dy_raw_per_h"]]).all().all()
        assert len(channels)==len(raw)
        assert np.allclose(channels.time_h,raw.time_h)
        assert np.allclose(channels.y_raw,raw.y)
        info=json.loads((folder/"shape_graph_diagnostics.json").read_text())
        assert info["eigen_residual_max"]<1e-9
        if "clock_replicas" in metric["batch"]:
            replica=pd.read_csv(folder/"replica_invariance_checks.csv")
            assert replica.iloc[:,2:].all().all()
        checks.append({"batch":metric["batch"],"artifact_checks_pass":True,"n":len(a)})
    meta=json.loads((output/"existing_400"/"metadata_posthoc.json").read_text())
    assert all(baseline.sha(ROOT/m["source"])==m["sha256"] for m in meta)
    # Regression reproduces the exact serialization sensitivity that motivated
    # the numerical amendment. Both fits still receive only observations.
    original,truth,_=baseline.new_batch("independent",31001)
    restored,truth2,_=load_prior("independent_31001")
    labels=[]; distances=[]
    for curves in (original,restored):
        _,scaled,_,_=baseline.representations(curves)
        distance=baseline.accelerated_dtw(scaled[:,:,:3],[.5,.375,.125])
        pred,_=stable_shape_partition(distance)
        labels.append(pred); distances.append(distance)
    assert np.array_equal(truth,truth2)
    agreement=float(adjusted_rand_score(*labels))
    assert agreement==1.
    # CLI input files were sorted, whereas benchmark inputs were shuffled.
    # Compare assignments by source path, not by arbitrary cluster IDs.
    smoke=HERE/"cli_smoke"
    smoke_a=pd.read_csv(smoke/"hour_assignments.csv")
    smoke_meta=json.loads((smoke/"source_manifest.json").read_text())
    by_path={m["path"]:smoke_a.iloc[i].shape_cluster for i,m in enumerate(smoke_meta)}
    benchmark=pd.read_csv(output/"existing_400"/"hour_assignments.csv")
    aligned=[by_path[str((ROOT/m["source"]).resolve())] for m in meta]
    cli_agreement=float(adjusted_rand_score(benchmark.shape_cluster,aligned))
    assert cli_agreement==1.
    report={"batches":checks,"total_observations_as_curves":sum(m["n"] for m in summary),
            "original_400_hashes_unchanged":True,"roundtrip_shape_partition_ari":agreement,
            "roundtrip_max_distance_difference":float(np.max(np.abs(distances[0]-distances[1]))),
            "cli_vs_benchmark_shape_partition_ari":cli_agreement,
            "python":platform.python_version(),
            "packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas","scikit-learn","matplotlib")},
            "code_hashes":{p.name:baseline.sha(p) for p in HERE.glob("*.py")}}
    dump(output/"verification.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
