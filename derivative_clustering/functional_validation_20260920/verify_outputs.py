"""Independent checks of emitted observations, assignments, metrics and lineage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import validate as v


def main():
    out=Path(__file__).resolve().parent/"outputs"
    metrics=pd.read_csv(out/"metrics.csv")
    totals=0; batches=0; comparisons=0
    for folder in sorted(out.iterdir()):
        if not (folder/"assignments.csv").exists():
            continue
        batches+=1
        obs=pd.read_csv(folder/"observations.csv")
        assignments=pd.read_csv(folder/"assignments.csv")
        meta=json.loads((folder/"metadata_posthoc.json").read_text())
        audit=json.loads((folder/"audit.json").read_text())
        assert audit["observations_sha256"]==v.sha(folder/"observations.csv")
        assert len(meta)==len(assignments)==obs.curve_id.nunique()
        assert assignments.curve_id.tolist()==[m["curve_id"] for m in meta]
        assert "target_posthoc" not in obs
        assert len(set(assignments.curve_id))==len(assignments)
        family,seed=folder.name.rsplit("_",1)
        truth=assignments.target_posthoc.to_numpy()
        mask=np.isin(truth,v.CLASSES)
        for _,row in metrics.loc[(metrics.family==family)&(metrics.data_seed==int(seed))].iterrows():
            column=row.method if row.method.endswith("_k8") else f"{row.method}_seed{int(row.model_seed)}"
            ari=adjusted_rand_score(truth[mask],assignments[column].to_numpy()[mask])
            assert abs(ari-row.ari)<1e-12
            comparisons+=1
        cache=np.load(folder/"distances.npz")
        for name in cache.files:
            matrix=cache[name]
            assert matrix.shape==(len(meta),len(meta))
            assert np.isfinite(matrix).all() and np.all(matrix>=0)
            np.testing.assert_array_equal(matrix,matrix.T)
            np.testing.assert_array_equal(np.diag(matrix),0)
        if family=="existing":
            for m in meta:
                assert m["sha256"]==v.sha(v.ROOT/m["source"])
        totals+=len(meta)
    assert batches==16 and totals==4360 and comparisons==len(metrics)
    summary=pd.read_csv(out/"summary.csv")
    actual=metrics.groupby(["family","data_seed","method"]).ari.mean().groupby(["family","method"]).mean()
    for row in summary.itertuples():
        assert abs(row.ari_mean-actual.loc[(row.family,row.method)])<1e-12
    existing=pd.read_csv(out/"existing_0/assignments.csv")
    metadata=json.loads((out/"existing_0/metadata_posthoc.json").read_text())
    existing["source_file"]=[m["source"] for m in metadata]
    cli=pd.read_csv(v.HERE/"cli_smoke/anonymous_clusters.csv")
    joined=existing.merge(cli,on="source_file",validate="one_to_one")
    assert len(joined)==400
    cli_agreement={}
    for method in ("level_graph","d1_graph","d12_graph","shared_graph","fusion_graph"):
        score=adjusted_rand_score(joined[method+"_seed0"],joined[method+"_0"])
        assert score==1
        cli_agreement[method]=score
    report={"batches":batches,"curve_observation_instances":totals,
            "metrics_recomputed":comparisons,"original_400_files_unchanged":True,
            "all_distance_matrices_valid":True,
            "unlabeled_cli_agreement_after_row_permutation":cli_agreement,
            "final_source_sha256":{str(p.name):v.sha(p) for p in [v.HERE/"validate.py",v.HERE/"dtw.cpp",v.HERE/"test_validation.py"]},
            "post_run_source_changes":"Only plot title font sizes, summary plotted methods, removal of unused validation call, and added identifiability test; model fitting unchanged."}
    (out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
