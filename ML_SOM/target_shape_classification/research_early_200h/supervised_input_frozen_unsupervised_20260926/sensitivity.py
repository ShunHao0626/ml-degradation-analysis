"""Refit four anonymous groups after perturbing original PCE observations."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from run import CONFIG, HERE, OUT, encode, fit, original_observations, read_rows


def main() -> None:
    baseline = read_rows(OUT / "anonymous_assignments.csv")
    index = {r["curve_id"]: r for r in read_rows((HERE / CONFIG["input_index"]).resolve())}
    originals = [original_observations(index[r["curve_id"]]) for r in baseline]
    reference = np.array([int(r["anonymous_cluster"]) for r in baseline])
    report = []
    for noise_sd in (0.003, 0.005):
        for seed in range(5):
            rng = np.random.default_rng(21000 + seed)
            perturbed = []
            for values in originals:
                noisy = values + rng.normal(0, noise_sd, len(values))
                noisy = noisy / noisy[0]
                rank = np.linspace(0, len(values) - 1, 64)
                perturbed.append(np.interp(rank, np.arange(len(values)), noisy))
            embedding, _, _, _ = encode(np.array(perturbed))
            new, _, _ = fit(embedding)
            jaccard = []
            for group in sorted(set(reference)):
                old_members = reference == group
                jaccard.append(max(np.sum(old_members & (new == k)) /
                                   np.sum(old_members | (new == k)) for k in set(new)))
            report.append(dict(noise_sd=noise_sd, seed=seed,
                               adjusted_rand_index=adjusted_rand_score(reference, new),
                               median_best_cluster_jaccard=float(np.median(jaccard)),
                               best_cluster_jaccard=[float(v) for v in jaccard],
                               cluster_sizes=np.bincount(new).tolist()))
    summary = {}
    for noise_sd in (0.003, 0.005):
        rows = [r for r in report if r["noise_sd"] == noise_sd]
        summary[str(noise_sd)] = dict(
            mean_ari=float(np.mean([r["adjusted_rand_index"] for r in rows])),
            min_ari=float(np.min([r["adjusted_rand_index"] for r in rows])),
            mean_median_best_cluster_jaccard=float(np.mean(
                [r["median_best_cluster_jaccard"] for r in rows])))
    result = dict(perturbation_unit="normalized_PCE", noise_is_hypothetical=True,
                  original_observations_perturbed=True, original_exclusion_set_fixed=True,
                  refit_from_scratch=True, summary=summary, trials=report)
    (OUT / "noise_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
