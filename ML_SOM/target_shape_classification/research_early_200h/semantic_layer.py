#!/usr/bin/env python3
"""E3/E4 after version-locked human references exist, with reviewer limits reported."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from early_data import CLASSES, HERE, digest, inside, rows, write_rows
from masked_som import MaskedSOM


def checked_references(path, index, config):
    found = {}
    for r in rows(path):
        fid = r["curve_id"]
        if fid in found or fid not in index:
            raise ValueError("duplicate or unknown reviewed curve: " + fid)
        source = index[fid]
        protocol = r.get("reference_protocol", "")
        reviewer_count = int(r["reviewer_count"]) if r["reviewer_count"] else 0
        if not ((protocol == "single_reviewer_with_delayed_recheck" and reviewer_count == 1) or
                (protocol == "two_independent_reviewers" and reviewer_count >= 2)):
            raise ValueError("reviewer count and declared protocol disagree: " + fid)
        if not r["strict_opinions_sha256"]:
            raise ValueError("strict blind opinions not locked: " + fid)
        recheck = r.get("recheck_status", "")
        if recheck not in ("", "not_selected", "pending", "agreed", "discordant_resolved", "discordant_unresolved"):
            raise ValueError("invalid recheck status: " + fid)
        if recheck in ("agreed", "discordant_resolved") and not r.get("recheck_opinion_sha256", ""):
            raise ValueError("completed recheck lacks locked opinion: " + fid)
        if recheck == "discordant_unresolved" and r["reference_strict_200h_status"].lower() == "confirmed":
            raise ValueError("unresolved self-disagreement cannot confirm a class: " + fid)
        for key in ("source_sha256", "image_sha256"):
            if r[key] != source[key]:
                raise ValueError("input hash changed; re-review: " + fid)
        if (digest(inside(config["source_root"], source["source_csv"])) != r["source_sha256"] or
                digest(inside(config["image_root"], source["source_image"])) != r["image_sha256"]):
            raise ValueError("source file changed after review: " + fid)
        if not np.isclose(float(r["normalization_divisor"]),
                          float(source["normalization_divisor"]), rtol=0, atol=1e-9):
            raise ValueError("normalization changed; re-review: " + fid)
        label = r["reference_strict_200h_class"].lower()
        status = r["reference_strict_200h_status"].lower()
        if label and label not in CLASSES:
            raise ValueError("invalid class: " + fid)
        if status == "confirmed" and not label:
            raise ValueError("confirmed review lacks a class: " + fid)
        if status != "confirmed" and label:
            raise ValueError("class requires confirmed evidence: " + fid)
        found[fid] = r
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", default=str(HERE / "reviews/adjudicated_references.csv"))
    parser.add_argument("--score-test", action="store_true", help="Use only after test blind references are locked")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    index = {r["curve_id"]: r for r in rows(HERE / "delivery/class_index.csv")}
    refs = checked_references(args.references, index, config)
    protocols = Counter(r["reference_protocol"] for r in refs.values())
    if len(protocols) > 1:
        raise ValueError("mixed review protocols need separate analyses")
    reference_protocol = next(iter(protocols), "single_reviewer_with_delayed_recheck")
    recheck_completed = sum(r.get("recheck_status") in ("agreed", "discordant_resolved") for r in refs.values())
    if args.score_test and reference_protocol == "single_reviewer_with_delayed_recheck":
        chosen = {r["curve_id"] for r in rows(HERE / "reviews/single_reviewer_train_recheck.csv")}
        if sum(refs.get(fid, {}).get("recheck_status") in ("agreed", "discordant_resolved") for fid in chosen) < 24:
            raise ValueError("finish and lock all 24 single-reviewer rechecks before test scoring")
    confirmed = {fid: r["reference_strict_200h_class"].lower() for fid, r in refs.items()
                 if r["reference_strict_200h_status"].lower() == "confirmed"}
    by_split = defaultdict(list)
    for fid in confirmed:
        by_split[index[fid]["split"]].append(fid)
    groups = defaultdict(set)
    for fid, label in confirmed.items():
        if index[fid]["split"] == "train":
            groups[label].add(index[fid]["source_group"])
    if any(len(groups[label]) < 10 for label in CLASSES) or len(by_split["validation"]) < 80:
        status = dict(state="waiting_for_human_references", reference_protocol=reference_protocol,
                      recheck_completed=recheck_completed,
                      confirmed_by_split={k: len(v) for k, v in by_split.items()},
                      train_source_groups_by_class={k: len(groups[k]) for k in CLASSES},
                      requirement="At least ten distinct training DOI groups per class and 80 confirmed validation curves.")
        (HERE / "evaluation").mkdir(exist_ok=True)
        (HERE / "evaluation/semantic_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status))
        return
    summary = json.loads((HERE / "runs/summary.json").read_text(encoding="utf-8"))
    saved = np.load(HERE / "runs" / f"{summary['selected_run']}.npz")
    stored = np.load(HERE / "manifests/strict_features.npz")
    ids = [str(x) for x in stored["curve_ids"]]
    data, mask = stored["data"], stored["mask"]
    model = MaskedSOM(summary["config"]["map_rows"], summary["config"]["map_columns"],
                      list(saved["feature_sizes"]), list(saved["weights"]), int(saved["seed"]))
    model.prototypes = saved["prototypes"]
    model.supported = saved["supported"]
    distances = np.array([model.distance_one(x, m) for x, m in zip(data, mask)])
    nearest = np.min(distances, axis=1)
    train_curve_ids = set(by_split["train"])
    train_positions = np.array([i for i, fid in enumerate(ids) if fid in train_curve_ids])
    tau = float(np.median(nearest[train_positions]))
    if tau <= 0:
        raise ValueError("invalid distance temperature")
    logits = -distances ** 2 / (2 * tau ** 2)
    logits -= np.max(logits, axis=1, keepdims=True)
    soft = np.exp(logits)
    soft /= np.sum(soft, axis=1, keepdims=True)
    e3 = np.column_stack([soft, nearest])
    # E4 keeps original feature dimensions and their missingness, with all fit
    # statistics learned from the labeled training DOI groups only.
    e4 = np.where(mask, data, np.nan)
    position = {fid: i for i, fid in enumerate(ids)}
    train_ids = [fid for fid in by_split["train"] if fid in position]
    val_ids = [fid for fid in by_split["validation"] if fid in position]
    test_ids = [fid for fid in by_split["test"] if fid in position]
    x_train = np.array([position[fid] for fid in train_ids])
    y_train = np.array([confirmed[fid] for fid in train_ids])
    x_val = np.array([position[fid] for fid in val_ids])
    if len(set(y_train)) != 4:
        raise ValueError("four training classes are required")
    results = {}
    output_rows = []
    for label, matrix, pipeline in (
        ("E3", e3, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("E4", e4, make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))):
        pipeline.fit(matrix[x_train], y_train)
        predicted = pipeline.predict(matrix)
        scores = pipeline.predict_proba(matrix)
        val_truth = [confirmed[fid] for fid in val_ids]
        val_pred = predicted[x_val]
        result = dict(validation_count=len(val_ids),
                      validation_macro_f1=f1_score(val_truth, val_pred, labels=CLASSES, average="macro", zero_division=0),
                      validation_balanced_accuracy=balanced_accuracy_score(val_truth, val_pred),
                      validation_confusion=confusion_matrix(val_truth, val_pred, labels=CLASSES).tolist(),
                      validation_report=classification_report(val_truth, val_pred, labels=CLASSES,
                                                              output_dict=True, zero_division=0))
        if args.score_test:
            if len(test_ids) < 100:
                raise ValueError("locked test reference set too small")
            test_truth = [confirmed[fid] for fid in test_ids]
            result["test_macro_f1"] = f1_score(test_truth, predicted[[position[f] for f in test_ids]],
                                                labels=CLASSES, average="macro", zero_division=0)
            result["test_count"] = len(test_ids)
        results[label] = result
        for i, fid in enumerate(ids):
            output_rows.append(dict(curve_id=fid, route=label, split=index[fid]["split"],
                                    predicted_class=predicted[i], model_score=float(np.max(scores[i])),
                                    reference_class=confirmed.get(fid, ""),
                                    reference_status=refs.get(fid, {}).get("reference_strict_200h_status", "")))
    output = HERE / "evaluation"
    write_rows(output / "semantic_predictions.csv", output_rows, list(output_rows[0]))
    metadata = dict(reference_file=args.references, reference_sha256=digest(args.references),
                    som_model_sha256=summary["selected_model_sha256"], distance_temperature=tau,
                    training_labels=len(train_ids), validation_labels=len(val_ids),
                    test_scores_included=bool(args.score_test), reference_protocol=reference_protocol,
                    recheck_completed=recheck_completed,
                    reference_limitation=("One person's judgments; delayed recheck measures within-reader consistency, not inter-reader agreement."
                                          if reference_protocol == "single_reviewer_with_delayed_recheck" else ""),
                    results=results)
    (output / "semantic_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"training_labels": len(train_ids), "validation_labels": len(val_ids),
                      "E3_macro_f1": results["E3"]["validation_macro_f1"],
                      "E4_macro_f1": results["E4"]["validation_macro_f1"]}))


if __name__ == "__main__":
    main()
