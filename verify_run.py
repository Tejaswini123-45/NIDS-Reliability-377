"""Verify exported seed metrics and uncertainty from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.evidence import KEY, METRICS, check_keys, file_digest, independent_metrics


def verify(directory):
    record = json.loads((directory / "prediction_validation.json").read_text())
    runs = pd.read_csv(directory / "verified_seed_metrics.csv")
    if record["status"] != "PASS":
        raise ValueError("Incomplete prediction audit")
    check_keys(runs, record["seeds"])
    if record["paper_mode"] and record["seeds"] != [42, 52, 62, 72, 82]:
        raise ValueError("Paper mode requires the five protocol seeds")
    run_config = json.loads((directory / "experiment_config.json").read_text())
    if run_config["paper_mode"] != record["paper_mode"] or run_config["seeds"] != record["seeds"]:
        raise ValueError("Run configuration disagrees with verified seed coverage")
    legacy_schema = json.loads((directory / "validation.json").read_text())
    population = legacy_schema.get("rows_final", legacy_schema.get("total_unified_rows"))
    split_ids = {}
    holdout_test = None
    for seed in record["seeds"]:
        for split in ("Reference", "Temporal"):
            with np.load(directory / f"split_{seed}_{split}.npz", allow_pickle=False) as artifact:
                train, test = artifact["train"], artifact["test"]
            combined = np.union1d(train, test)
            if len(combined) != len(train) + len(test) or combined[0] < 0 or combined[-1] >= population:
                raise ValueError("Invalid or overlapping split row identities")
            if record["paper_mode"] and len(combined) != population:
                raise ValueError("Paper-mode split does not cover the full population")
            if split == "Temporal":
                if holdout_test is None:
                    holdout_test = test
                else:
                    np.testing.assert_array_equal(test, holdout_test)
            split_ids[(seed, split)] = test
    mask_columns = ["Eligible stress cells", "Mask selected cells", "New missing cells"]
    if set(mask_columns).issubset(runs):
        if not runs.groupby(["Seed", "Condition"])[mask_columns].nunique().eq(1).all().all():
            raise ValueError("Mask counts differ across paired model or task evaluations")
        selected, new = runs["Mask selected cells"], runs["New missing cells"]
        if not ((new >= 0) & (new <= selected) & (selected <= runs["Eligible stress cells"])).all():
            raise ValueError("Invalid missingness cell counts")
        if not selected[runs["Missingness"].eq(0)].eq(0).all():
            raise ValueError("Clean evaluations contain synthetic corruption")

    observed_truth = {}
    for evaluation in runs.to_dict("records"):
        path = directory / evaluation["Prediction file"]
        if file_digest(path) != evaluation["Prediction SHA256"]:
            raise ValueError(f"Unexpected prediction checksum: {path}")
        split = "Reference" if evaluation["Condition"].startswith("Reference") else "Temporal"
        with np.load(path, allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["row_ids"], split_ids[(evaluation["Seed"], split)])
            truth, predicted, classes = artifact["truth"], artifact["predicted"], artifact["classes"]
            truth_key = (evaluation["Seed"], evaluation["Task"], split)
            np.testing.assert_array_equal(truth, observed_truth.setdefault(truth_key, truth))
            scores = independent_metrics(truth, predicted, classes, evaluation["Task"] == "binary")
            expected_cm = np.zeros((len(classes), len(classes)), dtype=int)
            order = np.argsort(classes)
            np.add.at(expected_cm, (order[np.searchsorted(classes[order], truth)],
                                   order[np.searchsorted(classes[order], predicted)]), 1)
        np.testing.assert_allclose([evaluation[m] for m in METRICS], list(scores.values()), atol=1e-12)
        matrix = pd.read_csv(path.with_name(path.stem + "_confusion.csv"), index_col=0)
        np.testing.assert_array_equal(matrix.to_numpy(), expected_cm)
        per_class = pd.read_csv(path.with_name(path.stem + "_recall.csv"))
        support = expected_cm.sum(axis=1)
        recall = np.divide(expected_cm.diagonal(), support, out=np.full(len(classes), np.nan), where=support > 0)
        np.testing.assert_array_equal(per_class["Test support"], support)
        np.testing.assert_allclose(per_class["Recall"], recall, atol=1e-12)

    summary = pd.read_csv(directory / "verified_summary.csv", header=[0, 1], index_col=[0, 1, 2])
    for keys, group in runs.groupby(KEY[1:]):
        values = group[METRICS].to_numpy()
        n = len(values)
        means = np.mean(values, axis=0)
        sd = np.std(values, ddof=1, axis=0) if n > 1 else np.full(len(METRICS), np.nan)
        margin = t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        saved = summary.loc[keys]
        for label, expected in [("Mean", means), ("Sample SD", sd),
                                ("CI95 lower", means - margin), ("CI95 upper", means + margin)]:
            np.testing.assert_allclose(saved[label][METRICS].to_numpy(), expected, atol=1e-12)
        if saved[("Count", "N")] != n:
            raise ValueError(f"Wrong summary count for {keys}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = verify(parser.parse_args().directory)
    print(f"Verified {result['evaluations']} evaluations for seeds {result['seeds']}")
