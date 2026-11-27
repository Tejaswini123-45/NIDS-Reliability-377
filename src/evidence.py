"""Prediction-based verification and reproducible run artifacts."""

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.dataset_audit import class_coverage


METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "FPR"]
KEY = ["Seed", "Task", "Model", "Condition"]
CONDITIONS = [
    f"{split}{suffix}" for split in ("Reference", "Temporal")
    for suffix in ("", "+10% Missingness", "+20% Missingness")
]


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_directory(dataset):
    base = Path(os.environ.get("NIDS_OUTPUT_ROOT", "verified_runs"))
    output = base / dataset
    output.mkdir(parents=True, exist_ok=False)
    return output


def verify_dataset(dataset, output):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/dataset_manifest.json").read_text())[dataset]
    directory = root / "data" / dataset
    paths = []
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if (not path.is_file() or path.stat().st_size != expected["bytes"]
                or file_digest(path) != expected["sha256"]):
            raise ValueError(f"Missing or mismatched benchmark file: {path}. See README acquisition.")
        paths.append(path)
    (output / "dataset_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths, manifest["repository"]


def independent_metrics(truth, predicted, classes, binary):
    if not len(truth) or truth.shape != predicted.shape:
        raise ValueError("Prediction and truth arrays must be aligned and nonempty")
    support = np.array([np.count_nonzero(truth == name) for name in classes])
    predicted_count = np.array([np.count_nonzero(predicted == name) for name in classes])
    correct = np.array([
        np.count_nonzero((truth == name) & (predicted == name)) for name in classes
    ])
    if support.sum() != len(truth) or predicted_count.sum() != len(truth):
        raise ValueError("Unexpected class in truth or decoded predictions")

    def zero_divide(top, bottom):
        return np.divide(top, bottom, out=np.zeros_like(top, dtype=float), where=bottom != 0)

    precision = zero_divide(correct, predicted_count)
    recall = zero_divide(correct, support)
    f1 = zero_divide(2 * correct, support + predicted_count)
    fprs = zero_divide(predicted_count - correct, len(truth) - support)
    return dict(zip(METRICS, [
        np.mean(truth == predicted), precision.mean(), recall.mean(), f1.mean(),
        np.average(f1, weights=support),
        fprs[list(classes).index(1)] if binary else fprs.mean(),
    ]))


def save_split(output, seed, split, train, test, labels):
    if len(np.unique(train)) != len(train) or len(np.union1d(train, test)) != len(train) + len(test):
        raise ValueError("Duplicate row identities within or across split")
    np.savez_compressed(output / f"split_{seed}_{split}.npz", train=train, test=test)
    class_coverage(train, test, labels).to_csv(output / f"coverage_{seed}_{split}.csv", index=False)


def save_predictions(output, row, truth, predicted, classes, seen, test_idx):
    name = "_".join(str(row[key]).replace(" ", "_") for key in KEY)
    directory = output / "predictions"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.npz"
    np.savez_compressed(path, truth=truth, predicted=predicted, classes=classes, row_ids=test_idx)
    checked = independent_metrics(truth, predicted, classes, row["Task"] == "binary")
    for metric, value in checked.items():
        np.testing.assert_allclose(row[metric], value, rtol=1e-12, atol=1e-12)
    matrix = row["Confusion Matrix"]
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(directory / f"{name}_confusion.csv")
    train_classes = set(seen)
    per_class = pd.DataFrame({
        "Class": classes, "Seen in training": [label in train_classes for label in classes],
        "Test support": matrix.sum(axis=1),
        "Recall": np.divide(matrix.diagonal(), matrix.sum(axis=1),
                            out=np.full(len(classes), np.nan), where=matrix.sum(axis=1) > 0),
    })
    per_class.to_csv(directory / f"{name}_recall.csv", index=False)
    observed_seen = per_class["Seen in training"] & (per_class["Test support"] > 0)
    row["Seen test macro recall"] = per_class.loc[observed_seen, "Recall"].mean()
    row["Unseen test rows"] = int(per_class.loc[~per_class["Seen in training"], "Test support"].sum())
    row["Prediction file"] = str(path.relative_to(output))
    row["Prediction SHA256"] = file_digest(path)
    metadata = {key: value for key, value in row.items()
                if key not in {"Confusion Matrix", "Per-class Recall"}}
    (directory / f"{name}_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def check_keys(results, seeds):
    expected = pd.MultiIndex.from_product([
        seeds, ["binary", "multiclass"], ["Logistic Regression", "Random Forest", "XGBoost"],
        CONDITIONS,
    ], names=KEY)
    actual = pd.MultiIndex.from_frame(results[KEY])
    if not actual.is_unique or set(actual) != set(expected):
        raise ValueError("Incomplete or duplicate seed, model, task, or condition keys")


def finalize(output, results, seeds, holdout, feature_columns, duplicate_ids=None):
    check_keys(results, seeds)
    columns = [column for column in results if column not in {"Confusion Matrix", "Per-class Recall", "LR_Diag"}]
    runs = results[columns].copy()
    runs.to_csv(output / "verified_seed_metrics.csv", index=False)
    runs = pd.read_csv(output / "verified_seed_metrics.csv")
    sensitivity = []
    for record in runs.to_dict("records"):
        path = output / record["Prediction file"]
        if file_digest(path) != record["Prediction SHA256"]:
            raise ValueError(f"Prediction artifact hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as artifact:
            scores = independent_metrics(
                artifact["truth"], artifact["predicted"], artifact["classes"], record["Task"] == "binary",
            )
            if duplicate_ids is not None and record["Condition"] == "Temporal":
                truth, predicted, classes = artifact["truth"], artifact["predicted"], artifact["classes"]
                keep = ~np.isin(artifact["row_ids"], duplicate_ids)
                filtered = independent_metrics(truth[keep], predicted[keep], classes, record["Task"] == "binary")
                sensitivity.append({
                    **{key: record[key] for key in KEY}, "Removed rows": int((~keep).sum()),
                    "Original rows": len(keep), "Filtered rows": int(keep.sum()),
                    **{f"Original {k}": v for k, v in scores.items()},
                    **{f"Filtered {k}": v for k, v in filtered.items()},
                    **{f"Change {k}": filtered[k] - v for k, v in scores.items()},
                })
                for label in classes:
                    original = truth == label
                    retained = original & keep
                    correct = predicted == label
                    sensitivity[-1][f"Original support {label}"] = int(original.sum())
                    sensitivity[-1][f"Filtered support {label}"] = int(retained.sum())
                    sensitivity[-1][f"Original recall {label}"] = correct[original].mean() if original.any() else np.nan
                    sensitivity[-1][f"Filtered recall {label}"] = correct[retained].mean() if retained.any() else np.nan
        np.testing.assert_allclose([record[m] for m in METRICS], list(scores.values()), atol=1e-12)
    grouped = runs.groupby(KEY[1:])[METRICS]
    n = grouped.size()
    mean, std = grouped.mean(), grouped.std(ddof=1)
    margin = std.mul(t.ppf(0.975, n - 1) / np.sqrt(n), axis=0)
    pd.concat({"Mean": mean, "Sample SD": std, "CI95 lower": mean - margin,
               "CI95 upper": mean + margin, "Count": n.rename("N")}, axis=1).to_csv(
        output / "verified_summary.csv")

    ref = runs[runs["Condition"] == "Reference"][KEY[:3] + ["Macro F1", "FPR"]]
    change = runs.merge(ref, on=KEY[:3], suffixes=("", " reference"), validate="many_to_one")
    change["Absolute macro F1 drop"] = change["Macro F1 reference"] - change["Macro F1"]
    change["Relative macro F1 drop (%)"] = (
        100 * change["Absolute macro F1 drop"] / change["Macro F1 reference"].replace(0, np.nan)
    )
    change["FPR change"] = change["FPR"] - change["FPR reference"]
    change.to_csv(output / "paired_degradation.csv", index=False)
    interaction = change.pivot(index=KEY[:3], columns="Condition", values="Absolute macro F1 drop")
    pd.concat({
        rate: interaction[f"Temporal+{rate} Missingness"]
        - interaction["Temporal"] - interaction[f"Reference+{rate} Missingness"]
        for rate in ("10%", "20%")
    }, names=["Missingness"]).rename("Absolute macro F1 interaction").to_csv(output / "paired_interaction.csv")
    if sensitivity:
        pd.DataFrame(sensitivity).to_csv(output / "duplicate_sensitivity.csv", index=False)
    (output / "prediction_validation.json").write_text(json.dumps({
        "status": "PASS", "evaluations": len(runs), "seeds": seeds, "holdout": holdout,
        "paper_mode": len(seeds) == 5, "predictors": feature_columns,
        "macro_policy": "Fixed dataset label universe, with zero for undefined class scores",
        "fpr_policy": "Binary attack FPR. Multiclass unweighted macro one-vs-rest FPR.",
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        **{p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]}},
    }, indent=2), encoding="utf-8")
