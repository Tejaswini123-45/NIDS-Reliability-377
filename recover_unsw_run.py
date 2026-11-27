"""Recover UNSW paper metrics from complete predictions after an export failure."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_audit import audit_reference_overlap
from src.evidence import CONDITIONS, file_digest, finalize


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def recover(source, timing_source):
    root = Path(__file__).resolve().parent
    shared = {"__file__": str(root / "run_unsw_experiments.py"), "__name__": "__main__"}
    for name in ["unsw_01_setup.py", "unsw_02_data_and_temporal.py", "unsw_03_preprocessing.py"]:
        path = root / "src" / name
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), shared)
    if not shared["PAPER_MODE"]:
        raise ValueError("Recovery requires the full five-seed protocol")
    output = shared["OUTPUT_DIR"]
    frame = shared["df_master"]
    numerical = shared["numeric_cols"]
    seeds = shared["SEEDS"]
    shutil.copytree(source / "predictions", output / "predictions")
    for pattern in ("split_*.npz", "coverage_*.csv"):
        for path in source.glob(pattern):
            shutil.copy2(path, output / path.name)
    if (source / "execution.log").exists():
        shutil.copy2(source / "execution.log", output / "execution.log")
    observed_manifest = json.loads((source / "dataset_provenance.json").read_text())
    if observed_manifest != json.loads((output / "dataset_provenance.json").read_text()):
        raise ValueError("The prediction run used a different dataset manifest")

    rows = []
    factories = dict(zip(shared["MODEL_NAMES"], [shared["make_lr"], shared["make_rf"], shared["make_xgb"]]))
    holdout_truth = {"binary": frame.label.to_numpy(), "multiclass": frame.attack_cat.to_numpy(dtype=str)}
    for seed in seeds:
        for condition in CONDITIONS:
            split = condition.split("+")[0]
            rate = 0.10 if "+10" in condition else 0.20 if "+20" in condition else 0.0
            with np.load(output / f"split_{seed}_{split}.npz", allow_pickle=False) as partition:
                train, test = partition["train"], partition["test"]
            _, masked, newly_missing = shared["inject_feature_missingness"](
                frame.iloc[test], rate, seed, numerical, counts=True)
            eligible = len(test) * len(numerical)
            for task, labels in holdout_truth.items():
                binary = task == "binary"
                for model, factory in factories.items():
                    name = f"{seed}_{task}_{model}_{condition}".replace(" ", "_")
                    path = output / "predictions" / f"{name}.npz"
                    with np.load(path, allow_pickle=False) as evidence:
                        truth, pred, classes = evidence["truth"], evidence["predicted"], evidence["classes"]
                        np.testing.assert_array_equal(evidence["row_ids"], test)
                        np.testing.assert_array_equal(truth, labels[test])
                        class_set = {0, 1} if binary else set(frame.attack_cat.unique())
                        if set(classes) != class_set or not set(pred).issubset(set(labels[train])):
                            raise ValueError(f"Unexpected labels in {path}")
                        scores = shared["evaluate_predictions"](truth, pred, classes, binary)
                    estimator = factory(seed, binary=binary) if model == "XGBoost" else factory(seed)
                    rows.append({
                        **scores, "Seed": seed, "Task": task, "Model": model, "Condition": condition,
                        "Missingness": rate, "Prediction file": str(path.relative_to(output)),
                        "Prediction SHA256": file_digest(path), "Eligible stress cells": eligible,
                        "Mask selected cells": masked, "New missing cells": newly_missing,
                        "Mask selected (%)": 100 * masked / eligible,
                        "Model parameters": json.dumps(estimator.get_params(), sort_keys=True),
                    })

    holdout = "Official Train/Test Holdout"
    summary = {"seeds": seeds, "paper_mode": True, "total_unified_rows": len(frame),
               "holdout": holdout, "recovered_from_predictions": True}
    write_json(output / "validation.json", summary)
    write_json(output / "experiment_config.json", {
        **summary, "missingness_rates": list(shared["MISSINGNESS_LEVELS"]),
        "missingness_eligible": numerical, "predictors": shared["feature_candidates"],
        "categorical_features": shared["categorical_cols"], "source": str(source),
        "settings_source": "Unchanged paper-mode model factories and split artifacts",
    })
    files = []
    for file_key, frame_key, role in [("train_file", "df_train_raw", "official_train"),
                                      ("test_file", "df_test_raw", "official_test")]:
        partition = shared[frame_key].drop(columns="_session_partition")
        files.append({
            "file": shared[file_key].name, "role": role, "rows": len(partition), "columns": list(partition),
            "missing_cells": int(partition.isna().sum().sum()),
            "infinite_numeric_cells": int(np.isinf(partition.select_dtypes("number").to_numpy()).sum()),
            "class_counts": partition.attack_cat.value_counts().to_dict(),
            "timestamp_columns": [column for column in partition if column in {"stime", "ltime", "timestamp"}],
        })
    write_json(output / "dataset_file_validation.json", files)
    audit_reference_overlap(output, output, shared["train_file"], shared["test_file"])
    if timing_source:
        timings = pd.read_csv(timing_source / "verified_seed_metrics.csv")
        fields = ["Seed", "Task", "Model", "Condition", "Fit seconds", "Predict seconds", "Records per second"]
        timings[fields].assign(**{"Timing protocol": "Seed 42, 50,000 fitted rows per split"}).to_csv(
            output / "timing_validation_seed42.csv", index=False)
        timing_metadata = json.loads((timing_source / "prediction_validation.json").read_text())
        write_json(output / "timing_validation_metadata.json", timing_metadata)
    finalize(output, pd.DataFrame(rows), seeds, holdout, shared["feature_candidates"], shared["duplicate_test_ids"])
    write_json(output / "recovery.json", {
        "source": str(source), "evaluations_recomputed_from_predictions": len(rows),
        "training_or_prediction_repeated": False,
        "unavailable_original_measurements": ["fit timing", "prediction timing", "peak memory", "LR convergence"],
        "missingness_counts": "Recomputed from the original masks and pinned inputs",
    })
    print(f"Recovered {len(rows)} evaluations in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--timing-source", type=Path)
    args = parser.parse_args()
    recover(args.source, args.timing_source)
