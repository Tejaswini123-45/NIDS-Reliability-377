"""Dataset-specific audit exports."""

import numpy as np
import pandas as pd


def class_coverage(train, test, labels):
    train_counts = pd.Series(labels[train]).value_counts()
    test_counts = pd.Series(labels[test]).value_counts()
    classes = np.unique(labels)
    return pd.DataFrame({
        "Class": classes, "Train": train_counts.reindex(classes, fill_value=0).to_numpy(),
        "Test": test_counts.reindex(classes, fill_value=0).to_numpy(),
    })


def audit_reference_overlap(output, source, train_path, test_path):
    """Measure reference value overlap separately from official holdout overlap."""
    train = pd.read_csv(train_path).drop(columns="id")
    test = pd.read_csv(test_path).drop(columns="id")
    merged = pd.concat([train, test], ignore_index=True)
    features = merged.drop(columns=["label", "attack_cat"])
    exact_hash = pd.util.hash_pandas_object(merged, index=False).to_numpy()
    feature_hash = pd.util.hash_pandas_object(features, index=False).to_numpy()
    results = []
    for path in sorted(source.glob("split_*_Reference.npz")):
        seed = int(path.stem.split("_")[1])
        with np.load(path, allow_pickle=False) as split:
            fit, evaluation = split["train"], split["test"]
        results.append({
            "Seed": seed, "Fit rows": len(fit), "Test rows": len(evaluation),
            "Exact value overlap rows": int(np.isin(exact_hash[evaluation], exact_hash[fit]).sum()),
            "Feature value overlap rows": int(np.isin(feature_hash[evaluation], feature_hash[fit]).sum()),
        })
    pd.DataFrame(results).to_csv(output / "reference_overlap_by_seed.csv", index=False)
