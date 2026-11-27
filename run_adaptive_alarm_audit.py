"""Controlled offline alarm replay on the verified UNSW-NB15 flow features."""

import json
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parent
    shared = {"__file__": str(root / "run_unsw_experiments.py"), "__name__": "__main__"}
    for name in ["unsw_01_setup.py", "unsw_02_data_and_temporal.py", "unsw_03_preprocessing.py"]:
        path = root / "src" / name
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), shared)

    dataset = shared["df_master"]
    train = shared["temp_train_idx_base"]
    test = np.setdiff1d(shared["temp_test_idx_base"], shared["duplicate_test_ids"])
    output = shared["OUTPUT_DIR"]
    rows = []
    window_size, fit_size, history_size = 5_000, 20_000, 8
    rate = 0.20
    for seed in shared["SEEDS"]:
        rng = np.random.default_rng(seed)
        ordered_train = rng.permutation(train)
        fit_rows = ordered_train[:fit_size]
        calibration_rows = ordered_train[fit_size:fit_size + history_size * window_size]
        test_rows = rng.permutation(test)
        windows = len(test_rows) // window_size
        split = windows // 2

        fitted = shared["fit_train_preprocessors"](
            dataset.iloc[fit_rows], shared["categorical_cols"], shared["numeric_cols"],
        )
        transform = shared["transform_feature_matrix"]
        fit = transform(fitted, dataset.iloc[fit_rows])
        scaler = StandardScaler().fit(fit)
        cluster = KMeans(n_clusters=10, random_state=seed, n_init=5).fit(scaler.transform(fit))

        def score(frame):
            features = scaler.transform(transform(fitted, frame))
            return float(np.min(cluster.transform(features), axis=1).mean())

        history = deque((
            score(dataset.iloc[indices]) for indices in np.split(calibration_rows, history_size)
        ), maxlen=history_size)
        fixed = float(np.mean(history) + 2 * np.std(history))
        for index in range(windows):
            shifted = index >= split
            indices = test_rows[index * window_size:(index + 1) * window_size]
            frame = dataset.iloc[indices]
            if shifted:
                frame = shared["inject_feature_missingness"](frame, rate, seed + index, shared["numeric_cols"])
            start = perf_counter()
            value = score(frame)
            adaptive = float(np.mean(history) + 2 * np.std(history))
            elapsed = perf_counter() - start
            rows.append({
                "Seed": seed, "Window": index, "Controlled stress": shifted,
                "Distance score": value, "Fixed threshold": fixed, "Adaptive threshold": adaptive,
                "Fixed alarm": value > fixed, "Adaptive alarm": value > adaptive,
                "Scoring and threshold seconds": elapsed,
            })
            history.append(value)
        np.savez_compressed(output / f"alarm_replay_split_{seed}.npz",
                            fit=fit_rows, calibration=calibration_rows, test=test_rows[:windows * window_size])

    windows = pd.DataFrame(rows)
    windows.to_csv(output / "alarm_replay_windows.csv", index=False)
    summary = []
    for seed, group in windows.groupby("Seed"):
        expected = group["Controlled stress"]
        onset = int(group.loc[expected, "Window"].iloc[0])
        for name in ("Fixed", "Adaptive"):
            alarm = group[f"{name} alarm"]
            hits = group.loc[expected & alarm, "Window"]
            summary.append({
                "Seed": seed, "Threshold": name,
                "Clean windows": int((~expected).sum()), "Stressed windows": int(expected.sum()),
                "False alarms": int((alarm & ~expected).sum()), "Stressed alarms": len(hits),
                "Precision": float((alarm & expected).sum() / alarm.sum()) if alarm.any() else np.nan,
                "Detection delay windows": int(hits.iloc[0]) - onset if len(hits) else np.nan,
            })
    pd.DataFrame(summary).to_csv(output / "alarm_replay_summary.csv", index=False)
    (output / "alarm_replay_protocol.json").write_text(json.dumps({
        "scope": "Offline alarm-only evaluation with an imposed feature-missingness change",
        "dataset_order": "Seeded replay order of official test records after exact overlap exclusion",
        "seeds": shared["SEEDS"], "window_size": window_size, "fit_rows": fit_size,
        "calibration_windows": history_size, "missingness_rate_after_midpoint": rate,
        "threshold": "Calibration mean plus two population SDs. Adaptive uses the preceding eight scores.",
        "scoring_time": "Feature transformation, clustering distances, and threshold calculation",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
