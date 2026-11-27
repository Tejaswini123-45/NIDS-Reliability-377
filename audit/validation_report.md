# Validation procedure

The execution log, `prediction_validation.json`, and `verify_run.py` provide
distinct checks. A successful run records 36 unique model, task and condition
keys per seed. The standalone check opens each prediction file, verifies its
hash, checks test-row alignment and split disjointness, recalculates all six
metrics and compares all summary means, sample SDs and Student-t intervals.
The split IDs refer to the deterministic post-cleaning dataset row order.

The model runners fit preprocessing exclusively on their actual saved training
IDs and apply it to clean and stressed evaluation rows. The model inputs have
finite-value checks. Predictor eligibility is enforced by the dataset-specific
schema exclusions. The code keeps prototype smoke tests outside the benchmark
runners.

A complete paper run needs five seeds and an independent verification report
for each dataset. Read the package's `RUN_STATUS.json` before using its tables.
Only verified runs enter the package builder, which requires a separately
declared partial mode for single-seed evidence.

If an export fails after fitting, retain the saved predictions and split IDs.
The UNSW recovery entry point used in this revision independently validates
the saved class labels against the pinned raw population, then recomputes
metrics and sensitivity without refitting. Runtime and convergence diagnostics
require original execution records.

## Delivered verification checks

- CICIDS2017 passed checksum, prediction, split, confusion-matrix and seed-summary checks for 180 evaluations in full-data paper mode. Seeds 42, 52, 62, 72, 82. [Validation metadata](../results/cicids2017/prediction_validation.json)
- UNSW-NB15 metrics were recovered from the five fitted seeds' saved predictions after a plot export failure. Fit timing and convergence diagnostics from that execution were unavailable. [Recovery record](../results/unsw_nb15/recovery.json)
- UNSW-NB15 passed checksum, prediction, split, confusion-matrix and seed-summary checks for 180 evaluations in full-data paper mode. Seeds 42, 52, 62, 72, 82. [Validation metadata](../results/unsw_nb15/prediction_validation.json)
- A fresh full-data UNSW-NB15 run passed the same checks and reproduced the previously published metrics across all 180 evaluations. It supplied fit and prediction timings and logged all 20 logistic regression fits without convergence warnings. [Timing run validation](../paper_assets/experiment_metadata/unsw_nb15_timing_prediction_validation.json) · [Convergence audit](../paper_assets/experiment_metadata/unsw_nb15_lr_convergence.json)
