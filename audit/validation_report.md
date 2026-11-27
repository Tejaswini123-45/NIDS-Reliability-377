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
