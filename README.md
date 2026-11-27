# NIDS Reliability · ICISAS 2026 paper 377

This project evaluates Logistic Regression, Random Forest and XGBoost on
CICIDS2017 and UNSW-NB15. It reports binary and multiclass results separately.
Historical exports in `results/` and `conference_paper_assets/` are legacy
material. The current results in `results/` are the `verified_*` exports and their prediction validation reports.

## Set up and acquire the data

Use Python 3.12 and install `requirements.txt`. The full CICIDS2017 run can take
substantial RAM and CPU time. Use a dedicated host with ample free memory and
disk for roughly 860 MB of source CSVs, prediction artifacts and the package.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe acquire_benchmarks.py
```

See [dataset provenance](audit/dataset_provenance.md) and the
[pinned revisions and checksums](configs/dataset_manifest.json). Acquisition
places data in `data/cicids2017` and `data/unsw_nb15`. The loaders accept only
the pinned file contents. Linux/macOS use `.venv/bin/python` and shell-appropriate
environment syntax.

## Execute and verify

From the repository root, set a new output root for each execution. Capture
stdout and stderr outside the run directories if you need execution logs.
An existing dataset output directory prevents an accidental overwrite.

```powershell
$env:NIDS_OUTPUT_ROOT = 'verified_runs/paper'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
.venv/Scripts/python.exe -u run_experiments.py
.venv/Scripts/python.exe -u run_unsw_experiments.py
.venv/Scripts/python.exe verify_run.py verified_runs/paper/cicids2017
.venv/Scripts/python.exe verify_run.py verified_runs/paper/unsw_nb15
.venv/Scripts/python.exe run_cross_analysis.py
.venv/Scripts/python.exe build_evidence.py verified_runs/paper NIDS-Reliability-Final
```

For a first-pass check, set `NIDS_QUICK=1` and select a fresh output root such
as `verified_runs/initial`. This runs seed 42 with training caps of 100,000
CICIDS2017 rows or 50,000 UNSW-NB15 rows per split. All evaluation rows in
each partition remain eligible. Remove the environment setting before a paper
run. Use `build_evidence.py --partial verified_runs/initial NIDS-Reliability-Final-Partial`
for a distinctly labelled package.
Use `run_cross_analysis.py --partial` for a preliminary comparison of
verified shared seeds, with `--fallback` if the datasets live in two run roots.
The resulting `cross_dataset_partial` directory records the two input modes
and stays separate from the full-data cross-dataset analysis.

## Protocol and outputs

The default uses seeds 42, 52, 62, 72 and 82 with full training partitions.
The random reference is a stratified 80/20 split. CICIDS2017 holds out Friday
after training on Monday–Thursday. UNSW-NB15 uses the official train/test
partition. It has unverified chronological ordering. The internal result key
`Temporal` maps to these two dataset-specific holdouts.

Both partitions are evaluated clean and at 10% and 20% controlled
numeric-feature dropout. Fit operations use training rows only. The saved
mask counts distinguish selected cells from newly missing cells. Binary attack
FPR and multiclass macro one-vs-rest FPR use different definitions.

Each completed run writes dataset provenance, per-file validation,
reproducible split indices, confusion matrices, class recalls, prediction
artifacts and their checksums, and `verified_seed_metrics.csv`,
`verified_summary.csv`, `paired_degradation.csv` and
`prediction_validation.json`. Paper mode requires five completed seeds.
The standalone verifier independently checks metrics and uncertainty.
See [results interpretation](RESULTS_EXPLANATION.md) and the
[author handoff](AUTHOR_HANDOFF.md) for scope and limitations.

## Revision evidence snapshot

[Run status](RUN_STATUS.json) identifies the verified seeds and run modes
behind the checked-in tables and figures in `paper_assets/`. The full
prediction archives and split IDs are generated locally by the commands
above. Both dataset snapshots contain 180 verified full-data evaluations
across five seeds. The [cross-dataset comparison](paper_assets/cross_dataset/comparison_metadata.json)
is descriptive. A fresh UNSW-NB15 run reproduced the published evaluation
metrics and supplied full-data timing and convergence records. The
[adaptive alarm audit](audit/adaptive_live_validation.md) is a separate
offline experiment on benchmark flows.
