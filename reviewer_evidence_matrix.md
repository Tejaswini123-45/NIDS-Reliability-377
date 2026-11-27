# Reviewer evidence matrix

Reviewer 1 appears to refer to an adaptive architecture and additional
experiments that the authors report were absent from the original six-page
submission. Treat the added benchmark scope as revision evidence and reconcile
the submission history against the original PDF before responding.

| Reviewer | Comment | Required Action | Files Modified | Experiment/Evidence | Status | Limitations | Manuscript Section to Update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1-A | Adaptive architecture | Distinguish prototype and benchmark | audit/adaptive_live_validation.md, run_adaptive_alarm_audit.py | Code audit and controlled offline comparison of fixed and rolling alarm thresholds | Alarm component evaluated | Full adaptive classification and replay pipeline remain open | Scope, limitations |
| R1-B | CICIDS2017 class coverage | Publish counts and supported-class recall | src/02_data_and_temporal.py, src/evidence.py | Dataset and split audit, prediction artifacts | Check run status for seed coverage | Friday-only classes are outside the fitted label set | Experimental design, results |
| R1-C | 5.16% duplicate overlap | Verify and rescore a filtered test subset | src/unsw_02_data_and_temporal.py, src/evidence.py | Duplicate audit and sensitivity CSV | Verified for all five UNSW seeds and six model-task combinations | Filtering changes prevalence | Dataset, sensitivity |
| R1-D | Real-world traffic | Specify evidence and future design | audit/adaptive_live_validation.md | Shadow-mode evaluation outline | Open | No confirmed authorized live trace or deployment | Limitations |
| R1-E | Scope relative to submission | Reconcile original and revision | reviewer_evidence_matrix.md | Scope statement above | Author verification pending | Original submission content needs author confirmation | Introduction, response letter |
| R2-A | Empirical results | Export verified seed metrics | src/evidence.py, verify_run.py | Prediction and summary audit | Check run status | Partial evidence must retain its seed and cap labels | Results |
| R2-B | Contribution | Frame controlled reliability evaluation | RESULTS_EXPLANATION.md | Fixed baseline model set and stress protocol | Documented | No new evaluated detector | Introduction |
| R2-C | UNSW external evidence | Validate official split | src/unsw_02_data_and_temporal.py | Pinned input audit and separate benchmark results | Check run status | Partition chronology unverified | Dataset, results |
| R2-D | Literature | Verify relevant primary sources | AUTHOR_HANDOFF.md | References and methodological distinctions | Documented | Protocols and taxonomies differ | Related work |
| R2-E | Stress levels | Justify and measure masks | src/03_preprocessing_models.py, src/unsw_03_preprocessing.py | Eligible, selected and newly missing numeric counts | Implemented | Independent cell dropout is a controlled proxy | Methodology |
| R2-F | Computational costs | Measure wall-clock fit and predict | src/04_experiments.py, src/unsw_04_experiments.py | Table 8, environment metadata, and UNSW convergence audit | Full-data measurements for both benchmarks | Prepared batch prediction excludes flow construction and end-to-end latency | Computational costs |
| R2-G | Task differences | Preserve distinct metrics | src/evidence.py | Binary attack FPR and multiclass macro one-vs-rest FPR | Implemented and verified for UNSW initial run | Class universes differ | Results |
| R2-H | Chronological validation | Describe distinct partitions | RESULTS_EXPLANATION.md | Capture-day versus official partition | Documented | Only one fixed CICIDS2017 holdout | Experimental design |
| R2-I | Unresolved scope | Document limitations | AUTHOR_HANDOFF.md | Open concerns | Documented | Live deployment, adaptation and chronology remain open | Limitations |
