# Metric integrity

The legacy CICIDS2017 final metric export has identical macro precision, macro
F1 and weighted F1 across all 36 rows. Its per-seed data are absent, so the old
mean, variation and confidence intervals cannot be verified from that export.
The fresh runner computes metrics from live estimates, checks them using per-class
counts, persists truth and predictions, and repeats the check from disk before
writing its prediction validation report. The historical conference figures,
result directories and draft reports are legacy material. Current evidence lives
under `verified_runs` and carries a run-specific prediction audit.

Fixed-label macro metrics average all dataset labels, with zero for undefined
terms. In the class-specific recall export, a missing recall indicates zero
test support. The seen-class macro recall diagnostic averages classes present in
both fitted training and test data. Compare unseen-class counts separately.

Binary FPR is false attack alarms divided by actual benign records. Multiclass
FPR is the unweighted mean of one-vs-rest class FPRs. For an empty negative
denominator a class contributes zero under the current rule. Binary and
multiclass metrics retain separate rows.

Uncertainty uses sample SD and a two-sided 95% Student-t interval. Single-seed
runs have blank uncertainty. Fixed-holdout intervals summarize variability
among seeded runs on the same holdout. They do not estimate uncertainty over
different capture days or benchmark populations.
