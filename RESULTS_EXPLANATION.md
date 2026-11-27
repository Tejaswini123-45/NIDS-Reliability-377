# Reading the evidence

Start with `RUN_STATUS.json` and the per-dataset `prediction_validation.json`.
The independent metrics and summary tables have different rows for each task.
Single-seed, capped-training evidence can establish pipeline execution and
offer preliminary patterns. It cannot support five-seed full-data estimates.

For CICIDS2017, compare `coverage_42_Temporal.csv` with the class recall files.
Classes present only on Friday are outside the training label vocabulary.
Classes missing on Friday have undefined test recall, even when they were
learned. The fixed-universe macro scores give zero to those undefined terms.
The seen-class macro recall diagnostic is conditional on classes with
support in both splits, and it can be dominated by benign traffic if benign
is the only shared class.

For UNSW-NB15, use the label “Official Train/Test Holdout”. The internal key
`Temporal` is retained in machine-readable results for compatibility. The
partition comparison reflects many possible differences in the underlying
evaluation populations. Duplicate sensitivity also changes the test class mix.

Ten and twenty percent missingness represent moderate and stronger controlled
numeric feature dropout. A seeded Bernoulli mask is shared by models of a
given split and seed. Thresholding the same random stream at both rates nests
the 10% mask within the 20% mask. Inference timing measures batch estimator
prediction on an already transformed matrix. Model-fit timing includes
model-specific scaling and excludes shared split-level preprocessing.
This is a compute comparison for the measured benchmark and hardware.
Peak resident memory covers the entire process and is shared across model
rows. The revision runs shared the host, so timing is subject to resource
contention. Repeat on an idle host for tighter timing comparisons.
Live deployment adds flow construction, queueing, feature extraction, alert
routing and traffic-dependent latency. Historical benchmark data provide no
live-traffic deployment evidence.
