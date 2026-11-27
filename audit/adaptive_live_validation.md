# Adaptive prototype and live-traffic validation

The revision's benchmark results apply to the three static classifiers.
The local A-NIDS prototype contains a neural classifier, a window-based
alarm, and a replay component. Its smoke tests exercise component execution
and use synthetic inputs for the alarm tests.

## Findings from code inspection

- The prototype loader discovers categorical levels using both official
  UNSW partitions before splitting. Its subsequent median calculation does
  not undo that use of held-out inputs.
- The adaptive alarm includes the current window's score in the threshold
  used to judge that window. The window is also used to fit the scaler and
  clustering model. These choices need an explicit deployment interpretation
  and a past-only comparator before evaluating detection delay. With the
  five-score default and population standard deviation, the largest possible
  standardized score within its own sample is two. The strict mean-plus-two-SD
  rule therefore cannot trigger after warmup in exact arithmetic.
- The pinned official UNSW partitions do not establish traffic chronology.
  CICIDS2017 capture days establish a day order but leave attack class and
  capture day confounded in the Friday holdout.

## Controlled alarm replay

`run_adaptive_alarm_audit.py` evaluates a past-only distance alarm on held-out
UNSW-NB15 features after exact training overlap exclusions. It uses the same
training-fitted feature transformations and centroids for its fixed and
rolling threshold variants. A separate historical calibration sample sets
both initial thresholds. A seeded replay creates clean windows followed by
windows with prescribed numeric feature missingness. Precision and delay
refer to that deliberately imposed change. The replay covers the alarm
component. Classifier updates and CTGAN replay require a separate evaluation.

Across five seeds, the rolling threshold fired on 4 of 40 stressed windows
and the fixed threshold fired on none. Neither fired on the 35 clean
windows. The rolling alarm detected the imposed change in three seeds,
with first alarms 1, 5, and 6 windows after onset. The other two seeds had
no detection. [Per-seed replay summary](adaptive_alarm_replay/alarm_replay_summary.csv)

## Evidence needed to close these points

For offline validation, define the label-availability schedule and predeclare
the change windows. Fit feature transformations on historical data only.
Compare a frozen classifier with the same classifier under the adaptive
policy. Evaluate each window before incorporating its labels or refitting.
Report alarm precision, false alarms per window, detection delay, post-update
performance, unseen-class coverage and runtime. Keep replay-based results
separate from the three static benchmarks.

For live validation, obtain an authorized ordered flow stream or capture,
independent adjudicated labels, and capture and alert timestamps. Run the
detector passively in shadow mode. Record label arrival and operational
latency separately from model inference. Keep the evaluation trace and
its provenance under the applicable access and retention rules.

No authorized live trace or deployment environment was supplied for this
revision. The adaptive and live evaluations remain open.
