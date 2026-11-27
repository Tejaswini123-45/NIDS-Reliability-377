# Author handoff · ICISAS 2026 paper 377

Read the delivered `RUN_STATUS.json` before quoting metrics. A missing
validation report indicates incomplete evidence. Single-seed evidence must
retain the split-specific training cap and seed label.

## Verified starting point

The pinned UNSW-NB15 partition has 175,341 training rows and 82,332 test rows.
The repeated raw-value audit found 4,247 test rows identical to a training
record after excluding ID and retaining features and both labels. This is
5.158% of the official test partition. Across the five full-data UNSW seeds,
removing those rows while keeping predictions fixed lowered mean macro F1
by 0.0017 to 0.0100 across the six model-task comparisons. Binary FPR
increased for Logistic Regression and XGBoost and decreased for Random
Forest. [Per-seed sensitivity table](paper_assets/tables/table7_unsw_duplicate_sensitivity.csv)

The CICIDS2017 mirror audit loaded 2,099,971 rows. After removing 313,362
within-file duplicates under this revision's 82-predictor and label
definition, the cleaned population contained 1,786,609 rows and 27 labels.
The capture-day split contains 1,416,350 training rows and 370,259 Friday
test rows before training caps. Benign is the only shared label. Friday's
four test-only attack labels are Botnet, Botnet - Attempted, DDoS and Portscan.
Together those labels account for 98,761 Friday rows.
Use [class coverage and recall](paper_assets/tables/table6_class_coverage_and_recall.csv)
with the capture-day results. The seen-class diagnostic on Friday measures
benign recall. The multiclass classifiers cannot emit an attack label outside
their fitted vocabulary.

## Place evidence where readers need it

- Methods and datasets. Include Tables 1 and 2 and the two holdout definitions.
- Results. Keep binary and multiclass scores separate. Prioritize the
  reference-versus-holdout figure, a missingness curve, the class-support
  table or class-recall figure, and the duplicate sensitivity comparison.
- Computational costs and limitations. Use Table 8 and specify that prediction
  times describe prepared offline batches. Tree model cost depends on tree
  count and depth. Logistic regression prediction scales with feature count
  and the number of fitted classes. Time the wider flow pipeline separately
  before making deployment claims.

## Scope and open concerns

Table 8 uses full-data measurements for both benchmarks. The UNSW rerun
reproduced the published metrics and recorded 20 Logistic Regression fits
without convergence warnings. The offline adaptive alarm detected the
controlled change in three of five seeded replays, usually after a delay.
[Computational costs](paper_assets/tables/table8_computational_costs.csv) · [Adaptive and live validation](audit/adaptive_live_validation.md)

The `src/anids/` prototype has a PyTorch MLP detector, K-means outlier-window
threshold variants, and an experimental CTGAN replay module. Its smoke
tests exercise components and do not provide comparative benchmark
evidence. Future work needs drift alarm precision and detection delay,
retraining and unseen-class performance, batch and end-to-end latency,
and comparisons with fixed classifiers. For live validation, run a
passive shadow evaluation on authorized traffic, confirm labels through
analyst triage and independent records, and measure alert burden and
processing latency by ordered traffic windows.

Keep the contribution scoped to controlled offline reliability evaluation.
Do not infer concept drift, chronological UNSW generalization, universal
model rankings, a validated adaptive model or production readiness. The
original submission is a separate artifact. Reconcile Reviewer 1's scope
description with the submitted version before drafting the response.

## Sources for the revision

- Cite Moustafa and Slay for UNSW-NB15 and review the publisher's additional
  citation requirements. [Dataset and citation list](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Engelen, Rimmer and Joosen (2021) document CICIDS2017 flow extraction
  and labelling corrections relevant to interpreting the selected mirror.
  [IEEE SPW paper](https://doi.org/10.1109/SPW53761.2021.00009)
- AnoShift (NeurIPS 2022) evaluates network traffic at multiple chronological
  distances on Kyoto-2006+. The present CICIDS2017 protocol uses one fixed
  capture-day boundary and a supervised attack taxonomy.
  [Primary paper](https://papers.nips.cc/paper_files/paper/2022/file/d3bcbcb2a7b0b4716bf24ce4b2ea8d60-Paper-Datasets_and_Benchmarks.pdf)
- The 2025 NetFlow temporal analysis examines distributions using temporal
  features. It offers context for assessing temporal dynamics with explicit
  timing fields. [Preprint](https://arxiv.org/abs/2503.04404)
- MAWIFlow (2026) constructs future-year holdouts with longitudinal flow
  data and offers a comparison for longer time horizons.
  [Primary paper](https://www.scitepress.org/Papers/2026/144639/144639.pdf)

## Verified runs in this package


### CICIDS2017

Full-data paper mode with seed(s) 42, 52, 62, 72, 82. [Prediction-audited metrics](results/cicids2017/verified_seed_metrics.csv)

| Task | Model | Reference macro F1 | Holdout macro F1 |
| --- | --- | ---: | ---: |
| binary | Logistic Regression | 0.9823 | 0.9852 |
| binary | Random Forest | 0.9993 | 0.9919 |
| binary | XGBoost | 0.9983 | 0.9896 |
| multiclass | Logistic Regression | 0.4559 | 0.0367 |
| multiclass | Random Forest | 0.8632 | 0.0328 |
| multiclass | XGBoost | 0.8128 | 0.0369 |

### UNSW-NB15

Full-data paper mode with seed(s) 42, 52, 62, 72, 82. [Prediction-audited metrics](results/unsw_nb15/verified_seed_metrics.csv)

| Task | Model | Reference macro F1 | Holdout macro F1 |
| --- | --- | ---: | ---: |
| binary | Logistic Regression | 0.8897 | 0.7935 |
| binary | Random Forest | 0.9356 | 0.9026 |
| binary | XGBoost | 0.9302 | 0.8509 |
| multiclass | Logistic Regression | 0.3769 | 0.3325 |
| multiclass | Random Forest | 0.5514 | 0.4586 |
| multiclass | XGBoost | 0.5369 | 0.4644 |
