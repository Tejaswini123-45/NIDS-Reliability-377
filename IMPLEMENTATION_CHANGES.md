# Implementation changes for paper 377

- Preserved the ordered extracted pipeline and the LR, RF and XGBoost model
  factories. The original notebook remains the baseline reference for the
  extracted code.
- Pinned dataset files by revision, byte length and SHA-256. The revision
  uses the specified CICIDS2017 mirror and no fallback repository.
- Excluded outcome metadata and raw endpoint identifiers from CICIDS2017.
  Preserved the explicit Monday–Thursday versus Friday protocol and recorded
  class coverage per split and seed.
- Scoped UNSW-NB15 to its official train/test partition and separated value
  overlap, feature conflicts and row identities. Duplicate sensitivity rescores
  the original predictions on retained test rows without retraining.
- Added prediction artifacts, independent metric checks, timing, actual mask
  counts, paired changes, seed summaries and a package builder. Binary FPR
  now uses the attack-positive class. Old figures and CSVs use historical
  definitions and are excluded from the current evidence pipeline.
