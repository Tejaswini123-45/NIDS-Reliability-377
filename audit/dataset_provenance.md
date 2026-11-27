# Benchmark provenance

The acquisition script and both loaders verify all file lengths and SHA-256
hashes against [the pinned manifest](../configs/dataset_manifest.json).
Each run exports its own manifest and dataset validation report.

- CICIDS2017 uses the five daily files at Hugging Face revision
  `97610b600aa10939fb13b8109c207a86bac94a21`. The files contain an attempted-attack
  taxonomy. The mirror card identifies the Engelen–Rimmer–Joosen corrected
  and relabelled release as its starting point and documents converting
  IP addresses to decimal and dropping some infinite-value rows.
  [Pinned dataset card](https://huggingface.co/datasets/kumaranimesh1901/CICIDS-2017/raw/97610b600aa10939fb13b8109c207a86bac94a21/README.md)
- UNSW-NB15 uses the partition files at Hugging Face revision
  `6f5f54594dfc80c84264aec4c7cc3d9b162f1e2f`. Names and counts agree
  with the official partition description. Mirror files lack row-level
  capture-time metadata. The publisher's file link redirected to an
  authentication page, so byte equivalence to the publisher download
  remains unverified. [Official UNSW-NB15 description](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Cleaning and feature eligibility

CICIDS2017 drops missing labels, uses the shared numeric schema and replaces
invalid or infinite numeric values with missing values. The label, attempted
category, timestamp, IP identifiers and raw endpoint ports are excluded.
Duplicates are removed within each capture file by matching normalized
predictor values and label. The file audit gives raw and retained counts,
headers, pre-intersection candidate counts, invalid and missing cells, raw
class counts and the number of duplicates removed. Day metadata remains
separate from the feature matrix. Explicitly inspect per-day audit files for
the actual executed feature and label counts.

UNSW-NB15 excludes row IDs, endpoints, ports and timestamp metadata where
present. Numeric missing values use the training median. Categorical features
are one-hot encoded from training levels, with unknown levels mapped to an
all-zero block. Numeric cells alone are eligible for simulated dropout.
Primary official holdout evaluation retains the published partition rows.
Sensitivity excludes official test rows with identical parsed feature and
label values in the official training file after dropping row IDs.
