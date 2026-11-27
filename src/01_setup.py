# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 1 =================
# 1. Configuration — final reliability experiment (optimized + validated)
from pathlib import Path
import os, gc, json, warnings, random, sys, subprocess
import numpy as np
import pandas as pd


SEED = 42
SEEDS = [42, 52, 62, 72, 82]

# ---------------- RUN MODES ----------------
# QUICK_VALIDATION: finishes much faster and verifies the entire pipeline.
# PAPER_MODE: uses the full verified training partitions and all five seeds.
QUICK_VALIDATION = os.environ.get("NIDS_QUICK") == "1"
PAPER_MODE = not QUICK_VALIDATION
RUN_EXPERIMENTS = True

# Bounded validation size. This is NOT used in PAPER_MODE.
FAST_TRAIN_SAMPLES = 100_000
# In quick validation, preserve approximate class proportions rather than
# using one-example-per-class plus random fill. This setting is NEVER used
# in PAPER_MODE.
FAST_SAMPLING_STRATIFIED = True

# Exact test-time stress rates used by the reliability matrix.
MISSINGNESS_LEVELS = (0.0, 0.10, 0.20)

# Statistical reporting for the five-seed final experiment.
CI_LEVEL = 0.95

# Full-data paper protocol.
# Repeated seeds are expensive, so they run only in PAPER_MODE.
REPEATED_SEEDS = SEEDS

# Never use these as temporal substitutes.
FORBIDDEN_TIMESTAMP_FEATURES = {
    "flow duration", "total tcp flow time", "total fwd packets",
    "total bwd packets", "flow iat mean", "flow iat std",
    "flow iat max", "flow iat min"
}

# Threading is intentionally conservative for shared Colab runtimes.
N_JOBS = max(1, min(2, (os.cpu_count() or 2)))
from src.evidence import output_directory
OUTPUT_DIR = output_directory("cicids2017")

np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)

FINAL_MODE = PAPER_MODE
if QUICK_VALIDATION and PAPER_MODE:
    raise ValueError("Choose either QUICK_VALIDATION=True or PAPER_MODE=True, not both.")

print("Configuration ready.")
print("QUICK_VALIDATION:", QUICK_VALIDATION)
print("PAPER_MODE:", PAPER_MODE)
print("RUN_EXPERIMENTS:", RUN_EXPERIMENTS)
print("N_JOBS:", N_JOBS)


# ================= NOTEBOOK CELL 2 =================
# 2. Imports with a safe XGBoost fallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from xgboost import XGBClassifier

XGB_AVAILABLE = True

print("Core imports loaded. XGBoost available:", XGB_AVAILABLE)


# Pinned mirror and exact file content are required for revision runs.
from src.evidence import verify_dataset

def normalize_header_name(value):
    return str(value).replace("\ufeff", "").replace("\u00a0", " ").strip().casefold()

def find_exact_column(columns, target):
    return next((c for c in columns if normalize_header_name(c) == target.casefold()), None)

csv_candidates, selected_repo = verify_dataset("cicids2017", OUTPUT_DIR)

# ================= NOTEBOOK CELL 5 =================
# 5. Efficient schema discovery + schema harmonization

def normalized_columns(columns):
    return [normalize_header_name(c) for c in columns]

schema_infos = []
candidate_diagnostics = []

for path in csv_candidates:
    try:
        head = pd.read_csv(path, nrows=0, encoding="cp1252", engine="c", low_memory=False)
        label_candidate = find_exact_column(head.columns, "Label")
        timestamp_candidate = find_exact_column(head.columns, "Timestamp")

        if label_candidate is None:
            candidate_diagnostics.append({
                "file": str(path),
                "status": "Missing Label",
                "columns": list(head.columns)
            })
            continue

        schema_infos.append({
            "path": path,
            "label": label_candidate,
            "timestamp": timestamp_candidate,
            "columns": list(head.columns)
        })
    except Exception as exc:
        candidate_diagnostics.append({
            "file": str(path),
            "status": f"Header read error: {type(exc).__name__}: {exc}",
            "columns": []
        })

if not schema_infos:
    raise ValueError(
        "No CICIDS2017 CSV containing a usable Label column was loaded. "
        f"Diagnostics: {candidate_diagnostics[:10]}"
    )

print("Usable CSV files:", len(schema_infos))
print("Files with actual Timestamp:", sum(x["timestamp"] is not None for x in schema_infos))

# We do not require identical raw column names because CICIDS2017 mirrors may
# rename/remove identity fields. The model feature intersection is established
# after loading, while Label remains mandatory.


# ================= NOTEBOOK CELL 6 =================
# 5b. Dataset schema checks
for info in schema_infos:
    if normalize_header_name(info["label"]) != "label":
        raise ValueError(f"Unexpected Label column in {info['path']}.")

timestamp_candidates = [
    info["timestamp"] for info in schema_infos if info["timestamp"] is not None
]

if timestamp_candidates:
    print("At least one source file exposes an actual Timestamp column.")
else:
    print("No source file exposes a row-level Timestamp.")
    print("Temporal evaluation will therefore use verified CICIDS2017 capture-day/file order.")
    print("No duration/flow feature will be substituted as a timestamp.")

