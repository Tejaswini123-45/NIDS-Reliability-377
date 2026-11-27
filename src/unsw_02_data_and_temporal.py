# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Data Loading & Temporal Provenance (02)
# ============================================================================

import time
from pathlib import Path
import numpy as np
import pandas as pd

def normalize_column_name(name):
    return (
        str(name)
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .strip()
        .replace(" ", "_")
        .lower()
    )

from src.evidence import verify_dataset

paths, selected_dataset_source = verify_dataset("unsw_nb15", OUTPUT_DIR)
train_file = next(path for path in paths if "training" in path.name)
test_file = next(path for path in paths if "testing" in path.name)

print("\nReading UNSW-NB15 partitioned CSV files...")
df_train_raw = pd.read_csv(train_file, low_memory=False)
df_test_raw = pd.read_csv(test_file, low_memory=False)

print(f"  Train file: {train_file.name} ({len(df_train_raw):,} rows, {len(df_train_raw.columns)} cols)")
print(f"  Test file:  {test_file.name}  ({len(df_test_raw):,} rows, {len(df_test_raw.columns)} cols)")

# Normalize column names across both partitions
df_train_raw.columns = [normalize_column_name(c) for c in df_train_raw.columns]
df_test_raw.columns = [normalize_column_name(c) for c in df_test_raw.columns]

# Ensure column sets match
if set(df_train_raw.columns) != set(df_test_raw.columns):
    diff1 = set(df_train_raw.columns) - set(df_test_raw.columns)
    diff2 = set(df_test_raw.columns) - set(df_train_raw.columns)
    raise ValueError(f"Schema mismatch between train and test partitions: train-only={diff1}, test-only={diff2}")

common_cols = [c for c in df_train_raw.columns if c in df_test_raw.columns]

# Add partition tracking flags (0 = train session file, 1 = test session file)
df_train_raw["_session_partition"] = 0
df_test_raw["_session_partition"] = 1

# Combine into master audit frame
df_master = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
n_master = len(df_master)
print(f"Total unified records: {n_master:,}")

# ============================================================================
# AUDIT: TIMESTAMPS & PROVENANCE POLICY (ISSUE 3 RESOLUTION)
# ============================================================================
# These partition files lack verified chronological ordering.
TEMPORAL_PROVENANCE_MODE = "official_train_test_holdout"
temporal_order_desc = "Official Train/Test Holdout (internal condition key: Temporal)"
print(temporal_order_desc)

# ============================================================================
# AUDIT: DUPLICATION ANALYSIS & CROSS-PARTITION OVERLAP (ISSUE 2 RESOLUTION)
# ============================================================================
# Identify feature-only columns (excluding 'id', '_session_partition', labels)
feature_cols_only = [
    c for c in df_train_raw.columns 
    if c not in {"id", "_session_partition", "label", "attack_cat"}
]
all_cols_excl_id = [
    c for c in df_train_raw.columns 
    if c not in {"id", "_session_partition"}
]

# 1. Intra-partition duplicate counts
dupes_with_id = int(df_master.duplicated().sum())
dupes_without_id = int(df_master.duplicated(subset=all_cols_excl_id).sum())
train_dupes_without_id = int(df_train_raw.duplicated(subset=all_cols_excl_id).sum())
test_dupes_without_id = int(df_test_raw.duplicated(subset=all_cols_excl_id).sum())

# 2. Inter-partition (cross-partition) duplicate overlap quantification
train_exact_tuples = set(df_train_raw[all_cols_excl_id].itertuples(index=False, name=None))
test_exact_tuples = list(df_test_raw[all_cols_excl_id].itertuples(index=False, name=None))
exact_duplicate_test = np.array([row in train_exact_tuples for row in test_exact_tuples])
exact_cross_overlap_count = int(exact_duplicate_test.sum())
exact_cross_overlap_pct = 100 * exact_duplicate_test.mean()

from collections import defaultdict
train_labels_by_features = defaultdict(set)
for row, labels in zip(df_train_raw[feature_cols_only].itertuples(index=False, name=None),
                       df_train_raw[["label", "attack_cat"]].itertuples(index=False, name=None)):
    train_labels_by_features[row].add(labels)

test_feature_rows = list(df_test_raw[feature_cols_only].itertuples(index=False, name=None))
test_labels = list(df_test_raw[["label", "attack_cat"]].itertuples(index=False, name=None))
feature_overlap = np.array([row in train_labels_by_features for row in test_feature_rows])
conflicting = np.array([
    bool(train_labels_by_features.get(row, set()) - {label})
    for row, label in zip(test_feature_rows, test_labels)
])
feat_cross_overlap_count = int(feature_overlap.sum())
feat_cross_overlap_pct = 100 * feature_overlap.mean()
label_conflict_count = int(conflicting.sum())
label_conflict_pct = 100 * conflicting.mean()
duplicate_test_ids = np.flatnonzero(exact_duplicate_test) + len(df_train_raw)
pd.DataFrame({
    "Class": df_test_raw["attack_cat"], "Test rows": 1, "Exact overlap": exact_duplicate_test,
    "Feature overlap": feature_overlap, "Conflicting labels": conflicting,
}).groupby("Class").sum().to_csv(OUTPUT_DIR / "duplicate_overlap_by_class.csv")
del train_exact_tuples, test_exact_tuples, train_labels_by_features, test_feature_rows, test_labels

print("\n" + "=" * 80)
print("[DUPLICATION AUDIT & CROSS-PARTITION OVERLAP QUANTIFICATION]")
print("=" * 80)
print(f"  Total records in unified master dataset:                 {n_master:,}")
print(f"  Exact duplicates across all columns (inc. 'id'):          {dupes_with_id:,}")
print(f"  Duplicates in unified dataset (features+labels):          {dupes_without_id:,} ({dupes_without_id / n_master:.2%})")
print(f"  Duplicates within train partition (excl. 'id'):           {train_dupes_without_id:,} ({train_dupes_without_id / len(df_train_raw):.2%})")
print(f"  Duplicates within test partition (excl. 'id'):            {test_dupes_without_id:,} ({test_dupes_without_id / len(df_test_raw):.2%})")
print(f"  CROSS-PARTITION EXACT OVERLAP (identical feats + labels): {exact_cross_overlap_count:,} ({exact_cross_overlap_pct:.2f}% of test set)")
print(f"  CROSS-PARTITION FEATURE COLLISION (identical feats only): {feat_cross_overlap_count:,} ({feat_cross_overlap_pct:.2f}% of test set)")
print(f"  FEATURE COLLISION WITH CONFLICTING LABELS:                {label_conflict_count:,} ({label_conflict_pct:.2f}% of test set)")

# 3. Export duplicate overlap audit artifact
duplicate_overlap_audit = {
    "total_unified_rows": int(n_master),
    "train_rows": int(len(df_train_raw)),
    "test_rows": int(len(df_test_raw)),
    "within_train_duplicates_no_id": int(train_dupes_without_id),
    "within_train_duplicate_pct": float(train_dupes_without_id / len(df_train_raw)),
    "within_test_duplicates_no_id": int(test_dupes_without_id),
    "within_test_duplicate_pct": float(test_dupes_without_id / len(df_test_raw)),
    "total_duplicates_no_id": int(dupes_without_id),
    "total_duplicate_pct": float(dupes_without_id / n_master),
    "cross_partition_exact_overlap_count": int(exact_cross_overlap_count),
    "cross_partition_exact_overlap_pct": float(exact_cross_overlap_pct),
    "cross_partition_feature_collision_count": int(feat_cross_overlap_count),
    "cross_partition_feature_collision_pct": float(feat_cross_overlap_pct),
    "cross_partition_label_conflict_count": int(label_conflict_count),
    "cross_partition_label_conflict_pct": float(label_conflict_pct),
    "handling_policy": "Core official results retain all supplied rows. Sensitivity removes test rows with matching parsed features and labels in the official training file, ignoring row ID.",
    "within_train_duplicates_with_id": int(df_train_raw.duplicated().sum()),
    "within_test_duplicates_with_id": int(df_test_raw.duplicated().sum()),
}

if "OUTPUT_DIR" in globals():
    dupe_audit_path = OUTPUT_DIR / "duplicate_overlap_audit.json"
    with open(dupe_audit_path, "w", encoding="utf-8") as f_dupe:
        json.dump(duplicate_overlap_audit, f_dupe, indent=2)
    print(f"  [SAVED] Duplicate overlap audit saved to: {dupe_audit_path}")

# ============================================================================
# LABEL STANDARDIZATION & AUDIT
# ============================================================================
# 1. Binary label
if "label" not in df_master.columns:
    raise ValueError("Mandatory binary label column 'label' not found in UNSW-NB15 schema.")
if not df_master["label"].isin([0, 1]).all():
    raise ValueError("Expected binary labels 0 and 1")
df_master["label"] = df_master["label"].astype(int)

# 2. Multiclass label standardization
if "attack_cat" not in df_master.columns:
    raise ValueError("Mandatory multiclass label column 'attack_cat' not found in UNSW-NB15 schema.")

# Clean whitespace, casing, and handle spelling variants
raw_cats = df_master["attack_cat"].fillna("").astype(str).str.strip()

# Standardize: empty/null strings where label==0 -> 'Normal'
normalized_cats = raw_cats.copy()
normalized_cats[normalized_cats == ""] = "Normal"
normalized_cats[normalized_cats.str.casefold() == "nan"] = "Normal"
normalized_cats[normalized_cats.str.casefold() == "backdoors"] = "Backdoor"
normalized_cats[normalized_cats.str.casefold() == "normal"] = "Normal"
normalized_cats[normalized_cats.str.casefold() == "generic"] = "Generic"
normalized_cats[normalized_cats.str.casefold() == "exploits"] = "Exploits"
normalized_cats[normalized_cats.str.casefold() == "fuzzers"] = "Fuzzers"
normalized_cats[normalized_cats.str.casefold() == "dos"] = "DoS"
normalized_cats[normalized_cats.str.casefold() == "reconnaissance"] = "Reconnaissance"
normalized_cats[normalized_cats.str.casefold() == "analysis"] = "Analysis"
normalized_cats[normalized_cats.str.casefold() == "backdoor"] = "Backdoor"
normalized_cats[normalized_cats.str.casefold() == "shellcode"] = "Shellcode"
normalized_cats[normalized_cats.str.casefold() == "worms"] = "Worms"

if not (normalized_cats.eq("Normal") == df_master["label"].eq(0)).all():
    raise ValueError("Binary and multiclass labels disagree")
df_master["attack_cat"] = normalized_cats

# Verify target classes
target_classes = sorted(df_master["attack_cat"].unique())
print("\n[CLASS AUDIT] Standardized target classes (10 total):")
for cls in target_classes:
    cnt = int(np.sum(df_master["attack_cat"] == cls))
    print(f"  {cls:20s}: {cnt:>8,} ({cnt / n_master:>6.2%})")

if len(target_classes) != 10:
    print(f"[WARN] Expected 10 classes, found {len(target_classes)}: {target_classes}")

# ============================================================================
# FEATURE PARTITIONING & JUSTIFICATIONS
# ============================================================================
excluded_set = {normalize_column_name(k) for k in EXCLUDED_FEATURES_RATIONALE.keys()}
excluded_present = [c for c in df_master.columns if c in excluded_set]

print("\n[FEATURE EXCLUSION AUDIT]")
for col in excluded_present:
    reason = EXCLUDED_FEATURES_RATIONALE.get(col, "Excluded by design.")
    print(f"  - '{col}': {reason}")

# Separate categorical and numeric feature candidates
feature_candidates = [
    c for c in df_master.columns
    if c not in excluded_set and c not in {"label", "attack_cat", "_session_partition"}
]

categorical_cols = [c for c in feature_candidates if c in CATEGORICAL_FEATURES]
numeric_cols = [c for c in feature_candidates if c not in CATEGORICAL_FEATURES]

print(f"\nRetained feature count:  {len(feature_candidates)}")
print(f"  Categorical features: {len(categorical_cols)} -> {categorical_cols}")
print(f"  Numeric features:     {len(numeric_cols)}")

# ============================================================================
# SPLIT CONSTRUCTION: REFERENCE VS. TEMPORAL
# ============================================================================
# Reference split (Stratified 80/20 on unified dataset)
all_indices = np.arange(n_master, dtype=np.int64)

# Seed 42 baseline reference split
ref_train_idx_base, ref_test_idx_base = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=df_master["attack_cat"]
)

# Temporal split (Discrete capture session partition holdout)
temp_train_idx_base = np.flatnonzero(df_master["_session_partition"] == 0).astype(np.int64)
temp_test_idx_base = np.flatnonzero(df_master["_session_partition"] == 1).astype(np.int64)

print("\n[SPLIT DIAGNOSTICS & COVERAGE]")
print(f"  Reference Split (Seed 42): Train={len(ref_train_idx_base):,} | Test={len(ref_test_idx_base):,}")
print(f"  Temporal Split:             Train={len(temp_train_idx_base):,} | Test={len(temp_test_idx_base):,}")

# Audit class distribution in both splits
split_audit_rows = []
for cls in target_classes:
    ref_tr_cnt = int(np.sum(df_master["attack_cat"].iloc[ref_train_idx_base] == cls))
    ref_te_cnt = int(np.sum(df_master["attack_cat"].iloc[ref_test_idx_base] == cls))
    temp_tr_cnt = int(np.sum(df_master["attack_cat"].iloc[temp_train_idx_base] == cls))
    temp_te_cnt = int(np.sum(df_master["attack_cat"].iloc[temp_test_idx_base] == cls))
    
    split_audit_rows.append({
        "Class": cls,
        "Ref_Train": ref_tr_cnt,
        "Ref_Test": ref_te_cnt,
        "Temp_Train": temp_tr_cnt,
        "Temp_Test": temp_te_cnt,
        "Temp_Covered": "YES" if (temp_tr_cnt > 0 and temp_te_cnt > 0) else "NO",
    })

df_split_audit = pd.DataFrame(split_audit_rows)
print("\nPer-class train/test distribution across splits:")
print(f"  {'Class':18s} {'Ref_Train':>10} {'Ref_Test':>10} {'Temp_Train':>11} {'Temp_Test':>10} {'Temp_Status':>12}")
print("-" * 78)
for _, r in df_split_audit.iterrows():
    print(f"  {r['Class']:18s} {r['Ref_Train']:>10,} {r['Ref_Test']:>10,} {r['Temp_Train']:>11,} {r['Temp_Test']:>10,} {r['Temp_Covered']:>12}")

# Scientific non-equivalence check:
absent_temp_classes = df_split_audit[df_split_audit["Temp_Covered"] == "NO"]["Class"].tolist()
print(f"Classes absent from either official partition: {absent_temp_classes}")

print("\nData loading & temporal provenance preparation complete.\n")
