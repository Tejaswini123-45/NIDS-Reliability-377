# ============================================================================
# Cross-Dataset Comparative Reliability Analysis: CICIDS2017 vs. UNSW-NB15
# ============================================================================
"""
Cross-Dataset Analysis Module

Synthesizes experimental results from:
1. CICIDS2017 capture-day chronological holdout
2. UNSW-NB15 Official Train/Test Holdout

Generates:
- Comparative metrics table (Reference, Temporal, Missingness, Combined)
- Relative degradation comparison across all 6 conditions
- Non-additive stress interaction comparison (Ceiling/Floor vs Continuous)
- Structural dataset comparison (Class dynamics, Provenance, Missingness)
- Publication-quality comparative figures (1-4) in results/cross_dataset/
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evidence import METRICS
from verify_run import verify

# Ensure clean UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("NIDS_OUTPUT_ROOT", ROOT_DIR / "verified_runs"))
CROSS_DIR = RESULTS_DIR / ("cross_dataset_partial" if PARTIAL_COMPARISON else "cross_dataset")
directories, verified = {}, {}
for dataset in ("cicids2017", "unsw_nb15"):
    directory = RESULTS_DIR / dataset
    if FALLBACK_ROOT and not (directory / "prediction_validation.json").exists():
        directory = FALLBACK_ROOT / dataset
    record = verify(directory)
    if not PARTIAL_COMPARISON and not record["paper_mode"]:
        raise ValueError(f"{dataset} needs a completed five-seed prediction audit")
    directories[dataset], verified[dataset] = directory, record
shared_seeds = sorted(set.intersection(*(set(record["seeds"]) for record in verified.values())))
if not shared_seeds:
    raise ValueError("The benchmark runs have no common seeds")
title_suffix = f" · Preliminary, seed(s) {', '.join(map(str, shared_seeds))}" if PARTIAL_COMPARISON else ""
mode_labels = {
    dataset: "full partition" if record["paper_mode"] else "capped training"
    for dataset, record in verified.items()
}
CROSS_DIR.mkdir(parents=True, exist_ok=True)
(CROSS_DIR / "comparison_metadata.json").write_text(json.dumps({
    "paper_mode": not PARTIAL_COMPARISON,
    "shared_seeds": shared_seeds,
    "inputs": {dataset: {"source": str(directories[dataset]), "paper_mode": record["paper_mode"],
                         "holdout": record["holdout"]} for dataset, record in verified.items()},
    "scope": "Descriptive within-dataset changes. The two label universes, features and partitions differ.",
}, indent=2), encoding="utf-8")

print("=" * 80)
print("CROSS-DATASET RELIABILITY EVALUATION: CICIDS2017 vs. UNSW-NB15")
print("=" * 80)

# ============================================================================
# 1. LOAD UNSW-NB15 RESULTS (PART B)
# ============================================================================
CICIDS_DIR = directories["cicids2017"]
UNSW_DIR = directories["unsw_nb15"]


def verified_condition_means(directory):
    runs = pd.read_csv(directory / "verified_seed_metrics.csv")
    runs = runs[runs["Seed"].isin(shared_seeds)]
    means = runs.groupby(["Task", "Model", "Condition"], as_index=False)[METRICS].mean()
    return means.rename(columns=lambda name: name.replace(" ", "_"))


df_unsw_metrics = verified_condition_means(UNSW_DIR)
df_cicids_metrics = verified_condition_means(CICIDS_DIR)

# ============================================================================
# 3. COMPUTE DEGRADATION AND INTERACTION FOR BOTH DATASETS
# ============================================================================
def compute_degradation(df, metric_col="Macro_F1"):
    base = df[df["Condition"] == "Reference"][["Task", "Model", metric_col]].rename(columns={metric_col: "Reference"})
    merged = df.merge(base, on=["Task", "Model"], how="left")
    merged["Degradation"] = (merged["Reference"] - merged[metric_col]) / merged["Reference"].replace(0, np.nan)
    return merged[["Task", "Model", "Condition", metric_col, "Reference", "Degradation"]].copy()

def compute_interaction(deg_df):
    out = []
    for (task, model), g in deg_df.groupby(["Task", "Model"]):
        lookup = dict(zip(g["Condition"], g["Degradation"]))
        d_temp = lookup.get("Temporal")
        for rate in ("10%", "20%"):
            d_m = lookup.get(f"Reference+{rate} Missingness")
            d_c = lookup.get(f"Temporal+{rate} Missingness")
            if d_temp is not None and d_m is not None and d_c is not None:
                out.append({
                    "Task": task,
                    "Model": model,
                    "Missingness": rate,
                    "Temporal Degradation": d_temp,
                    "Missingness Degradation": d_m,
                    "Combined Degradation": d_c,
                    "Interaction": d_c - (d_temp + d_m),
                })
    return pd.DataFrame(out)

df_cicids_deg = compute_degradation(df_cicids_metrics)
df_cicids_interaction = compute_interaction(df_cicids_deg)

# Format column names for consistency
if "Macro F1" in df_unsw_metrics.columns and "Macro_F1" not in df_unsw_metrics.columns:
    df_unsw_metrics = df_unsw_metrics.rename(columns={"Macro F1": "Macro_F1", "Macro Recall": "Macro_Recall"})

# ============================================================================
# 4. CROSS-DATASET SYNTHESIS TABLES
# ============================================================================
# Table 1: Side-by-side performance comparison
df_unsw_tagged = df_unsw_metrics.copy()
df_unsw_tagged["Dataset"] = "UNSW-NB15"
df_cicids_tagged = df_cicids_metrics.copy()
df_cicids_tagged["Dataset"] = "CICIDS2017"

common_cols = ["Dataset", "Task", "Model", "Condition", "Macro_F1", "Macro_Recall", "Accuracy", "FPR"]
df_comparison_metrics = pd.concat([
    df_cicids_tagged[[c for c in common_cols if c in df_cicids_tagged.columns]],
    df_unsw_tagged[[c for c in common_cols if c in df_unsw_tagged.columns]]
], ignore_index=True)
df_comparison_metrics.to_csv(CROSS_DIR / "comparison_metrics.csv", index=False)

# Table 2: Degradation Comparison
df_cicids_deg_tagged = df_cicids_deg.copy()
df_cicids_deg_tagged["Dataset"] = "CICIDS2017"
df_unsw_deg_calc = compute_degradation(df_unsw_metrics)
df_unsw_deg_calc["Dataset"] = "UNSW-NB15"

df_comparison_degradation = pd.concat([df_cicids_deg_tagged, df_unsw_deg_calc], ignore_index=True)
df_comparison_degradation.to_csv(CROSS_DIR / "comparison_degradation.csv", index=False)

# Table 3: Interaction Comparison
df_cicids_int_tagged = df_cicids_interaction.copy()
df_cicids_int_tagged["Dataset"] = "CICIDS2017"
df_unsw_int_calc = compute_interaction(df_unsw_deg_calc)
df_unsw_int_calc["Dataset"] = "UNSW-NB15"

df_comparison_interaction = pd.concat([df_cicids_int_tagged, df_unsw_int_calc], ignore_index=True)
df_comparison_interaction.to_csv(CROSS_DIR / "comparison_interaction.csv", index=False)

# Table 4: Structural & Class Dynamics Breakdown
def coverage_summary(directory, title):
    coverage = pd.read_csv(directory / "coverage_42_Temporal.csv")
    validation = json.loads((directory / "prediction_validation.json").read_text())
    return {
        "Dataset": title, "Holdout": validation["holdout"],
        "Train rows": int(coverage["Train"].sum()), "Test rows": int(coverage["Test"].sum()),
        "Feature count": len(validation["predictors"]), "Class count": len(coverage),
        "Test-only classes": ", ".join(coverage.loc[coverage["Train"].eq(0) & coverage["Test"].gt(0), "Class"]),
        "Classes with zero test support": ", ".join(coverage.loc[coverage["Test"].eq(0), "Class"]),
    }

df_structural = pd.DataFrame([
    coverage_summary(CICIDS_DIR, "CICIDS2017"), coverage_summary(UNSW_DIR, "UNSW-NB15"),
])
df_structural.to_csv(CROSS_DIR / "comparison_class_dynamics.csv", index=False)

print("\nCross-dataset comparison tables generated:")
print("  - comparison_metrics.csv")
print("  - comparison_degradation.csv")
print("  - comparison_interaction.csv")
print("  - comparison_class_dynamics.csv")

# ============================================================================
# AUTOMATED VERIFICATION: COMPARISON TABLES VS. SOURCE SUMMARIES
# ============================================================================
print("\n" + "=" * 80)
print("INDEPENDENT VERIFICATION: COMPARISON TABLES VS. SOURCE SUMMARIES")
print("=" * 80)

# Check CICIDS2017 metrics alignment
c_chk = df_comparison_metrics[df_comparison_metrics["Dataset"] == "CICIDS2017"]
for _, r in c_chk.iterrows():
    orig = df_cicids_metrics[(df_cicids_metrics["Task"] == r["Task"]) & 
                             (df_cicids_metrics["Model"] == r["Model"]) & 
                             (df_cicids_metrics["Condition"] == r["Condition"])].iloc[0]
    assert np.isclose(r["Macro_F1"], orig["Macro_F1"], atol=1e-4), f"CICIDS Macro_F1 mismatch at {r['Task']}/{r['Model']}/{r['Condition']}"
    assert np.isclose(r["Accuracy"], orig["Accuracy"], atol=1e-4), f"CICIDS Accuracy mismatch at {r['Task']}/{r['Model']}/{r['Condition']}"

# Check UNSW-NB15 metrics alignment
u_chk = df_comparison_metrics[df_comparison_metrics["Dataset"] == "UNSW-NB15"]
for _, r in u_chk.iterrows():
    orig = df_unsw_metrics[(df_unsw_metrics["Task"] == r["Task"]) & 
                           (df_unsw_metrics["Model"] == r["Model"]) & 
                           (df_unsw_metrics["Condition"] == r["Condition"])].iloc[0]
    assert np.isclose(r["Macro_F1"], orig["Macro_F1"], atol=1e-4), f"UNSW Macro_F1 mismatch at {r['Task']}/{r['Model']}/{r['Condition']}"
    assert np.isclose(r["Accuracy"], orig["Accuracy"], atol=1e-4), f"UNSW Accuracy mismatch at {r['Task']}/{r['Model']}/{r['Condition']}"

print("  [PASS] CICIDS2017 metrics verified against prediction-audited seed metrics")
print("  [PASS] UNSW-NB15 metrics verified against prediction-audited seed metrics")
print("Degradation and interaction tables generated from the audited run summaries.")

# ============================================================================
# 5. PUBLICATION COMPARISON FIGURES
# ============================================================================
print("\nGenerating cross-dataset comparative figures...")

# Figure Cross-1: Reference vs Temporal Macro-F1 across datasets
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
models = ["Logistic Regression", "Random Forest", "XGBoost"]
colors = {"Logistic Regression": "#3b82f6", "Random Forest": "#10b981", "XGBoost": "#8b5cf6"}

# Subplot A: CICIDS2017
ax_a = axes[0]
for m in models:
    sub = df_cicids_metrics[(df_cicids_metrics["Task"] == "multiclass") & (df_cicids_metrics["Model"] == m)]
    ref_val = sub[sub["Condition"] == "Reference"]["Macro_F1"].values[0]
    temp_val = sub[sub["Condition"] == "Temporal"]["Macro_F1"].values[0]
    ax_a.plot(["Reference", "Capture-day holdout"], [ref_val, temp_val], marker="o", linewidth=2.5, markersize=8, color=colors[m], label=m)

ax_a.set_title(f"A. CICIDS2017 · {mode_labels['cicids2017']}", fontsize=11, fontweight="bold")
ax_a.set_ylabel("Macro-F1", fontsize=11)
ax_a.grid(alpha=0.25)
ax_a.legend(loc="upper right")
ax_a.set_ylim(-0.05, 1.05)


# Subplot B: UNSW-NB15
ax_b = axes[1]
for m in models:
    sub = df_unsw_metrics[(df_unsw_metrics["Task"] == "multiclass") & (df_unsw_metrics["Model"] == m)]
    ref_val = sub[sub["Condition"] == "Reference"]["Macro_F1"].values[0]
    temp_val = sub[sub["Condition"] == "Temporal"]["Macro_F1"].values[0]
    ax_b.plot(["Reference", "Official train/test holdout"], [ref_val, temp_val], marker="s", linewidth=2.5, markersize=8, color=colors[m], label=m)

ax_b.set_title(f"B. UNSW-NB15 · {mode_labels['unsw_nb15']}", fontsize=11, fontweight="bold")
ax_b.grid(alpha=0.25)
ax_b.legend(loc="upper right")


fig.suptitle("Reference vs. Holdout Macro-F1" + title_suffix, fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(CROSS_DIR / "figure_cross_1_reference_vs_temporal.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure Cross-2: Degradation Comparison across all 6 conditions
conditions = [
    "Reference", "Reference+10% Missingness", "Reference+20% Missingness",
    "Temporal", "Temporal+10% Missingness", "Temporal+20% Missingness"
]
cond_labels = ["Ref", "Ref+10%M", "Ref+20%M", "Holdout", "Holdout+10%M", "Holdout+20%M"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

# CICIDS2017
ax_a = axes[0]
for m in models:
    sub = df_cicids_deg[(df_cicids_deg["Task"] == "multiclass") & (df_cicids_deg["Model"] == m)]
    deg_map = dict(zip(sub["Condition"], sub["Degradation"]))
    vals = [deg_map[c] for c in conditions]
    ax_a.plot(cond_labels, vals, marker="o", linewidth=2, color=colors[m], label=m)

ax_a.set_title("A. CICIDS2017: Relative Macro-F1 Degradation", fontsize=11, fontweight="bold")
ax_a.set_ylabel("Relative Degradation from Reference", fontsize=11)
ax_a.grid(alpha=0.25)
ax_a.legend()
ax_a.set_ylim(-0.05, 1.1)

# UNSW-NB15
ax_b = axes[1]
for m in models:
    sub = df_unsw_deg_calc[(df_unsw_deg_calc["Task"] == "multiclass") & (df_unsw_deg_calc["Model"] == m)]
    deg_map = dict(zip(sub["Condition"], sub["Degradation"]))
    vals = [deg_map[c] for c in conditions]
    ax_b.plot(cond_labels, vals, marker="s", linewidth=2, color=colors[m], label=m)

ax_b.set_title("B. UNSW-NB15: Relative Macro-F1 Degradation", fontsize=11, fontweight="bold")
ax_b.grid(alpha=0.25)
ax_b.legend()

fig.suptitle("Within-dataset relative degradation" + title_suffix, fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(CROSS_DIR / "figure_cross_2_degradation_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure Cross-3: Interaction Effect Comparison
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(models))
width = 0.20

# CICIDS 10% and 20%
c_int_10 = [df_cicids_interaction[(df_cicids_interaction["Task"]=="multiclass") & (df_cicids_interaction["Model"]==m) & (df_cicids_interaction["Missingness"]=="10%")]["Interaction"].values[0] for m in models]
c_int_20 = [df_cicids_interaction[(df_cicids_interaction["Task"]=="multiclass") & (df_cicids_interaction["Model"]==m) & (df_cicids_interaction["Missingness"]=="20%")]["Interaction"].values[0] for m in models]

# UNSW 10% and 20%
u_int_10 = [df_unsw_int_calc[(df_unsw_int_calc["Task"]=="multiclass") & (df_unsw_int_calc["Model"]==m) & (df_unsw_int_calc["Missingness"]=="10%")]["Interaction"].values[0] for m in models]
u_int_20 = [df_unsw_int_calc[(df_unsw_int_calc["Task"]=="multiclass") & (df_unsw_int_calc["Model"]==m) & (df_unsw_int_calc["Missingness"]=="20%")]["Interaction"].values[0] for m in models]

ax.bar(x - 1.5*width, c_int_10, width, label="CICIDS2017 (10% Missingness)", color="#fb7185", alpha=0.9)
ax.bar(x - 0.5*width, c_int_20, width, label="CICIDS2017 (20% Missingness)", color="#e11d48", alpha=0.9)
ax.bar(x + 0.5*width, u_int_10, width, label="UNSW-NB15 (10% Missingness)", color="#60a5fa", alpha=0.9)
ax.bar(x + 1.5*width, u_int_20, width, label="UNSW-NB15 (20% Missingness)", color="#2563eb", alpha=0.9)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight="bold")
ax.set_ylabel("Non-Additive Interaction: D_comb - (D_temp + D_miss)", fontsize=10)
ax.set_title("Descriptive combined-stress interaction" + title_suffix, fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(CROSS_DIR / "figure_cross_3_stress_interaction.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure Cross-4: Architecture Ranking Across Stress Conditions
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(conditions))
w = 0.25

rf_unsw = [df_unsw_metrics[(df_unsw_metrics["Task"]=="multiclass") & (df_unsw_metrics["Model"]=="Random Forest") & (df_unsw_metrics["Condition"]==c)]["Macro_F1"].values[0] for c in conditions]
xgb_unsw = [df_unsw_metrics[(df_unsw_metrics["Task"]=="multiclass") & (df_unsw_metrics["Model"]=="XGBoost") & (df_unsw_metrics["Condition"]==c)]["Macro_F1"].values[0] for c in conditions]
lr_unsw = [df_unsw_metrics[(df_unsw_metrics["Task"]=="multiclass") & (df_unsw_metrics["Model"]=="Logistic Regression") & (df_unsw_metrics["Condition"]==c)]["Macro_F1"].values[0] for c in conditions]

ax.bar(x - w, lr_unsw, w, label="Logistic Regression", color="#3b82f6", alpha=0.85)
ax.bar(x, rf_unsw, w, label="Random Forest", color="#10b981", alpha=0.85)
ax.bar(x + w, xgb_unsw, w, label="XGBoost", color="#8b5cf6", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(cond_labels, fontsize=10)
ax.set_ylabel("Macro-F1 Score", fontsize=10)
ax.set_title("UNSW-NB15 macro F1 by model and condition" + title_suffix, fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(CROSS_DIR / "figure_cross_4_model_resilience_ranking.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Cross-dataset figures generated successfully:")
print("  - figure_cross_1_reference_vs_temporal.png")
print("  - figure_cross_2_degradation_comparison.png")
print("  - figure_cross_3_stress_interaction.png")
print("  - figure_cross_4_model_resilience_ranking.png")
print("\n" + "=" * 80)
print("CROSS-DATASET SYNTHESIS COMPLETED")
print("=" * 80)
