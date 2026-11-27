"""Build a publication evidence package from independently verified runs."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evidence import KEY
from verify_run import verify


DATASETS = {"cicids2017": "CICIDS2017", "unsw_nb15": "UNSW-NB15"}
TASKS = ("binary", "multiclass")


def figures(runs, directory, title, holdout, seed_count):
    for task in TASKS:
        data = runs[runs.Task.eq(task)].copy()
        data["Partition"] = data.Condition.str.split("+", regex=False).str[0]
        data.Partition = data.Partition.replace({"Temporal": holdout})
        for metric in ["Macro F1"] if task == "multiclass" else ["Macro F1", "Attack recall", "FPR"]:
            if metric not in data:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
            for ax, (partition, subset) in zip(axes, data.groupby("Partition", sort=False)):
                for model, group in subset.groupby("Model"):
                    means = group.groupby("Missingness")[metric].mean()
                    ax.plot(means.index * 100, means, marker="o", label=model)
                ax.set(title=partition, xlabel="Missing numeric feature values (%)", xticks=[0, 10, 20])
                ax.grid(alpha=0.25)
            axes[0].set_ylabel(metric)
            axes[1].legend(fontsize=8)
            fig.suptitle(f"{title} · {task} · {seed_count} seed(s)")
            fig.tight_layout()
            name = f"{task}_{metric.lower().replace(' ', '_')}"
            fig.savefig(directory / f"{name}.png", dpi=300)
            fig.savefig(directory / f"{name}.pdf")
            plt.close(fig)


def diagnostics(runs, source, directory, title, holdout, seed_count):
    clean = runs[runs.Missingness.eq(0)]
    multiclass = clean[clean.Task.eq("multiclass")]
    labels = {"Reference": "Reference", "Temporal": holdout}
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, group in multiclass.groupby("Model"):
        stats = group.groupby("Condition")["Macro F1"].mean()
        ax.plot([labels[key] for key in stats.index], stats.values, marker="o", label=model)
    ax.set(title=f"{title} · multiclass · {seed_count} seed(s)", ylabel="Macro F1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "reference_vs_holdout.png", dpi=300)
    plt.close(fig)

    seed42 = multiclass[multiclass.Seed.eq(42) & multiclass.Model.eq("Random Forest")]
    recalls = []
    for _, row in seed42.iterrows():
        name = Path(row["Prediction file"])
        matrix = pd.read_csv(source / name.with_name(name.stem + "_confusion.csv"), index_col=0)
        recall = pd.read_csv(source / name.with_name(name.stem + "_recall.csv")).set_index("Class")
        recalls.append(recall["Recall"].rename(labels[row.Condition]))
        rates = matrix.div(matrix.sum(axis=1).replace(0, float("nan")), axis=0)
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        image = ax.imshow(rates, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=list(range(len(matrix))), yticks=list(range(len(matrix))),
               xlabel="Predicted class", ylabel="True class",
               title=f"{title} · Random Forest · {labels[row.Condition]} · seed 42")
        ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=7)
        ax.set_yticklabels(matrix.index, fontsize=7)
        fig.colorbar(image, ax=ax, label="Fraction of true class")
        fig.tight_layout()
        fig.savefig(directory / f"confusion_{row.Condition.lower()}_seed42.png", dpi=300)
        plt.close(fig)
    class_recall = pd.concat(recalls, axis=1).drop(index=["BENIGN", "Benign", "Normal"], errors="ignore")
    ax = class_recall.plot.barh(figsize=(8, max(4, 0.28 * len(class_recall))), width=0.75)
    ax.set(xlabel="Recall", xlim=(0, 1), ylabel="Attack category",
           title=f"{title} · Random Forest · seed 42")
    ax.figure.text(0.99, 0.01, "Refer to the class-support table for zero-support categories.",
                   ha="right", fontsize=8)
    ax.figure.tight_layout(rect=(0, 0.035, 1, 1))
    ax.figure.savefig(directory / "attack_recall_seed42.png", dpi=300)
    plt.close(ax.figure)

    pivot = runs[runs.Task.eq("multiclass")].pivot(index=["Seed", "Model"], columns="Condition",
                                                   values="Macro F1")
    fig, axis = plt.subplots(figsize=(7, 4))
    effects = pd.DataFrame({
        rate: pivot["Temporal"] + pivot[f"Reference+{rate}% Missingness"]
        - pivot[f"Temporal+{rate}% Missingness"] - pivot["Reference"]
        for rate in [10, 20]
    }).groupby("Model").mean()
    for model, values in effects.iterrows():
        axis.plot(values.index, values.to_numpy(), marker="o", label=model)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(title=f"{title} · {seed_count} seed(s)", xlabel="Missing numeric feature values (%)",
             ylabel="Absolute macro F1 interaction", xticks=[10, 20])
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "descriptive_interaction.png", dpi=300)
    plt.close(fig)


def uncertainty(summary, directory, title, holdout):
    report = summary.reset_index()
    report = report[report.Task.eq("multiclass") & report.Condition.isin(["Reference", "Temporal"])]
    fig, axis = plt.subplots(figsize=(7, 4))
    for model, group in report.groupby("Model"):
        group = group.sort_values("Condition")
        unit = group["Mean Macro F1"] - group["CI95 lower Macro F1"]
        axis.errorbar(group.Condition.replace({"Temporal": holdout}), group["Mean Macro F1"],
                      yerr=unit.clip(lower=0), capsize=4, marker="o", label=model)
    axis.set(title=f"{title} · 95% Student-t intervals", ylabel="Mean macro F1")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "repeated_seed_uncertainty.png", dpi=300)
    plt.close(fig)


def write_tables(package, inputs):
    tables = package / "paper_assets/tables"
    tables.mkdir(parents=True)
    all_coverage, all_stats, all_config, all_drops, all_costs = [], [], [], [], []
    for dataset, directory, metadata in inputs:
        title = DATASETS[dataset]
        run_copy = package / "results" / dataset
        shutil.copytree(directory, run_copy, ignore=shutil.ignore_patterns("figure*.png"))
        runs = pd.read_csv(directory / "verified_seed_metrics.csv")
        paths = runs.loc[runs.Task.eq("binary"), "Prediction file"]
        runs.loc[paths.index, "Attack recall"] = [
            pd.read_csv(directory / Path(name).with_name(Path(name).stem + "_recall.csv"))
            .set_index("Class").at[1, "Recall"] for name in paths
        ]

        figures_out = package / "paper_assets/figures" / dataset
        figures_out.mkdir(parents=True)
        figures(runs, figures_out, title, metadata["holdout"], len(metadata["seeds"]))
        diagnostics(runs, directory, figures_out, title, metadata["holdout"], len(metadata["seeds"]))
        summary = pd.read_csv(directory / "verified_summary.csv", header=[0, 1], index_col=[0, 1, 2])
        summary.columns = [" ".join(column) for column in summary.columns]
        if metadata["paper_mode"]:
            uncertainty(summary, figures_out, title, metadata["holdout"])
        number = 3 if dataset == "cicids2017" else 4
        summary.to_csv(tables / f"table{number}_{dataset}.csv")
        coverage_path = directory / "full_holdout_coverage.csv"
        coverage = pd.read_csv(coverage_path if coverage_path.exists() else directory / "coverage_42_Temporal.csv")
        coverage["Dataset"] = title
        coverage["Fit cap in coverage"] = not metadata["paper_mode"] and not coverage_path.exists()
        coverage["Holdout"] = metadata["holdout"]
        coverage["Test only"] = coverage.Train.eq(0) & coverage.Test.gt(0)
        coverage["Zero test support"] = coverage.Test.eq(0)
        coverage["Predictor count"] = len(metadata["predictors"])
        coverage["Total in holdout split"] = coverage.Train.sum() + coverage.Test.sum()
        if dataset == "unsw_nb15":
            audit = json.loads((directory / "duplicate_overlap_audit.json").read_text())
            coverage["Exact overlap test rows"] = audit["cross_partition_exact_overlap_count"]
            coverage["Duplicates excluded in core experiment"] = 0
        elif (directory / "validation.json").exists():
            audit = json.loads((directory / "validation.json").read_text())
            coverage["Duplicates excluded in core experiment"] = sum(
                item["duplicates_removed"] for item in audit["file_validation"])
        all_stats.append(coverage)
        for _, row in runs[runs.Task.eq("multiclass")].iterrows():
            name = Path(row["Prediction file"])
            recall = pd.read_csv(directory / name.with_name(name.stem + "_recall.csv"))
            split = "Reference" if row.Condition.startswith("Reference") else "Temporal"
            fit_coverage = pd.read_csv(directory / f"coverage_{row.Seed}_{split}.csv")
            recall = recall.merge(fit_coverage[["Class", "Train"]], on="Class", validate="one_to_one")
            recall["Test only"] = recall.Train.eq(0) & recall["Test support"].gt(0)
            for key in KEY:
                recall[key] = row[key]
            recall["Dataset"] = title
            all_coverage.append(recall)
        config = runs.loc[runs.Missingness.eq(0), KEY + ["Model parameters"]].copy()
        config["Dataset"] = title
        config["Holdout"] = metadata["holdout"]
        config["Seeds"] = ", ".join(map(str, metadata["seeds"]))
        config["Missingness rates"] = "0%, 10%, 20% of eligible numeric features"
        config["Fit rows"] = [
            int(pd.read_csv(directory / f"coverage_{row.Seed}_{row.Condition}.csv")["Train"].sum())
            for row in config.itertuples(index=False)
        ]
        config["Environment"] = json.dumps(metadata["environment"])
        all_config.append(config)
        drops = pd.read_csv(directory / "paired_degradation.csv")
        all_drops.append(drops.assign(Dataset=title))
        cost_fields = KEY + ["Fit seconds", "Predict seconds", "Records per second"]
        timing_file = directory / "timing_validation_seed42.csv"
        if timing_file.exists():
            costs = pd.read_csv(timing_file)
            timing_metadata = json.loads((directory / "timing_validation_metadata.json").read_text())
            costs["Environment"] = json.dumps(timing_metadata["environment"])
        else:
            costs = runs[cost_fields].copy()
            costs["Timing protocol"] = "Full-data paper mode" if metadata["paper_mode"] else "Capped-training validation"
            costs["Environment"] = json.dumps(metadata["environment"])
            runtime_file = directory / "process_runtime.json"
            if runtime_file.exists():
                costs["Run peak resident bytes"] = json.loads(runtime_file.read_text())["peak_resident_bytes"]
        all_costs.append(costs.assign(Dataset=title))
        sensitivity = directory / "duplicate_sensitivity.csv"
        if sensitivity.exists():
            shutil.copy2(sensitivity, tables / "table7_unsw_duplicate_sensitivity.csv")
    for frames, filename in [
        (all_stats, "table1_class_and_holdout_counts.csv"),
        (all_config, "table2_configuration.csv"),
        (all_drops, "table5_paired_degradation.csv"),
        (all_coverage, "table6_class_coverage_and_recall.csv"),
        (all_costs, "table8_computational_costs.csv"),
    ]:
        pd.concat(frames, ignore_index=True).to_csv(tables / filename, index=False)


def document_runs(package, inputs):
    handoff = ["\n## Verified runs in this package\n"]
    validation = ["\n## Delivered verification checks\n"]
    for dataset, directory, metadata in inputs:
        seeds = ", ".join(map(str, metadata["seeds"]))
        mode = "full-data paper mode" if metadata["paper_mode"] else "capped-training validation"
        runs = pd.read_csv(directory / "verified_seed_metrics.csv")
        clean = runs[runs.Missingness.eq(0)].groupby(["Task", "Model", "Condition"])["Macro F1"].mean()
        handoff += [
            f"\n### {DATASETS[dataset]}\n",
            f"{mode.capitalize()} with seed(s) {seeds}. "
            f"[Prediction-audited metrics](results/{dataset}/verified_seed_metrics.csv)\n",
            "| Task | Model | Reference macro F1 | Holdout macro F1 |",
            "| --- | --- | ---: | ---: |",
        ]
        if (directory / "recovery.json").exists():
            validation.append(
                f"- {DATASETS[dataset]} metrics were recovered from the five fitted seeds' saved "
                f"predictions after a plot export failure. Fit timing and convergence diagnostics "
                f"from that execution were unavailable. [Recovery record](../results/{dataset}/recovery.json)"
            )
        for task, model in clean.index.droplevel("Condition").unique():
            handoff.append(
                f"| {task} | {model} | {clean.loc[(task, model, 'Reference')]:.4f} "
                f"| {clean.loc[(task, model, 'Temporal')]:.4f} |"
            )
        validation.append(
            f"- {DATASETS[dataset]} passed checksum, prediction, split, confusion-matrix and "
            f"seed-summary checks for {len(runs)} evaluations in {mode}. Seeds {seeds}. "
            f"[Validation metadata](../results/{dataset}/prediction_validation.json)"
        )
    for name, lines in [("AUTHOR_HANDOFF.md", handoff), ("audit/validation_report.md", validation)]:
        with (package / name).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("--partial", action="store_true", help="Package explicitly labelled partial evidence")
    parser.add_argument("--fallback", type=Path, help="Alternate verified root for a partial package")
    arguments = parser.parse_args()
    inputs = []
    for name in DATASETS:
        directory = arguments.runs / name
        if arguments.partial and arguments.fallback and not (directory / "prediction_validation.json").exists():
            directory = arguments.fallback / name
        if arguments.partial and not (directory / "prediction_validation.json").exists():
            continue
        metadata = verify(directory)
        if not metadata["paper_mode"] and not arguments.partial:
            parser.error(f"{name} is a single-seed validation run. Use --partial to label it.")
        inputs.append((name, directory, metadata))
    if not inputs:
        parser.error("No verified runs found")
    root = Path(__file__).resolve().parent
    package = arguments.package
    package.mkdir(parents=True, exist_ok=False)
    for name in ["src", "configs", "audit", "tests", "notebooks"]:
        shutil.copytree(root / name, package / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["README.md", "requirements.txt", "acquire_benchmarks.py", "verify_run.py", "build_evidence.py",
                 "run_experiments.py", "run_unsw_experiments.py", "run_cross_analysis.py",
                 "recover_unsw_run.py", "diagnostic_audit.py",
                 "reviewer_evidence_matrix.md", "IMPLEMENTATION_CHANGES.md",
                 "RESULTS_EXPLANATION.md", "AUTHOR_HANDOFF.md"]:
        shutil.copy2(root / name, package / name)
    write_tables(package, inputs)
    document_runs(package, inputs)
    cross = arguments.runs / "cross_dataset"
    partial_cross = arguments.runs / "cross_dataset_partial"
    if cross.is_dir() and all(metadata["paper_mode"] for _, _, metadata in inputs):
        shutil.copytree(cross, package / "results/cross_dataset")
    elif arguments.partial and partial_cross.is_dir():
        shutil.copytree(partial_cross, package / "results/cross_dataset")
    metadata_dir = package / "paper_assets/experiment_metadata"
    metadata_dir.mkdir()
    for dataset, directory, metadata in inputs:
        for filename in ["dataset_provenance.json", "dataset_file_validation.json", "experiment_config.json",
                         "prediction_validation.json"]:
            shutil.copy2(directory / filename, metadata_dir / f"{dataset}_{filename}")
        if (directory / "process_runtime.json").exists():
            shutil.copy2(directory / "process_runtime.json", metadata_dir / f"{dataset}_process_runtime.json")
        if dataset == "cicids2017":
            full = directory / "full_holdout_coverage.csv"
            shutil.copy2(full if full.exists() else directory / "coverage_42_Temporal.csv",
                         package / "audit/temporal_class_coverage.csv")
        elif (directory / "duplicate_sensitivity.csv").is_file():
            shutil.copy2(directory / "duplicate_sensitivity.csv", package / "audit/duplicate_sensitivity.csv")
    (package / "RUN_STATUS.json").write_text(json.dumps({
        "partial": arguments.partial,
        "datasets": {dataset: metadata for dataset, _, metadata in inputs},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
