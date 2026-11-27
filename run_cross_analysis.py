# ============================================================================
# Cross-Dataset Reliability Evaluation — Master Execution Runner
# ============================================================================

from pathlib import Path
import argparse
import sys

# Ensure UTF-8 clean output across Windows console runtimes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
CROSS_SCRIPT = ROOT / "src" / "cross_dataset_analysis.py"
parser = argparse.ArgumentParser(description="Compare prediction-audited benchmark trends.")
parser.add_argument("--partial", action="store_true", help="Compare shared seeds with an explicit preliminary label")
parser.add_argument("--fallback", type=Path, help="Alternate verified run root in partial mode")
options = parser.parse_args()
if options.fallback and not options.partial:
    parser.error("--fallback requires --partial")

print("=" * 80)
print("EXECUTING CROSS-DATASET RELIABILITY EVALUATION PIPELINE")
print("=" * 80)

if not CROSS_SCRIPT.exists():
    raise FileNotFoundError(f"Script not found: {CROSS_SCRIPT}")

code = compile(CROSS_SCRIPT.read_text(encoding="utf-8"), str(CROSS_SCRIPT), "exec")
shared_namespace = {
    "__file__": str(CROSS_SCRIPT),
    "__name__": "__main__",
    "PARTIAL_COMPARISON": options.partial,
    "FALLBACK_ROOT": options.fallback,
}
exec(code, shared_namespace, shared_namespace)

print("\n" + "=" * 80)
print("CROSS-DATASET PIPELINE COMPLETED")
print("=" * 80)
