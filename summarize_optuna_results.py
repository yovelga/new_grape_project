"""
Summarize Optuna full-image experiment results for thesis comparison.
Reads final_metrics.json from each run and writes a single CSV
with all 16 combinations: 4 features × 2 (early/late) × 2 (balance/unbalance).

Selection logic:
  - For "early" runs → use the *_new folders.
  - For "late" runs  → use *_new if available, else fall back to old folders.
  - 159_full_balance_new is treated as 159_late_balance_new.
"""

import json
import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent / "experiments" / "optuna_full_image"
OUTPUT_CSV = Path(__file__).parent / "experiments" / "optuna_full_image" / "optuna_results_summary.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_folder_name(name: str):
    """
    Return (n_features, timing, balance, is_new) from a folder name.
    Examples:
      run_2026-02-04_23-06-59_09_early_unbalance_new  → (9, 'early', 'unbalance', True)
      run_2026-02-04_21-03-49_09_late_unbalance       → (9, 'late',  'unbalance', False)
      run_2026-02-06_12-37-22_159_full_balance_new     → (159,'late', 'balance',   True)   # full→late
    """
    parts = name.split("_")
    # Find the feature-count token (pure digits after the timestamp tokens)
    # Folder format: run_YYYY-MM-DD_HH-MM-SS_<features>_<timing>_<balance>[_new]
    # timestamp uses 3 underscore-separated groups: date, time, features…
    # e.g. ['run', '2026-02-04', '23-06-59', '09', 'early', 'unbalance', 'new']
    n_features = int(parts[3])

    timing = parts[4]           # early / late / full
    balance = parts[5]          # balance / unbalance
    is_new = "new" in parts

    # Treat "full" as "late" (159_full_balance_new is the late-balance run)
    if timing == "full":
        timing = "late"

    return n_features, timing, balance, is_new


def read_metrics(folder_path: Path):
    """Read final_metrics.json and return (calibration_metrics, test_metrics)."""
    fp = folder_path / "final_metrics.json"
    with open(fp, "r") as f:
        data = json.load(f)
    return data["splits"]["calibration"], data["splits"]["test"], data["metadata"]


def read_dataset_names(folder_path: Path):
    """Extract calibration and test CSV file names from optuna_run.log."""
    log_file = folder_path / "optuna_run.log"
    calib_name, test_name = "", ""
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Calibration CSV:"):
                    calib_name = Path(line.split(":", 1)[1].strip()).name
                elif line.startswith("Test CSV:"):
                    test_name = Path(line.split(":", 1)[1].strip()).name
                if calib_name and test_name:
                    break
    return calib_name, test_name


# ── collect every run ────────────────────────────────────────────────────────

runs = {}  # key = (n_features, timing, balance)  →  (folder_path, is_new)

for entry in sorted(os.listdir(BASE_DIR)):
    full_path = BASE_DIR / entry
    if not full_path.is_dir() or not entry.startswith("run_"):
        continue
    try:
        n_feat, timing, balance, is_new = parse_folder_name(entry)
    except (IndexError, ValueError):
        continue

    key = (n_feat, timing, balance)

    # If we already have a new version, skip old; if we have old, overwrite with new
    if key in runs:
        existing_is_new = runs[key][1]
        if is_new and not existing_is_new:
            runs[key] = (full_path, is_new)
        # else keep the existing (already new, or both old — keep first)
    else:
        runs[key] = (full_path, is_new)

# ── Also keep old 159 runs from the first experiment batch (run_2026-01-30) ──
# Those are the original 159-feature runs; they may or may not have been
# superseded by _new folders.  The logic above already handles this.

# ── build summary rows ──────────────────────────────────────────────────────

FEATURES_ORDER = [9, 11, 30, 159]
TIMINGS = ["early", "late"]
BALANCES = ["balance", "unbalance"]

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "precision", "recall",
    "f1_score", "f2_score", "specificity", "npv", "mcc",
]

rows = []
missing = []

for n_feat in FEATURES_ORDER:
    for timing in TIMINGS:
        for balance in BALANCES:
            key = (n_feat, timing, balance)
            if key not in runs:
                missing.append(key)
                continue

            folder_path, is_new = runs[key]
            cal, test, meta = read_metrics(folder_path)
            calib_ds, test_ds = read_dataset_names(folder_path)

            row = {
                "n_features": n_feat,
                "timing": timing,
                "balance": "balanced" if balance == "balance" else "unbalanced",
                "source": "new" if is_new else "old",
                "folder": folder_path.name,
                "calib_dataset": calib_ds,
                "test_dataset": test_ds,
                "best_trial": meta["best_trial"],
                "best_cv_score": round(meta["best_score"], 4),
            }

            # Confusion matrix – test
            cm = test["confusion_matrix"]
            row["test_TP"] = cm["TP"]
            row["test_FP"] = cm["FP"]
            row["test_TN"] = cm["TN"]
            row["test_FN"] = cm["FN"]

            # Test metrics
            for m in METRIC_COLS:
                row[f"test_{m}"] = round(test[m], 4)

            # Calibration metrics (for reference)
            for m in METRIC_COLS:
                row[f"cal_{m}"] = round(cal[m], 4)

            rows.append(row)

# ── sort for readability ────────────────────────────────────────────────────
rows.sort(key=lambda r: (r["n_features"], r["timing"], r["balance"]))

# ── write CSV ────────────────────────────────────────────────────────────────

fieldnames = list(rows[0].keys()) if rows else []

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅  Wrote {len(rows)} rows to {OUTPUT_CSV}")
if missing:
    print(f"⚠️  Missing combinations ({len(missing)}):")
    for m in missing:
        print(f"     features={m[0]}, timing={m[1]}, balance={m[2]}")
