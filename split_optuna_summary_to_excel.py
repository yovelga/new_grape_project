"""
Split the optuna_results_summary.csv into an Excel workbook with two sheets:
  - Balanced
  - Unbalanced
Sorted by n_features and timing for easy thesis comparison.
"""

import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent / "experiments" / "optuna_full_image" / "optuna_results_summary.csv"
XLSX_PATH = CSV_PATH.with_name("optuna_results_summary.xlsx")

df = pd.read_csv(CSV_PATH)

# Sort for readability
df = df.sort_values(["n_features", "timing"]).reset_index(drop=True)

balanced = df[df["balance"] == "balanced"].reset_index(drop=True)
unbalanced = df[df["balance"] == "unbalanced"].reset_index(drop=True)

print(f"Balanced   rows: {len(balanced)}")
print(f"Unbalanced rows: {len(unbalanced)}")

with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
    balanced.to_excel(writer, sheet_name="Balanced", index=False)
    unbalanced.to_excel(writer, sheet_name="Unbalanced", index=False)

print(f"\n✅  Wrote {XLSX_PATH}")
