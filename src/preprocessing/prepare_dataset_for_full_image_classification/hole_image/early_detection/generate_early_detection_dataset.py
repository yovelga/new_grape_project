"""
Prepare an EARLY-detection dataset CSV from an Excel file of crack observations by week.

For each grape cluster this script finds the first week (left-to-right) where a crack
was observed (using `is_cracked`). If a cluster ever cracked, the per-cluster chosen
week is that first-crack week and `label=1`. If a cluster never cracked, the chosen
week is `LAST_WEEK_COL` and `label=0`.

Saves a CSV with columns: grape_id, row, image_path, label

TODO: Set LAST_WEEK_COL to the exact last-week column name (e.g. "25.09.24") or enable AUTO_DETECT_LAST_WEEK.

Usage:
    python generate_early_detection_dataset.py

Configurable constants are at the top of the file.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import sys

# ====== CONFIG ======
EXCEL_PATH = "crack_by_weeks_for_dataset.xlsx"
SHEET_NAME: Optional[str] = None  # None -> first sheet; or set to a specific sheet name
GRAPE_ID_COL = "Grape ID"  # original column name in Excel
LAST_WEEK_COL = "25.09.24"  # TODO: set the exact last-week column name (e.g. "25.09.24")
OUTPUT_CSV = "early_detection_dataset.csv"
BASE_RAW_DIR = r"/data/raw"  # base folder for raw data
AUTO_DETECT_LAST_WEEK = False  # if True, pick the right-most non-empty week column automatically
# ====================


def read_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read an Excel file into a DataFrame and normalize column names (strip whitespace).

    If `sheet` is None this will read the first sheet (sheet_name=0) to ensure a DataFrame
    is returned instead of a dict (pandas returns a dict when sheet_name=None). Prints columns.
    """
    if sheet is None:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    print("Columns found in Excel:")
    for i, c in enumerate(df.columns):
        print(f"  {i:02d}: {c}")
    return df


def find_last_week_column(df: pd.DataFrame, exclude_cols: Optional[set] = None) -> Optional[str]:
    """Attempt to find a reasonable 'last week' column automatically.

    Strategy: pick the right-most column that is not in exclude_cols and not completely empty.
    """
    exclude = set(exclude_cols or [])
    for col in reversed(list(df.columns)):
        if col in exclude:
            continue
        if df[col].isna().all():
            continue
        return col
    return None


def is_cracked(value) -> bool:
    """Return True if the cell value should be considered a crack observation.

    Rules:
    - NaN -> False
    - Empty string -> False
    - Non-empty string -> True
    - Numeric: 0 or 0.0 -> False, other numbers -> True
    - Any other non-null value -> True
    """
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    try:
        return float(value) != 0.0
    except Exception:
        return True


def build_folder_path(base_raw_dir: str, grape_id: str, week_col_name: str) -> str:
    """Build the folder path for the grape cluster given a week column name.

    Example: base_raw_dir / grape_id / week_col_name
    """
    base = Path(base_raw_dir)
    return str(base / str(grape_id) / str(week_col_name))


def extract_row_from_grape_id(grape_id: str) -> Optional[int]:
    """Extract the integer row component from grape_id assumed to be like '1_07'.

    Returns an integer or None if it cannot be parsed.
    """
    if not isinstance(grape_id, str):
        return None
    parts = grape_id.split("_")
    if not parts:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


def prepare_dataset(
    excel_path: str,
    sheet_name: Optional[str],
    grape_col: str,
    last_week_col: str,
    base_raw_dir: str,
    output_csv: str,
    auto_detect_last_week: bool = False,
) -> pd.DataFrame:
    """Prepare and save the EARLY-detection dataset and return the output DataFrame."""
    df = read_excel(excel_path, sheet_name)

    if grape_col not in df.columns:
        raise ValueError(f"GRAPE_ID_COL '{grape_col}' not found in columns: {list(df.columns)}")

    # create grape_id column and normalize
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    # optionally drop rows with missing grape_id
    df = df[df["grape_id"].notna() & (df["grape_id"].str.strip() != "")]

    # validate LAST_WEEK_COL exists or auto-detect it
    chosen_last_week_global = last_week_col
    if not chosen_last_week_global or str(chosen_last_week_global).startswith("<"):
        if auto_detect_last_week:
            candidate = find_last_week_column(df, exclude_cols={grape_col, "grape_id"})
            if candidate is None:
                raise ValueError("AUTO_DETECT_LAST_WEEK failed: could not find a suitable last-week column.")
            print(f"AUTO_DETECT_LAST_WEEK: using column '{candidate}' as global last-week column.")
            chosen_last_week_global = candidate
        else:
            raise ValueError("LAST_WEEK_COL is not set. Set LAST_WEEK_COL to the exact last-week column name or enable AUTO_DETECT_LAST_WEEK.")

    if chosen_last_week_global not in df.columns:
        raise ValueError(f"LAST_WEEK_COL '{chosen_last_week_global}' not found in columns: {list(df.columns)}")

    print(f"Using LAST_WEEK_COL (fallback for never-cracked clusters): '{chosen_last_week_global}'")

    # identify week columns (assumes order in DataFrame is chronological left->right)
    non_week_cols = {grape_col, "grape_id"}
    week_cols = [c for c in df.columns if c not in non_week_cols]

    if len(week_cols) == 0:
        raise ValueError("No week columns found in the Excel file. Check your columns and GRAPE_ID_COL setting.")

    print(f"Identified {len(week_cols)} week columns (left-to-right): {week_cols}")

    # helper: find the first week column where the row indicates a crack
    def find_first_crack_col(row, week_columns):
        for col in week_columns:
            # protect against missing columns in row
            if col not in row.index:
                continue
            if is_cracked(row[col]):
                return col
        return None

    df["first_crack_col"] = df.apply(lambda r: find_first_crack_col(r, week_cols), axis=1)
    df["ever_cracked"] = df["first_crack_col"].notna()

    # choose per-row week: first_crack_col if cracked, otherwise the global LAST_WEEK_COL
    def choose_week_col(row):
        if pd.notna(row.get("first_crack_col")):
            return row["first_crack_col"]
        return chosen_last_week_global

    df["chosen_week_col"] = df.apply(choose_week_col, axis=1)

    # build label: 1 if ever_cracked, else 0
    df["label"] = df["ever_cracked"].apply(lambda x: 1 if x else 0)

    # extract row integer from grape_id
    df["row"] = df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
    try:
        df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
    except Exception:
        pass

    # build image_path per-row using chosen_week_col
    df["image_path"] = df.apply(lambda r: build_folder_path(base_raw_dir, r["grape_id"], r["chosen_week_col"]), axis=1)

    # final output
    output_df = df[["grape_id", "row", "image_path", "label"]].copy()

    # drop rows with empty image_path and duplicated grape_id
    output_df = output_df[output_df["image_path"].notna() & (output_df["image_path"].str.strip() != "")]
    output_df = output_df.drop_duplicates(subset=["grape_id"])  # keep first occurrence

    # save
    output_df.to_csv(output_csv, index=False)
    print(f"Saved early detection dataset to {output_csv} with {len(output_df)} rows.")
    return output_df


if __name__ == "__main__":
    try:
        prepare_dataset(
            EXCEL_PATH,
            SHEET_NAME,
            GRAPE_ID_COL,
            LAST_WEEK_COL,
            BASE_RAW_DIR,
            OUTPUT_CSV,
            auto_detect_last_week=AUTO_DETECT_LAST_WEEK,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
