"""
Prepare a late-detection dataset CSV from an Excel file of crack observations by week.

Saves a CSV with columns: grape_id, row, image_path, label (1 = cracked in last week, 0 = not cracked).

TODO: Set LAST_WEEK_COL to the exact last-week column name (e.g. "25.09.24") or enable AUTO_DETECT_LAST_WEEK.

Usage:
    python generate_late_detection_dataset.py

Configurable constants are at the top of the file.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import sys

# ====== CONFIG ======
EXCEL_PATH = "crack_in_image_by_weeks_for_dataset.xlsx"
SHEET_NAME: Optional[str] = None  # None -> first sheet; or set to a specific sheet name
GRAPE_ID_COL = "Grape ID"  # original column name in Excel
LAST_WEEK_COL = "18.09.24"  # TODO: set the exact last-week column name (e.g. "25.09.24")
FIRST_WEEK_COL = "08.08.24"
OUTPUT_CSV = "late_detection_dataset.csv"
BASE_RAW_DIR = r"C:\Users\yovel\Desktop\Grape_Project\data\raw" # base folder for raw data
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
    """Return True if the cell value is 1 (main grape cracked) or 3 (periphery grape cracked).

    Rules:
    - NaN -> False
    - Value is 1 or 3 -> True
    - Any other value -> False
    """
    if pd.isna(value):
        return False
    try:
        num_value = float(value)
        return num_value == 1.0 or num_value == 3.0
    except Exception:
        # If it's a string, check if it's "1" or "3"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped == "1" or stripped == "3"
        return False


def build_folder_path(base_raw_dir: str, grape_id: str, last_week_col_name: str) -> str:
    """Build the folder path for the grape cluster last-week raw data.

    Example: base_raw_dir / grape_id / last_week_col_name
    """
    base = Path(base_raw_dir)
    return str(base / str(grape_id) / str(last_week_col_name))


def extract_row_from_grape_id(grape_id: str) -> Optional[int]:
    """Extract the integer row component from grape_id assumed to be like '1_07'.

    Returns an integer or pandas.NA if it cannot be parsed.
    """
    if not isinstance(grape_id, str):
        return pd.NA
    parts = grape_id.split("_")
    if not parts:
        return pd.NA
    try:
        return int(parts[0])
    except Exception:
        return pd.NA


def prepare_dataset(
    excel_path: str,
    sheet_name: Optional[str],
    grape_col: str,
    last_week_col: str,
    base_raw_dir: str,
    output_csv: str,
    auto_detect_last_week: bool = False,
) -> pd.DataFrame:
    """Prepare and save the late-detection dataset and return the output DataFrame."""
    df = read_excel(excel_path, sheet_name)

    if grape_col not in df.columns:
        raise ValueError(f"GRAPE_ID_COL '{grape_col}' not found in columns: {list(df.columns)}")

    # create grape_id column and normalize
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    # optionally drop rows with missing grape_id
    df = df[df["grape_id"].notna() & (df["grape_id"].str.strip() != "")]

    # determine last week column
    chosen_last_week = last_week_col
    if not chosen_last_week or str(chosen_last_week).startswith("<"):
        if auto_detect_last_week:
            candidate = find_last_week_column(df, exclude_cols={grape_col, "grape_id"})
            if candidate is None:
                raise ValueError("AUTO_DETECT_LAST_WEEK failed: could not find a suitable last-week column.")
            print(f"AUTO_DETECT_LAST_WEEK: using column '{candidate}' as last-week column.")
            chosen_last_week = candidate
        else:
            raise ValueError("LAST_WEEK_COL is not set. Set LAST_WEEK_COL to the exact last-week column name or enable AUTO_DETECT_LAST_WEEK.")

    if chosen_last_week not in df.columns:
        raise ValueError(f"LAST_WEEK_COL '{chosen_last_week}' not found in columns: {list(df.columns)}")

    print(f"Using last-week column: '{chosen_last_week}'")

    # extract row from grape_id
    df["row"] = df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
    # convert row to nullable integer dtype (Int64) for safety
    try:
        df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
    except Exception:
        # leave as-is if conversion fails
        pass

    # compute label from chosen last-week column
    # label: 1 = cracked (either main grape or periphery grape), 0 = not cracked
    # Cell values: 1 = main grape cracked, 3 = periphery grape cracked -> both get label 1
    def get_label(v):
        if pd.isna(v):
            return 0
        try:
            num_value = float(v)
            if num_value == 1.0 or num_value == 3.0:
                return 1  # both 1 and 3 are cracked
            else:
                return 0
        except Exception:
            if isinstance(v, str):
                stripped = v.strip()
                if stripped == "1" or stripped == "3":
                    return 1  # both 1 and 3 are cracked
            return 0

    df["label"] = df[chosen_last_week].apply(get_label)

    # build image_path as a FOLDER path: BASE_RAW_DIR / grape_id / chosen_last_week
    df["image_path"] = df["grape_id"].apply(lambda gid: build_folder_path(base_raw_dir, gid, chosen_last_week))

    # final columns
    output_df = df[["grape_id", "row", "image_path", "label"]].copy()

    # drop rows where image_path is missing/empty (unlikely since we constructed it), and drop duplicates
    output_df = output_df[output_df["image_path"].notna() & (output_df["image_path"].str.strip() != "")]
    output_df = output_df.drop_duplicates(subset=["grape_id"])  # keep first

    # save
    output_df.to_csv(output_csv, index=False)
    print(f"Saved late detection dataset to {output_csv} with {len(output_df)} rows.")
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
