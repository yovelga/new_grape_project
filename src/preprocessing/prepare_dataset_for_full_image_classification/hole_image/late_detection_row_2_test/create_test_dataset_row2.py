"""
Prepare a late-detection dataset CSV from an Excel file of crack observations by week.

Saves a CSV with columns: grape_id, row, image_path, label (1 = cracked in last week, 0 = not cracked).

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
LAST_WEEK_COL = "25.09.24"
FIRST_WEEK_COL = "01.08.24"
OUTPUT_CSV = "late_detection_dataset.csv"
# Use the Windows raw folder so generated full_path matches your example
BASE_RAW_DIR = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"
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
    print(f"Read Excel: {len(df)} rows x {len(df.columns)} columns")
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
    """Prepare and save the late-detection dataset and return the output DataFrame."""
    df = read_excel(excel_path, sheet_name)

    print(f"Starting dataset preparation from '{excel_path}'")

    if grape_col not in df.columns:
        raise ValueError(f"GRAPE_ID_COL '{grape_col}' not found in columns: {list(df.columns)}")

    # create grape_id column and normalize
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    # optionally drop rows with missing grape_id
    before = len(df)
    df = df[df["grape_id"].notna() & (df["grape_id"].str.strip() != "")]
    after = len(df)
    print(f"Normalized grape_id column: kept {after} / {before} rows")

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

    print(f"Using last-week column: '{chosen_last_week}' (non-null count: {df[chosen_last_week].notna().sum()})")

    # extract row from grape_id
    df["row"] = df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
    # convert row to nullable integer dtype (Int64) for safety
    try:
        df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
    except Exception:
        # leave as-is if conversion fails
        pass

    # Keep only row 2 samples
    total_before_row_filter = len(df)
    df = df[df["row"] == 2]
    print(f"Filtered to row 2: {len(df)} / {total_before_row_filter} rows remain")

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
    cracked_count = int(df["label"].sum())
    print(f"Labeling complete (row 2 only): cracked={cracked_count}, not_cracked={len(df)-cracked_count}")

    # build image_path as a FOLDER path: BASE_RAW_DIR / grape_id / chosen_last_week
    df["image_path"] = df["grape_id"].apply(lambda gid: build_folder_path(base_raw_dir, gid, chosen_last_week))
    # add explicit full_path column (same as image_path here — if you have filenames add them here)
    df["full_path"] = df["image_path"].astype(str)

    # final columns
    output_df = df[["grape_id", "row", "full_path", "label"]].copy()

    # drop rows where full_path is missing/empty and drop duplicates
    output_df = output_df[output_df["full_path"].notna() & (output_df["full_path"].str.strip() != "")]
    output_df = output_df.drop_duplicates(subset=["grape_id"])  # keep first

    # save
    print(f"Saving dataset CSV to '{output_csv}' ({len(output_df)} rows)")
    output_df.to_csv(output_csv, index=False)
    print(f"Saved late detection dataset to {output_csv} with {len(output_df)} rows.")
    return output_df


def prepare_test_csvs(
    excel_path: str,
    sheet_name: Optional[str],
    grape_col: str,
    first_week_col: str,
    last_week_col: str,
    base_raw_dir: str,
    output_prefix: str = "late_detection_test",
    auto_detect_last_week: bool = False,
) -> None:
    """Create two CSVs for testing the model:
    - <output_prefix>_august.csv: all image paths built from `first_week_col` (e.g. 08.08.24)
    - <output_prefix>_sept_cracked.csv: only image paths where `last_week_col` is cracked (1 or 3)
    """
    df = read_excel(excel_path, sheet_name)
    print(f"Preparing test CSVs from '{excel_path}' (rows={len(df)})")

    if grape_col not in df.columns:
        raise ValueError(f"GRAPE_ID_COL '{grape_col}' not found in columns: {list(df.columns)}")

    # normalize grape_id
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    df = df[df["grape_id"].notna() & (df["grape_id"].str.strip() != "")]
    print(f"Normalized grape_id for test CSVs: {len(df)} rows remain")

    # determine last week column
    chosen_last_week = last_week_col
    if (not chosen_last_week or str(chosen_last_week).startswith("<")):
        if auto_detect_last_week:
            candidate = find_last_week_column(df, exclude_cols={grape_col, "grape_id"})
            if candidate is None:
                raise ValueError("AUTO_DETECT_LAST_WEEK failed: could not find a suitable last-week column.")
            print(f"AUTO_DETECT_LAST_WEEK: using column '{candidate}' as last-week column.")
            chosen_last_week = candidate
        else:
            raise ValueError("LAST_WEEK_COL is not set. Set LAST_WEEK_COL or enable AUTO_DETECT_LAST_WEEK.")

    if chosen_last_week not in df.columns:
        raise ValueError(f"LAST_WEEK_COL '{chosen_last_week}' not found in columns: {list(df.columns)}")

    # Prepare August CSV (all grapes, use first_week_col)
    if first_week_col not in df.columns:
        print(f"Warning: FIRST_WEEK_COL '{first_week_col}' not found in Excel columns. Skipping August CSV.")
    else:
        aug_df = df.copy()
        aug_df["row"] = aug_df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
        try:
            aug_df["row"] = pd.to_numeric(aug_df["row"], errors="coerce").astype("Int64")
        except Exception:
            pass
        # filter to row 2 only
        before_aug = len(aug_df)
        aug_df = aug_df[aug_df["row"] == 2]
        print(f"August: filtered to row 2 -> {len(aug_df)} / {before_aug} rows remain")

        aug_df["image_path"] = aug_df["grape_id"].apply(lambda gid: build_folder_path(base_raw_dir, gid, first_week_col))
        aug_df["full_path"] = aug_df["image_path"].astype(str)
        # compute label from chosen last-week column (so August CSV includes the future label)
        aug_df["label"] = aug_df[chosen_last_week].apply(is_cracked) if chosen_last_week in aug_df.columns else 0
        aug_out = f"{output_prefix}_august_{first_week_col.replace('.', '-')}.csv"
        aug_df_out = aug_df[["grape_id", "row", "full_path", "label"]].drop_duplicates(subset=["grape_id"])
        print(f"Preparing August CSV -> {aug_out} (unique grapes: {len(aug_df_out)})")
        aug_df_out.to_csv(aug_out, index=False)
        print(f"Saved August dataset to {aug_out} with {len(aug_df_out)} rows.")

    # Prepare September cracked-only CSV (only rows where last_week_col indicates crack)
    def _is_cracked_val(v):
        # reuse is_cracked helper
        return 1 if is_cracked(v) else 0

    non_null_last = df[chosen_last_week].notna().sum()
    print(f"Last-week column '{chosen_last_week}' non-null entries: {non_null_last}")

    df["_last_label"] = df[chosen_last_week].apply(_is_cracked_val)
    cracked_df = df[df["_last_label"] == 1].copy()
    print(f"Found {len(cracked_df)} cracked rows in last-week column (will build cracked CSV)")
    cracked_df["row"] = cracked_df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
    try:
        cracked_df["row"] = pd.to_numeric(cracked_df["row"], errors="coerce").astype("Int64")
    except Exception:
        pass
    # filter to row 2 only
    before_cracked = len(cracked_df)
    cracked_df = cracked_df[cracked_df["row"] == 2]
    print(f"Cracked (Sept): filtered to row 2 -> {len(cracked_df)} / {before_cracked} rows remain")

    cracked_df["image_path"] = cracked_df["grape_id"].apply(lambda gid: build_folder_path(base_raw_dir, gid, chosen_last_week))
    cracked_df["full_path"] = cracked_df["image_path"].astype(str)
    # ensure label column exists (should be 1 for these rows)
    cracked_df["label"] = cracked_df[chosen_last_week].apply(_is_cracked_val)

    sept_out = f"{output_prefix}_sept_cracked_{str(chosen_last_week).replace('.', '-')}.csv"
    cracked_df_out = cracked_df[["grape_id", "row", "full_path", "label"]].drop_duplicates(subset=["grape_id"])
    print(f"Preparing September cracked CSV -> {sept_out} (unique cracked grapes: {len(cracked_df_out)})")
    cracked_df_out.to_csv(sept_out, index=False)
    print(f"Saved September cracked dataset to {sept_out} with {len(cracked_df_out)} rows.")


def prepare_single_test_dataset(
    excel_path: str,
    sheet_name: Optional[str],
    grape_col: str,
    early_date_col: str,
    late_date_col: str,
    base_raw_dir: str,
    output_csv: str = "test_dataset_row2.csv",
) -> None:
    """Create a single combined test dataset CSV:
    - Early date (e.g., '01.08.24'): ALL samples from row 2 -> new_label = 0
    - Late date (e.g., '25.09.24'): ONLY CRACK samples from row 2 -> new_label = 1
    """
    df = read_excel(excel_path, sheet_name)
    print(f"\n=== Creating Single Test Dataset from '{excel_path}' ===")

    if grape_col not in df.columns:
        raise ValueError(f"GRAPE_ID_COL '{grape_col}' not found in columns: {list(df.columns)}")

    # normalize grape_id
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    df = df[df["grape_id"].notna() & (df["grape_id"].str.strip() != "")]
    print(f"Normalized grape_id: {len(df)} rows")

    # extract row from grape_id and filter to row 2
    df["row"] = df["grape_id"].apply(lambda gid: extract_row_from_grape_id(gid))
    try:
        df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
    except Exception:
        pass

    before_row = len(df)
    df = df[df["row"] == 2]
    print(f"Filtered to row 2: {len(df)} / {before_row} rows")

    # === EARLY DATE GROUP: ALL samples, label=0 ===
    if early_date_col not in df.columns:
        print(f"Warning: FIRST_WEEK_COL '{early_date_col}' not found. Skipping early group.")
        early_group = pd.DataFrame()
    else:
        early_group = df.copy()
        early_group["full_path"] = early_group["grape_id"].apply(
            lambda gid: build_folder_path(base_raw_dir, gid, early_date_col)
        )
        early_group["new_label"] = 0
        early_group = early_group[["grape_id", "row", "full_path", "new_label"]].drop_duplicates(subset=["grape_id"])
        print(f"Early date ({early_date_col}): {len(early_group)} samples (ALL labels) -> new_label=0")

    # === LATE DATE GROUP: CRACK only, label=1 ===
    if late_date_col not in df.columns:
        print(f"Warning: LAST_WEEK_COL '{late_date_col}' not found. Skipping late group.")
        late_group = pd.DataFrame()
    else:
        # Check if late_date_col has crack values (1 or 3)
        late_df = df.copy()
        late_df["_is_crack"] = late_df[late_date_col].apply(lambda v: 1 if is_cracked(v) else 0)
        late_group = late_df[late_df["_is_crack"] == 1].copy()

        late_group["full_path"] = late_group["grape_id"].apply(
            lambda gid: build_folder_path(base_raw_dir, gid, late_date_col)
        )
        late_group["new_label"] = 1
        late_group = late_group[["grape_id", "row", "full_path", "new_label"]].drop_duplicates(subset=["grape_id"])
        print(f"Late date ({late_date_col}): {len(late_group)} samples (CRACK only) -> new_label=1")

    # === COMBINE ===
    combined = pd.concat([early_group, late_group], ignore_index=True)

    # Rename columns to match requested format
    combined.rename(columns={"full_path": "image_path", "new_label": "label"}, inplace=True)

    # Reorder columns: [grape_id, row, image_path, label]
    final = combined[["grape_id", "row", "image_path", "label"]]

    # Save
    final.to_csv(output_csv, index=False)
    print(f"\n✓ Saved combined test dataset to: {output_csv}")
    print(f"  Total rows: {len(final)}")
    print(f"  Early (label=0): {(final['label'] == 0).sum()}")
    print(f"  Late (label=1): {(final['label'] == 1).sum()}")


if __name__ == "__main__":
    try:
        # Create single combined test dataset
        prepare_single_test_dataset(
            EXCEL_PATH,
            SHEET_NAME,
            GRAPE_ID_COL,
            FIRST_WEEK_COL,
            LAST_WEEK_COL,
            BASE_RAW_DIR,
            output_csv="test_dataset_row2.csv",
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
