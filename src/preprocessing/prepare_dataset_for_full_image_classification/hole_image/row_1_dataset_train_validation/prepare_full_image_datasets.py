"""
Prepare multiple CSV files for full-image dataset building from weekly crack annotations.

This script generates 5 CSV outputs from an Excel file containing grape crack observations by week:
1. Row 1 - all weeks: All grape IDs from vineyard row 1 across all week columns
2. Row 2 - all weeks: All grape IDs from vineyard row 2 across all week columns
3. Row 1 - only cracked: Subset of (1) where label==1
4. Row 2 - TEST LATE: Row 2 grapes for a specific late week
5. Row 2 - TEST EARLY: Row 2 grapes at their earliest crack detection point (or fallback week)

Each CSV contains: grape_id, row, week_date, label, image_path

Usage:
    python prepare_full_image_datasets.py
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import sys
import re

# ====== CONFIGURATION ======
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
EXCEL_PATH = SCRIPT_DIR / "crack_in_image_by_weeks_for_dataset.xlsx"  # Excel file in same folder as script
SHEET_NAME: Optional[str] = None  # None -> first sheet
GRAPE_ID_COL = "Grape ID"  # Column name in Excel for grape identifiers
BASE_RAW_DIR = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"  # Base folder for raw images
OUT_DIR = SCRIPT_DIR / "dataset_csvs"  # Output directory for generated CSVs (in same folder as script)

# Specific week for "late detection" test
LATE_WEEK_COL = "18.09.24"  # TODO: Adjust this to your desired late week column

# Row 1 filtering: only include weeks from this date onwards
ROW1_START_WEEK = "16.08.24"  # For Row 1, only include this week and later weeks

# Output filenames
OUT_ROW1_ALL = "row1_all_weeks.csv"
OUT_ROW2_ALL = "row2_all_weeks.csv"
OUT_ROW1_CRACKED = "row1_only_cracked.csv"
OUT_ROW2_LATE = "row2_test_late.csv"
OUT_ROW2_EARLY = "row2_test_early.csv"
# ===========================


def read_excel(path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read Excel file and normalize column names (strip whitespace)."""
    # Convert Path to string if needed
    path_str = str(path)

    if sheet is None:
        df = pd.read_excel(path_str, sheet_name=0)
    else:
        df = pd.read_excel(path_str, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
    print("Columns found:")
    for i, c in enumerate(df.columns):
        print(f"  {i:02d}: {c}")
    return df


def is_week_column(col_name: str) -> bool:
    """Check if a column name looks like a date/week column (dd.mm.yy format)."""
    # Match patterns like "24.06.24", "18.09.24", etc.
    pattern = r'^\d{2}\.\d{2}\.\d{2}$'
    return bool(re.match(pattern, col_name))


def get_week_columns(df: pd.DataFrame) -> List[str]:
    """Extract all week/date columns from the DataFrame."""
    week_cols = [c for c in df.columns if is_week_column(c)]
    print(f"\nDetected {len(week_cols)} week columns: {week_cols}")
    return week_cols


def filter_weeks_from_date(week_cols: List[str], start_week: str) -> List[str]:
    """
    Filter week columns to only include start_week and later weeks.

    Assumes week_cols are in chronological order (left to right).

    Args:
        week_cols: List of all week column names
        start_week: The starting week (e.g., "16.08.24")

    Returns:
        List of week columns from start_week onwards
    """
    if start_week not in week_cols:
        print(f"WARNING: Start week '{start_week}' not found in week columns.")
        return week_cols

    start_idx = week_cols.index(start_week)
    filtered = week_cols[start_idx:]
    print(f"Filtered weeks from '{start_week}': {len(filtered)} weeks selected")
    return filtered


def is_cracked(value) -> bool:
    """
    Return True if the cell value indicates a crack, False otherwise.

    Rules:
    - NaN / None -> False
    - Empty string / whitespace -> False
    - Zero (0, 0.0, "0") -> False
    - Any other non-empty value -> True
    """
    if pd.isna(value):
        return False

    # Handle string values
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped == "0":
            return False
        return True

    # Handle numeric values
    try:
        num_value = float(value)
        return num_value != 0.0
    except (ValueError, TypeError):
        # If it's some other object type, treat non-empty as cracked
        return True


def extract_row_from_grape_id(grape_id: str) -> Optional[int]:
    """
    Extract the vineyard row number from grape_id (e.g., "1_07" -> 1).

    Returns the integer row number or None if parsing fails.
    """
    if not isinstance(grape_id, str):
        return None

    parts = grape_id.split("_")
    if not parts:
        return None

    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def build_image_path(base_raw_dir: str, grape_id: str, week_date: str) -> str:
    """
    Build the image folder path: BASE_RAW_DIR / grape_id / week_date
    """
    base = Path(base_raw_dir)
    return str(base / grape_id / week_date)


def prepare_all_weeks_dataset(
    df: pd.DataFrame,
    week_cols: List[str],
    row_filter: Optional[int],
    base_raw_dir: str,
    grape_col: str
) -> pd.DataFrame:
    """
    Prepare a dataset with all grape IDs across all week columns.

    Args:
        df: Source DataFrame
        week_cols: List of week column names
        row_filter: If specified, only include grapes from this vineyard row (1 or 2)
        base_raw_dir: Base directory for image paths
        grape_col: Column name for grape IDs

    Returns:
        DataFrame with columns: grape_id, row, week_date, label, image_path
    """
    records = []

    for idx, row_data in df.iterrows():
        grape_id = str(row_data[grape_col]).strip()
        if not grape_id or grape_id.lower() == 'nan':
            continue

        vineyard_row = extract_row_from_grape_id(grape_id)
        if vineyard_row is None:
            continue

        # Apply row filter if specified
        if row_filter is not None and vineyard_row != row_filter:
            continue

        # Create a record for each week
        for week_col in week_cols:
            value = row_data[week_col]
            label = 1 if is_cracked(value) else 0
            image_path = build_image_path(base_raw_dir, grape_id, week_col)

            records.append({
                'grape_id': grape_id,
                'row': vineyard_row,
                'week_date': week_col,
                'label': label,
                'image_path': image_path
            })

    result_df = pd.DataFrame(records)

    # Remove duplicates (keep first occurrence)
    result_df = result_df.drop_duplicates(subset=['grape_id', 'week_date'], keep='first')

    return result_df


def prepare_late_week_dataset(
    df: pd.DataFrame,
    week_col: str,
    row_filter: Optional[int],
    base_raw_dir: str,
    grape_col: str
) -> pd.DataFrame:
    """
    Prepare a dataset for a specific week column.

    Returns:
        DataFrame with columns: grape_id, row, week_date, label, image_path
    """
    records = []

    for idx, row_data in df.iterrows():
        grape_id = str(row_data[grape_col]).strip()
        if not grape_id or grape_id.lower() == 'nan':
            continue

        vineyard_row = extract_row_from_grape_id(grape_id)
        if vineyard_row is None:
            continue

        # Apply row filter if specified
        if row_filter is not None and vineyard_row != row_filter:
            continue

        value = row_data[week_col]
        label = 1 if is_cracked(value) else 0
        image_path = build_image_path(base_raw_dir, grape_id, week_col)

        records.append({
            'grape_id': grape_id,
            'row': vineyard_row,
            'week_date': week_col,
            'label': label,
            'image_path': image_path
        })

    result_df = pd.DataFrame(records)

    # Remove duplicates (keep first)
    result_df = result_df.drop_duplicates(subset=['grape_id'], keep='first')

    return result_df


def prepare_early_detection_dataset(
    df: pd.DataFrame,
    week_cols: List[str],
    row_filter: Optional[int],
    base_raw_dir: str,
    grape_col: str
) -> pd.DataFrame:
    """
    Prepare "early detection" dataset where each grape appears once.

    Logic:
    - If grape has any crack weeks, choose the FIRST (leftmost) week where crack is detected
    - If grape never cracks, choose the last week column as fallback

    Returns:
        DataFrame with columns: grape_id, row, week_date, label, image_path
    """
    records = []

    for idx, row_data in df.iterrows():
        grape_id = str(row_data[grape_col]).strip()
        if not grape_id or grape_id.lower() == 'nan':
            continue

        vineyard_row = extract_row_from_grape_id(grape_id)
        if vineyard_row is None:
            continue

        # Apply row filter if specified
        if row_filter is not None and vineyard_row != row_filter:
            continue

        # Find first crack week or use last week as fallback
        chosen_week = None
        chosen_label = 0

        for week_col in week_cols:
            value = row_data[week_col]
            if is_cracked(value):
                chosen_week = week_col
                chosen_label = 1
                break

        # If no crack found, use last week column
        if chosen_week is None:
            chosen_week = week_cols[-1]
            chosen_label = 0

        image_path = build_image_path(base_raw_dir, grape_id, chosen_week)

        records.append({
            'grape_id': grape_id,
            'row': vineyard_row,
            'week_date': chosen_week,
            'label': chosen_label,
            'image_path': image_path
        })

    result_df = pd.DataFrame(records)

    # Remove duplicates (keep first)
    result_df = result_df.drop_duplicates(subset=['grape_id'], keep='first')

    return result_df


def print_dataset_summary(df: pd.DataFrame, description: str):
    """Print a summary of the dataset: row count and label distribution."""
    total = len(df)
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        cracked = label_counts.get(1, 0)
        not_cracked = label_counts.get(0, 0)
        print(f"{description}: {total} rows | Cracked: {cracked} | Not cracked: {not_cracked}")
    else:
        print(f"{description}: {total} rows")


def validate_inputs(df: pd.DataFrame, grape_col: str, late_week_col: str, week_cols: List[str]):
    """Validate critical assumptions about the input data."""
    # Check grape ID column exists
    if grape_col not in df.columns:
        raise ValueError(f"Grape ID column '{grape_col}' not found in Excel. Available columns: {list(df.columns)}")

    # Check late week column exists
    if late_week_col not in df.columns:
        raise ValueError(f"Late week column '{late_week_col}' not found in Excel. Available columns: {list(df.columns)}")

    # Check that late week is among detected week columns
    if late_week_col not in week_cols:
        print(f"WARNING: Late week column '{late_week_col}' is not detected as a week column by pattern matching.")
        print(f"Detected week columns: {week_cols}")

    # Check we have at least one week column
    if not week_cols:
        raise ValueError("No week columns detected. Expected columns matching pattern dd.mm.yy (e.g., '24.06.24')")

    print(f"\n✓ Validation passed")
    print(f"  - Grape ID column: '{grape_col}'")
    print(f"  - Late week column: '{late_week_col}'")
    print(f"  - Week columns detected: {len(week_cols)}")


def main():
    """Main entry point for the script."""
    print("=" * 80)
    print("Full-Image Dataset CSV Generator")
    print("=" * 80)

    # Read Excel file
    print(f"\nReading Excel file: {EXCEL_PATH}")
    df = read_excel(EXCEL_PATH, SHEET_NAME)

    # Detect week columns
    week_cols = get_week_columns(df)

    # Validate inputs
    validate_inputs(df, GRAPE_ID_COL, LATE_WEEK_COL, week_cols)

    # Create output directory
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir.absolute()}")

    # Generate datasets
    print("\n" + "=" * 80)
    print("Generating datasets...")
    print("=" * 80)

    # 1. Row 1 - All weeks (filtered from ROW1_START_WEEK onwards)
    print(f"\n[1/5] Generating Row 1 - All weeks (from {ROW1_START_WEEK} onwards)...")
    row1_week_cols = filter_weeks_from_date(week_cols, ROW1_START_WEEK)
    df_row1_all = prepare_all_weeks_dataset(df, row1_week_cols, row_filter=1, base_raw_dir=BASE_RAW_DIR, grape_col=GRAPE_ID_COL)
    output_path = out_dir / OUT_ROW1_ALL
    df_row1_all.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print_dataset_summary(df_row1_all, "  Summary")

    # 2. Row 2 - All weeks
    print("\n[2/5] Generating Row 2 - All weeks...")
    df_row2_all = prepare_all_weeks_dataset(df, week_cols, row_filter=2, base_raw_dir=BASE_RAW_DIR, grape_col=GRAPE_ID_COL)
    output_path = out_dir / OUT_ROW2_ALL
    df_row2_all.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print_dataset_summary(df_row2_all, "  Summary")

    # 3. Row 1 - Only cracked
    print("\n[3/5] Generating Row 1 - Only cracked...")
    df_row1_cracked = df_row1_all[df_row1_all['label'] == 1].copy()
    output_path = out_dir / OUT_ROW1_CRACKED
    df_row1_cracked.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print_dataset_summary(df_row1_cracked, "  Summary")

    # 4. Row 2 - TEST LATE
    print(f"\n[4/5] Generating Row 2 - TEST LATE (week: {LATE_WEEK_COL})...")
    df_row2_late = prepare_late_week_dataset(df, LATE_WEEK_COL, row_filter=2, base_raw_dir=BASE_RAW_DIR, grape_col=GRAPE_ID_COL)
    output_path = out_dir / OUT_ROW2_LATE
    df_row2_late.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print_dataset_summary(df_row2_late, "  Summary")

    # 5. Row 2 - TEST EARLY
    print("\n[5/5] Generating Row 2 - TEST EARLY (first crack detection or last week)...")
    df_row2_early = prepare_early_detection_dataset(df, week_cols, row_filter=2, base_raw_dir=BASE_RAW_DIR, grape_col=GRAPE_ID_COL)
    output_path = out_dir / OUT_ROW2_EARLY
    df_row2_early.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print_dataset_summary(df_row2_early, "  Summary")

    # Final summary
    print("\n" + "=" * 80)
    print("✓ All datasets generated successfully!")
    print("=" * 80)
    print(f"\nOutput files in '{OUT_DIR}':")
    print(f"  1. {OUT_ROW1_ALL} ({len(df_row1_all)} rows)")
    print(f"  2. {OUT_ROW2_ALL} ({len(df_row2_all)} rows)")
    print(f"  3. {OUT_ROW1_CRACKED} ({len(df_row1_cracked)} rows)")
    print(f"  4. {OUT_ROW2_LATE} ({len(df_row2_late)} rows)")
    print(f"  5. {OUT_ROW2_EARLY} ({len(df_row2_early)} rows)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
