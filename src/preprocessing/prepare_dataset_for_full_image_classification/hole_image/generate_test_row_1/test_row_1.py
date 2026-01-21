"""
Generate NOISY and CLEAN CSV datasets for Row 1 Full-Image Classification.

This script creates two controlled datasets for vineyard ROW 1:
1. NOISY Dataset: August+ weeks (cartesian product of all row-1 grapes × all weeks >= August)
2. CLEAN Dataset: Pre-August negatives (label=0) + tagged crack positives (label=1)

Labeling:
- label=1 ONLY for (grape_id, week_date) pairs that appear in the TXT crack list
- label=0 for all other combinations
- The Excel is used ONLY to get week columns and row-1 grape IDs

Usage:
    python test_row_1.py
"""

from pathlib import Path
from typing import Set, Tuple, List, Optional
from datetime import datetime
import pandas as pd
import sys
import re

# ====== CONFIGURATION ======
SCRIPT_DIR = Path(__file__).parent.absolute()

# Excel file for extracting week columns and row-1 grape IDs
EXCEL_PATH = SCRIPT_DIR.parent / "late_detection_test" / "crack_in_image_by_weeks_for_dataset.xlsx"
SHEET_NAME: Optional[str] = None  # None -> first sheet
GRAPE_ID_COL = "Grape ID"  # Column name in Excel for grape identifiers

# TXT file with tagged crack paths (112 clusters)
TXT_CRACK_LIST_PATH = SCRIPT_DIR / "row_1_taged_with_crack_112_clusters.txt"

# Base directory for building image paths
BASE_RAW_DIR = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"

# Output directory and filenames
OUT_DIR = SCRIPT_DIR / "dataset_csvs"
OUT_NOISY = "row1_noisy_august_plus.csv"
OUT_CLEAN = "row1_clean_pre_august_plus_tagged_cracks.csv"

# August cutoff date (inclusive) - dates >= this are "August+"
AUGUST_CUTOFF = "01.08.24"  # dd.mm.yy format
# ===========================


def parse_date(date_str: str) -> datetime:
    """
    Parse a date string in dd.mm.yy format to a datetime object.

    Args:
        date_str: Date string like "01.08.24"

    Returns:
        datetime object
    """
    return datetime.strptime(date_str, "%d.%m.%y")


def is_week_column(col_name: str) -> bool:
    """Check if a column name looks like a date/week column (dd.mm.yy format)."""
    pattern = r'^\d{2}\.\d{2}\.\d{2}$'
    return bool(re.match(pattern, col_name))


def get_week_columns(df: pd.DataFrame) -> List[str]:
    """Extract all week/date columns from the DataFrame."""
    week_cols = [c for c in df.columns if is_week_column(c)]
    print(f"Detected {len(week_cols)} week columns: {week_cols}")
    return week_cols


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


def get_row1_grape_ids(df: pd.DataFrame, grape_col: str) -> List[str]:
    """
    Extract all grape IDs belonging to row 1 from the DataFrame.

    Args:
        df: Source DataFrame
        grape_col: Column name for grape IDs

    Returns:
        List of grape IDs starting with "1_"
    """
    row1_grapes = []
    for grape_id in df[grape_col].dropna().unique():
        grape_id_str = str(grape_id).strip()
        if grape_id_str.lower() == 'nan':
            continue
        vineyard_row = extract_row_from_grape_id(grape_id_str)
        if vineyard_row == 1:
            row1_grapes.append(grape_id_str)

    print(f"Found {len(row1_grapes)} row-1 grape IDs")
    return sorted(row1_grapes)


def parse_crack_txt_file(txt_path: Path) -> Set[Tuple[str, str]]:
    r"""
    Parse the TXT file containing crack image folder paths.

    Each line format: C:\Users\yovel\Desktop\Grape_Project\data\raw\1_04\05.09.24
    Extracts (grape_id, week_date) pairs.

    Args:
        txt_path: Path to the TXT file

    Returns:
        Set of (grape_id, week_date) tuples marked as cracked
    """
    crack_set = set()

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse path to extract grape_id and week_date
            # Example: C:\Users\yovel\Desktop\Grape_Project\data\raw\1_04\05.09.24
            path = Path(line)
            parts = path.parts

            # Find "raw" in path and extract subsequent folders
            try:
                raw_idx = parts.index("raw")
                grape_id = parts[raw_idx + 1]  # e.g., "1_04"
                week_date = parts[raw_idx + 2]  # e.g., "05.09.24"
                crack_set.add((grape_id, week_date))
            except (ValueError, IndexError) as e:
                print(f"WARNING: Could not parse line: {line} - {e}")
                continue

    print(f"Parsed {len(crack_set)} unique (grape_id, week_date) crack pairs from TXT")
    return crack_set


def split_weeks_by_august(week_cols: List[str], cutoff: str) -> Tuple[List[str], List[str]]:
    """
    Split week columns into pre-August and August+ lists.

    Args:
        week_cols: List of all week column names (dd.mm.yy format)
        cutoff: Cutoff date string (dd.mm.yy), dates >= cutoff are "August+"

    Returns:
        Tuple of (pre_august_weeks, august_plus_weeks)
    """
    cutoff_date = parse_date(cutoff)

    pre_august = []
    august_plus = []

    for week in week_cols:
        try:
            week_date = parse_date(week)
            if week_date >= cutoff_date:
                august_plus.append(week)
            else:
                pre_august.append(week)
        except ValueError as e:
            print(f"WARNING: Could not parse week '{week}': {e}")
            continue

    print(f"Pre-August weeks ({len(pre_august)}): {pre_august}")
    print(f"August+ weeks ({len(august_plus)}): {august_plus}")

    return pre_august, august_plus


def build_image_path(base_raw_dir: str, grape_id: str, week_date: str) -> str:
    """
    Build the image folder path: BASE_RAW_DIR / grape_id / week_date
    """
    base = Path(base_raw_dir)
    return str(base / grape_id / week_date)


def generate_noisy_dataset(
    row1_grapes: List[str],
    august_plus_weeks: List[str],
    crack_set: Set[Tuple[str, str]],
    base_raw_dir: str
) -> pd.DataFrame:
    """
    Generate NOISY dataset: Cartesian product of row-1 grapes × August+ weeks.

    Label=1 only if (grape_id, week_date) is in crack_set, else 0.

    Args:
        row1_grapes: List of row-1 grape IDs
        august_plus_weeks: List of week columns >= August cutoff
        crack_set: Set of (grape_id, week_date) tuples marked as cracked
        base_raw_dir: Base directory for image paths

    Returns:
        DataFrame with columns: grape_id, row, week_date, label, image_path
    """
    records = []

    for grape_id in row1_grapes:
        for week_date in august_plus_weeks:
            label = 1 if (grape_id, week_date) in crack_set else 0
            image_path = build_image_path(base_raw_dir, grape_id, week_date)

            records.append({
                'grape_id': grape_id,
                'row': 1,
                'week_date': week_date,
                'label': label,
                'image_path': image_path
            })

    return pd.DataFrame(records)


def generate_clean_dataset(
    row1_grapes: List[str],
    pre_august_weeks: List[str],
    crack_set: Set[Tuple[str, str]],
    base_raw_dir: str
) -> pd.DataFrame:
    """
    Generate CLEAN dataset:
    - Part 1: All (row-1 grapes × pre-August weeks) with label=0
    - Part 2: All crack_set entries with label=1
    - Duplicates resolved by keeping label=1

    Args:
        row1_grapes: List of row-1 grape IDs
        pre_august_weeks: List of week columns < August cutoff
        crack_set: Set of (grape_id, week_date) tuples marked as cracked
        base_raw_dir: Base directory for image paths

    Returns:
        DataFrame with columns: grape_id, row, week_date, label, image_path
    """
    # Use dict to handle duplicates: key=(grape_id, week_date), value=record
    # label=1 should override label=0
    records_dict = {}

    # Part 1: Pre-August negatives (all label=0)
    for grape_id in row1_grapes:
        for week_date in pre_august_weeks:
            key = (grape_id, week_date)
            image_path = build_image_path(base_raw_dir, grape_id, week_date)

            records_dict[key] = {
                'grape_id': grape_id,
                'row': 1,
                'week_date': week_date,
                'label': 0,
                'image_path': image_path
            }

    # Part 2: Tagged crack positives (label=1 overrides)
    for grape_id, week_date in crack_set:
        key = (grape_id, week_date)
        image_path = build_image_path(base_raw_dir, grape_id, week_date)

        records_dict[key] = {
            'grape_id': grape_id,
            'row': 1,
            'week_date': week_date,
            'label': 1,
            'image_path': image_path
        }

    return pd.DataFrame(list(records_dict.values()))


def print_dataset_summary(df: pd.DataFrame, description: str):
    """Print a summary of the dataset: row count and label distribution."""
    total = len(df)
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        cracked = label_counts.get(1, 0)
        not_cracked = label_counts.get(0, 0)
        print(f"{description}: {total} rows | label=1 (cracked): {cracked} | label=0 (not cracked): {not_cracked}")
    else:
        print(f"{description}: {total} rows")


def read_excel(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read Excel file and normalize column names (strip whitespace)."""
    path_str = str(path)

    if sheet is None:
        df = pd.read_excel(path_str, sheet_name=0)
    else:
        df = pd.read_excel(path_str, sheet_name=sheet)

    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
    return df


def main():
    """Main entry point for the script."""
    print("=" * 80)
    print("Row 1 NOISY & CLEAN Dataset Generator")
    print("=" * 80)

    # Read Excel file
    print(f"\n[1] Reading Excel file: {EXCEL_PATH}")
    df = read_excel(EXCEL_PATH, SHEET_NAME)

    # Get week columns
    print(f"\n[2] Extracting week columns...")
    week_cols = get_week_columns(df)

    # Get row-1 grape IDs
    print(f"\n[3] Extracting row-1 grape IDs...")
    row1_grapes = get_row1_grape_ids(df, GRAPE_ID_COL)

    # Parse TXT crack list
    print(f"\n[4] Parsing TXT crack list: {TXT_CRACK_LIST_PATH}")
    crack_set = parse_crack_txt_file(TXT_CRACK_LIST_PATH)

    # Split weeks by August cutoff
    print(f"\n[5] Splitting weeks by August cutoff ({AUGUST_CUTOFF})...")
    pre_august_weeks, august_plus_weeks = split_weeks_by_august(week_cols, AUGUST_CUTOFF)

    # Create output directory
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[6] Output directory: {out_dir.absolute()}")

    # Generate datasets
    print("\n" + "=" * 80)
    print("Generating datasets...")
    print("=" * 80)

    # Dataset A: NOISY (August+ only)
    print(f"\n[A] Generating NOISY dataset (August+ only)...")
    df_noisy = generate_noisy_dataset(row1_grapes, august_plus_weeks, crack_set, BASE_RAW_DIR)
    output_path_noisy = out_dir / OUT_NOISY
    df_noisy.to_csv(output_path_noisy, index=False)
    print(f"  ✓ Saved to: {output_path_noisy}")
    print_dataset_summary(df_noisy, "  NOISY Summary")

    # Dataset B: CLEAN (pre-August negatives + tagged cracks)
    print(f"\n[B] Generating CLEAN dataset (pre-August negatives + tagged cracks)...")
    df_clean = generate_clean_dataset(row1_grapes, pre_august_weeks, crack_set, BASE_RAW_DIR)
    output_path_clean = out_dir / OUT_CLEAN
    df_clean.to_csv(output_path_clean, index=False)
    print(f"  ✓ Saved to: {output_path_clean}")
    print_dataset_summary(df_clean, "  CLEAN Summary")

    # Final summary
    print("\n" + "=" * 80)
    print("✓ All datasets generated successfully!")
    print("=" * 80)
    print(f"\nOutput files in '{OUT_DIR}':")
    print(f"  1. {OUT_NOISY} ({len(df_noisy)} rows)")
    print(f"  2. {OUT_CLEAN} ({len(df_clean)} rows)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
