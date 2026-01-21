"""
Build full-image classification datasets (CSV files) for EARLY and LATE detection experiments.

This script generates 8 dataset CSV files in one run:
  EARLY: early_row1_noisy.csv, early_row2_noisy.csv, early_row1_curated.csv, early_row2_curated.csv
  LATE:  late_row1_noisy.csv,  late_row2_noisy.csv,  late_row1_curated.csv,  late_row2_curated.csv

Usage:
    python build_full_image_datasets.py

Configurable constants are at the top of the file. All required constants marked with TODO must be filled in.

Output CSV schema:
    grape_id, row, image_path, label, chosen_week_col, experiment_name, dataset_mode
"""
from pathlib import Path
from typing import Optional, List, Dict
import argparse
import pandas as pd
import sys

# ============================================================================
# CONFIGURATION - USER MUST FILL THESE IN
# ============================================================================

# Path to the Excel file with crack observations per week
EXCEL_PATH: str = r"C:\Users\yovel\Desktop\Grape_Project\src\datasets\taged_grape_clusters\crack_in_image_by_weeks_for_dataset.xlsx"

# Sheet name (None = first sheet, or set to specific sheet name)
SHEET_NAME: Optional[str] = None

# Column name in Excel that contains the grape ID
GRAPE_ID_COL: str = "Grape ID"

# Base directory containing raw images organized as: BASE_RAW_DIR/<grape_id>/<week_col_name>/
BASE_RAW_DIR: str = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"

# Output directory for generated CSV files
OUTPUT_DIR: str = r"C:\Users\yovel\Desktop\Grape_Project\src\datasets\full_image\builders\output"

# ----------------------------------------------------------------------------
# EARLY DETECTION CONFIG
# ----------------------------------------------------------------------------
# The last week column used for EARLY detection (for never-cracked samples).
# For EARLY positives: we find the FIRST week where crack occurred.
# For EARLY negatives (noisy): we use this column as chosen_week_col.
EARLY_LAST_WEEK_COL: str = "10.07.24"

# For EARLY curated negatives: grape_ids with all zeros up to this column are considered "clean negatives".
EARLY_CLEAN_NEG_MAX_WEEK_COL: str = "10.07.24"

# ----------------------------------------------------------------------------
# LATE DETECTION CONFIG
# ----------------------------------------------------------------------------
# The last week column used for LATE detection (all samples use this column).
LATE_LAST_WEEK_COL: str = "25.09.24"

# For LATE curated negatives: grape_ids with all zeros up to this column are considered "clean negatives".
LATE_CLEAN_NEG_MAX_WEEK_COL: str = "25.09.24"

# ----------------------------------------------------------------------------
# CURATED DATASET SETTINGS
# ----------------------------------------------------------------------------
# Maximum number of negative samples per row in curated datasets.
# If there are more clean negatives, we randomly sample this many.
MAX_NEGATIVES: int = 60

# Random seed for reproducibility when sampling negatives.
RANDOM_SEED: int = 42

# ============================================================================
# END CONFIGURATION
# ============================================================================


def read_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read an Excel file into a DataFrame and normalize column names (strip whitespace).

    If `sheet` is None, reads the first sheet (sheet_name=0) to ensure a DataFrame
    is returned instead of a dict. Prints all columns found for verification.

    Args:
        path: Path to the Excel file.
        sheet: Sheet name to read (None = first sheet).

    Returns:
        DataFrame with normalized column names.
    """
    if sheet is None:
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    print("=" * 60)
    print("Columns found in Excel:")
    print("=" * 60)
    for i, c in enumerate(df.columns):
        print(f"  [{i:02d}] {c}")
    print("=" * 60)
    return df


def is_cracked(value) -> bool:
    """Return True if the cell value indicates a crack was observed.

    Rules:
        - NaN -> False
        - Numeric 1 (or 1.0) -> True
        - String "1" -> True
        - Any other value -> False

    Args:
        value: Cell value to check.

    Returns:
        True if crack observed, False otherwise.
    """
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value == 1 or value == 1.0
    if isinstance(value, str):
        return value.strip() == "1"
    return False


def extract_row(grape_id: str) -> Optional[int]:
    """Extract the row number from a grape_id string.

    Assumes grape_id format is like "1_07" where the prefix before "_" is the row number.

    Args:
        grape_id: Grape ID string (e.g., "1_07").

    Returns:
        Row number as integer, or None if parsing fails.
    """
    if not isinstance(grape_id, str):
        return None
    parts = grape_id.split("_")
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def validate_columns(df: pd.DataFrame, required_cols: List[str], context: str = "") -> None:
    """Validate that all required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        required_cols: List of column names that must exist.
        context: Optional context string for error messages.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Missing required columns{ctx}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def get_week_columns(df: pd.DataFrame, grape_col: str) -> List[str]:
    """Get the list of week columns from the DataFrame.

    Assumes all columns except the grape ID column are week columns,
    and they are in chronological order left-to-right.
    Filters out any columns starting with 'Unnamed:' (Excel artifacts).

    Args:
        df: DataFrame with week columns.
        grape_col: Name of the grape ID column (excluded from week columns).

    Returns:
        List of week column names in order.
    """
    return [c for c in df.columns if c != grape_col and not c.startswith("Unnamed:")]


def get_week_columns_up_to(all_weeks: List[str], up_to_col: str) -> List[str]:
    """Get week columns from the start up to (and including) a specified column.

    Args:
        all_weeks: List of all week columns in order.
        up_to_col: The column to stop at (inclusive).

    Returns:
        List of week columns up to and including up_to_col.

    Raises:
        ValueError: If up_to_col is not in all_weeks.
    """
    if up_to_col not in all_weeks:
        raise ValueError(f"Column '{up_to_col}' not found in week columns: {all_weeks}")
    idx = all_weeks.index(up_to_col)
    return all_weeks[: idx + 1]


def build_image_path(base_raw_dir: str, grape_id: str, week_col: str) -> str:
    """Build the folder path for images of a grape cluster at a specific week.

    Args:
        base_raw_dir: Base directory for raw images.
        grape_id: Grape cluster ID.
        week_col: Week/date column name (used as folder name).

    Returns:
        Full path as string: base_raw_dir/grape_id/week_col
    """
    return str(Path(base_raw_dir) / str(grape_id) / str(week_col))


def find_first_crack_week(row: pd.Series, week_cols: List[str]) -> Optional[str]:
    """Find the first week column (left-to-right) where value == 1 for a row.

    Args:
        row: DataFrame row (as Series).
        week_cols: List of week column names in chronological order.

    Returns:
        Column name of the first crack week, or None if never cracked.
    """
    for col in week_cols:
        if col in row.index and is_cracked(row[col]):
            return col
    return None


def has_any_crack(row: pd.Series, week_cols: List[str]) -> bool:
    """Check if the row has any crack in the specified week columns.

    Args:
        row: DataFrame row (as Series).
        week_cols: List of week column names to check.

    Returns:
        True if any week has a crack (value == 1), False otherwise.
    """
    for col in week_cols:
        if col in row.index and is_cracked(row[col]):
            return True
    return False


def is_clean_negative(row: pd.Series, week_cols: List[str]) -> bool:
    """Check if the row has NO cracks in all specified week columns.

    Args:
        row: DataFrame row (as Series).
        week_cols: List of week column names to check.

    Returns:
        True if no cracks (all values != 1), False otherwise.
    """
    for col in week_cols:
        if col in row.index and is_cracked(row[col]):
            return False
    return True


def prepare_base_dataframe(df: pd.DataFrame, grape_col: str) -> pd.DataFrame:
    """Prepare the base DataFrame with grape_id and row columns.

    Args:
        df: Raw DataFrame from Excel.
        grape_col: Name of the grape ID column.

    Returns:
        DataFrame with cleaned grape_id and extracted row number.
    """
    df = df.copy()
    df["grape_id"] = df[grape_col].astype(str).str.strip()
    df = df[df["grape_id"].notna() & (df["grape_id"] != "") & (df["grape_id"].str.lower() != "nan")]
    df["row"] = df["grape_id"].apply(extract_row)
    df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
    return df


def build_early_noisy(
    df: pd.DataFrame,
    grape_col: str,
    week_cols: List[str],
    early_last_week_col: str,
    base_raw_dir: str,
) -> pd.DataFrame:
    """Build EARLY noisy dataset.

    Logic:
        - For each grape_id: find the FIRST week column where value == 1.
        - If found: chosen_week_col = that first-crack week, label=1.
        - If never cracked: chosen_week_col = early_last_week_col, label=0.

    Args:
        df: Base DataFrame with grape_id and row.
        grape_col: Name of grape ID column.
        week_cols: List of all week columns.
        early_last_week_col: Last week column for EARLY (fallback for negatives).
        base_raw_dir: Base directory for raw images.

    Returns:
        DataFrame with columns: grape_id, row, image_path, label, chosen_week_col, experiment_name, dataset_mode
    """
    df = df.copy()

    # Find first crack week for each row
    df["first_crack_week"] = df.apply(lambda r: find_first_crack_week(r, week_cols), axis=1)
    df["ever_cracked"] = df["first_crack_week"].notna()

    # Determine chosen_week_col and label
    def get_chosen_week(row):
        if pd.notna(row["first_crack_week"]):
            return row["first_crack_week"]
        return early_last_week_col

    df["chosen_week_col"] = df.apply(get_chosen_week, axis=1)
    df["label"] = df["ever_cracked"].apply(lambda x: 1 if x else 0)
    df["image_path"] = df.apply(
        lambda r: build_image_path(base_raw_dir, r["grape_id"], r["chosen_week_col"]), axis=1
    )
    df["experiment_name"] = "early"
    df["dataset_mode"] = "noisy"

    return df[["grape_id", "row", "image_path", "label", "chosen_week_col", "experiment_name", "dataset_mode"]]


def build_late_noisy(
    df: pd.DataFrame,
    grape_col: str,
    week_cols: List[str],
    late_last_week_col: str,
    base_raw_dir: str,
) -> pd.DataFrame:
    """Build LATE noisy dataset.

    Logic:
        - For each grape_id: use the late_last_week_col as chosen_week_col.
        - label = 1 if value == 1 in that column, else 0.

    Args:
        df: Base DataFrame with grape_id and row.
        grape_col: Name of grape ID column.
        week_cols: List of all week columns.
        late_last_week_col: Last week column for LATE detection.
        base_raw_dir: Base directory for raw images.

    Returns:
        DataFrame with columns: grape_id, row, image_path, label, chosen_week_col, experiment_name, dataset_mode
    """
    df = df.copy()

    df["chosen_week_col"] = late_last_week_col
    df["label"] = df[late_last_week_col].apply(lambda v: 1 if is_cracked(v) else 0)
    df["image_path"] = df.apply(
        lambda r: build_image_path(base_raw_dir, r["grape_id"], r["chosen_week_col"]), axis=1
    )
    df["experiment_name"] = "late"
    df["dataset_mode"] = "noisy"

    return df[["grape_id", "row", "image_path", "label", "chosen_week_col", "experiment_name", "dataset_mode"]]


def build_early_curated(
    df: pd.DataFrame,
    grape_col: str,
    week_cols: List[str],
    early_last_week_col: str,
    early_clean_neg_max_week_col: str,
    base_raw_dir: str,
    max_negatives: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build EARLY curated dataset.

    Logic:
        - Positives: grape_ids with ANY crack (value==1) in weeks up to early_last_week_col.
          chosen_week_col = first crack week, label=1.
        - Negatives: grape_ids with NO crack in weeks up to early_clean_neg_max_week_col.
          chosen_week_col = early_last_week_col, label=0.
        - If more negatives than max_negatives per row, randomly sample.

    Args:
        df: Base DataFrame with grape_id and row.
        grape_col: Name of grape ID column.
        week_cols: List of all week columns.
        early_last_week_col: Last week column for EARLY positives/fallback.
        early_clean_neg_max_week_col: Max week for clean negatives check.
        base_raw_dir: Base directory for raw images.
        max_negatives: Max number of negatives per row.
        random_seed: Random seed for sampling.

    Returns:
        DataFrame with columns: grape_id, row, image_path, label, chosen_week_col, experiment_name, dataset_mode
    """
    df = df.copy()

    weeks_up_to_early = get_week_columns_up_to(week_cols, early_last_week_col)
    weeks_up_to_clean = get_week_columns_up_to(week_cols, early_clean_neg_max_week_col)

    # Find first crack week in the relevant window
    df["first_crack_week"] = df.apply(lambda r: find_first_crack_week(r, weeks_up_to_early), axis=1)

    # Positives: any crack in weeks_up_to_early
    df["is_positive"] = df["first_crack_week"].notna()

    # Negatives: no crack in weeks_up_to_clean
    df["is_clean_negative"] = df.apply(lambda r: is_clean_negative(r, weeks_up_to_clean), axis=1)

    # Build positives
    positives = df[df["is_positive"]].copy()
    positives["chosen_week_col"] = positives["first_crack_week"]
    positives["label"] = 1

    # Build negatives
    negatives = df[df["is_clean_negative"] & ~df["is_positive"]].copy()
    negatives["chosen_week_col"] = early_last_week_col
    negatives["label"] = 0

    # Sample negatives per row if needed
    sampled_negatives = []
    for row_num in negatives["row"].dropna().unique():
        row_negs = negatives[negatives["row"] == row_num]
        if len(row_negs) > max_negatives:
            row_negs = row_negs.sample(n=max_negatives, random_state=random_seed)
        sampled_negatives.append(row_negs)

    if sampled_negatives:
        negatives = pd.concat(sampled_negatives, ignore_index=True)
    else:
        negatives = negatives.iloc[0:0]  # empty DataFrame with same columns

    # Combine
    result = pd.concat([positives, negatives], ignore_index=True)
    result["image_path"] = result.apply(
        lambda r: build_image_path(base_raw_dir, r["grape_id"], r["chosen_week_col"]), axis=1
    )
    result["experiment_name"] = "early"
    result["dataset_mode"] = "curated"

    return result[["grape_id", "row", "image_path", "label", "chosen_week_col", "experiment_name", "dataset_mode"]]


def build_late_curated(
    df: pd.DataFrame,
    grape_col: str,
    week_cols: List[str],
    late_last_week_col: str,
    late_clean_neg_max_week_col: str,
    base_raw_dir: str,
    max_negatives: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build LATE curated dataset.

    Logic:
        - Positives: grape_ids with value==1 in late_last_week_col.
          chosen_week_col = late_last_week_col, label=1.
        - Negatives: grape_ids with NO crack in weeks up to late_clean_neg_max_week_col.
          chosen_week_col = late_last_week_col, label=0.
        - If more negatives than max_negatives per row, randomly sample.

    Args:
        df: Base DataFrame with grape_id and row.
        grape_col: Name of grape ID column.
        week_cols: List of all week columns.
        late_last_week_col: Last week column for LATE detection.
        late_clean_neg_max_week_col: Max week for clean negatives check.
        base_raw_dir: Base directory for raw images.
        max_negatives: Max number of negatives per row.
        random_seed: Random seed for sampling.

    Returns:
        DataFrame with columns: grape_id, row, image_path, label, chosen_week_col, experiment_name, dataset_mode
    """
    df = df.copy()

    weeks_up_to_clean = get_week_columns_up_to(week_cols, late_clean_neg_max_week_col)

    # Positives: cracked in late_last_week_col
    df["is_positive"] = df[late_last_week_col].apply(lambda v: is_cracked(v))

    # Negatives: no crack in weeks_up_to_clean
    df["is_clean_negative"] = df.apply(lambda r: is_clean_negative(r, weeks_up_to_clean), axis=1)

    # Build positives
    positives = df[df["is_positive"]].copy()
    positives["chosen_week_col"] = late_last_week_col
    positives["label"] = 1

    # Build negatives
    negatives = df[df["is_clean_negative"] & ~df["is_positive"]].copy()
    negatives["chosen_week_col"] = late_last_week_col
    negatives["label"] = 0

    # Sample negatives per row if needed
    sampled_negatives = []
    for row_num in negatives["row"].dropna().unique():
        row_negs = negatives[negatives["row"] == row_num]
        if len(row_negs) > max_negatives:
            row_negs = row_negs.sample(n=max_negatives, random_state=random_seed)
        sampled_negatives.append(row_negs)

    if sampled_negatives:
        negatives = pd.concat(sampled_negatives, ignore_index=True)
    else:
        negatives = negatives.iloc[0:0]  # empty DataFrame with same columns

    # Combine
    result = pd.concat([positives, negatives], ignore_index=True)
    result["image_path"] = result.apply(
        lambda r: build_image_path(base_raw_dir, r["grape_id"], r["chosen_week_col"]), axis=1
    )
    result["experiment_name"] = "late"
    result["dataset_mode"] = "curated"

    return result[["grape_id", "row", "image_path", "label", "chosen_week_col", "experiment_name", "dataset_mode"]]


def save_dataset(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save a dataset DataFrame to CSV and print statistics.

    Args:
        df: Dataset DataFrame.
        output_dir: Output directory path.
        filename: Output filename (CSV).
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    positives = (df["label"] == 1).sum()
    negatives = (df["label"] == 0).sum()

    print(f"  Saved: {output_path}")
    print(f"    Total: {total}, Positives: {positives}, Negatives: {negatives}")


def split_by_row(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Split DataFrame by row number (1 and 2).

    Args:
        df: DataFrame with 'row' column.

    Returns:
        Dictionary mapping row number to filtered DataFrame.
    """
    return {
        1: df[df["row"] == 1].copy(),
        2: df[df["row"] == 2].copy(),
    }


def main(
    excel_path: str = EXCEL_PATH,
    sheet_name: Optional[str] = SHEET_NAME,
    grape_col: str = GRAPE_ID_COL,
    base_raw_dir: str = BASE_RAW_DIR,
    output_dir: str = OUTPUT_DIR,
    early_last_week_col: str = EARLY_LAST_WEEK_COL,
    late_last_week_col: str = LATE_LAST_WEEK_COL,
    early_clean_neg_max_week_col: str = EARLY_CLEAN_NEG_MAX_WEEK_COL,
    late_clean_neg_max_week_col: str = LATE_CLEAN_NEG_MAX_WEEK_COL,
    max_negatives: int = MAX_NEGATIVES,
    random_seed: int = RANDOM_SEED,
) -> None:
    """Main entry point: build and save all 8 datasets.

    Args:
        excel_path: Path to the Excel file.
        sheet_name: Sheet name (None = first sheet).
        grape_col: Grape ID column name.
        base_raw_dir: Base directory for raw images.
        output_dir: Output directory for CSV files.
        early_last_week_col: Last week column for EARLY detection.
        late_last_week_col: Last week column for LATE detection.
        early_clean_neg_max_week_col: Max week for EARLY curated clean negatives.
        late_clean_neg_max_week_col: Max week for LATE curated clean negatives.
        max_negatives: Max number of negatives per row in curated datasets.
        random_seed: Random seed for sampling.
    """
    print("\n" + "=" * 60)
    print("Building Full-Image Classification Datasets")
    print("=" * 60)
    print(f"Excel file: {excel_path}")
    print(f"Output directory: {output_dir}")
    print(f"EARLY_LAST_WEEK_COL: {early_last_week_col}")
    print(f"LATE_LAST_WEEK_COL: {late_last_week_col}")
    print(f"EARLY_CLEAN_NEG_MAX_WEEK_COL: {early_clean_neg_max_week_col}")
    print(f"LATE_CLEAN_NEG_MAX_WEEK_COL: {late_clean_neg_max_week_col}")
    print(f"MAX_NEGATIVES: {max_negatives}")
    print(f"RANDOM_SEED: {random_seed}")
    print("=" * 60 + "\n")

    # Read Excel
    df = read_excel(excel_path, sheet_name)

    # Validate grape ID column
    validate_columns(df, [grape_col], "Grape ID column")

    # Get week columns
    week_cols = get_week_columns(df, grape_col)
    if not week_cols:
        raise ValueError("No week columns found. Check GRAPE_ID_COL setting.")

    print(f"\nIdentified {len(week_cols)} week columns (left-to-right):")
    print(f"  {week_cols}\n")

    # Validate required week columns exist
    required_weeks = [
        early_last_week_col,
        late_last_week_col,
        early_clean_neg_max_week_col,
        late_clean_neg_max_week_col,
    ]
    validate_columns(df, required_weeks, "Week columns")
    print("All required week columns validated successfully.\n")

    # Prepare base DataFrame
    base_df = prepare_base_dataframe(df, grape_col)
    print(f"Total grape clusters loaded: {len(base_df)}")
    print(f"  Row 1: {(base_df['row'] == 1).sum()}")
    print(f"  Row 2: {(base_df['row'] == 2).sum()}")
    print()

    # ========================================================================
    # Build EARLY NOISY datasets
    # ========================================================================
    print("-" * 60)
    print("Building EARLY NOISY datasets...")
    print("-" * 60)
    early_noisy = build_early_noisy(base_df, grape_col, week_cols, early_last_week_col, base_raw_dir)
    early_noisy_by_row = split_by_row(early_noisy)
    save_dataset(early_noisy_by_row[1], output_dir, "early_row1_noisy.csv")
    save_dataset(early_noisy_by_row[2], output_dir, "early_row2_noisy.csv")
    print()

    # ========================================================================
    # Build LATE NOISY datasets
    # ========================================================================
    print("-" * 60)
    print("Building LATE NOISY datasets...")
    print("-" * 60)
    late_noisy = build_late_noisy(base_df, grape_col, week_cols, late_last_week_col, base_raw_dir)
    late_noisy_by_row = split_by_row(late_noisy)
    save_dataset(late_noisy_by_row[1], output_dir, "late_row1_noisy.csv")
    save_dataset(late_noisy_by_row[2], output_dir, "late_row2_noisy.csv")
    print()

    # ========================================================================
    # Build EARLY CURATED datasets
    # ========================================================================
    print("-" * 60)
    print("Building EARLY CURATED datasets...")
    print("-" * 60)
    early_curated = build_early_curated(
        base_df,
        grape_col,
        week_cols,
        early_last_week_col,
        early_clean_neg_max_week_col,
        base_raw_dir,
        max_negatives,
        random_seed,
    )
    early_curated_by_row = split_by_row(early_curated)
    save_dataset(early_curated_by_row[1], output_dir, "early_row1_curated.csv")
    save_dataset(early_curated_by_row[2], output_dir, "early_row2_curated.csv")
    print()

    # ========================================================================
    # Build LATE CURATED datasets
    # ========================================================================
    print("-" * 60)
    print("Building LATE CURATED datasets...")
    print("-" * 60)
    late_curated = build_late_curated(
        base_df,
        grape_col,
        week_cols,
        late_last_week_col,
        late_clean_neg_max_week_col,
        base_raw_dir,
        max_negatives,
        random_seed,
    )
    late_curated_by_row = split_by_row(late_curated)
    save_dataset(late_curated_by_row[1], output_dir, "late_row1_curated.csv")
    save_dataset(late_curated_by_row[2], output_dir, "late_row2_curated.csv")
    print()

    print("=" * 60)
    print("All 8 datasets generated successfully!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (optional overrides for configuration).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build full-image classification datasets for EARLY and LATE detection experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_full_image_datasets.py
    python build_full_image_datasets.py --excel-path data.xlsx --output-dir ./output
    python build_full_image_datasets.py --early-last-week "10.07.24" --late-last-week "25.09.24"
        """,
    )
    parser.add_argument("--excel-path", type=str, default=EXCEL_PATH, help="Path to the Excel file")
    parser.add_argument("--sheet-name", type=str, default=SHEET_NAME, help="Sheet name (default: first sheet)")
    parser.add_argument("--grape-col", type=str, default=GRAPE_ID_COL, help="Grape ID column name")
    parser.add_argument("--base-raw-dir", type=str, default=BASE_RAW_DIR, help="Base directory for raw images")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for CSV files")
    parser.add_argument("--early-last-week", type=str, default=EARLY_LAST_WEEK_COL, help="Last week column for EARLY")
    parser.add_argument("--late-last-week", type=str, default=LATE_LAST_WEEK_COL, help="Last week column for LATE")
    parser.add_argument(
        "--early-clean-neg-max-week",
        type=str,
        default=EARLY_CLEAN_NEG_MAX_WEEK_COL,
        help="Max week for EARLY curated clean negatives",
    )
    parser.add_argument(
        "--late-clean-neg-max-week",
        type=str,
        default=LATE_CLEAN_NEG_MAX_WEEK_COL,
        help="Max week for LATE curated clean negatives",
    )
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES, help="Max negatives per row (curated)")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed for sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(
            excel_path=args.excel_path,
            sheet_name=args.sheet_name,
            grape_col=args.grape_col,
            base_raw_dir=args.base_raw_dir,
            output_dir=args.output_dir,
            early_last_week_col=args.early_last_week,
            late_last_week_col=args.late_last_week,
            early_clean_neg_max_week_col=args.early_clean_neg_max_week,
            late_clean_neg_max_week_col=args.late_clean_neg_max_week,
            max_negatives=args.max_negatives,
            random_seed=args.random_seed,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
