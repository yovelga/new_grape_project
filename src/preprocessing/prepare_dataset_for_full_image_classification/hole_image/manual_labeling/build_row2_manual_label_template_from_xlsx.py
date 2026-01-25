"""
Build Row 2 Manual Labeling Template from Excel

Purpose:
--------
Transform Excel crack tracking data (wide format) into a CSV template 
(long format) for manual image labeling in a UI.

Output Schema:
--------------
grape_id,row,week_date,label,image_path

Where:
- grape_id: Extracted from Excel "Grape ID" column (e.g., "2_01")
- row: Row number extracted from grape_id (e.g., 2)
- week_date: Week in dd.mm.yy format (from Excel column headers)
- label: Crack status (1=crack, 0=no crack) from Excel cells
- image_path: Absolute path to image directory

Usage:
------
Simply run the script with no arguments:
    python build_row2_manual_label_template_from_xlsx.py

The script will:
1. Load the configured Excel file (wide format with weeks as columns)
2. Filter records for the specified row (default: Row 2)
3. Transform to long format (one row per grape-week combination)
4. Generate absolute image paths
5. Fill missing labels with 0 (for manual review)

Configuration:
--------------
All settings are configured as constants at the top of the file.
Edit INPUT_XLSX, OUTPUT_CSV, BASE_RAW_DIR, or filtering parameters as needed.

Assumptions:
------------
- Excel has "Grape ID" column and week columns (dd.mm.yy format)
- Grape IDs follow pattern: {row}_{id} (e.g., "2_01" for row 2)
- Excel cells contain 1 for crack, 0 or NaN for no crack
- Images are stored at: {BASE_RAW_DIR}\{grape_id}\{week_date}\
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set
import pandas as pd
import re
import datetime as dt

# =========================
# CONFIGURATION (EDIT HERE)
# =========================

# Input Excel file path
INPUT_XLSX = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\crack_by_weeks_for_dataset.xlsx"

# Output CSV path
OUTPUT_CSV = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\prepare_dataset_for_full_image_classification\manual_labeling\output\row2_manual_label_template.csv"

# Base directory for raw images
BASE_RAW_DIR = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"

# Which row to export (vineyard row number)
ROW_TO_EXPORT = 2

# Optional date filtering (set to None to disable)
# Format: "dd.mm.yy" (e.g., "01.08.24")
DATE_FROM: Optional[str] = "01.08.24"  # Only include weeks from this date onwards
DATE_TO: Optional[str] = None

# Default value for missing/empty labels (NaN cells become 0)
DEFAULT_LABEL_VALUE = 0

# =========================
# Column Detection Patterns
# =========================

# Patterns for detecting grape ID column (case-insensitive)
GRAPE_ID_PATTERNS = ["grape_id", "grapeid", "grape id", "cluster_id", "clusterid", "cluster id"]

# Patterns for detecting path columns (case-insensitive)
PATH_PATTERNS = ["path", "image_path", "file_path", "rgb_path", "hsi_path", "folder_path", "dir_path"]

# Patterns for detecting label/tag columns (case-insensitive)
LABEL_PATTERNS = ["crack", "label", "target", "y", "class", "gt", "annotation", "tag", "category"]

# Pattern for detecting date columns (dd.mm.yy format)
DATE_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")


# =========================
# Helper Functions
# =========================

def normalize_column_name(col: str) -> str:
    """Normalize column name for matching (lowercase, strip spaces)."""
    return str(col).strip().lower().replace(" ", "_")


def parse_date(date_str: str) -> dt.date:
    """Parse date in dd.mm.yy format."""
    try:
        return dt.datetime.strptime(date_str, "%d.%m.%y").date()
    except Exception:
        raise ValueError(f"Invalid date format: {date_str} (expected dd.mm.yy)")


def is_date_column(col_name: str) -> bool:
    """Check if column name looks like a date (dd.mm.yy)."""
    return bool(DATE_PATTERN.match(str(col_name).strip()))


def extract_row_from_grape_id(grape_id: str) -> Optional[int]:
    """Extract row number from grape_id.
    
    Example: "2_001" -> 2, "1_042" -> 1
    
    Returns:
        Row number or None if extraction fails
    """
    try:
        return int(str(grape_id).split("_")[0])
    except Exception:
        return None


def build_image_path(base_dir: str, grape_id: str, week_date: str) -> str:
    """Build absolute Windows path to image directory."""
    return str(Path(base_dir) / grape_id / week_date)


def is_crack_cell(value) -> int:
    """Convert Excel cell value to label (1=crack, 0=no crack).
    
    Args:
        value: Excel cell value
    
    Returns:
        1 if cracked, 0 otherwise
    """
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        val_str = value.strip()
        if val_str == "" or val_str == "0":
            return 0
        try:
            return 1 if float(val_str) != 0.0 else 0
        except Exception:
            return 1  # Non-numeric non-empty string = crack
    try:
        return 1 if float(value) != 0.0 else 0
    except Exception:
        return 0


def select_best_sheet(excel_path: Path) -> Tuple[str, pd.ExcelFile]:
    """
    Select the most appropriate sheet from Excel file.
    Prefers sheets that contain a 'Grape ID' or similar column.
    
    Returns:
        (sheet_name, excel_file)
    """
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names
    
    if len(sheet_names) == 1:
        return sheet_names[0], excel_file
    
    # Check each sheet for grape ID column
    for sheet_name in sheet_names:
        df_preview = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0)
        normalized_cols = [normalize_column_name(c) for c in df_preview.columns]
        
        # Check if any grape ID pattern matches
        for pattern in GRAPE_ID_PATTERNS:
            if pattern in normalized_cols:
                return sheet_name, excel_file
    
    # Default to first sheet if no match
    print(f"⚠ Warning: No sheet with 'Grape ID' column found. Using first sheet: {sheet_names[0]}")
    return sheet_names[0], excel_file


def detect_column(df: pd.DataFrame, patterns: List[str], col_type: str) -> Optional[str]:
    """
    Detect a column matching any of the given patterns.
    
    Args:
        df: DataFrame to search
        patterns: List of pattern strings to match (case-insensitive)
        col_type: Description of column type (for error messages)
    
    Returns:
        Original column name (preserving case) or None if not found
    """
    normalized_cols = {normalize_column_name(c): c for c in df.columns}
    
    for pattern in patterns:
        if pattern in normalized_cols:
            return normalized_cols[pattern]
    
    return None


def detect_all_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """
    Detect all columns matching any of the given patterns.
    
    Returns:
        List of original column names (preserving case)
    """
    normalized_cols = {normalize_column_name(c): c for c in df.columns}
    matched = []
    
    for pattern in patterns:
        if pattern in normalized_cols:
            matched.append(normalized_cols[pattern])
    
    return matched


def build_dedup_key(row: pd.Series, key_columns: List[str]) -> str:
    """Build a stable deduplication key from available columns."""
    parts = []
    for col in key_columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                parts.append(str(val))
            else:
                parts.append("NA")
    return "|".join(parts)


def deduplicate_records(df: pd.DataFrame, available_cols: Set[str]) -> pd.DataFrame:
    """
    Deduplicate records based on strongest available key.
    
    Priority:
    1. (date, cluster_id, path)
    2. (date, cluster_id)
    3. (date, path)
    4. (cluster_id, path)
    5. All columns (last resort)
    """
    # Try to find best key columns
    possible_date_cols = [c for c in df.columns if is_date_column(c)]
    possible_id_cols = [c for c in available_cols if "cluster" in normalize_column_name(c) or "grape" in normalize_column_name(c)]
    possible_path_cols = detect_all_columns(df, PATH_PATTERNS)
    
    # Build key column list
    key_cols = []
    if possible_date_cols:
        key_cols.append(possible_date_cols[0])  # Use first date column
    if possible_id_cols:
        key_cols.append(possible_id_cols[0])  # Use first ID column
    if possible_path_cols:
        key_cols.append(possible_path_cols[0])  # Use first path column
    
    if not key_cols:
        # Last resort: use all columns
        print("⚠ Warning: No standard key columns found. Deduplicating by all columns.")
        return df.drop_duplicates()
    
    print(f"  Deduplicating by: {', '.join(key_cols)}")
    return df.drop_duplicates(subset=key_cols, keep="first")


# =========================
# Main Processing
# =========================

def main() -> None:
    print("=" * 80)
    print("Row 2 Manual Labeling Template Builder")
    print("=" * 80)
    
    # Validate input file
    input_path = Path(INPUT_XLSX)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_path}")
    
    print(f"Input:  {input_path}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Row to export: {ROW_TO_EXPORT}")
    print(f"Base raw dir: {BASE_RAW_DIR}")
    
    if DATE_FROM or DATE_TO:
        print(f"Date filter: {DATE_FROM or 'any'} to {DATE_TO or 'any'}")
    
    # Select best sheet
    print("\n" + "-" * 80)
    print("STEP 1: Excel Sheet Selection")
    print("-" * 80)
    
    sheet_name, excel_file = select_best_sheet(input_path)
    print(f"✓ Selected sheet: '{sheet_name}'")
    
    # Load data
    print("\n" + "-" * 80)
    print("STEP 2: Load and Parse Excel")
    print("-" * 80)
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Detect grape ID column
    print("\n" + "-" * 80)
    print("STEP 3: Detect Key Columns")
    print("-" * 80)
    
    grape_id_col = detect_column(df, GRAPE_ID_PATTERNS, "grape_id")
    if grape_id_col is None:
        raise ValueError(
            f"Could not find 'Grape ID' column. Available columns: {', '.join(df.columns.tolist())}\n"
            f"Expected one of: {', '.join(GRAPE_ID_PATTERNS)}"
        )
    print(f"✓ Grape ID column: '{grape_id_col}'")
    
    # Detect week columns (date format dd.mm.yy)
    week_cols = [col for col in df.columns if is_date_column(col)]
    week_cols = sorted(week_cols, key=lambda x: parse_date(x))
    print(f"✓ Detected {len(week_cols)} week columns: {week_cols[0]} to {week_cols[-1]}")
    
    # Apply date filtering to week columns
    if DATE_FROM or DATE_TO:
        date_from_obj = parse_date(DATE_FROM) if DATE_FROM else None
        date_to_obj = parse_date(DATE_TO) if DATE_TO else None
        
        filtered_weeks = []
        for week in week_cols:
            week_date = parse_date(week)
            if date_from_obj and week_date < date_from_obj:
                continue
            if date_to_obj and week_date > date_to_obj:
                continue
            filtered_weeks.append(week)
        
        week_cols = filtered_weeks
        print(f"✓ After date filter: {len(week_cols)} weeks ({week_cols[0]} to {week_cols[-1]})")
    
    # Filter by row
    print("\n" + "-" * 80)
    print("STEP 4: Filter to Row {ROW_TO_EXPORT}")
    print("-" * 80)
    
    # Extract row numbers from grape IDs
    df['_row_number'] = df[grape_id_col].apply(extract_row_from_grape_id)
    
    # Check extraction success
    null_rows = df['_row_number'].isna().sum()
    if null_rows > 0:
        print(f"⚠ Warning: Could not extract row number from {null_rows} grape IDs")
    
    unique_rows = sorted(df['_row_number'].dropna().unique().astype(int).tolist())
    print(f"  Detected row numbers: {unique_rows}")
    
    # Filter to target row
    df_row = df[df['_row_number'] == ROW_TO_EXPORT].copy()
    print(f"✓ Filtered to Row {ROW_TO_EXPORT}: {len(df_row)} grapes (from {len(df)} total)")
    
    if len(df_row) == 0:
        raise ValueError(f"No records found for Row {ROW_TO_EXPORT}. Available rows: {unique_rows}")
    
    # Transform from wide to long format
    print("\n" + "-" * 80)
    print("STEP 5: Transform Wide to Long Format")
    print("-" * 80)
    
    records = []
    for _, row in df_row.iterrows():
        grape_id = str(row[grape_id_col]).strip()
        row_num = extract_row_from_grape_id(grape_id)
        
        if row_num is None:
            print(f"⚠ Warning: Skipping grape_id '{grape_id}' - cannot extract row number")
            continue
        
        # For each week column, create a record
        for week_date in week_cols:
            label = is_crack_cell(row[week_date])
            image_path = build_image_path(BASE_RAW_DIR, grape_id, week_date)
            
            records.append({
                "grape_id": grape_id,
                "row": row_num,
                "week_date": week_date,
                "label": label,
                "image_path": image_path,
            })
    
    print(f"✓ Created {len(records)} records (grapes × weeks)")
    
    # Create DataFrame with exact schema
    df_output = pd.DataFrame(records, columns=["grape_id", "row", "week_date", "label", "image_path"])
    
    # Save output
    print("\n" + "-" * 80)
    print("STEP 6: Save CSV Template")
    print("-" * 80)
    
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✓ Saved: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Excel sheet:         {sheet_name}")
    print(f"Row {ROW_TO_EXPORT} grapes:        {len(df_row)}")
    print(f"Week columns:        {len(week_cols)} ({week_cols[0]} to {week_cols[-1]})")
    print(f"Total records:       {len(df_output)}")
    print(f"Label distribution:")
    print(f"  Crack (1):         {(df_output['label'] == 1).sum()}")
    print(f"  No crack (0):      {(df_output['label'] == 0).sum()}")
    print(f"\nOutput schema: grape_id, row, week_date, label, image_path")
    print("=" * 80)
    print("✓ Template ready for manual labeling!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
