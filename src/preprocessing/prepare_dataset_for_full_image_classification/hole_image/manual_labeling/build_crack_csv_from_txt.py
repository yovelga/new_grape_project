"""
Script to build a CSV file from the txt files containing crack labels.
The output format matches crack_by_weeks_for_dataset.xlsx structure.

Input:
- row_1_taged_with_crack_112_clusters.txt
- row_2_taged_with_crack_84_clusters.txt

Output:
- crack_labels_from_txt.csv
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Define paths
SCRIPT_DIR = Path(__file__).parent
ROW1_FILE = SCRIPT_DIR / "row_1_taged_with_crack_110_clusters.txt"
ROW2_FILE = SCRIPT_DIR / "row_2_taged_with_crack_109_clusters.txt"
OUTPUT_CSV = SCRIPT_DIR / "crack_labels_from_txt.csv"

# Date columns in the desired order (matching the xlsx)
DATE_COLUMNS = [
    '24.06.24', '01.07.24', '10.07.24', '15.07.24', '25.07.24',
    '01.08.24', '08.08.24', '16.08.24', '22.08.24',
    '01.09.24', '05.09.24', '11.09.24', '18.09.24', '25.09.24'
]


def parse_txt_file(file_path):
    """
    Parse a txt file and extract grape_id and date pairs.

    Each line is a path like:
    C:/Users/.../data/raw/2_01/01.08.24

    Returns:
        list of tuples: [(grape_id, date), ...]
    """
    crack_entries = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse the path to extract grape_id and date
            # Path format: ...\data\raw\{grape_id}\{date}
            path = Path(line)
            date = path.name  # e.g., "01.08.24"
            grape_id = path.parent.name  # e.g., "2_01"

            crack_entries.append((grape_id, date))

    return crack_entries


def get_all_grape_ids():
    """
    Generate all possible grape IDs for rows 1 and 2 (1_01 to 1_60, 2_01 to 2_60).
    Adjust the range if needed based on your data.
    """
    grape_ids = []
    for row in [1, 2]:
        for num in range(1, 61):  # 1 to 60
            grape_ids.append(f"{row}_{num:02d}")
    return grape_ids


def build_dataframe(crack_entries, all_grape_ids, date_columns):
    """
    Build a DataFrame in the format of the xlsx file.

    Args:
        crack_entries: list of (grape_id, date) tuples indicating crack presence
        all_grape_ids: list of all grape IDs
        date_columns: list of date column names

    Returns:
        pd.DataFrame with Grape ID as first column and dates as other columns
    """
    # Create a set for quick lookup
    crack_set = set(crack_entries)

    # Build data dictionary
    data = {'Grape ID': all_grape_ids}

    # Initialize all date columns with None (NaN)
    for date in date_columns:
        data[date] = [None] * len(all_grape_ids)

    # Create grape_id to index mapping
    grape_id_to_idx = {gid: idx for idx, gid in enumerate(all_grape_ids)}

    # Fill in the crack labels
    for grape_id, date in crack_entries:
        if grape_id in grape_id_to_idx and date in data:
            idx = grape_id_to_idx[grape_id]
            data[date][idx] = 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Reorder columns: Grape ID first, then dates in order
    columns = ['Grape ID'] + date_columns
    df = df[columns]

    return df


def main():
    print("Building crack labels CSV from txt files...")
    print(f"Row 1 file: {ROW1_FILE}")
    print(f"Row 2 file: {ROW2_FILE}")

    # Parse both txt files
    crack_entries = []

    if ROW1_FILE.exists():
        entries1 = parse_txt_file(ROW1_FILE)
        print(f"  - Parsed {len(entries1)} entries from row 1 file")
        crack_entries.extend(entries1)
    else:
        print(f"  - Warning: {ROW1_FILE} not found")

    if ROW2_FILE.exists():
        entries2 = parse_txt_file(ROW2_FILE)
        print(f"  - Parsed {len(entries2)} entries from row 2 file")
        crack_entries.extend(entries2)
    else:
        print(f"  - Warning: {ROW2_FILE} not found")

    print(f"\nTotal crack entries: {len(crack_entries)}")

    # Get all grape IDs
    all_grape_ids = get_all_grape_ids()
    print(f"Total grape IDs: {len(all_grape_ids)}")

    # Find any dates in the txt files that aren't in our predefined list
    unique_dates = set(date for _, date in crack_entries)
    new_dates = unique_dates - set(DATE_COLUMNS)
    if new_dates:
        print(f"\nNote: Found dates not in predefined list: {sorted(new_dates)}")
        # Add new dates to the column list
        all_dates = DATE_COLUMNS + sorted(list(new_dates))
    else:
        all_dates = DATE_COLUMNS

    # Build the DataFrame
    df = build_dataframe(crack_entries, all_grape_ids, all_dates)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"DataFrame shape: {df.shape}")

    # Summary statistics
    crack_count = df.iloc[:, 1:].notna().sum().sum()
    print(f"Total crack labels in CSV: {int(crack_count)}")

    # Show a preview
    print("\nPreview (first 10 rows, showing only columns with data):")
    # Get columns that have at least one non-NaN value
    cols_with_data = ['Grape ID'] + [col for col in all_dates if df[col].notna().any()]
    print(df[cols_with_data].head(10).to_string())

    return df


if __name__ == "__main__":
    df = main()
