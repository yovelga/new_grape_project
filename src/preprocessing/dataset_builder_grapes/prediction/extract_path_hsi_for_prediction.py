"""
Description:
    Generates file paths for hyperspectral images based on grape cracking dates from an Excel file.

Main Functionality:
    - Reads grape data from Excel and identifies cracking dates.
    - Constructs paths for both crack and previous dates.
    - Saves all relevant paths to a text file.

Usage Notes:
    - Requires environment variables: BASE_PATH, DATA_FILE_PATH, BASE_DEST_PATH, OUTPUT_PATH.
    - Depends on pandas and dotenv.
"""



BASE_PATH = Path(get_env_path("BASE_PATH"))
DATA_FILE_PATH = BASE_PATH / get_env_path("DATA_FILE_PATH")
BASE_DEST_PATH = BASE_PATH / get_env_path("BASE_DEST_PATH")
OUTPUT_PATH = BASE_PATH / get_env_path("OUTPUT_PATH")

# Read the Excel file
df = pd.read_excel(DATA_FILE_PATH)
date_cols = [col for col in df.columns if col != "Grape ID"]

paths = []
for _, row in df.iterrows():
    grape_id = row["Grape ID"]
    crack_idx = None
    for i, date in enumerate(date_cols):
        val = str(row[date]).strip()
        if val and ("1" in val):
            crack_idx = i
            break
    if crack_idx is not None and crack_idx > 0:
        crack_date = date_cols[crack_idx]
        prev_date = date_cols[crack_idx - 1]
        # Type 30 (crack date)
        crack_path = BASE_DEST_PATH / str(grape_id) / str(crack_date)
        paths.append(str(crack_path))
        # Type 20 (previous date)
        prev_path = BASE_DEST_PATH / str(grape_id) / str(prev_date)
        paths.append(str(prev_path))

# Save paths to txt file
txt_output = OUTPUT_PATH / "paths.txt"
txt_output.parent.mkdir(parents=True, exist_ok=True)
with open(txt_output, "w", encoding="utf-8") as f:
    for path in paths:
        f.write(f"{path}\n")

print(f"Paths saved to: {txt_output}")
