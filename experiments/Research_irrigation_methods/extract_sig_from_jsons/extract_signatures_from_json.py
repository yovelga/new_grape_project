"""
Description:
    Extracts hyperspectral signatures from JSON files and corresponding mask images, saving results to a Parquet file.

Main Functionality:
    - Iterates through JSON files and extracts raw hyperspectral signatures for masked pixels.
    - Loads mask and HSI cube data, matches coordinates, and writes results to Parquet.
    - Handles environment variables for input/output paths.

Usage Notes:
    - Requires .env file with BASE_PATH, MASKS_DIR, OUTPUT_MASKS_PATH, and OUTPUT_PARQUET_PATH.
    - Depends on numpy, pandas, spectral, PIL, pyarrow, tqdm, and dotenv.
"""

import os
import json
import numpy as np
import pandas as pd
from spectral import io as spectral_io
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

# Load environment variables from .env file
load_dotenv()


def extract_signatures_to_parquet(json_dir: str, masks_dir: str, output_parquet_path: str):
    """
    Iterates through a directory of JSON files, extracts raw hyperspectral signatures
    for all masked pixels, and writes them iteratively to a Parquet file.

    Args:
        json_dir (str): Path to the directory containing the JSON files.
        masks_dir (str): Path to the directory containing the mask image files.
        output_parquet_path (str): Path to the output Parquet file.
    """
    print(f"Starting signature extraction from: {json_dir}")
    print(f"Output will be saved to: {output_parquet_path}")

    # Ensure output directory exists and remove old file if it exists
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Removing existing output file: {output_path}")
        output_path.unlink()

    json_files = sorted(list(Path(json_dir).glob('*.json')))
    if not json_files:
        print("No JSON files found in the specified directory.")
        return

    parquet_writer = None

    try:
        for json_path in tqdm(json_files, desc="Processing JSON files"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # --- 1. Extract metadata & paths ---
                image_path = Path(data.get("image_path"))
                mask_filename = Path(data.get("mask_path")).name
                mask_path = Path(masks_dir) / mask_filename

                if not image_path.exists() or not mask_path.exists():
                    tqdm.write(f"⚠️ Skipping {json_path.name}: Image or Mask not found.")
                    continue

                # --- 2. Load Mask and HSI Cube ---
                with Image.open(mask_path) as mask_img:
                    mask = np.array(mask_img.convert("L"))
                mask_coords = np.argwhere(mask > 0)

                if mask_coords.shape[0] == 0:
                    continue

                results_dir = image_path.parent / "results"
                hdr_files = list(results_dir.glob('*.hdr'))
                if not hdr_files:
                    tqdm.write(f"⚠️ Skipping {json_path.name}: No .hdr file.")
                    continue

                cube = spectral_io.envi.open(str(hdr_files[0])).load().astype(np.float32)

                # --- 3. Extract signatures ---
                signatures = []
                cube_h, cube_w, num_bands = cube.shape
                for y_mask, x_mask in mask_coords:
                    hsi_row, hsi_col = cube_w - x_mask - 1, y_mask
                    if 0 <= hsi_row < cube_w and 0 <= hsi_col < cube_h:
                        signatures.append(cube[hsi_col, hsi_row, :])

                if not signatures:
                    continue

                # --- 4. Prepare DataFrame ---
                df_sigs = pd.DataFrame(signatures, columns=[f'band_{i}' for i in range(num_bands)])
                df_sigs['date'] = data.get("date", "N/A")
                df_sigs['cluster_id'] = data.get("cluster_id", "N/A")
                df_sigs['irrigation_color'] = data.get("irrigation_color", "N/A")
                df_sigs['source_json'] = json_path.name

                # --- 5. Convert to Arrow Table and Write ---
                table = pa.Table.from_pandas(df_sigs, preserve_index=False)

                if parquet_writer is None:
                    # On the first successful file, create the writer with the table's schema
                    parquet_writer = pq.ParquetWriter(output_path, table.schema)

                parquet_writer.write_table(table)

            except Exception as e:
                tqdm.write(f"❌ Error processing {json_path.name}: {e}")
                continue

    finally:
        # --- 6. IMPORTANT: Close the writer to save the file ---
        if parquet_writer:
            parquet_writer.close()
            print(f"\n✅ Processing complete! All extracted signatures saved to: {output_path}")
        else:
            print("\nNo data was written to the output file.")


if __name__ == "__main__":
    BASE_DIR_ENV = os.getenv('BASE_PATH')
    if not BASE_DIR_ENV:
        sys.exit("Error: BASE_PATH environment variable not set.")
    BASE_DIR = Path(BASE_DIR_ENV)

    MASKS_DIR_ENV = os.getenv('MASKS_DIR')
    if not MASKS_DIR_ENV:
        sys.exit("Error: MASKS_DIR environment variable not set.")
    MASKS_DIR = BASE_DIR / MASKS_DIR_ENV

    OUTPUT_MASKS_PATH_ENV = os.getenv('OUTPUT_MASKS_PATH')
    if not OUTPUT_MASKS_PATH_ENV:
        sys.exit("Error: OUTPUT_MASKS_PATH environment variable not set.")
    JSON_DIR = BASE_DIR / OUTPUT_MASKS_PATH_ENV / 'jsons'

    OUTPUT_FILE_ENV = os.getenv('OUTPUT_PARQUET_PATH')
    if not OUTPUT_FILE_ENV:
        sys.exit("Error: OUTPUT_PARQUET_PATH environment variable not set.")

    parquet_output_path = BASE_DIR / OUTPUT_FILE_ENV

    for path, name in [(MASKS_DIR, "MASKS_DIR"), (JSON_DIR, "JSON_DIR")]:
        if not path.exists():
            sys.exit(f"Error: {name} does not exist: {path}")

    extract_signatures_to_parquet(
        json_dir=str(JSON_DIR),
        masks_dir=str(MASKS_DIR),
        output_parquet_path=str(parquet_output_path),
    )