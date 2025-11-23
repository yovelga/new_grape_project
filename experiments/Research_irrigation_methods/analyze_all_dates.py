"""
Description:
    Analyzes hyperspectral data to compute mean and standard deviation of NIR reflectance by irrigation color and date.

Main Functionality:
    - Loads spectral data from a Parquet file.
    - Removes outliers and computes statistics for NIR bands.
    - Displays results as trend tables.

Usage Notes:
    - Requires .env file with BASE_PATH, WAVELENGTHS_PATH, INPUT_PARQUET_PATH.
    - Depends on pandas, numpy, scipy, tqdm, dotenv.
"""

import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import importlib.util
from tqdm import tqdm
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()


def load_wavelengths_from_path(path_str: str) -> dict:
    """Dynamically loads the WAVELENGTHS dictionary from a given .py file path."""
    path = Path(path_str)
    if not path.is_file():
        sys.exit(f"Error: Wavelengths file not found at {path}")

    spec = importlib.util.spec_from_file_location("wavelengths_module", path)
    wavelengths_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wavelengths_module)

    if not hasattr(wavelengths_module, 'WAVELENGTHS'):
        sys.exit(f"Error: 'WAVELENGTHS' dictionary not found in {path}")

    return wavelengths_module.WAVELENGTHS


def remove_outliers(df_signatures: pd.DataFrame, percentile_to_keep=95) -> pd.DataFrame:
    """
    Removes outliers based on Euclidean distance from the mean signature.
    """
    if len(df_signatures) < 10:  # Don't filter if there are too few samples
        return df_signatures

    mean_sig = df_signatures.mean(axis=0).to_numpy().reshape(1, -1)
    distances = distance.cdist(df_signatures.to_numpy(), mean_sig, 'euclidean').flatten()
    threshold = np.percentile(distances, percentile_to_keep)

    return df_signatures[distances <= threshold]


def analyze_all_dates_stats(parquet_path: str, wavelengths_dict: dict, percentile_to_keep=95):
    """
    Analyzes all dates to find the mean and standard deviation of NIR reflectance
    for each irrigation color and displays them as trend tables after removing outliers.
    """
    print(f"Loading data from: {parquet_path}")
    if not Path(parquet_path).is_file():
        sys.exit(f"Error: Input Parquet file not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Convert date column to datetime objects
    df['date_dt'] = pd.to_datetime(df['date'], format='%d.%m.%y')

    # Identify band columns that fall within the 700-900 nm range
    nir_band_cols = []
    for band_index, wavelength in wavelengths_dict.items():
        if 700 <= wavelength <= 900:
            col_name = f"band_{band_index - 1}"
            if col_name in df.columns:
                nir_band_cols.append(col_name)

    if not nir_band_cols:
        sys.exit("Error: Could not find any band columns in the 700-900nm range.")

    print(f"Found {len(nir_band_cols)} bands in the 700-900nm NIR range for analysis.")

    # --- NEW: Apply outlier removal before any calculations ---
    print(f"\nApplying outlier removal (keeping {percentile_to_keep}th percentile of data for each group)...")

    filtered_dfs = []
    grouped_for_filtering = df.groupby(['date_dt', 'irrigation_color'])

    for name, group in tqdm(grouped_for_filtering, desc="Filtering outliers"):
        signatures = group[nir_band_cols]
        filtered_signatures = remove_outliers(signatures, percentile_to_keep=percentile_to_keep)
        # Keep the original rows that correspond to the filtered signatures
        filtered_group = group.loc[filtered_signatures.index]
        filtered_dfs.append(filtered_group)

    # Create a new, clean DataFrame from the filtered groups
    df_filtered = pd.concat(filtered_dfs)
    print("Outlier removal complete.")

    # --- All subsequent calculations are on the filtered DataFrame ---
    grouped = df_filtered.groupby(['date_dt', 'irrigation_color'])

    # --- 1. Calculate Mean Reflectance ---
    df_filtered['mean_nir'] = df_filtered[nir_band_cols].mean(axis=1)
    mean_results = df_filtered.groupby(['date_dt', 'irrigation_color'])['mean_nir'].mean()
    mean_trend_table = mean_results.unstack(level='irrigation_color').sort_index()
    mean_trend_table.index = mean_trend_table.index.strftime('%d.%m.%Y')

    # --- 2. Calculate Mean Standard Deviation ---
    std_results = grouped[nir_band_cols].std().mean(axis=1)
    std_trend_table = std_results.unstack(level='irrigation_color').sort_index()
    std_trend_table.index = std_trend_table.index.strftime('%d.%m.%Y')

    # --- 3. Print Results ---
    print("\n--- ANALYSIS RESULTS (AFTER OUTLIER REMOVAL) ---")

    print("\nTable 1: Mean Reflectance (Higher is generally better)")
    print("----------------------------------------------------")
    print(mean_trend_table.to_string())

    print("\n\nTable 2: Mean Standard Deviation (Lower means more uniform)")
    print("----------------------------------------------------------")
    print(std_trend_table.to_string())

    print("\n--- END OF RESULTS ---")

    # --- Export results to CSV ---
    export_trend_tables(mean_trend_table, std_trend_table, Path(os.getenv('BASE_PATH')))

    # --- Print irrigation color descriptions ---
    print("\nIrrigation Color Descriptions:")
    for color in mean_trend_table.columns:
        print(f"{color}: {get_irrigation_color_description(color)}")


def get_irrigation_color_description(color: str) -> str:
    """
    Returns a description for each irrigation color.
    """
    descriptions = {
        "yellow": "Low irrigation, increasing over time.",
        "blue": "High irrigation throughout.",
        "white": "Low irrigation throughout.",
        "red": "High irrigation, decreasing over time."
    }
    return descriptions.get(color.lower(), "No description available.")


def plot_trend_table(trend_table, ylabel, title, output_path):
    """
    Plots the trend table (mean or std) by date for each irrigation color.
    """
    plt.figure(figsize=(10, 6))
    for color in trend_table.columns:
        plt.plot(trend_table.index, trend_table[color], marker='o', label=f"{color} ({get_irrigation_color_description(color)})")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def export_trend_tables(mean_trend_table, std_trend_table, base_dir):
    """
    Exports the mean and std trend tables to CSV files and adds irrigation color descriptions.
    Also creates and saves plots for both tables.
    """
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    mean_csv_path = results_dir / "mean_nir_trend_table.csv"
    std_csv_path = results_dir / "std_nir_trend_table.csv"
    desc_csv_path = results_dir / "irrigation_color_descriptions.csv"
    mean_plot_path = results_dir / "mean_nir_trend_plot.png"
    std_plot_path = results_dir / "std_nir_trend_plot.png"

    mean_trend_table.to_csv(mean_csv_path)
    std_trend_table.to_csv(std_csv_path)

    # Create a DataFrame for irrigation color descriptions
    desc_df = pd.DataFrame({
        "irrigation_color": mean_trend_table.columns,
        "description": [get_irrigation_color_description(c) for c in mean_trend_table.columns]
    })
    desc_df.to_csv(desc_csv_path, index=False)

    print(f"\nExported mean trend table to: {mean_csv_path}")
    print(f"Exported std trend table to: {std_csv_path}")
    print(f"Exported irrigation color descriptions to: {desc_csv_path}")

    # Plot and save figures
    plot_trend_table(mean_trend_table, ylabel="Mean NIR Reflectance", title="Mean NIR Reflectance by Date and Irrigation Color", output_path=mean_plot_path)
    plot_trend_table(std_trend_table, ylabel="Mean NIR Std", title="Mean NIR Std by Date and Irrigation Color", output_path=std_plot_path)


if __name__ == "__main__":
    # Load configuration from .env file
    BASE_PATH_ENV = os.getenv('BASE_PATH')
    WAVELENGTHS_PATH_REL = os.getenv('WAVELENGTHS_PATH')
    INPUT_PARQUET_PATH_REL = os.getenv('INPUT_PARQUET_PATH')

    # Validate and build absolute paths
    if not BASE_PATH_ENV:
        sys.exit("Error: BASE_PATH environment variable is not set.")

    if not all([WAVELENGTHS_PATH_REL, INPUT_PARQUET_PATH_REL]):
        sys.exit("Error: Required path variables are missing from .env file (WAVELENGTHS_PATH, INPUT_PARQUET_PATH).")

    BASE_DIR = Path(BASE_PATH_ENV)

    abs_wavelengths_path = BASE_DIR / WAVELENGTHS_PATH_REL
    abs_input_path = BASE_DIR / INPUT_PARQUET_PATH_REL

    # Load wavelengths dynamically
    WAVELENGTHS = load_wavelengths_from_path(str(abs_wavelengths_path))

    # Execution
    analyze_all_dates_stats(
        parquet_path=str(abs_input_path),
        wavelengths_dict=WAVELENGTHS,
        percentile_to_keep=0.80  # You can change this value as needed
    )