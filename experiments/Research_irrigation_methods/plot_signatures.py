"""
Description:
    Plots mean and standard deviation of spectral signatures by irrigation color and date from Parquet data.

Main Functionality:
    - Loads spectral data and wavelength mapping.
    - Removes outliers and generates grid plots for raw and normalized data.
    - Saves plots to image files.

Usage Notes:
    - Requires .env file with BASE_PATH, WAVELENGTHS_PATH, INPUT_PARQUET_PATH, OUTPUT_IMAGE_PATH.
    - Depends on pandas, numpy, matplotlib, tqdm, dotenv, scipy.
"""
from pathlib import Path
import pandas as pd

# --- NEW: Color mapping for plots ---
COLOR_MAP = {
    'BLUE': 'blue',
    'RED': 'red',
    'YELLOW': 'gold',  # Gold is more visible than yellow
    'WHITE': 'gray',
    'DEFAULT': 'purple'  # Fallback for any other color
}


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


def remove_outliers(df_signatures: pd.DataFrame, percentile_to_keep=90) -> pd.DataFrame:
    """
    Removes outliers based on Euclidean distance from the mean signature.
    """
    if len(df_signatures) < 10:
        return df_signatures
    mean_sig = df_signatures.mean(axis=0).to_numpy().reshape(1, -1)
    distances = distance.cdist(df_signatures.to_numpy(), mean_sig, 'euclidean').flatten()
    threshold = np.percentile(distances, percentile_to_keep)
    return df_signatures[distances <= threshold]


def plot_mean_std_grid(parquet_path: str, wavelengths_dict: dict, output_image_path: str, normalize: bool = False):
    """
    Loads spectral data and generates a grid of plots.
    If normalize is True, applies Min-Max scaling to each signature.
    """
    print(f"\n--- Generating plot for {'Normalized' if normalize else 'Raw'} data ---")
    print(f"Loading data from {parquet_path}...")
    if not Path(parquet_path).is_file():
        sys.exit(f"Error: Input Parquet file not found at {parquet_path}")

    df = pd.read_parquet(parquet_path).copy()  # Use a copy to avoid side effects

    # --- 1. Prepare Data and Metadata ---
    band_cols = [col for col in df.columns if col.startswith('band_')]
    band_cols.sort(key=lambda x: int(x.split('_')[1]))

    # --- NEW: Optional Min-Max Normalization ---
    if normalize:
        print("Applying Min-Max normalization to each signature...")
        signatures = df[band_cols]
        min_vals = signatures.min(axis=1)
        max_vals = signatures.max(axis=1)
        # Handle cases where min equals max to avoid division by zero
        range_vals = (max_vals - min_vals).replace(0, 1)
        df[band_cols] = signatures.sub(min_vals, axis=0).div(range_vals, axis=0)
        plot_title_suffix = 'Normalized'
    else:
        plot_title_suffix = 'Raw'

    wavelength_values = [wavelengths_dict.get(i + 1, 0) for i in range(len(band_cols))]

    irrigation_colors = sorted(df['irrigation_color'].unique())
    df['date_dt'] = pd.to_datetime(df['date'], format='%d.%m.%y')
    dates = sorted(df['date_dt'].unique())

    if not dates or not irrigation_colors:
        print("No data to plot.")
        return

    # --- 2. Set up the Plot Grid ---
    num_rows, num_cols = len(irrigation_colors), len(dates)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows),
                             sharex=True, squeeze=False)

    fig.suptitle(f'Mean {plot_title_suffix} Spectral Signatures by Irrigation and Date', fontsize=20, y=0.98)

    # --- 3. Iterate and Plot with TQDM ---
    plot_combinations = list(itertools.product(enumerate(irrigation_colors), enumerate(dates)))

    for (row_idx, color), (date_idx, date) in tqdm(plot_combinations, desc=f"Generating {plot_title_suffix} Plots"):
        col_idx = num_cols - 1 - date_idx
        ax = axes[row_idx, col_idx]

        subset = df[(df['irrigation_color'] == color) & (df['date_dt'] == date)]
        plot_color = COLOR_MAP.get(color, COLOR_MAP['DEFAULT'])  # Get color from map

        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            signatures = subset[band_cols]
            filtered_signatures = remove_outliers(signatures, percentile_to_keep=99)
            mean_sig = filtered_signatures.mean(axis=0).values
            std_sig = filtered_signatures.std(axis=0).values

            ax.plot(wavelength_values, mean_sig, label=f"Mean ({len(filtered_signatures)} pixels)", color=plot_color)
            ax.fill_between(wavelength_values, mean_sig - std_sig, mean_sig + std_sig, color=plot_color, alpha=0.15)
            ax.grid(True)
            ax.legend()

        # Set Y-axis limit for all plots (normalized or raw)
        ax.set_ylim(0, 1.0)

        date_str = pd.to_datetime(str(date)).strftime('%d.%m.%y')
        ax.set_title(f"{color} - {date_str}")
        if row_idx == num_rows - 1: ax.set_xlabel('Wavelength (nm)')
        if col_idx == 0: ax.set_ylabel('Reflectance' if not normalize else 'Normalized Reflectance')

    # --- 4. Final Touches and Saving ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving final plot to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    # Load configuration from .env file
    BASE_PATH_ENV = os.getenv('BASE_PATH')
    WAVELENGTHS_PATH_REL = os.getenv('WAVELENGTHS_PATH')
    INPUT_PARQUET_PATH_REL = os.getenv('INPUT_PARQUET_PATH')
    OUTPUT_IMAGE_PATH_REL = os.getenv('OUTPUT_IMAGE_PATH')

    # Validate and build absolute paths
    if not BASE_PATH_ENV:
        sys.exit("Error: BASE_PATH environment variable is not set.")
    if not all([WAVELENGTHS_PATH_REL, INPUT_PARQUET_PATH_REL, OUTPUT_IMAGE_PATH_REL]):
        sys.exit("Error: One or more relative path variables are missing from .env file.")

    BASE_DIR = Path(BASE_PATH_ENV)
    abs_wavelengths_path = BASE_DIR / WAVELENGTHS_PATH_REL
    abs_input_path = BASE_DIR / INPUT_PARQUET_PATH_REL

    # --- NEW: Create two different output paths ---
    base_output_path = BASE_DIR / OUTPUT_IMAGE_PATH_REL
    raw_output_path = base_output_path.with_name(f"{base_output_path.stem}_raw.png")
    normalized_output_path = base_output_path.with_name(f"{base_output_path.stem}_normalized.png")

    WAVELENGTHS = load_wavelengths_from_path(str(abs_wavelengths_path))

    # --- Execution: Run for both Raw and Normalized data ---
    # 1. Run for RAW data
    plot_mean_std_grid(
        parquet_path=str(abs_input_path),
        wavelengths_dict=WAVELENGTHS,
        output_image_path=str(raw_output_path),
        normalize=False
    )

    # 2. Run for NORMALIZED data
    plot_mean_std_grid(
        parquet_path=str(abs_input_path),
        wavelengths_dict=WAVELENGTHS,
        output_image_path=str(normalized_output_path),
        normalize=True
    )