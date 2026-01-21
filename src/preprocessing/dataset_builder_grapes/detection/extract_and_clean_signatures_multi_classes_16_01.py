"""
extract_and_clean_grape_signatures.py

Extracts grape signatures from JSON/mask/image data and optionally cleans them using Mahalanobis outlier detection.

Parameters:
- CLEAN_DATA: Set True to clean signatures, False to only extract.
- P_LOSS: Fraction of worst outliers to remove if cleaning.

Outputs:
- all_origin_signatures_results.csv: All extracted signatures.
- all_classes_cleaned_<date>.csv: Cleaned signatures (if CLEAN_DATA is True).
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.covariance import MinCovDet
from spectral import io as spectral_io
from PIL import Image
from datetime import datetime
import logging
import pathlib
import json
import matplotlib.pyplot as plt

# --- USER PARAMETERS ---
CLEAN_DATA = False  # Set to True to clean, False to only extract
P_LOSS = 0.00      # Fraction of worst outliers to remove
SUBSAMPLE = 20000  # rows for MinCovDet fitting
SUPPORT = 0.8      # support_fraction for MinCovDet

# --- CLASS DEFINITIONS ---
# All class folders to process (if they exist)
ALL_CLASS_FOLDERS = [
    "BACKGROUND",
    "BRANCH",
    "BURNT_PIXEL",
    "CRACK",
    "IRON",
    "LEAF",
    "PLASTIC",
    "REGULAR",
    "TRIPOD",
    "WHITE_REFERENCE",
]

# 3-class mapping: folder_name -> (label_3class_name, label_3class_id)
THREE_CLASS_MAPPING = {
    "REGULAR": ("REGULAR", 1),
    "CRACK": ("CRACK", 2),
    # All others map to "not_grape"
}
NOT_GRAPE_LABEL = ("not_grape", 3)

# --- GLOBALS ---
today_str = datetime.now().strftime("%Y-%m-%d")

# --- PATHS ---
SAM2_RESULTS_BASE = r"C:\Users\yovel\Desktop\Grape_Project\ui\pixel_picker\sam2_results"
RAW_EXPORT_DIR = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data"
RESULTS_CSV_PATH_multiclass = os.path.join(RAW_EXPORT_DIR, f"all_origin_signatures_results_multiclass_{today_str}.csv")
RESULTS_CSV_PATH_3class = os.path.join(RAW_EXPORT_DIR, f"all_origin_signatures_results_3class_{today_str}.csv")
OUTPUT_DIR = pathlib.Path(rf"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\dataset\cleaned_{P_LOSS}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HDR_PATH = r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\WL.hdr"

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- STAGE 1: EXTRACT SIGNATURES ---

def get_3class_labels(original_label):
    """Map original label to 3-class labels."""
    if original_label in THREE_CLASS_MAPPING:
        return THREE_CLASS_MAPPING[original_label]
    return NOT_GRAPE_LABEL

def extract_all_signatures_from_class_folders(base_dir, class_folders, output_multiclass_csv, output_3class_csv):
    """
    Extract signatures from all class folders under base_dir.
    Generates two CSVs:
    1. Multi-class CSV with original labels
    2. 3-class CSV with mapped labels (REGULAR=1, CRACK=2, not_grape=3)
    """
    all_sigs, all_meta = [], []
    total_extracted = 0
    wavelengths = None  # Will be extracted from first valid HSI cube

    # Build list of (json_dir, label) pairs from class folders
    json_dirs_labels = []
    for class_folder in class_folders:
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            json_dirs_labels.append((class_path, class_folder))
            logging.info("Found class folder: %s", class_path)
        else:
            logging.warning("Class folder not found, skipping: %s", class_path)

    for json_dir, label in json_dirs_labels:
        logging.info("Processing directory: %s (label=%s)", json_dir, label)
        try:
            dir_contents = sorted(os.listdir(json_dir))
        except Exception as e:
            logging.warning("Failed to list directory %s: %s", json_dir, e)
            continue

        for fn in dir_contents:
            if not fn.lower().endswith(".json"):
                continue
            json_path = os.path.join(json_dir, fn)
            logging.info("Processing JSON: %s", json_path)
            per_file_count = 0
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logging.warning("Failed to read JSON %s: %s", json_path, e)
                continue
            image_path = data.get("image_path")
            mask_path = data.get("mask_path")
            if mask_path and "\\pixel_picker\\sam2_results" in mask_path and not "\\ui\\pixel_picker\\sam2_results" in mask_path:
                mask_path = mask_path.replace("\\pixel_picker\\sam2_results", "\\ui\\pixel_picker\\sam2_results")
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                logging.warning(
                    "Missing image or mask for %s. image: %s exists=%s, mask: %s exists=%s",
                    fn,
                    image_path,
                    os.path.exists(image_path),
                    mask_path,
                    os.path.exists(mask_path),
                )
                continue
            rgb = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            coords = np.column_stack(np.where(mask > 0))
            if coords.size == 0:
                logging.info("No masked pixels found in %s", fn)
                continue
            hs_dir = os.path.dirname(image_path)
            results_dir = os.path.join(hs_dir, "results")
            if not os.path.isdir(results_dir):
                logging.warning("Results directory missing: %s", results_dir)
                continue
            hdr_file = next((f for f in os.listdir(results_dir) if f.lower().endswith(".hdr")), None)
            if not hdr_file:
                logging.warning("No HDR file found in results dir: %s", results_dir)
                continue
            hdr_path = os.path.join(results_dir, hdr_file)
            try:
                hsi_img = spectral_io.envi.open(hdr_path)
                cube = hsi_img.load().astype(np.float32)

                # Extract wavelengths from metadata if not already extracted
                if wavelengths is None:
                    hsi_metadata = hsi_img.metadata
                    wl_from_meta = hsi_metadata.get("wavelength") or hsi_metadata.get("wavelengths")
                    if wl_from_meta:
                        try:
                            wavelengths = [float(w) for w in wl_from_meta]
                            logging.info("Extracted %d wavelengths from ENVI header", len(wavelengths))
                        except Exception as e:
                            logging.warning("Failed to parse wavelengths: %s", e)
                    else:
                        logging.warning("No wavelength metadata found in ENVI header")
            except Exception as e:
                logging.warning("Failed to load HSI cube %s: %s", hdr_path, e)
                continue
            mh, mw = mask.shape
            ch, cw, _ = cube.shape
            if (mh, mw) == (cw, ch):
                cube = cube.transpose(0, 1, 2)
            elif (mh, mw) != (ch, cw):
                logging.debug("Mask and cube dimensions differ: mask=%s, cube=%s", (mh, mw), (ch, cw))
            for y0, x0 in coords:
                H, W, _ = cube.shape
                hsi_row = W - x0 - 1
                hsi_col = y0
                raw_sig = cube[hsi_row, hsi_col, :]
                all_sigs.append(raw_sig)
                all_meta.append({
                    "json_file": fn,
                    "hs_dir": hs_dir,
                    "x": int(x0),
                    "y": int(y0),
                    "timestamp": datetime.now().isoformat(),
                    "mask_path": mask_path,
                    "label": label,
                })
                per_file_count += 1
                total_extracted += 1
            logging.info("Extracted %d signatures from %s", per_file_count, fn)

    # Create column names using wavelengths if available
    if all_sigs:
        if wavelengths and len(wavelengths) == all_sigs[0].shape[0]:
            column_names = [f"{wl:.2f}nm" for wl in wavelengths]
            logging.info("Using wavelength-based column names")
        else:
            column_names = [f"band_{i}" for i in range(all_sigs[0].shape[0])]
            logging.warning("Using band index column names (wavelengths not available or mismatch)")
        df_sigs = pd.DataFrame(all_sigs, columns=column_names)
    else:
        df_sigs = None
    df_meta = pd.DataFrame(all_meta) if all_meta else None
    full_df = pd.concat([df_meta, df_sigs], axis=1) if df_sigs is not None and df_meta is not None else None

    if full_df is not None:
        os.makedirs(os.path.dirname(output_multiclass_csv), exist_ok=True)

        # Save multi-class CSV (original labels)
        full_df.to_csv(output_multiclass_csv, index=False)
        logging.info("Saved multi-class signatures to %s (total %d)", output_multiclass_csv, total_extracted)

        # Create 3-class CSV with mapped labels
        df_3class = full_df.copy()
        df_3class["label_3class_name"] = df_3class["label"].apply(lambda x: get_3class_labels(x)[0])
        df_3class["label_3class_id"] = df_3class["label"].apply(lambda x: get_3class_labels(x)[1])

        os.makedirs(os.path.dirname(output_3class_csv), exist_ok=True)
        df_3class.to_csv(output_3class_csv, index=False)
        logging.info("Saved 3-class signatures to %s (total %d)", output_3class_csv, total_extracted)
    else:
        logging.info("No signatures extracted. Nothing to save.")
    return full_df

# --- STAGE 2: OPTIONAL CLEANING ---
def load_wavelengths(hdr_path, start=0, end=203):
    try:
        hdr = spectral_io.envi.read_envi_header(hdr_path)
        wavelengths = hdr.get("wavelength", None)
        if wavelengths is None:
            raise ValueError("No 'wavelength' field in HDR header.")
        waves = [float(w) for w in wavelengths]
        return waves[start : end + 1]
    except Exception as e:
        print(f"⚠️ Failed to read wavelengths: {e}")
        return list(range(start, end + 1))

def mahalanobis_mask(X, p=P_LOSS, subsample=SUBSAMPLE, support=SUPPORT):
    if 0 < subsample < X.shape[0]:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X
    mcd = MinCovDet(support_fraction=support, random_state=0)
    mcd.fit(X_fit)
    center = mcd.location_
    cov_inv = mcd.precision_
    d2 = np.empty(X.shape[0], dtype=float)
    for i in tqdm(range(0, X.shape[0], 50000), desc="Mahalanobis dist"):
        delta = X[i : i + 50000] - center
        d2[i : i + 50000] = np.einsum("ij,jk,ik->i", delta, cov_inv, delta)
    thresh = np.percentile(d2, 100 * (1 - p))
    mask = d2 <= thresh
    return mask, center, d2

def plot_removed_signatures(removed_df, bands, n=100, title="Filtered-out signatures"):
    sample = removed_df[bands].iloc[:n].values
    plt.figure(figsize=(8, 3))
    for sig in sample:
        plt.plot(sig, alpha=0.6)
    plt.title(title or f"{n} filtered-out signatures (first {n})")
    plt.xlabel("band")
    plt.ylabel("value")
    plt.tight_layout()
    plt.show()

def plot_class_mean_std(df, bands, class_name, wavelengths=None):
    data = df[bands].values
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    x = wavelengths if wavelengths is not None else np.arange(len(mean))
    plt.figure(figsize=(10, 4))
    plt.plot(x, mean, label=f"Mean: {class_name}", color="blue")
    plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.2, label="±1 std")
    plt.title(f"Mean ±1 std per wavelength: {class_name}")
    plt.xlabel("Wavelength" if wavelengths is not None else "Band index")
    plt.ylabel("Reflectance value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def clean_all_classes(df, hdr_path):
    wavelengths = load_wavelengths(hdr_path, start=0, end=203)
    cleaned_frames = []
    for class_name in df['label'].unique():
        class_df = df[df['label'] == class_name].copy()
        # Support both wavelength-based and band-based column names
        band_cols = [c for c in class_df.columns if c.endswith("nm") or c.startswith("band_")]
        if not band_cols:
            logging.warning(f"No spectral columns found for class {class_name}.")
            continue

        X = class_df[band_cols].values.astype(np.float32)
        mask, center, d2 = mahalanobis_mask(X)
        class_df["is_outlier"] = ~mask
        cleaned_frames.append(class_df[mask])
        # Plot filtered-out signatures for this class
        removed_df = class_df[~mask]
        if not removed_df.empty:
            plot_removed_signatures(removed_df, band_cols, n=100, title=f"Filtered-out signatures: {class_name}")
        # Plot mean ±1 std for cleaned signatures
        plot_class_mean_std(class_df[mask], band_cols, class_name, wavelengths)
        logging.info(f"Cleaned {class_name}: {mask.sum()} inliers, {len(mask)-mask.sum()} outliers removed.")
    if cleaned_frames:
        all_cleaned = pd.concat(cleaned_frames, ignore_index=True)
        return all_cleaned
    else:
        return None

if __name__ == "__main__":
    logging.info("Starting signature extraction run.")
    logging.info("Multi-class output: %s", RESULTS_CSV_PATH_multiclass)
    logging.info("3-class output: %s", RESULTS_CSV_PATH_3class)

    df_all = extract_all_signatures_from_class_folders(
        base_dir=SAM2_RESULTS_BASE,
        class_folders=ALL_CLASS_FOLDERS,
        output_multiclass_csv=RESULTS_CSV_PATH_multiclass,
        output_3class_csv=RESULTS_CSV_PATH_3class
    )

    if CLEAN_DATA and df_all is not None:
        logging.info("Starting cleaning stage...")
        cleaned_df = clean_all_classes(df_all, HDR_PATH)
        if cleaned_df is not None:
            cleaned_path = OUTPUT_DIR / f"all_classes_cleaned_{today_str}.csv"
            cleaned_df.to_csv(cleaned_path, index=False)
            logging.info(f"✅ Saved all cleaned classes to {cleaned_path} ({len(cleaned_df)} rows)")
        else:
            logging.info("No cleaned dataframes to save.")
    else:
        logging.info("Cleaning stage skipped or no data extracted.")
