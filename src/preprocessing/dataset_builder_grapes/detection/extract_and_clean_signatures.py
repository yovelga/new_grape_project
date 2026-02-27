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
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# --- USER PARAMETERS ---
CLEAN_DATA = True  # Set to True to clean, False to only extract
P_LOSS = 0.05      # Fraction of worst outliers to remove
SUBSAMPLE = 20000  # rows for MinCovDet fitting
SUPPORT = 0.8      # support_fraction for MinCovDet

# --- PATHS ---
JSON_DIR_branch = r"/ui/pixel_picker/sam2_results/old/BRANCH"
JSON_DIR_regular = r"/ui/pixel_picker/sam2_results/old/REGULAR"
JSON_DIR_cracked = r"/ui/pixel_picker/sam2_results/old/CRACK"
JSON_DIR_plastic = r"/ui/pixel_picker/sam2_results/old/PLASTIC"
JSON_DIR_background = r"/ui/pixel_picker/sam2_results/old/BACKGROUND"
RAW_EXPORT_DIR = str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data")
RESULTS_CSV_PATH_all = os.path.join(RAW_EXPORT_DIR, "all_origin_signatures_results.csv")
OUTPUT_DIR = pathlib.Path(rFstr(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/dataset/cleaned_{P_LOSS}"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HDR_PATH = str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/WL.hdr")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- STAGE 1: EXTRACT SIGNATURES ---
def normalization_min_max(signature: np.ndarray) -> np.ndarray:
    return signature

def extract_all_signatures_from_multiple_json_dirs(json_dirs_labels, output_results_csv):
    all_sigs, all_meta = [], []
    total_extracted = 0
    for json_dir, label in json_dirs_labels:
        logging.info("Processing directory: %s (label=%s)", json_dir, label)
        for fn in sorted(os.listdir(json_dir)):
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
                cube = spectral_io.envi.open(hdr_path).load().astype(np.float32)
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
                norm_sig = normalization_min_max(raw_sig)
                all_sigs.append(norm_sig)
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
    df_sigs = pd.DataFrame(all_sigs, columns=[f"band_{i}" for i in range(all_sigs[0].shape[0])]) if all_sigs else None
    df_meta = pd.DataFrame(all_meta) if all_meta else None
    full_df = pd.concat([df_meta, df_sigs], axis=1) if df_sigs is not None and df_meta is not None else None
    if full_df is not None:
        os.makedirs(os.path.dirname(output_results_csv), exist_ok=True)
        full_df.to_csv(output_results_csv, index=False)
        logging.info("Saved all normalized signatures to %s (total %d)", output_results_csv, total_extracted)
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
        if class_name.upper() == "BACKGROUND":
            logging.info(f"Skipping cleaning for BACKGROUND class.")
            # Directly append all BACKGROUND rows without cleaning
            class_df = df[df['label'] == class_name].copy()
            class_df["is_outlier"] = False
            cleaned_frames.append(class_df)
            continue
        class_df = df[df['label'] == class_name].copy()
        band_cols = [c for c in class_df.columns if c.startswith("band_")]
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
    json_dirs_labels = [
        (JSON_DIR_branch, "BRANCH"),
        (JSON_DIR_plastic, "PLASTIC"),
        (JSON_DIR_regular, "REGULAR"),
        (JSON_DIR_cracked, "CRACK"),
        (JSON_DIR_background, "BACKGROUND"),
    ]
    logging.info("Starting signature extraction run. Output will be saved to: %s", RESULTS_CSV_PATH_all)
    df_all = extract_all_signatures_from_multiple_json_dirs(json_dirs_labels, RESULTS_CSV_PATH_all)
    if CLEAN_DATA and df_all is not None:
        logging.info("Starting cleaning stage...")
        cleaned_df = clean_all_classes(df_all, HDR_PATH)
        if cleaned_df is not None:
            today_str = datetime.now().strftime("%Y-%m-%d")
            cleaned_path = OUTPUT_DIR / f"all_classes_cleaned_{today_str}.csv"
            cleaned_df.to_csv(cleaned_path, index=False)
            logging.info(f"✅ Saved all cleaned classes to {cleaned_path} ({len(cleaned_df)} rows)")
        else:
            logging.info("No cleaned dataframes to save.")
    else:
        logging.info("Cleaning stage skipped or no data extracted.")
