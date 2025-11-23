import os
import numpy as np
import pandas as pd
import spectral.io.envi as envi
from tqdm import tqdm
from PIL import Image


def normalize_signature_minmax(signature: np.ndarray) -> np.ndarray:
    signature_min = signature.min()
    signature_max = signature.max()
    if signature_max != signature_min:
        return (signature - signature_min) / (signature_max - signature_min)
    else:
        return np.zeros_like(signature)


def run_extraction_to_parquet(masks_dir, hsi_hdr_path, label_dict, output_dir):
    print("Loading HSI cube...")
    hsi = envi.open(hsi_hdr_path)
    hsi_cube = hsi.load().astype(np.float32)
    print(f"Loaded HSI cube with shape {hsi_cube.shape}")

    all_signatures = []
    all_metadata = []

    path_parts = masks_dir.split(os.sep)
    cluster_id = path_parts[-3]  # e.g., '2_57'
    date = path_parts[-2]  # e.g., '05.09.24'

    for mask_file in tqdm(sorted(os.listdir(masks_dir))):
        if not mask_file.lower().endswith(".tif"):
            continue

        mask_path = os.path.join(masks_dir, mask_file)
        label = label_dict.get(mask_file)
        if label != "Grape":
            continue

        try:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
        except Exception as e:
            print(f"⚠️ Failed to open mask {mask_file}: {e}")
            continue

        if mask.shape[:2] != hsi_cube.shape[:2]:
            print(f"⚠️ Skipping {mask_file}: mask and HSI dimensions do not match.")
            continue

        mask_pixels = np.column_stack(np.where(mask > 0))
        np.random.shuffle(mask_pixels)

        sampling_ratio = 0.01
        # max_pixels_per_mask = np.inf
        min_pixels_per_mask = 10

        # n_pixels = int(len(mask_pixels) * sampling_ratio)
        n_pixels = int(1)
        # n_pixels = max(min_pixels_per_mask, min(n_pixels, max_pixels_per_mask))
        # n_pixels = max(min_pixels_per_mask, n_pixels)

        sampled_pixels = mask_pixels[:n_pixels]

        for y, x in sampled_pixels:
            spectrum = hsi_cube[y, x, :]
            norm_spectrum = normalize_signature_minmax(spectrum)
            all_signatures.append(norm_spectrum)
            all_metadata.append(
                {"date": date, "id": cluster_id, "x": x, "y": y, "mask_file": mask_file}
            )

    if not all_signatures:
        print("⚠️ No spectral signatures were collected – skipping Parquet save.")
        return

    df = pd.DataFrame(
        all_signatures, columns=[f"band_{i}" for i in range(hsi_cube.shape[2])]
    )
    meta_df = pd.DataFrame(all_metadata)
    df = pd.concat([meta_df, df], axis=1)

    hsi_name = (
        os.path.basename(hsi_hdr_path).replace("REFLECTANCE_", "").replace(".hdr", "")
    )
    date = hsi_name.split("_")[0]
    # output_dir_for_date = os.path.join(output_dir,date)
    output_dir_for_date = output_dir
    os.makedirs(output_dir_for_date, exist_ok=True)
    # parquet_path = os.path.join(output_dir_for_date, f"signatures_all_{date}_{cluster_id}.parquet")
    parquet_path = os.path.join(output_dir_for_date, f"signatures_4_PCA.csv")

    if os.path.exists(parquet_path):
        existing_df = pd.read_csv(parquet_path)  # Use read_parquet, not read_csv
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(parquet_path, index=False)
    print(
        f"✅Last parquet saved {len(df)} samples to {parquet_path} ,date: {date}, cluster_id: {cluster_id}"
    )
