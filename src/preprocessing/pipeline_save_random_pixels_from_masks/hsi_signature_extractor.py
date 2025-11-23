# hsi_signature_extractor.py

import os
import numpy as np
import tifffile as tiff
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image


# פונקציה לטעינת קובץ HSI מפורמט .dat
def load_hsi(dat_path, samples=512, lines=512, bands=204):
    """
    טוענת קובץ HSI (.dat) ומחזירה מערך numpy עם המימדים הנתונים.
    """
    data = np.fromfile(dat_path, dtype=np.float32)
    hsi = data.reshape((lines, samples, bands))
    return hsi


def extract_spectral_signatures(mask_tiff_path, hsi_dat_path, output_csv):
    """
    - טוענת מסכת TIFF (שהתקבלה מתהליך SAM) וקובץ HSI (.dat).
    - הקובץ של המסכה מכיל מטא־דטה עם bounding box.
    - חותכים את אזור המסכה מתוך קובץ ה-HSI.
    - עבור כל פיקסל בתוך המסכה (אשר המסכה True), יוצרים רשומה:
         image_name, x, y, signature (וקטור ספקטרלי)
    - התוצאות נשמרות לקובץ CSV.
    """
    # טען מסכת TIFF והפק מטא־דטה
    mask, meta = tiff.imread(mask_tiff_path, asdict=True)
    metadata = json.loads(meta.get("ImageDescription", "{}"))
    bbox = metadata.get("original_bbox", [0, 0, 0, 0])
    x_min, y_min, x_max, y_max = map(int, bbox)

    # טען קובץ HSI
    hsi = load_hsi(hsi_dat_path)

    # חתוך את אזור ה-HSI לפי ה-bbox
    hsi_crop = hsi[y_min:y_max, x_min:x_max, :]  # (crop_H, crop_W, bands)

    # טען את המסכה כמערך numpy (אם היא לא כבר)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask_crop = mask[y_min:y_max, x_min:x_max].astype(bool)

    # עבור כל פיקסל בתוך המסכה, איסוף חתימה
    results = []
    crop_h, crop_w = mask_crop.shape
    for i in tqdm(range(crop_h), desc="Extracting signatures", unit="row"):
        for j in range(crop_w):
            if mask_crop[i, j]:
                global_x = x_min + j
                global_y = y_min + i
                signature = hsi_crop[i, j, :]  # וקטור ספקטרלי
                results.append(
                    {
                        "image": metadata.get("image_name", "unknown"),
                        "x": int(global_x),
                        "y": int(global_y),
                        "signature": signature.tolist(),
                    }
                )
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved spectral signatures to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_tiff", type=str, required=True, help="Path to mask TIFF file"
    )
    parser.add_argument(
        "--hsi_dat", type=str, required=True, help="Path to corresponding HSI .dat file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save spectral signatures CSV",
    )
    args = parser.parse_args()

    extract_spectral_signatures(args.mask_tiff, args.hsi_dat, args.output_csv)
