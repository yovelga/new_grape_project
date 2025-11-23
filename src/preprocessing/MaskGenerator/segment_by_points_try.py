import sys, os, json, pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# make sure Python can import SAM-2 helpers (edit if your path differs)
SAM2_MODULES = (
    r"C:\Users\yovel\Desktop\Grape_Project\MaskGenerator\segment-anything-2\sam2"
)
sys.path.append(os.path.abspath(SAM2_MODULES))

from segment_object_module import create_point_segmenter
from mask_generator_module import initial_settings, initialize_sam2_predictor

# --------------------------------------------------------------------------
# --- user settings --------------------------------------------------------
image_path = r"/all_raw_data\1_04\01.08.24\HS\2024-08-01_006.png"  # <-- change
points = [(150, 220)]  # <-- change

# where to save results if you choose “yes”
JSON_DIR = (
    r"C:\Users\yovel\Desktop\Grape_Project\dataset_builder_grapes\output\cracks\jsons"
)
MASK_DIR = (
    r"C:\Users\yovel\Desktop\Grape_Project\dataset_builder_grapes\output\cracks\masks"
)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
# --------------------------------------------------------------------------

# load predictor & segment
initial_settings()
predictor = initialize_sam2_predictor()
segmenter = create_point_segmenter(predictor)

image_rgb, mask_bool = segmenter.segment_object(image_path, points)

# show overlay
overlay = image_rgb.copy()
overlay[mask_bool] = overlay[mask_bool] * 0.7 + np.array([255, 0, 0]) * 0.3
plt.figure(figsize=(10, 10))
plt.imshow(overlay.astype(np.uint8))
plt.title("Segmented Object")
plt.axis("off")
plt.show()

# --------------------------------------------------------------------------
# ask the user
ans = input("\nSave mask & JSON?  [y/N]  ").strip().lower()
if ans in {"y", "yes"}:
    stem = pathlib.Path(image_path).stem
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")
    json_path = os.path.join(JSON_DIR, f"{stem}.json")

    # save mask (as 0/255 PNG)
    cv2.imwrite(mask_path, (mask_bool.astype(np.uint8) * 255))

    # save JSON with absolute paths
    record = {
        "image_path": os.path.abspath(image_path),
        "mask_path": os.path.abspath(mask_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"✅ saved\n  mask ➜ {mask_path}\n  json ➜ {json_path}")
else:
    print("ℹ️  nothing saved.")
