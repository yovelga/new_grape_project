import os
import sys
import torch
from PIL import Image
from data_transforms import get_test_transforms
import classifier_inference

from extract_spectral_signatures import run_extraction_to_csv

# הוספת נתיבים לפרויקט
mask_module_path = "/storage/yovelg/Grape/MaskGenerator"
project_path = "/storage/yovelg/Grape/spectral_anomaly"
for path in [mask_module_path, project_path]:
    if path not in sys.path:
        sys.path.insert(0, path)
import mask_generator_module


def find_hsi_hdr(results_dir):
    """מחפש קובץ HSI שנגמר ב-.hdr"""
    for file in os.listdir(results_dir):
        if file.lower().endswith(".hdr"):
            return os.path.join(results_dir, file)
    raise FileNotFoundError(f"No HSI .hdr file found in {results_dir}")


def main():
    # נתיב בסיסי לכל התמונות וה-HSI
    base_path = "/storage/yovelg/Grape/data/2_01/05.09.24"

    images_path = os.path.join(base_path, "HS")  # תמונות RGB
    masks_output_path = os.path.join(base_path, "output")  # מסכות SAM
    results_path = os.path.join(base_path, "HS", "results")  # תיקיית תוצאות HSI

    # חיפוש קובץ ה-HSI באופן דינמי
    hsi_hdr_path = find_hsi_hdr(results_path)
    output_csv = "/storage/yovelg/Grape/spectral_anomaly/checks/output/CSV"

    # שלב 1: הרצת SAM והפקת מסכות
    print("🔹 Generating masks with SAM...")
    mask_generator_module.run(images_path, masks_output_path, images_path)

    # שלב 2: סיווג חתכים מהמסכות
    print("🔹 Classifying masks with EfficientNet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = classifier_inference.load_classifier_model()
    transform = get_test_transforms()

    label_dict = {}  # mask_file_name -> "Grape"/"Not Grape"
    for file in os.listdir(masks_output_path):
        if file.lower().endswith(".tif"):
            mask_path = os.path.join(masks_output_path, file)
            try:
                label, conf, image_name = classifier_inference.classify_crop_from_mask(
                    mask_path, images_path, model, transform, device
                )
                label_dict[file] = label
                print(f"[DEBUG] {file} → {label} ({conf:.2f})")
            except Exception as e:
                print(f"⚠️ Failed to classify {file}: {e}")

    # שלב 3: חילוץ חתימות ספקטרליות ושמירה ל-CSV
    print(f"🔹 Using HSI file: {hsi_hdr_path}")
    print("🔹 Extracting spectral signatures for 'Grape' masks...")
    run_extraction_to_csv(
        masks_dir=masks_output_path,
        hsi_hdr_path=hsi_hdr_path,
        label_dict=label_dict,
        output_dir=output_csv,
    )

    print("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
