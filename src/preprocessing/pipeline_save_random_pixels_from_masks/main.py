# main.py

import os
import argparse
from tqdm import tqdm
from mask_generator_module import (
    generate_mask_tiff,
    prepare_directories,
    ensure_sam2_repo,
    download_sam_weights,
)
from hsi_signature_extractor import extract_spectral_signatures
from PIL import Image
import tifffile as tiff
import torch
import json

# נייבא פונקציות מהמודול של הסיווג – יש להתאים את הנתיבים בהתאם
from classifier_inference import load_classifier_model, predict_label_and_confidence
from data_transforms import get_test_transforms


def check_mask_is_grape(rgb_image_path, mask_tiff_path, classifier, device):
    """
    טוען את קובץ המסכה, מחלץ את bounding box המקורי,
    מוציא את החתך המתאים מתמונת ה-RGB,
    ומריץ עליו את המודל המסווג.
    מחזיר True אם התווית היא "Grape", False אחרת.
    """
    # טען את המסכה עם מטא־דטה
    mask_data, meta = tiff.imread(mask_tiff_path, asdict=True)
    meta_json = meta.get("ImageDescription", "{}")
    metadata = json.loads(meta_json)
    bbox = metadata.get("original_bbox")
    if not bbox:
        print("No bounding box found in mask metadata.")
        return False
    x_min, y_min, x_max, y_max = map(int, bbox)
    # טען את תמונת ה-RGB כ-PIL Image
    rgb_img = Image.open(rgb_image_path).convert("RGB")
    # חתוך את האזור לפי ה-bounding box
    crop = rgb_img.crop((x_min, y_min, x_max, y_max))
    # הרץ את המודל המסווג על החתך – השתמש בטרנספורם בדיקה
    transform = get_test_transforms()
    label, conf = predict_label_and_confidence(classifier, crop, transform, device)
    print(
        f"Classification result for mask from {os.path.basename(rgb_image_path)}: {label} (conf: {conf:.2f})"
    )
    return label == "Grape"


def process_folder(
    rgb_folder,
    hsi_folder,
    masks_output_folder,
    signatures_output_folder,
    classifier,
    device,
    padding_px=10,
):
    """
    עבור כל תמונת RGB בתיקייה:
      - יוצר קובץ מסכה TIFF (באמצעות SAM) ושומר אותו.
      - בודק באמצעות מודל הסיווג האם המסכה מתאימה (כלומר, מדובר באזור ענבים).
      - אם כן, מחפש קובץ HSI תואם (לפי שם) ומריץ חילוץ חתימות ספקטרליות,
        ששומר את התוצאות לקובץ CSV.
    """
    rgb_images = [
        f
        for f in os.listdir(rgb_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not rgb_images:
        print("No RGB images found in the input folder.")
        return

    for image_file in tqdm(rgb_images, desc="Processing RGB images", unit="image"):
        rgb_path = os.path.join(rgb_folder, image_file)
        mask_tiff_name = os.path.splitext(image_file)[0] + "_mask.tif"
        mask_tiff_path = os.path.join(masks_output_folder, mask_tiff_name)
        # הפעלת יצירת המסכה
        generate_mask_tiff(rgb_path, mask_tiff_path, padding_px)

        # בדיקה באמצעות מודל הסיווג – האם המסכה שייכת לענב
        is_grape = check_mask_is_grape(rgb_path, mask_tiff_path, classifier, device)
        if not is_grape:
            print(
                f"Skipping HSI signature extraction for {image_file} as it is not classified as 'Grape'."
            )
            continue

        # מציאת קובץ HSI תואם – נניח שהשם תואם (עם סיומת .dat)
        hsi_filename = os.path.splitext(image_file)[0] + ".dat"
        hsi_path = os.path.join(hsi_folder, hsi_filename)
        if not os.path.exists(hsi_path):
            print(
                f"HSI file not found for {image_file}. Skipping signature extraction."
            )
            continue

        output_csv = os.path.join(
            signatures_output_folder,
            os.path.splitext(image_file)[0] + "_signatures.csv",
        )
        extract_spectral_signatures(mask_tiff_path, hsi_path, output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb_folder", type=str, required=True, help="Folder containing RGB images"
    )
    parser.add_argument(
        "--hsi_folder", type=str, required=True, help="Folder containing HSI .dat files"
    )
    parser.add_argument(
        "--masks_folder",
        type=str,
        required=True,
        help="Folder to save output mask TIFF files",
    )
    parser.add_argument(
        "--signatures_folder",
        type=str,
        required=True,
        help="Folder to save output spectral signatures CSV files",
    )
    parser.add_argument(
        "--padding", type=int, default=10, help="Padding in pixels for bounding box"
    )
    args = parser.parse_args()

    os.makedirs(args.masks_folder, exist_ok=True)
    os.makedirs(args.signatures_folder, exist_ok=True)

    prepare_directories()
    ensure_sam2_repo()
    download_sam_weights()

    # טען את מודל הסיווג – נניח שיש לך פונקציה load_classifier_model
    classifier = load_classifier_model()
    # נניח שהמודל נטען ומוכן לעבודה
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    process_folder(
        args.rgb_folder,
        args.hsi_folder,
        args.masks_folder,
        args.signatures_folder,
        classifier,
        device,
        args.padding,
    )


if __name__ == "__main__":
    main()
