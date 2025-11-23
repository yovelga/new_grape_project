import os
import argparse
from tqdm import tqdm
from mask_generator_module import (
    generate_mask_tiff,
    prepare_directories,
    ensure_sam2_repo,
    download_sam_weights,
)


def process_folder(input_folder, output_folder, padding_px=10):
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return
    os.makedirs(output_folder, exist_ok=True)
    # קבלת רשימת תמונות (JPG/PNG)
    images = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        print("No images found in the input folder.")
        return
    for image_name in tqdm(images, desc="Processing images", unit="image"):
        input_image_path = os.path.join(input_folder, image_name)
        output_tiff_path = os.path.join(
            output_folder, os.path.splitext(image_name)[0] + "_mask.tif"
        )
        generate_mask_tiff(input_image_path, output_tiff_path, padding_px)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing input JPG/PNG images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save output TIFF masks",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding in pixels to add to bounding box",
    )
    args = parser.parse_args()

    prepare_directories()
    ensure_sam2_repo()
    download_sam_weights()
    process_folder(args.input_folder, args.output_folder, args.padding)


if __name__ == "__main__":
    main()
