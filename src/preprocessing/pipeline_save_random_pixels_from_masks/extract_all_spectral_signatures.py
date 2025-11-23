import os
import sys
import torch
from data_transforms import get_test_transforms
import classifier_inference
from extract_spectral_signatures import run_extraction_to_parquet

mask_module_path = "/storage/yovelg/Grape/MaskGenerator"

for path in [mask_module_path]:
    if path not in sys.path:
        sys.path.insert(0, path)
import mask_generator_module


def find_hsi_hdr(results_dir):
    for file in os.listdir(results_dir):
        if file.lower().endswith(".hdr"):
            print(os.path.join(results_dir, file))
            return os.path.join(results_dir, file)
    raise FileNotFoundError(f"No HSI .hdr file found in {results_dir}")


def process_single_folder(base_path, output_parquet_dir):
    images_path = os.path.join(base_path, "HS")
    masks_output_path = os.path.join(base_path, "output")
    results_path = os.path.join(base_path, "HS", "results")

    if not os.path.exists(results_path):
        print(f"⚠️ Skipping {base_path} – no results folder.")
        return

    try:
        hsi_hdr_path = find_hsi_hdr(results_path)
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return

    # Step 1: Generate masks
    print(f"🔹 Generating masks for {base_path}...")
    mask_generator_module.run(images_path, masks_output_path, images_path)

    # Step 2: Classify masks
    print("🔹 Classifying masks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = classifier_inference.load_classifier_model()
    transform = get_test_transforms()

    label_dict = {}
    for file in os.listdir(masks_output_path):
        if file.lower().endswith(".tif"):
            mask_path = os.path.join(masks_output_path, file)
            try:
                label, conf, image_name = classifier_inference.classify_crop_from_mask(
                    mask_path, images_path, model, transform, device
                )
                label_dict[file] = label
            except Exception as e:
                print(f"⚠️ Failed to classify {file}: {e}")

    # Step 3: Extract spectral signatures
    print("🔹 Extracting spectral signatures...")
    run_extraction_to_parquet(
        masks_dir=masks_output_path,
        hsi_hdr_path=hsi_hdr_path,
        label_dict=label_dict,
        output_dir=output_parquet_dir,
    )


def main():
    base_data_dir = "/storage/yovelg/Grape/data"
    output_parquet_dir = "/storage/yovelg/Grape/PCA"
    os.makedirs(output_parquet_dir, exist_ok=True)

    # Loop over clusters with tqdm for progress
    for i in range(1, 2):
        for j in range(1, 60):
            cluster_id = f"{i}_{j:02}"
            cluster_path = os.path.join(base_data_dir, cluster_id)
            print(cluster_path)
            if not os.path.isdir(cluster_path):
                continue

            # Get a list of date directories and wrap with tqdm
            date_dirs = sorted(
                [
                    d
                    for d in os.listdir(cluster_path)
                    if os.path.isdir(os.path.join(cluster_path, d))
                ]
            )
            for date in date_dirs:

                date_parts = date.split(".")
                print(f"date: {date} and Month: {date_parts[1]}")
                if len(date_parts) < 2 or int(date_parts[1]) < 9:
                    print("next item focus on september and after")
                    continue

                date_path = os.path.join(cluster_path, date)
                print(f"\n🚀 Processing: {cluster_id} / {date}")
                process_single_folder(date_path, output_parquet_dir)


if __name__ == "__main__":
    main()
