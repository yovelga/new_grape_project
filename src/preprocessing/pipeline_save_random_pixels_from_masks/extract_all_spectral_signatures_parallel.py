import os
import sys
import logging
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_transforms import get_test_transforms
import classifier_inference
from extract_spectral_signatures_parallel import run_extraction_to_parquet
import glob
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Attributes
mask_module_path = "/storage/yovelg/Grape/MaskGenerator"
for path in [mask_module_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

import mask_generator_module


# -------------------------------------------------------------
# find_hsi_hdr
# -------------------------------------------------------------
def find_hsi_hdr(results_dir):
    logging.info("Searching for HSI .hdr file in '%s'", results_dir)
    for file in os.listdir(results_dir):
        if file.lower().endswith(".hdr"):
            found_path = os.path.join(results_dir, file)
            logging.info("Found HDR file: %s", found_path)
            return found_path
    logging.warning("No HSI .hdr file found in '%s'", results_dir)
    raise FileNotFoundError(f"No HSI .hdr file found in {results_dir}")


# -------------------------------------------------------------
# process_single_folder
# -------------------------------------------------------------
def process_single_folder(base_path, output_parquet_dir):
    images_path = os.path.join(base_path, "HS")
    masks_output_path = os.path.join(base_path, "output")
    results_path = os.path.join(base_path, "HS", "results")

    if not os.path.exists(results_path):
        logging.warning("Skipping '%s' – no results folder found.", base_path)
        return

    try:
        hsi_hdr_path = find_hsi_hdr(results_path)
    except FileNotFoundError as e:
        logging.warning("Skipping '%s'. Reason: %s", base_path, e)
        return

    # Step 1: Generate masks
    logging.info("Generating masks for '%s'...", base_path)
    try:
        # Look for all .tif files in the masks_output_path
        tif_files = glob.glob(os.path.join(masks_output_path, "*.tif"))

        # Only run SAM if fewer than 10 .tif masks
        if len(tif_files) < 100:
            mask_generator_module.run(images_path, masks_output_path, images_path)
            logging.info("Finished generating masks for '%s'", base_path)
    except Exception as e:
        logging.error("Error during mask generation in '%s': %s", base_path, e)
        return
    else:
        # This runs if there was no exception
        if len(tif_files) >= 100:
            logging.info(
                "Skipping mask generation for '%s': Found %d TIF mask(s).",
                base_path,
                len(tif_files),
            )

    # Step 2: Classify masks
    logging.info("Classifying masks...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = classifier_inference.load_classifier_model()
    # transform = get_test_transforms()

    label_dict = {}
    for file in os.listdir(masks_output_path):
        if file.lower().endswith(".tif"):

            ####
            label_dict[file] = "none"
            ###

            mask_path = os.path.join(masks_output_path, file)
            # try:
            #     label, conf, image_name = classifier_inference.classify_crop_from_mask(
            #         mask_path, images_path, model, transform, device
            #     )
            #     label_dict[file] = label
            #     logging.debug("Classified '%s' as '%s' with conf=%.2f", file, label, conf)
            # except Exception as e:
            #     logging.warning("Failed to classify '%s': %s", file, e)

    # Step 3: Extract spectral signatures
    logging.info("Extracting spectral signatures for '%s'...", base_path)
    try:
        run_extraction_to_parquet(
            masks_dir=masks_output_path,
            hsi_hdr_path=hsi_hdr_path,
            label_dict=label_dict,
            output_dir=output_parquet_dir,
        )
        logging.info("Spectral signatures extraction complete for '%s'", base_path)
    except Exception as e:
        logging.error("Spectral signature extraction error in '%s': %s", base_path, e)


def merge_csv_files(output_dir, merged_filename="signatures_4_PCA_merged.csv"):
    csv_files = glob.glob(os.path.join(output_dir, "signatures_4_PCA_*.csv"))
    if not csv_files:
        print("No CSV files found in", output_dir)
        return

    print(f"Merging {len(csv_files)} CSV files...")
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_output_file = os.path.join(output_dir, merged_filename)
    merged_df.to_csv(merged_output_file, index=False)
    print(f"✅ Merged all files into {merged_output_file}")


# -------------------------------------------------------------
# main
# -------------------------------------------------------------
def main(base_data_dir, output_parquet_dir):

    os.makedirs(output_parquet_dir, exist_ok=True)

    # Collect all date paths we want to process
    date_paths = []
    for i in range(2):
        for j in range(60):
            cluster_id = f"{i+1}_{j+1:02}"
            # print(cluster_id)
            cluster_path = os.path.join(base_data_dir, cluster_id)
            if not os.path.isdir(cluster_path):
                continue

            date_dirs = sorted(
                [
                    d
                    for d in os.listdir(cluster_path)
                    if os.path.isdir(os.path.join(cluster_path, d))
                ]
            )

            for date in date_dirs:
                date_parts = date.split(".")
                # If date format not as expected, skip
                if len(date_parts) < 2 or int(date_parts[1]) < 9:
                    logging.info("Skipping '%s/%s' because month < 9", cluster_id, date)
                    continue

                full_date_path = os.path.join(cluster_path, date)
                logging.info(
                    "Prepared cluster '%s', date '%s' for processing...",
                    cluster_id,
                    date,
                )
                date_paths.append(full_date_path)

    # Process items in parallel with 16 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_path = {
            executor.submit(process_single_folder, path, output_parquet_dir): path
            for path in date_paths
        }

        for future in as_completed(future_to_path):
            folder_path = future_to_path[future]
            try:
                future.result()
                logging.info("Finished processing '%s'", folder_path)
            except Exception as exc:
                logging.error("Error while processing '%s': %s", folder_path, exc)

    # After all parallel tasks finish, merge the resulting CSV files
    merge_csv_files(output_parquet_dir, "signatures_4_PCA_merged_all.csv")


if __name__ == "__main__":
    base_data_dir = "/storage/yovelg/Grape/data"
    output_parquet_dir = "/storage/yovelg/Grape/PCA/csv/Grapes"
    main(base_data_dir, output_parquet_dir)
