import subprocess
import cv2
import torch
import os
import requests
import shutil
import tifffile as tiff
import supervision as sv
import sys
from tqdm import tqdm
import time


# Define global variables
HOME = "/storage/yovelg/Grape/"  # Current directory: ObjectJsonizer
# HOME =os.path.join(os.getcwd(), "MaskGenerator")
ITEMS_DIR = os.path.join(
    os.path.dirname(HOME), "items_for_cnn_train"
)  # General directory for all data
SAM2_DIR = os.path.join(HOME, "segment-anything-2")
SAM_WEIGHTS = f"{HOME}/checkpoints/sam2_hiera_large.pt"
print(f"SAM_WEIGHTS:{SAM_WEIGHTS}")
IMAGES_PATH = os.path.join(ITEMS_DIR, "images")
MASKS_PATH = os.path.join(ITEMS_DIR, "masks")
USED_IMAGES_PATH = os.path.join(ITEMS_DIR, "used")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MaskGenerator/segment-anything-2/sam2/configs/sam2/sam2_hiera_l.yaml
# CONFIG = f"{SAM2_DIR}/sam2/configs/sam2/sam2_hiera_l.yaml"
# CONFIG = f"{HOME}/MaskGenerator/segment-anything-2/sam2/configs/sam2/sam2_hiera_l.yaml"
CONFIG = "/storage/yovelg/Grape/MaskGenerator/segment-anything-2/sam2/configs/sam2/sam2_hiera_l.yaml"

print(f"CONFIG:{CONFIG}")


# Ensure directories exist
def prepare_directories():
    """Create required directories."""
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    os.makedirs(USED_IMAGES_PATH, exist_ok=True)


# Call this function in your main workflow
prepare_directories()


# Check and print CUDA availability
def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")


# Clone the 'segment-anything-2' repository if it doesn't exist
def ensure_sam2_repo():
    if not os.path.exists(SAM2_DIR):
        print("'segment-anything-2' directory not found. Cloning the repository...")
        repo_url = "https://github.com/facebookresearch/sam2"
        try:
            subprocess.run(["git", "clone", repo_url, SAM2_DIR], check=True)
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error while cloning the repository: {e}")
            sys.exit(1)
    else:
        print("'segment-anything-2' directory already exists. Skipping clone.")


# Download SAM weights if not available
def download_sam_weights():
    if not os.path.exists(SAM_WEIGHTS):
        os.makedirs(os.path.dirname(SAM_WEIGHTS), exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(SAM_WEIGHTS, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Checkpoint downloaded successfully to {SAM_WEIGHTS}")
        else:
            print(f"Error while downloading: {response.status_code}")
    else:
        print("SAM weights already exist. Skipping download.")


import torch


import torch


def clean_checkpoint(ckpt_path):
    """Remove unexpected keys from the checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # מפתחות בלתי צפויים שיש להסיר
    unexpected_keys = [
        "no_obj_embed_spatial",
        "obj_ptr_tpos_proj.weight",
        "obj_ptr_tpos_proj.bias",
    ]

    for key in unexpected_keys:
        if key in state_dict:
            del state_dict[key]
            print(f"Removed key: {key}")

    cleaned_ckpt_path = ckpt_path.replace(".pt", "_cleaned.pt")
    torch.save({"model": state_dict}, cleaned_ckpt_path)
    print(f"Saved cleaned checkpoint: {cleaned_ckpt_path}")

    return cleaned_ckpt_path


# Initialize SAM2 model
def initialize_sam2():
    sys.path.append(os.path.abspath(SAM2_DIR))

    import hydra
    from hydra.core.global_hydra import GlobalHydra

    # לוודא ש-Hydra מאותחל נכון
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(
            config_path="segment-anything-2/sam2/configs/sam2", version_base=None
        )

    print(
        "✅ HYDRA CONFIG PATH SET:",
        "/storage/yovelg/Grape/MaskGenerator/segment-anything-2/sam2/configs/sam2",
    )

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    cleaned_weights = clean_checkpoint(SAM_WEIGHTS)  # השתמש בקובץ הנקי
    sam2_model = build_sam2(
        "sam2_hiera_l", cleaned_weights, device=DEVICE, apply_postprocessing=False
    )

    return SAM2AutomaticMaskGenerator(sam2_model)


# Prepare directories
def prepare_directories():
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    os.makedirs(USED_IMAGES_PATH, exist_ok=True)


# Add padding to bounding boxes
def padding_mask(x_min, y_min, x_max, y_max, px_size, image_w, image_h):
    xmin = max(0, x_min - px_size)
    ymin = max(0, y_min - px_size)
    xmax = min(image_w, x_max + px_size)
    ymax = min(image_h, y_max + px_size)
    return [xmin, ymin, xmax, ymax]


def get_image_list():
    """Retrieve the list of images in the images directory."""
    images_list = [
        f for f in os.listdir(IMAGES_PATH) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images_list:
        print("No images found in the base path. Exiting...")
    return images_list


def preprocess_image(image_path):
    """Read and preprocess an image."""
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (512, 512))
    return image, image_h, image_w


def move_image_to_used(image_path, image_name):
    """Move the processed image to the 'used' folder."""
    shutil.move(image_path, os.path.join(USED_IMAGES_PATH, image_name))


def generate_detections(mask_generator, image):
    """Generate detections using the SAM2 model."""
    sam2_result = mask_generator.generate(image)
    return sv.Detections.from_sam(sam_result=sam2_result)


def save_detection_mask(image_name_without_extension, i, detection, image_w, image_h):
    """Save detection masks as TIF files."""
    mask = detection.mask[0]  # Extract the mask
    bbox_original = detection.xyxy[0]
    x_min, y_min, x_max, y_max = map(int, bbox_original)
    bbox_extra = padding_mask(x_min, y_min, x_max, y_max, 10, image_w, image_h)

    # Save mask as TIF
    mask_tif_path = os.path.join(
        MASKS_PATH, f"{image_name_without_extension}_mask_{i}.tif"
    )
    metadata = {
        "image_name": image_name_without_extension,
        "original_bbox": bbox_original.tolist(),
        "padded_bbox": bbox_extra,
        "tag": "none",
    }
    tiff.imwrite(
        mask_tif_path, mask, dtype="bool", compression="LZW", metadata=metadata
    )


def process_images(mask_generator):
    """Main function to process images and generate masks with timing and progress bar."""
    # Start timing
    start_time = time.time()

    # Get the list of images
    images_list = get_image_list()
    if not images_list:
        print("No images found to process. Exiting...")
        return

    print(f"Processing {len(images_list)} images...")

    # Initialize progress bar
    for image_name in tqdm(images_list, desc="Processing Images", unit="image"):
        image_name_without_extension = os.path.splitext(image_name)[0]
        image_path = os.path.join(IMAGES_PATH, image_name)

        # Read and preprocess image
        image, image_h, image_w = preprocess_image(image_path)

        # Generate detections
        detections = generate_detections(mask_generator, image)

        # Save detection masks
        for i, detection in enumerate(detections):
            detection = detections[i]
            save_detection_mask(
                image_name_without_extension, i, detection, image_w, image_h
            )

        # Move image to 'used' folder
        move_image_to_used(image_path, image_name)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing completed in {total_time:.2f} seconds.")


# Main function
def main():
    print(f"HOME: {HOME}")
    check_cuda()
    ensure_sam2_repo()
    download_sam_weights()
    prepare_directories()
    mask_generator = initialize_sam2()
    process_images(mask_generator)


if __name__ == "__main__":
    main()
