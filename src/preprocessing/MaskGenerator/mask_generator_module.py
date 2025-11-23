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

# Set up correct SAM2 path for imports
SAM2_DIR = os.path.join(os.path.dirname(__file__), "segment-anything-2")
sys.path.append(os.path.abspath(SAM2_DIR))

# Global paths for images, masks, used images
IMAGES_PATH = ""
MASKS_PATH = ""
USED_IMAGES_PATH = ""


def set_paths(images_path, masks_path, used_images_path):
    global IMAGES_PATH, MASKS_PATH, USED_IMAGES_PATH
    IMAGES_PATH = images_path
    MASKS_PATH = masks_path
    USED_IMAGES_PATH = used_images_path


def initial_settings():
    global HOME, SAM2_DIR, SAM_WEIGHTS, DEVICE, CONFIG
    SAM2_DIR = os.path.join(os.path.dirname(__file__), "segment-anything-2")
    SAM_WEIGHTS = os.path.join(
        SAM2_DIR, "checkpoints", "sam2.1_hiera_large_cleaned.pt"
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG = os.path.join(SAM2_DIR, "sam2", "sam2_hiera_l.yaml")
    print(f"SAM_WEIGHTS: {SAM_WEIGHTS}")


def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")


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


def clean_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
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


def initialize_sam2():
    sys.path.append(os.path.abspath(SAM2_DIR))
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="sam2/configs/sam2", version_base=None)

    print("✅ HYDRA CONFIG PATH SET:", CONFIG)

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    cleaned_weights = clean_checkpoint(SAM_WEIGHTS)
    sam2_model = build_sam2(
        "sam2_hiera_l", cleaned_weights, device=DEVICE, apply_postprocessing=False
    )

    return SAM2AutomaticMaskGenerator(sam2_model)


def prepare_directories():
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    os.makedirs(USED_IMAGES_PATH, exist_ok=True)


def padding_mask(x_min, y_min, x_max, y_max, px_size, image_w, image_h):
    xmin = max(0, x_min - px_size)
    ymin = max(0, y_min - px_size)
    xmax = min(image_w, x_max + px_size)
    ymax = min(image_h, y_max + px_size)
    return [xmin, ymin, xmax, ymax]


def get_image_list():
    images_list = [
        f for f in os.listdir(IMAGES_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images_list:
        print("No images found in the base path. Exiting...")
    return images_list


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (512, 512))
    return image, image_h, image_w


def move_image_to_used(image_path, image_name):
    shutil.move(image_path, os.path.join(USED_IMAGES_PATH, image_name))


def save_detection_mask(image_name_without_extension, i, detection, image_w, image_h):
    mask = detection.mask[0]
    bbox_original = detection.xyxy[0]
    x_min, y_min, x_max, y_max = map(int, bbox_original)
    bbox_extra = padding_mask(x_min, y_min, x_max, y_max, 10, image_w, image_h)
    mask_tif_path = os.path.join(MASKS_PATH, f"{image_name_without_extension}_mask_{i}.tif")
    metadata = {
        "image_name": image_name_without_extension,
        "original_bbox": bbox_original.tolist(),
        "padded_bbox": bbox_extra,
        "tag": "none",
    }
    tiff.imwrite(mask_tif_path, mask, dtype="bool", compression="LZW", metadata=metadata)


def process_images(mask_generator):
    start_time = time.time()
    images_list = get_image_list()
    if not images_list:
        print("No images found to process. Exiting...")
        return
    print(f"Processing {len(images_list)} images...")
    for image_name in tqdm(images_list, desc="Processing Images", unit="image"):
        image_name_without_extension = os.path.splitext(image_name)[0]
        image_path = os.path.join(IMAGES_PATH, image_name)
        image, image_h, image_w = preprocess_image(image_path)
        detections = generate_detections(mask_generator, image)
        for i, detection in enumerate(detections):
            save_detection_mask(image_name_without_extension, i, detection, image_w, image_h)
        move_image_to_used(image_path, image_name)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds.")


def generate_detections(mask_generator, image, min_area=200, max_area=5000):
    print("CALLED generate_detections WITH max_area!", min_area, max_area)
    sam2_result = mask_generator.generate(image)
    detections = sv.Detections.from_sam(sam_result=sam2_result)
    filtered_detections = []
    num_masks = len(detections.mask)
    for i in range(num_masks):
        single_mask = detections.mask[i]
        single_bbox = detections.xyxy[i]
        area = single_mask.sum()
        if min_area <= area <= max_area:
            filtered_detections.append(type("Detection", (), {"mask": [single_mask], "xyxy": [single_bbox]}))
    print(f"Returning {len(filtered_detections)} detections")
    return filtered_detections


def main():
    initial_settings()
    check_cuda()
    ensure_sam2_repo()
    download_sam_weights()
    prepare_directories()
    mask_generator = initialize_sam2()
    process_images(mask_generator)


def run(images_path, masks_path, used_images_path):
    set_paths(images_path, masks_path, used_images_path)
    main()


def initialize_sam2_predictor():
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    SAM2_DIR = os.path.join(os.path.dirname(__file__), "segment-anything-2")
    sys.path.append(os.path.abspath(SAM2_DIR))
    config_path = os.path.join("segment-anything-2", "sam2", "configs", "sam2")
    if not GlobalHydra.instance().is_initialized():
        print(f"[SAM2] Initializing hydra with config_path: {config_path}")
        hydra.initialize(config_path=config_path, version_base=None)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cleaned_weights = clean_checkpoint(SAM_WEIGHTS)
    sam_model = build_sam2("sam2_hiera_l", cleaned_weights, device=DEVICE, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor
