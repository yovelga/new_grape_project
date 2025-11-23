# classifier_inference.py

import torch
import torch.nn.functional as F
from model import get_model
from PIL import Image
import json
import os
import tifffile as tiff


# נתיב למשקולות של המודל המסווג
MODEL_WEIGHTS_PATH = "/storage/yovelg/Grape/training_classification_model_cnn_for_grapes_berry/model_weights/best_model_original.pth"


def load_classifier_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Classifier model loaded from {MODEL_WEIGHTS_PATH} on device {device}")
    return model


def predict_label_and_confidence(model, image, transform, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, dim=1)
        conf = conf.item()
        pred = pred.item()
    label = "Grape" if pred == 1 else "Not Grape"
    return label, conf


def get_bbox_from_tiff(tiff_path):
    with tiff.TiffFile(tiff_path) as tif:
        tags = tif.pages[0].tags
        description = tags.get("ImageDescription")
        metadata = json.loads(description.value)
    bbox = metadata.get("original_bbox")  # [x_min, y_min, x_max, y_max]
    image_name = metadata.get("image_name")
    return bbox, image_name


def classify_crop_from_mask(mask_path, images_dir, model, transform, device):
    bbox, image_name = get_bbox_from_tiff(mask_path)

    # איתור קובץ התמונה
    for ext in [".png", ".jpg", ".jpeg"]:
        image_path = os.path.join(images_dir, image_name + ext)
        if os.path.exists(image_path):
            break
    else:
        raise FileNotFoundError(f"Image {image_name} not found in {images_dir}")

    # פתיחת התמונה וחתך לפי bbox
    image = Image.open(image_path).convert("RGB")
    cropped = image.crop(bbox)

    # סיווג
    label, conf = predict_label_and_confidence(model, cropped, transform, device)
    print(f"Image: {image_name}, Label: {label}, Confidence: {conf:.2f}")
    return label, conf, image_name
