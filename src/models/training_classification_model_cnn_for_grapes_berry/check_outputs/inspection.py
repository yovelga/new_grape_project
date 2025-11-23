import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())


import os
import cv2
import json
import torch
import numpy as np
import streamlit as st
from PIL import Image

from config import MODEL_SAVE_PATH, IMAGES_DIR
from back import (
    load_tif_files,
    get_current_tif,
    load_tif_metadata,
    decode_mask,
    find_image_with_extension,
    save_metadata_to_tif,
    next_tif,
    prev_tif,
    next_untagged_tif,
    current_tif_index,
)
from data_transforms import get_test_transforms
from model import get_model

print(f"file directory model: {MODEL_SAVE_PATH}")
MODEL_WEIGHTS_PATH = MODEL_SAVE_PATH
IMAGE_DIR = IMAGES_DIR

tif_files = load_tif_files()
if not tif_files:
    st.error("No TIF files found in the masks directory.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    model = get_model(num_classes=2)
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


model = load_model()


def preprocess_image_for_inference(image: np.ndarray) -> torch.Tensor:
    transform = get_test_transforms()
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0)
    return image_tensor


def predict_tag_and_confidence(image: np.ndarray) -> (str, float):
    image_tensor = preprocess_image_for_inference(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        st.write("Logits:", outputs)
        probabilities = torch.softmax(outputs, dim=1)
        st.write("Probabilities:", probabilities)
        confidence, predicted_class = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_class = predicted_class.item()
    predicted_label = "Grape" if predicted_class == 1 else "Not Grape"
    return predicted_label, confidence


def create_purple_mask(mask: np.ndarray) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)
    purple_mask = cv2.merge([binary_mask * 255, binary_mask * 0, binary_mask * 255])
    return purple_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    purple_mask = create_purple_mask(mask)
    overlayed = cv2.addWeighted(image.copy(), 1 - alpha, purple_mask, alpha, 0)
    return overlayed


st.sidebar.header("Navigation")
if st.sidebar.button("Next TIF"):
    next_tif()
    st.rerun()
if st.sidebar.button("Previous TIF"):
    prev_tif()
    st.rerun()
if st.sidebar.button("Next Untagged TIF"):
    next_untagged_tif()
    st.rerun()
if st.sidebar.button("Reload Data"):
    st.rerun()

mode = st.sidebar.radio(
    "Select mode", options=["Browse All", "Show Misclassified Only"]
)

current_data = get_current_tif()
tif_path = current_data["path"]
metadata = current_data["metadata"]

if not metadata or "image_name" not in metadata:
    st.error("Metadata or image name is missing in the TIF file.")
    st.stop()

image_name = metadata.get("image_name")
original_image_path = find_image_with_extension(
    image_name, IMAGE_DIR, [".png", ".jpg", ".jpeg"]
)
if not original_image_path:
    st.error(f"Original image for '{image_name}' not found in {IMAGE_DIR}.")
    st.stop()

original_image = cv2.imread(original_image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

mask_full = decode_mask(tif_path)
if mask_full is None:
    st.error("Failed to load mask from the TIF file.")
    st.stop()

annotated_full = overlay_mask(original_image, mask_full)
predicted_tag, confidence = predict_tag_and_confidence(original_image)
ground_truth_tag = metadata.get("tag", "none")

if mode == "Show Misclassified Only" and predicted_tag == ground_truth_tag:
    st.info(
        f"Current image is classified correctly with confidence {confidence:.2f}. Moving to next..."
    )
    next_tif()
    st.rerun()

st.markdown("## Image Inspection")
st.write(f"**File:** {tif_path}")
st.write(f"**Ground Truth Tag:** {ground_truth_tag}")
st.write(f"**Model Prediction:** {predicted_tag}")
st.write(f"**Confidence:** {confidence:.2f}")

col1, col2 = st.columns(2)
with col1:
    st.image(original_image, caption="Original Image (PNG)", use_container_width=True)
with col2:
    st.image(
        annotated_full,
        caption="Image with Full Segmentation Overlay",
        use_container_width=True,
    )

st.markdown("## Update Tag")
if st.button("Tag as Grape"):
    metadata["tag"] = "Grape"
    save_metadata_to_tif(tif_path, metadata)
    st.success("Tagged as Grape and saved.")
    next_tif()
    st.rerun()
if st.button("Tag as Not Grape"):
    metadata["tag"] = "Not Grape"
    save_metadata_to_tif(tif_path, metadata)
    st.success("Tagged as Not Grape and saved.")
    next_tif()
    st.rerun()
if st.button("Skip / Next"):
    next_tif()
    st.rerun()
