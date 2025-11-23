import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import tifffile as tiff
import json
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from config import TEST_DIR, BATCH_SIZE, IMAGES_DIR
from data_transforms import get_test_transforms
from dataset_multi import GrapeDataset
from model import get_model

# ---------------------------
# פונקציות עזר לאינפרנס
# ---------------------------


def get_original_metadata(mask_path):
    with tiff.TiffFile(mask_path) as tif:
        tags = tif.pages[0].tags
        description = tags.get("ImageDescription")
        metadata = json.loads(description.value)
    return metadata


def find_original_image_path(
    image_name, images_dir, extensions=[".png", ".jpg", ".jpeg"]
):
    for ext in extensions:
        path = os.path.join(images_dir, image_name + ext)
        if os.path.exists(path):
            return path
    return None


def preprocess_image_for_inference(image: np.ndarray, transform):
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0)
    return image_tensor


def predict_tag_and_confidence(model, image: np.ndarray, transform, device):
    image_tensor = preprocess_image_for_inference(image, transform).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, pred_class = probabilities.max(dim=1)
        conf = conf.item()
        pred_class = pred_class.item()
    predicted_label = "Grape" if pred_class == 1 else "Not Grape"
    return predicted_label, conf


def evaluate_on_test(model, test_loader, transform, device):
    all_labels = []
    all_predictions = []
    for batch in tqdm(test_loader, desc="Running inference", unit="batch"):
        for img, label in batch:
            img_np = np.array(img).astype("uint8")
            pred, _ = predict_tag_and_confidence(model, img_np, transform, device)
            all_labels.append(label)
            all_predictions.append(1 if pred == "Grape" else 0)
    return all_labels, all_predictions


# ---------------------------
# פונקציות לאיסוף misclassified
# ---------------------------


def collect_misclassified_examples(model, dataset, transform, device):
    misclassified = []
    for i in tqdm(
        range(len(dataset)), desc="Collecting misclassified examples", unit="sample"
    ):
        output, label = dataset[i]
        original_img = np.array(output).astype("uint8")
        predicted, conf = predict_tag_and_confidence(
            model, original_img, get_test_transforms(), device
        )
        ground_truth = "Grape" if label == 1 else "Not Grape"
        if predicted != ground_truth:
            mask_path, _ = dataset.base_samples[i]
            metadata = get_original_metadata(mask_path)
            img_name = metadata.get("image_name")
            orig_path = find_original_image_path(img_name, IMAGES_DIR)
            misclassified.append(
                {
                    "Image Name": img_name,
                    "Original Image Path": orig_path,
                    "Ground Truth": ground_truth,
                    "Prediction": predicted,
                    "Confidence": conf,
                    "Mask Path": mask_path,
                }
            )
    return misclassified


def plot_confusion_matrix(all_labels, all_predictions):
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix (rows = Actual, columns = Predicted):")
    print("             Predicted: Not Grape     Predicted: Grape")
    print(f"Actual: Not Grape       {cm[0,0]}                  {cm[0,1]}")
    print(f"Actual: Grape           {cm[1,0]}                  {cm[1,1]}")
    return cm


# ---------------------------
# פונקציות להפקת overlay באמצעות PIL בלבד
# ---------------------------


def create_purple_overlay(orig_crop, mask_crop, alpha=0.3):
    purple_img = Image.new("RGB", orig_crop.size, (255, 0, 255))
    mask_img = Image.fromarray((mask_crop * 255).astype(np.uint8))
    blended = Image.blend(orig_crop, purple_img, alpha)
    overlay_img = Image.composite(blended, orig_crop, mask_img)
    return overlay_img


def create_full_overlay(orig_img, mask):
    orig_np = np.array(orig_img)
    if mask.shape[0] != orig_np.shape[0] or mask.shape[1] != orig_np.shape[1]:
        mask = np.array(
            Image.fromarray(mask).resize(
                (orig_np.shape[1], orig_np.shape[0]), Image.NEAREST
            )
        )
    mask_binary = (mask > 0).astype(np.uint8)
    purple_img = Image.new("RGB", orig_img.size, (255, 0, 255))
    mask_img = Image.fromarray((mask_binary * 255).astype(np.uint8))
    blended = Image.blend(orig_img, purple_img, 0.3)
    overlay = Image.composite(blended, orig_img, mask_img)
    return overlay


# ---------------------------
# פונקציה להצגת misclassified לפי סוג טעויות
# ---------------------------
def plot_misclassified_images_by_type(misclassified, input_mode):
    false_pos = [
        ex
        for ex in misclassified
        if ex["Prediction"] == "Grape" and ex["Ground Truth"] == "Not Grape"
    ]
    false_neg = [
        ex
        for ex in misclassified
        if ex["Prediction"] == "Not Grape" and ex["Ground Truth"] == "Grape"
    ]

    for group_name, examples in [
        ("False Positives", false_pos),
        ("False Negatives", false_neg),
    ]:
        if len(examples) == 0:
            print(f"No {group_name} found.")
            continue
        num_examples = min(5, len(examples))
        # שימוש ב-squeeze=False כדי להבטיח ש-axes הוא מערך דו-ממדי
        fig, axes = plt.subplots(
            2, num_examples, figsize=(num_examples * 3, 6), squeeze=False
        )
        for idx in range(num_examples):
            ex = examples[idx]
            orig_path = ex["Original Image Path"]
            mask_path = ex["Mask Path"]
            try:
                orig_img = Image.open(orig_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {orig_path}: {e}")
                continue
            metadata = get_original_metadata(mask_path)
            # בחר bbox בהתאם למצב הקלט (אם eval_mode == "enlarged", השתמש ב-padded_bbox אם זמין)
            if input_mode == "enlarged":
                bbox = metadata.get("padded_bbox", metadata.get("original_bbox"))
            else:
                bbox = metadata.get("original_bbox")
            if bbox is None:
                continue
            crop = orig_img.crop(tuple(bbox))
            with tiff.TiffFile(mask_path) as tif:
                mask = tif.pages[0].asarray()
            overlay_full = create_full_overlay(orig_img, mask)
            title_text = f"GT: {ex['Ground Truth']}\nPred: {ex['Prediction']} ({ex['Confidence']:.2f})"
            axes[0, idx].imshow(crop)
            axes[0, idx].set_title(title_text, fontsize=8)
            axes[0, idx].axis("off")
            axes[1, idx].imshow(overlay_full)
            axes[1, idx].set_title("Full Overlay", fontsize=8)
            axes[1, idx].axis("off")
        plt.suptitle(f"{group_name} (Input mode: {input_mode})", fontsize=12)
        plt.tight_layout()
        plt.show()


# ---------------------------
# פונקציה להערכת המודל עבור מצב קלט נתון
# ---------------------------
def evaluate_model_for_input_mode(model_mode, eval_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transform = get_test_transforms()

    # טען את סט הבדיקה עם eval_mode
    identity = lambda x: x
    test_dataset = GrapeDataset(
        TEST_DIR, transform=identity, balance_mode=None, input_mode=eval_mode
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x
    )

    model = get_model(num_classes=2)
    model_weight_path = f"/storage/yovelg/Grape/training_classification_model/model_weights/best_model_{model_mode}.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels, all_predictions = evaluate_on_test(
        model, test_loader, test_transform, device
    )
    cm = plot_confusion_matrix(all_labels, all_predictions)

    misclassified = collect_misclassified_examples(
        model, test_dataset, test_transform, device
    )
    df = pd.DataFrame(misclassified)
    excel_save_path = (
        f"misclassified_examples_{model_mode}_evaluated_as_{eval_mode}.xlsx"
    )
    df.to_excel(excel_save_path, index=False)
    print(f"Misclassified examples saved to {excel_save_path}")

    plot_misclassified_images_by_type(misclassified, eval_mode)
    return misclassified


# ---------------------------
# פונקציה ראשית
# ---------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_mode",
        type=str,
        default="all",
        choices=["original", "enlarged", "segmentation", "all"],
        help="Model mode used in training (which weights to load)",
    )
    args = parser.parse_args()

    # נריץ הערכה עבור כל מצבי הקלט
    input_modes = ["original", "enlarged", "segmentation"]
    for mode in input_modes:
        print("=" * 40)
        print(f"Evaluating model trained on '{args.model_mode}' for input mode: {mode}")
        print("=" * 40)
        evaluate_model_for_input_mode(args.model_mode, eval_mode=mode)


if __name__ == "__main__":
    main()
