import os
import numpy as np
import pandas as pd
import tifffile as tiff
import json
from tqdm import tqdm
from PIL import Image


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


def predict_tag_and_confidence(model, image: np.ndarray, transform, device):
    """
    מפעילה את המודל על תמונת NumPy (RGB) ומחזירה את התווית (Grape/Not Grape) ואת רמת הביטחון.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, pred_class = probabilities.max(dim=1)
        conf = conf.item()
        pred_class = pred_class.item()
    predicted_label = "Grape" if pred_class == 1 else "Not Grape"
    return predicted_label, conf


def collect_misclassified_examples(model, dataset, transform, device, set_name="Train"):
    """
    אוסף דוגמאות בהן המודל טעה (misclassified).
    מניחים ש-input_mode="all", כך שכל דגימה בסיסית מתורגמת ל-3 מופעים (original, enlarged, segmentation).
    אנו בוחרים את ה- original crop (modality 0) לצורך בדיקת התחזית.
    """
    misclassified = []
    base_count = len(dataset.base_samples)  # מספר הדגימות הבסיסיות
    # שימוש ב-tqdm להצגת התקדמות
    for i in tqdm(
        range(base_count),
        desc=f"Collecting misclassified examples for {set_name}",
        unit="sample",
    ):
        # modality 0 => original crop
        output, label = dataset[i * 3]
        original_img = np.array(output)  # PIL -> NumPy

        predicted, conf = predict_tag_and_confidence(
            model, original_img, transform, device
        )
        ground_truth = "Grape" if label == 1 else "Not Grape"

        if predicted != ground_truth:
            # שליפת מטא-דאטה מה-TIF
            mask_path, _ = dataset.base_samples[i]
            metadata = get_original_metadata(mask_path)
            image_name = metadata.get("image_name")
            orig_path = find_original_image_path(image_name, dataset.IMAGES_DIR)

            misclassified.append(
                {
                    "Input Mode": dataset.input_mode,
                    "Set": set_name,
                    "Image Name": image_name,
                    "Original Image Path": orig_path,
                    "Ground Truth": ground_truth,
                    "Prediction": predicted,
                    "Confidence": conf,
                    "Mask Path": mask_path,
                }
            )

    return misclassified
