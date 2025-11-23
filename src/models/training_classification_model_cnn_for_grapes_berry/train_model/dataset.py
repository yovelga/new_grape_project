import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
from config import IMAGES_DIR


class GrapeDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance_mode=None):
        """
        balance_mode: "downsample" - לבחור את מספר הדוגמאות לפי הקלאס עם הכי פחות מופעים.
                      "oversample"  - לשכפל את הקלאס עם פחות מופעים כדי להגיע למספר המקסימלי.
                      None          - לא לבצע איזון, להשתמש בכל הדוגמאות.
        """
        print(
            f"Initializing dataset for {root_dir} with balance_mode = {balance_mode}..."
        )
        self.root_dir = root_dir
        self.transform = transform
        self.balance_mode = balance_mode
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from {root_dir}.")

    def _load_samples(self):
        class_samples = {}
        for class_name in ["Grape", "Not Grape"]:
            class_dir = os.path.join(self.root_dir, class_name)
            label = 1 if class_name == "Grape" else 0
            samples = []
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(".tif"):
                    file_path = os.path.join(class_dir, file_name)
                    samples.append((file_path, label))
            class_samples[class_name] = samples

        # הדפסת מספר הדוגמאות המקורי לכל קלאס
        for class_name, samples in class_samples.items():
            print(f"Original sample count for {class_name}: {len(samples)}")

        # אם אין איזון – מחזירים את כל הדוגמאות
        if self.balance_mode is None:
            balanced_samples = []
            for samples in class_samples.values():
                balanced_samples.extend(samples)
            return balanced_samples

        # קביעת target_count לפי מצב האיזון
        if self.balance_mode == "downsample":
            target_count = min(
                len(class_samples["Grape"]), len(class_samples["Not Grape"])
            )
        elif self.balance_mode == "oversample":
            target_count = max(
                len(class_samples["Grape"]), len(class_samples["Not Grape"])
            )
        else:
            raise ValueError(
                "balance_mode must be either 'downsample', 'oversample' or None"
            )

        balanced_samples = []
        # איזון עבור כל קלאס
        for class_name, samples in class_samples.items():
            if len(samples) < target_count:
                added = target_count - len(samples)
                print(
                    f"Under-sampled class '{class_name}': original count = {len(samples)}, added {added} samples."
                )
                balanced_samples.extend(samples + random.choices(samples, k=added))
            else:
                balanced_samples.extend(random.sample(samples, target_count))
        return balanced_samples  # ← הסרנו את הפסיק כאן

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mask_path, label = self.samples[idx]

        # קריאת מטא-דאטה מקובץ ה-TIF
        with tiff.TiffFile(mask_path) as tif:
            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")
            metadata = json.loads(description.value)

        # שליפת שם התמונה ללא סיומת
        image_name = metadata.get("image_name")

        # חיפוש קובץ התמונה בתיקייה
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            potential_path = os.path.join(IMAGES_DIR, image_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            raise FileNotFoundError(f"Image not found for {image_name} in {IMAGES_DIR}")

        # טעינת התמונה
        image = Image.open(image_path).convert("RGB")

        # חיתוך התמונה לפי BBOX
        bbox = metadata.get("original_bbox")  # [x_min, y_min, x_max, y_max]
        cropped_image = image.crop(bbox)

        # יישום טרנספורמציות במידה וקיימות
        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label
