import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff
import numpy as np
from config import IMAGES_DIR
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class GrapeDataset(Dataset):
    @staticmethod
    def compute_context_aware_square_crop(bbox, image_width, image_height, padding_factor=0.4):
        """
        Compute a square crop with context padding to prevent distortion.

        Args:
            bbox: Tight bounding box [x_min, y_min, x_max, y_max]
            image_width: Width of the source image
            image_height: Height of the source image
            padding_factor: Amount of context to add (0.4 = 40% extra space)

        Returns:
            Square bbox [x1, y1, x2, y2] with context padding

        Logic:
            1. Calculate center of tight bbox
            2. Find longest side (max of width/height)
            3. Expand by padding_factor to include background context
            4. Create square coordinates centered on object
            5. Apply boundary clamping for safety
        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate tight bbox dimensions
        w_tight = x_max - x_min
        h_tight = y_max - y_min

        # Find center of tight bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # Calculate new square size (longest side + expansion)
        # This prevents distortion when resizing to 224x224
        max_dim = max(w_tight, h_tight)
        square_size = int(max_dim * (1 + padding_factor))

        # Create square coordinates centered on object
        half_size = square_size / 2.0
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = int(cx + half_size)
        y2 = int(cy + half_size)

        # BOUNDARY SAFETY: Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return [x1, y1, x2, y2]
    def __init__(
        self, root_dir, transform=None, balance_mode=None, input_mode="original"
    ):
        """
        input_mode:
            "original"       - מחזיר את החתך לפי original_bbox.
            "enlarged"       - מחזיר את החתך לפי padded_bbox.
            "segmentation"   - מחזיר את ה-segmentation overlay (חתך לפי original_bbox כאשר מופיעים רק הפיקסלים המסומנים, השאר שחור).
            "context_square" - מחזיר חתך מרובע עם 40% רקע נוסף (מונע עיוות).
            "context_square_segmentation" - מחזיר חתך מרובע עם מסכת סגמנטציה (מרובע + מסכה).
            "all"            - מחזיר כל דגימה כ-3 דגימות נפרדות (כל אחת עם אחת מהאפשרויות).
        balance_mode:
            "downsample", "oversample" או None.
        """
        print(
            f"Initializing dataset for {root_dir} with balance_mode = {balance_mode} and input_mode = {input_mode}..."
        )
        self.root_dir = root_dir
        self.transform = transform
        self.balance_mode = balance_mode
        self.input_mode = input_mode
        self.base_samples = self._load_samples()
        if self.input_mode == "all":
            self.length = 3 * len(self.base_samples)
        else:
            self.length = len(self.base_samples)
        print(
            f"Loaded {len(self.base_samples)} base samples from {root_dir}. Total samples (after input_mode expansion): {self.length}"
        )

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

        if self.balance_mode is None:
            balanced_samples = []
            for samples in class_samples.values():
                balanced_samples.extend(samples)
            return balanced_samples

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
        for class_name, samples in class_samples.items():
            if len(samples) < target_count:
                added = target_count - len(samples)
                print(
                    f"Under-sampled class '{class_name}': original count = {len(samples)}, added {added} samples."
                )
                balanced_samples.extend(samples + random.choices(samples, k=added))
            else:
                balanced_samples.extend(random.sample(samples, target_count))
        return balanced_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # אם input_mode=="all", נחשב את האינדקס הבסיסי וסוג המודאליות:
        if self.input_mode == "all":
            base_index = idx // 3
            modality = idx % 3  # 0: original, 1: enlarged, 2: segmentation
        else:
            base_index = idx

        mask_path, label = self.base_samples[base_index]

        # קריאת המטא-דאטה מקובץ ה-TIF
        with tiff.TiffFile(mask_path) as tif:
            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")
            metadata = json.loads(description.value)
        image_name = metadata.get("image_name")

        # חיפוש קובץ התמונה המקורית בתיקייה (לדוגמה PNG/JPG)
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            potential_path = os.path.join(IMAGES_DIR, image_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        if image_path is None:
            raise FileNotFoundError(f"Image not found for {image_name} in {IMAGES_DIR}")
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # חשב את החתכים:
        # Original crop
        original_bbox = metadata.get("original_bbox")  # [x_min, y_min, x_max, y_max]
        cropped_image = image.crop(original_bbox)

        # Enlarged crop
        padded_bbox = metadata.get("padded_bbox", original_bbox)
        enlarged_image = image.crop(padded_bbox)

        # Context-aware square crop (NEW: prevents distortion, adds 40% background context)
        context_square_bbox = self.compute_context_aware_square_crop(
            original_bbox, image_width, image_height, padding_factor=0.4
        )
        context_square_image = image.crop(tuple(context_square_bbox))

        # Segmentation overlay: חתך לפי original_bbox עם מסכה בלבד
        with tiff.TiffFile(mask_path) as tif:
            mask = tif.pages[0].asarray()
        bbox = list(map(int, original_bbox))
        mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        mask_crop = (mask_crop > 0).astype(np.uint8)
        cropped_np = np.array(cropped_image)
        mask_crop_3 = np.stack([mask_crop, mask_crop, mask_crop], axis=-1)
        segmentation_overlay_np = cropped_np * mask_crop_3
        segmentation_overlay = Image.fromarray(segmentation_overlay_np)

        # Context-aware square segmentation (NEW: square crop + segmentation mask)
        context_square_bbox_int = list(map(int, context_square_bbox))
        mask_crop_square = mask[
            context_square_bbox_int[1]:context_square_bbox_int[3],
            context_square_bbox_int[0]:context_square_bbox_int[2]
        ]
        mask_crop_square = (mask_crop_square > 0).astype(np.uint8)
        context_square_np = np.array(context_square_image)
        mask_crop_square_3 = np.stack([mask_crop_square, mask_crop_square, mask_crop_square], axis=-1)
        context_square_segmentation_np = context_square_np * mask_crop_square_3
        context_square_segmentation = Image.fromarray(context_square_segmentation_np)

        # בהתאם ל-input_mode, נחזיר את התוצאה:
        if self.input_mode == "original":
            output = cropped_image
        elif self.input_mode == "enlarged":
            output = enlarged_image
        elif self.input_mode == "segmentation":
            output = segmentation_overlay
        elif self.input_mode == "context_square":
            output = context_square_image
        elif self.input_mode == "context_square_segmentation":
            output = context_square_segmentation
        elif self.input_mode == "all":
            if modality == 0:
                output = cropped_image
            elif modality == 1:
                output = enlarged_image
            elif modality == 2:
                output = segmentation_overlay
            else:
                raise ValueError("Invalid modality index.")
        else:
            raise ValueError(
                "input_mode must be one of: original, enlarged, segmentation, context_square, context_square_segmentation, all"
            )

        if self.transform:
            output = self.transform(output)
        return output, label

#     C:\\Users\\yovel\\Desktop\\Grape_Project\\src\\preprocessing\\items_for_cnn_train\\Data_for_train_and_val_cnn\\items_for_cnn_train\\Data_for_train_and_val_cnn\\Train\\Grape'
#     C:\Users\yovel\OneDrive\Desktop\Grape_Project\src\preprocessing\items_for_cnn_train\Data_for_train_and_val_cnn\Train\Grape
