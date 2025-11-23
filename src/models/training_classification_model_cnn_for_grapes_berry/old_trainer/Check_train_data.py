import os
import json
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff

# הגדרת נתיבים לקבצים
TEST_DIR = r"/storage/yovelg/Grape/items/Data_for_train_and_val/Val"
TRAIN_DIR = r"/storage/yovelg/Grape/items/Data_for_train_and_val/Train"
IMAGES_DIR = r"/storage/yovelg/Grape/items/used"
print(f"TRAIN_DIR: {TRAIN_DIR}, IMAGES_DIR: {IMAGES_DIR}")

# הגדרת טרנספורמציות (data augmentation והנרמול)
transform_pipeline = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_mask_metadata(mask_path):
    """
    טוען את המידע המטא-דאטה מקובץ ה-TIFF (מהתיאור שבתוכו).
    """
    with tiff.TiffFile(mask_path) as tif:
        img_desc = tif.pages[0].tags.get("ImageDescription")
        metadata = json.loads(img_desc.value)
    return metadata


class GrapeDataset(Dataset):
    def __init__(self, masks_dir, images_dir, transform=None):
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        self.transform = transform
        self.samples = self._collect_mask_paths()

    def _collect_mask_paths(self):
        """
        עובר על תיקיות "Grape" ו-"Not Grape" ואוסף את הנתיבים לקבצי המסק.
        """
        paths = []
        for category in ["Grape", "Not Grape"]:
            cat_folder = os.path.join(self.masks_dir, category)
            if os.path.isdir(cat_folder):
                for filename in os.listdir(cat_folder):
                    if filename.lower().endswith(".tif"):
                        full_path = os.path.join(cat_folder, filename)
                        paths.append(full_path)
                        print(f"Loaded mask: {full_path}")
        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mask_path = self.samples[index]
        metadata = load_mask_metadata(mask_path)

        # שליפת פרטי התמונה וה-BBOX מתוך המטא-דאטה
        image_filename = metadata.get("image_name").strip("/ ")  # הסרת תווים מיותרים
        bbox = metadata.get("original_bbox")  # [x_min, y_min, x_max, y_max]

        # בניית הנתיב לקובץ התמונה על ידי הוספת הסיומת ".png"
        image_path = os.path.join(self.images_dir, image_filename + ".png")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path} for mask {mask_path}")
            return None

        # טעינת התמונה, המרה ל-RGB וחיתוך לפי ה-BBOX
        image = Image.open(image_path).convert("RGB")
        cropped_image = image.crop(bbox)

        # החלת טרנספורמציות אם הוגדרו
        if self.transform:
            cropped_image = self.transform(cropped_image)

        # קביעת תווית: אם ה-tag שווה (לא תלוי רישיות) ל-"GRAPE" אז 1, אחרת 0
        tag_value = metadata.get("tag", "").strip().upper()
        label = 1 if tag_value == "GRAPE" else 0

        return cropped_image, label, mask_path


# ---------------------------------------------------------
# הצגת מספר דוגמאות מהדאטהסט
# ---------------------------------------------------------

# יצירת דאטהסט ראשוני וה-Dataloader להצגת דוגמאות
dataset = GrapeDataset(TRAIN_DIR, IMAGES_DIR, transform=transform_pipeline)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"Number of samples in the dataset: {len(dataset)}")

# איסוף 5 דוגמאות עבור LABEL 0 ו-5 דוגמאות עבור LABEL 1
examples_label_0 = []
examples_label_1 = []

for i in range(len(dataset)):
    sample = dataset[i]
    if sample is None:
        continue
    image_tensor, label, mask_path = sample
    if label == 0 and len(examples_label_0) < 5:
        examples_label_0.append((image_tensor, label, mask_path))
    elif label == 1 and len(examples_label_1) < 5:
        examples_label_1.append((image_tensor, label, mask_path))
    if len(examples_label_0) >= 5 and len(examples_label_1) >= 5:
        break

# הצגת 5 דוגמאות עבור LABEL 0 ו-5 דוגמאות עבור LABEL 1 בתמונה אחת עם 10 תת-תמונות
plt.figure(figsize=(12, 6))
# שורה ראשונה - LABEL 0
for idx, (image_tensor, label, mask_path) in enumerate(examples_label_0):
    plt.subplot(2, 5, idx + 1)
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(image_np)
    plt.title(f"Label: {label}")
    plt.axis("off")

# שורה שנייה - LABEL 1
for idx, (image_tensor, label, mask_path) in enumerate(examples_label_1):
    plt.subplot(2, 5, idx + 6)
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(image_np)
    plt.title(f"Label: {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Displayed 5 examples for LABEL 0 and 5 examples for LABEL 1.")

# ---------------------------------------------------------
# טעינת המודל ושימוש בו על דאטהסט האימון
# ---------------------------------------------------------

# Load the saved model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)

# יצירת דאטהסטים לאימון ולבדיקה
print("Creating Train and Test datasets...")
train_dataset = GrapeDataset(TRAIN_DIR, IMAGES_DIR, transform=transform_pipeline)
test_dataset = GrapeDataset(TEST_DIR, IMAGES_DIR, transform=transform_pipeline)

# יצירת DataLoader לאימון (ולבדיקה, אם תרצה בהמשך)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# טעינת משקולות המודל
model.load_state_dict(
    torch.load(
        r"/storage/yovelg/Grape/training_classification_model_cnn_for_grapes_berry/efficientnet_classifier_weights.pth",
        weights_only=True,
    )
)

model.to(device="cuda")
model.eval()
print("Model weights loaded successfully.")

# Evaluate on the training set to find misclassified examples
misclassified = []

with torch.no_grad():
    for batch_idx, (images, labels, mask_paths) in enumerate(train_loader):
        # מעבירים את התמונות והתוויות ל-GPU
        images, labels = images.to(device="cuda"), labels.to(device="cuda")

        # מעבירים את התמונות דרך המודל ומקבלים את הפלט
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # בדיקה אילו דוגמאות סווגו לא נכון
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified.append(
                    {
                        "batch_index": batch_idx,
                        "image_index": i,
                        "predicted": predicted[i].item(),
                        "true_label": labels[i].item(),
                        "mask_path": mask_paths[i],
                    }
                )

print(f"Total misclassified examples: {len(misclassified)}")

# שמירת הדוגמאות השגויות ל-Excel
excel_path = r"/storage/yovelg/Grape/training_classification_model_cnn_for_grapes_berry/misclassified_examples.xlsx"
df = pd.DataFrame(misclassified)
df.to_excel(excel_path, index=False)
print(f"Misclassified examples saved to {excel_path}")

# הצגת 5 הדוגמאות השגויות הראשונות
print("Displaying top 5 biggest failures...")
for i, mis in enumerate(misclassified[:5]):  # Show first 5 misclassified
    mask_path = mis["mask_path"]
    predicted = mis["predicted"]
    true_label = mis["true_label"]

    # טוענים את קובץ ה-mask כדי לשלוף מטא-דאטה
    with tiff.TiffFile(mask_path) as tif:
        tags = tif.pages[0].tags
        description = tags.get("ImageDescription")
        metadata = json.loads(description.value)

    image_name = metadata.get("image_name")
    bbox = metadata.get("original_bbox")

    # טוענים את התמונה המקורית
    image_path = os.path.join(IMAGES_DIR, image_name)
    image = Image.open(image_path).convert("RGB")

    # חותכים ומציגים
    cropped_image = image.crop(bbox)
    plt.imshow(cropped_image)
    plt.title(f"Predicted: {predicted}, True: {true_label}")
    plt.axis("off")
    plt.show()
