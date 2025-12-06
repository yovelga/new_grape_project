import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import tifffile as tiff

# Paths for Train and Test directories
TRAIN_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Train"
TEST_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Val"
# Path to the original images
IMAGES_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/used"

# Data augmentation and normalization
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Custom dataset class
class GrapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print(f"Initializing dataset for {root_dir}...")
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from {root_dir}.")

    def _load_samples(self):
        samples = []
        for class_name in ["Grape", "Not Grape"]:
            class_dir = os.path.join(self.root_dir, class_name)
            label = 1 if class_name == "Grape" else 0
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(".tif"):
                    file_path = os.path.join(class_dir, file_name)
                    samples.append((file_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mask_path, label = self.samples[idx]

        # Load metadata from the mask file
        with tiff.TiffFile(mask_path) as tif:
            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")
            metadata = json.loads(description.value)

        # Get image name without extension
        image_name = metadata.get("image_name")

        # Search for the image file in the directory
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            potential_path = os.path.join(IMAGES_DIR, image_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            raise FileNotFoundError(f"Image not found for {image_name} in {IMAGES_DIR}")

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Crop image based on BBOX
        bbox = metadata.get("original_bbox")  # [x_min, y_min, x_max, y_max]
        cropped_image = image.crop(bbox)

        # Apply transform if specified
        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label


# Create datasets for Train and Test
print("Creating Train and Test datasets...")
train_dataset = GrapeDataset(TRAIN_DIR, transform=transform)
test_dataset = GrapeDataset(TEST_DIR, transform=transform)

# Create DataLoaders for Train and Test
print("Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("DataLoaders created successfully.")

# Load pre-trained EfficientNet model
print("Loading pre-trained EfficientNet model...")
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)  # 2 classes: Grape, Not Grape
print("Model loaded and modified for 2 classes.")

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 10

print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} started...")
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch_idx + 1}/{len(train_loader)} processed...")

    print(
        f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {running_loss / len(train_loader):.4f}"
    )

    # Evaluate the model on the test set
    print(f"Evaluating model after epoch {epoch + 1}...")
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, zero_division=1)

    print(f"Metrics after Epoch {epoch + 1}:")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

print("Training completed.")

torch.save(model.state_dict(), r"efficientnet_classifier_weights.pth")
