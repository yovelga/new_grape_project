from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
    QComboBox,
    QGridLayout,
    QListWidget,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
import joblib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

# For EfficientNet-B0
from torchvision.models import efficientnet_b0


# Import functions from backend
from backend import (
    load_rgb_image_from_folder,
    load_hsi_image_from_folder,
    load_canon_rgb_image,
    compute_signatures_for_plot,
)


from pathlib import Path
import sys, os, json, pathlib
import re


# load_dotenv(dotenv_path=str(_PROJECT_ROOT / r".env"))
load_dotenv()

# Correctly set the project root (one level up from current working directory)


BASE_PATH = Path(os.getenv("BASE_PATH"))

AUTOENCODER_MODEL_PATH = BASE_PATH / os.getenv("AUTOENCODER_MODEL_PATH")


SAM2_MODULES = BASE_PATH / os.getenv("SAM_MODEL_PATH")


segmet_classes = json.loads(os.getenv("CLASSES_SEGMENT"))
sys.path.append(os.path.abspath(SAM2_MODULES))

from MaskGenerator.segment_object_module import create_point_segmenter
from MaskGenerator.mask_generator_module import (
    initial_settings,
    initialize_sam2_predictor,
)
from MaskGenerator.mask_generator_module import generate_detections
from MaskGenerator.mask_generator_module import initialize_sam2


# ===== Utility functions =====
# def load_grape_classifier(device):
#     # Build model architecture
#     model = efficientnet_b0(
#         weights=None
#     )  # or weights="IMAGENET1K_V1" if you used pretrained
#     model.classifier[1] = torch.nn.Linear(
#         model.classifier[1].in_features, 2
#     )  # 2 classes: grape/not grape
#
#     # Load weights
#     model_path = BASE_PATH / os.getenv("GRAPE_MODEL_CALSSIFIER_PATH")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model


def extract_cluster_id_from_path(path):
    p = Path(path)
    for parent in p.parents:
        if re.match(r"^\d{1,2}_\d{2}$", parent.name):
            return parent.name
    match = re.search(r"(\d{1,2}_\d{2})", str(path))
    return match.group(1) if match else None


# transform_rgb = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),  # הפוך לגרייסקל!
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # או סטטיסטיקות שחישבת
])


#
# def predict_grape_confidence(rgb_img, mask, model, transform, device):
#     y_indices, x_indices = np.where(mask)
#     if len(y_indices) == 0 or len(x_indices) == 0:
#         return None, None
#     y_min, y_max = y_indices.min(), y_indices.max()
#     x_min, x_max = x_indices.min(), x_indices.max()
#     crop = rgb_img[y_min : y_max + 1, x_min : x_max + 1, :]
#
#     # Optional: mask out background if you want (set to black)
#     mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
#     crop[~mask_crop] = 0
#
#     input_tensor = transform(crop).unsqueeze(0).to(device)
#     with torch.no_grad():
#         logits = model(input_tensor)
#         probs = F.softmax(logits, dim=1)
#         confidence = probs[0, 1].item()  # Assuming class 1 is "grape"
#     return confidence, probs[0].cpu().numpy()



def predict_grape_confidence(rgb_img, mask, model, transform, device):
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # גבולות - לא לעבור 512 ולא לעבור את גודל התמונה!
    H, W = rgb_img.shape[:2]
    y_max = min(y_max, H - 1, 511)
    x_max = min(x_max, W - 1, 511)
    y_min = max(0, y_min)
    x_min = max(0, x_min)

    crop = rgb_img[y_min : y_max + 1, x_min : x_max + 1, :]
    mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]

    # אפשר להחזיר את ניקוי הרקע אם תרצה:
    # crop[~mask_crop] = 0

    # אם crop ב-[H,W], הפוך ל-[H,W,1]
    if crop.ndim == 2:
        crop = np.expand_dims(crop, axis=-1)

    input_tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence = probs[0, 1].item()
    return confidence, probs[0].cpu().numpy()




def get_hsi_pixel_from_mask(mask_pixel, hsi_image):
    y, x = mask_pixel
    img_width = hsi_image.shape[1]
    hsi_row = img_width - x - 1
    hsi_col = y
    return hsi_image[hsi_row, hsi_col, :]


def get_center_pixel_from_mask(mask):
    """
    Returns the center-most pixel (row, col) from a binary mask.
    If the mask is empty, returns None.
    """
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    centroid = np.mean(indices, axis=0)
    center_idx = np.argmin(np.sum((indices - centroid) ** 2, axis=1))
    return tuple(indices[center_idx])


def save_last_progress_to_env(cluster_id, date, env_path=None):
    """
    Save the current cluster ID and date to the .env file (as LAST_CLUSTER_PROGRESS).
    """
    if env_path is None:
        env_path = BASE_PATH / ".env"
    else:
        env_path = Path(env_path)
    key = "LAST_CLUSTER_PROGRESS"
    new_line = f"{key}={cluster_id},{date}\n"
    lines = []
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    found = False
    for idx, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[idx] = new_line
            found = True
            break
    if not found:
        lines.append(new_line)
    with open(env_path, "w") as f:
        f.writelines(lines)


def extract_date_from_path(path):
    """
    Extracts date in the format DD.MM.YY from a given file or folder path.
    Returns the first match found, or None if not found.
    """
    pattern = r"\b\d{2}\.\d{2}\.\d{2}\b"
    match = re.search(pattern, path)
    if match:
        return match.group(0)
    return None


class SegmentDecisionDialog(QDialog):
    def __init__(self, overlay_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segment Decision")
        layout = QVBoxLayout(self)
        # Show overlay image
        label = QLabel(self)
        height, width, channels = overlay_img.shape
        bytes_per_line = channels * width
        q_image = QImage(
            overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        layout.addWidget(label)
        # Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("SAVE") # Save the segment
        self.next_btn = QPushButton("NEXT") # Skip to next segment
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)
        self.save_btn.clicked.connect(lambda: self.done(1))
        self.next_btn.clicked.connect(lambda: self.done(0))


def load_cluster_irrigation_mapping():
    """
    Loads cluster-irrigation mapping from .env variable CLUSTER_IRRIGATION.
    Returns a dictionary: {cluster_id: irrigation_color}
    """
    mapping_str = os.getenv("CLUSTER_IRRIGATION")
    mapping = {}
    if not mapping_str:
        print("CLUSTER_IRRIGATION is not set or empty!")
        return mapping
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            mapping[k] = v
        else:
            print(f"Warning: Invalid pair '{pair}' in CLUSTER_IRRIGATION")
    return mapping


mapping = None  # הגדרה מחוץ לפונקציה


def get_irrigation_color_for_cluster(cluster_id):
    """
    Returns the irrigation color for a given cluster_id
    """
    global mapping
    if mapping is None:
        mapping = load_cluster_irrigation_mapping()
    return mapping.get(cluster_id, "UNKNOWN")


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.array([0, 0])
    return np.array([int(np.mean(ys)), int(np.mean(xs))])


# def load_grape_classifier(device):
#     from torchvision.models import efficientnet_b0
#     import torch.nn as nn
#
#     # בנה EfficientNet עם conv ראשון לערוץ בודד
#     model = efficientnet_b0(weights=None)
#     old_conv = model.features[0][0]
#     model.features[0][0] = nn.Conv2d(
#         1,
#         old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=old_conv.bias is not None
#     )
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
#
#     # טען משקלים של המודל המאומן שלך
#     model_path = BASE_PATH / os.getenv("GRAPE_MODEL_CALSSIFIER_PATH")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model


def load_grape_classifier(device):
    from torchvision.models import efficientnet_b0
    import torch.nn as nn
    model = efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model_path = BASE_PATH / os.getenv("GRAPE_MODEL_CALSSIFIER_GRAY_COLORS_PATH")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model



class HSI_RGB_Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rgb_image_path = None
        self.last_clicked_pixel = None
        self.sam2_segmenter = None
        self.sam2_auto_generator = None
        self.current_folder_path = None
        self.current_folder_path = None
        self.sam_count = None
        # self.rgb_image_path = None
        self.rgb_image = None
        self.auto_sam_detections = []  # This is where we will store all the segments.
        self.current_detection_index = -1  # Current segment index
        self.count_segment_saved = 0  # Counter for saved segments
        self.segments2saved = 7

        self.setWindowTitle("HSI and RGB Viewer")
        self.setGeometry(100, 100, 1600, 800)
        self.installEventFilter(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grape_classifier = load_grape_classifier(self.device)

        try:
            self.autoencoder_model = joblib.load(AUTOENCODER_MODEL_PATH)
        except Exception as e:
            self.autoencoder_model = None

        # Initialize variables
        self.rgb_image = None
        self.hsi_image = None
        self.current_band = 0
        self.saved_pixels = []  # List of saved pixels info
        self.folder_list = []  # List of folder paths
        self.current_folder_index = -1  # Current folder index
        self.current_capture_index = -1
        self.capture_references = []

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Grid layout for images
        self.image_grid = QGridLayout()
        self.layout.addLayout(self.image_grid)

        self.folder_path_label = QLabel("Current Folder: None")
        self.folder_path_label.setAlignment(Qt.AlignCenter)
        self.folder_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.folder_path_label)

        # RGB Image Label
        self.rgb_label = QLabel("RGB Image (HS folder) will be displayed here")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rgb_label.setMinimumSize(512, 512)
        self.rgb_label.setMaximumSize(512, 512)
        self.image_grid.addWidget(self.rgb_label, 0, 0)

        # HSI Image Label
        self.hsi_label = QLabel("HSI Image will be displayed here")
        self.hsi_label.setAlignment(Qt.AlignCenter)
        self.hsi_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hsi_label.setMinimumSize(512, 512)
        self.hsi_label.setMaximumSize(512, 512)
        self.image_grid.addWidget(self.hsi_label, 0, 1)

        # Canon RGB Image Label
        self.canon_rgb_label = QLabel("Canon RGB Image will be displayed here")
        self.canon_rgb_label.setAlignment(Qt.AlignCenter)
        self.canon_rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canon_rgb_label.setMinimumSize(512, 512)
        self.canon_rgb_label.setMaximumSize(512, 512)
        self.image_grid.addWidget(self.canon_rgb_label, 0, 2)

        # Connect mouse click events for HSI and RGB images
        self.hsi_label.mousePressEvent = self.handle_hsi_click
        self.rgb_label.mousePressEvent = self.handle_rgb_click

        # Controls
        self.controls_layout = QHBoxLayout()
        self.layout.addLayout(self.controls_layout)

        self.load_folder_button = QPushButton("Load Folder")
        self.load_folder_button.clicked.connect(self.handle_load_folder)
        self.controls_layout.addWidget(self.load_folder_button)

        self.sam2_button = QPushButton("SAM2")
        self.sam2_button.clicked.connect(self.handle_sam2)
        self.controls_layout.addWidget(self.sam2_button)

        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setValue(0)
        self.band_slider.valueChanged.connect(self.update_hsi_band)
        self.controls_layout.addWidget(self.band_slider)

        self.save_button = QPushButton("Export Histogram")
        self.save_button.clicked.connect(self.handle_export_histogram)
        self.controls_layout.addWidget(self.save_button)

        # Add UI for saved pixels
        self.saved_pixels_list = QListWidget()
        self.saved_pixels_list.itemClicked.connect(self.display_saved_pixel_histogram)
        self.saved_pixels_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.saved_pixels_list)

        # Additional control buttons
        self.save_pixel_button = QPushButton("Save Pixel")
        self.save_pixel_button.clicked.connect(self.save_pixel)
        self.controls_layout.addWidget(self.save_pixel_button)

        self.delete_pixels_button = QPushButton("Delete Selected Pixels")
        self.delete_pixels_button.clicked.connect(self.delete_selected_pixels)
        self.controls_layout.addWidget(self.delete_pixels_button)

        self.export_csv_button = QPushButton("Export to CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.controls_layout.addWidget(self.export_csv_button)

        # Plot widget for embedded plots in the UI
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(700, 500)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.layout.addWidget(self.plot_widget)

        # Status bar
        self.status_bar = self.statusBar()

        # ComboBox for pixel type
        self.pixel_type_combo = QComboBox()
        self.pixel_type_combo.addItems(["Regular", "Crack"])
        self.controls_layout.addWidget(self.pixel_type_combo)

        # Buttons for managing folders
        self.add_folder_button = QPushButton("Add list")
        self.add_folder_button.clicked.connect(self.handle_add_folder)
        self.controls_layout.addWidget(self.add_folder_button)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.handle_previous_folder)
        self.controls_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.handle_next_folder)
        self.controls_layout.addWidget(self.next_button)

        self.auto_sam_button = QPushButton("Auto SAM")
        self.auto_sam_button.clicked.connect(self.handle_auto_sam)
        self.controls_layout.addWidget(self.auto_sam_button)

        self.prev_button_c = QPushButton("Prev cap")
        self.prev_button_c.clicked.connect(self.handle_previous_capture)
        self.controls_layout.addWidget(self.prev_button_c)

        self.next_button_c = QPushButton("Next cap")
        self.next_button_c.clicked.connect(self.handle_next_capture)
        self.controls_layout.addWidget(self.next_button_c)

        self.save_segment_button = QPushButton("Save Segment")
        self.save_segment_button.clicked.connect(self.ask_save_current_detection)
        self.controls_layout.addWidget(self.save_segment_button)



    def ask_segment_class(self):
        classes = segmet_classes
        classes_display = classes + ["Other..."]
        # QInputDialog for selection
        item, ok = QInputDialog.getItem(
            self,
            "Select Segment Class",
            "Choose the segment class:",
            classes_display,
            editable=False,
        )
        if ok:
            if item == "Other...":
                text, ok2 = QInputDialog.getText(
                    self, "New Class", "Enter new class name:"
                )
                if ok2 and text:
                    # Save the new class
                    if text not in classes:
                        classes.append(text)
                        self.save_classes(classes)
                    return text
            else:
                return item
        return None

    def handle_load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            try:
                print(f"Selected folder path: {folder_path}")
                folder_main = os.path.basename(os.path.dirname(folder_path))
                folder_date = os.path.basename(folder_path)
                print(f"Extracted folder_main: {folder_main}")
                print(f"Extracted folder_date: {folder_date}")
                self.folder_main = folder_main if folder_main else "Unknown"
                self.folder_date = folder_date if folder_date else "Unknown"

                # Load RGB Image (from HS folder)
                # self.rgb_image,self.rgb_image_path = load_rgb_image_from_folder(folder_path)
                self.rgb_image, self.rgb_image_path = load_rgb_image_from_folder(
                    folder_path
                )
                # print("heyyyyy")
                print(f"iam hare:  {self.rgb_image_path}")
                # self.rgb_image = load_rgb_image_from_folder(folder_path)
                self.display_image(self.rgb_image, is_hsi=False)

                # Load HSI Image
                self.hsi_image = load_hsi_image_from_folder(folder_path)
                self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
                self.update_hsi_band()

                # Load Canon RGB Image (from RGB folder)
                self.canon_rgb_image = load_canon_rgb_image(folder_path)
                self.display_image(
                    self.canon_rgb_image,
                    is_hsi=False,
                    target_label=self.canon_rgb_label,
                )

                self.status_bar.showMessage(
                    f"Images Loaded Successfully from Folder: {folder_path}"
                )
                self.current_folder_path = folder_path
            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")

    def update_hsi_band(self):
        if self.hsi_image is not None:
            self.current_band = self.band_slider.value()
            band_image = self.hsi_image[:, :, self.current_band]
            band_image = cv2.normalize(
                band_image, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            band_image = cv2.rotate(band_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(band_image, is_hsi=True)
            self.status_bar.showMessage(f"Displaying HSI Band {self.current_band}")

    def display_image(self, image, is_hsi=False, target_label=None):
        label = (
            target_label
            if target_label
            else (self.hsi_label if is_hsi else self.rgb_label)
        )
        if is_hsi:
            height, width = image.shape
            q_image = QImage(image.data, width, height, QImage.Format_Grayscale8)
        else:
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def handle_export_histogram(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Histogram", "", "PNG Files (*.png)"
        )
        if file_path:
            self.status_bar.showMessage(f"Histogram exported to {file_path}")

    def handle_rgb_click(self, event):
        if self.rgb_image is not None and self.hsi_image is not None:
            img_height, img_width, _ = self.rgb_image.shape
            label_width = self.rgb_label.width()
            label_height = self.rgb_label.height()
            x = int((event.pos().x() / label_width) * img_width)
            y = int((event.pos().y() / label_height) * img_height)
            if 0 <= x < img_width and 0 <= y < img_height:
                # Map click coordinates to HSI image coordinates (adjust if needed)
                intensity_values = self.hsi_image[img_width - x - 1, y, :]
                self.last_clicked_pixel = (x, y, intensity_values)
                print(f"x: {x},y: {y}")
                # Instead of calling backend plt.show() function, use the UI embedding function:
                self.plot_pixel_signatures_in_ui(intensity_values)
                self.status_bar.showMessage(
                    f"Selected pixel ({x}, {y}) from RGB. Plots displayed."
                )
                self.mark_pixel_on_hsi_image(x, y)
                self.mark_pixel_on_hsi_rgb_image(x, y)
            else:
                QMessageBox.warning(self, "Warning", "Selected pixel is out of bounds.")

    def handle_hsi_click(self, event):
        if self.hsi_image is not None:
            img_height, img_width, _ = self.hsi_image.shape
            label_width = self.hsi_label.width()
            label_height = self.hsi_label.height()
            x = int((event.pos().x() / label_width) * img_width)
            y = int((event.pos().y() / label_height) * img_height)
            if 0 <= x < img_width and 0 <= y < img_height:
                intensity_values = self.hsi_image[img_width - x - 1, y, :]
                self.last_clicked_pixel = (x, y, intensity_values)
                print(f"x: {x},y: {y}")
                # Use the UI embedded plotting function:
                self.plot_pixel_signatures_in_ui(intensity_values)
                self.status_bar.showMessage(
                    f"Selected pixel ({x}, {y}) from HSI. Plots displayed."
                )
                self.mark_pixel_on_hsi_image(x, y)
                self.mark_pixel_on_hsi_rgb_image(x, y)
            else:
                QMessageBox.warning(self, "Warning", "Selected pixel is out of bounds.")

    def mark_pixel_on_hsi_rgb_image(self, x, y):
        if self.rgb_image is not None:
            rgb_copy = self.rgb_image.copy()
            cv2.circle(rgb_copy, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            self.display_image(rgb_copy, is_hsi=False, target_label=self.rgb_label)

    def mark_pixel_on_hsi_image(self, x, y):
        if self.hsi_image is not None:
            band_image = self.hsi_image[:, :, self.current_band].copy()
            band_image = cv2.normalize(
                band_image, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            cv2.circle(
                band_image,
                (y, self.hsi_label.width() - x - 1),
                radius=5,
                color=(255, 0, 0),
                thickness=-1,
            )
            band_image = cv2.rotate(band_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(band_image, is_hsi=True)

    def save_pixel(self):
        if hasattr(self, "last_clicked_pixel"):
            x, y, intensity = self.last_clicked_pixel
            folder_main = getattr(self, "folder_main", "Unknown")
            folder_date = getattr(self, "folder_date", "Unknown")
            pixel_type = self.pixel_type_combo.currentText()
            self.saved_pixels.append(
                {
                    "position": (x, y),
                    "intensity": intensity,
                    "folder_main": folder_main,
                    "folder_date": folder_date,
                    "type": pixel_type,
                }
            )
            display_text = f"Pixel ({x}, {y}) - {pixel_type} [Folder: {folder_main}, Subfolder: {folder_date}]"
            self.saved_pixels_list.addItem(display_text)
            self.status_bar.showMessage(f"Pixel ({x}, {y}) saved as {pixel_type}")
        else:
            QMessageBox.warning(self, "Warning", "No pixel selected to save.")

    def delete_selected_pixels(self):
        selected_items = self.saved_pixels_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No pixels selected for deletion.")
            return
        for item in selected_items:
            index = self.saved_pixels_list.row(item)
            self.saved_pixels.pop(index)
            self.saved_pixels_list.takeItem(index)
        self.status_bar.showMessage("Selected pixels deleted.")

    def export_to_csv(self):
        if not self.saved_pixels:
            QMessageBox.warning(self, "Warning", "No pixels to export.")
            return

        csv_file_name = f"detected_pixels{datetime.now().strftime('%Y%m%d')}.csv"
        with open(csv_file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["Folder_Main", "Folder_Date", "Type", "X", "Y"]
                + [f"Band_{i}" for i in range(len(self.saved_pixels[0]["intensity"]))]
            )
            for pixel in self.saved_pixels:
                x, y = pixel["position"]
                folder_main = pixel["folder_main"]
                folder_date = pixel["folder_date"]
                pixel_type = pixel["type"]
                row = [folder_main, folder_date, pixel_type, x, y] + list(
                    pixel["intensity"]
                )
                writer.writerow(row)
        QMessageBox.information(
            self, "Export Successful", f"Pixels exported to {csv_file_name}"
        )

    def display_saved_pixel_histogram(self, item):
        index = self.saved_pixels_list.row(item)
        pixel_data = self.saved_pixels[index]
        intensity_values = pixel_data["intensity"]
        self.plot_pixel_signatures_in_ui(intensity_values)
        self.status_bar.showMessage(
            f"Displaying histogram for Pixel {pixel_data['position']} [Folder: {pixel_data['folder_main']}, Subfolder: {pixel_data['folder_date']}]"
        )

    def handle_add_folder(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Add Folders", "Paste folder paths (one per line):"
        )
        if ok and text.strip():
            folder_paths = [line.strip() for line in text.split("\n") if line.strip()]
            for folder_path in folder_paths:
                if os.path.exists(folder_path):
                    self.folder_list.append(folder_path)
                    self.status_bar.showMessage(f"Added folder: {folder_path}")
                else:
                    QMessageBox.warning(
                        self, "Warning", f"Folder not found: {folder_path}"
                    )
            if len(self.folder_list) == len(folder_paths):
                self.current_folder_index = 0
                self.load_current_folder()

    def handle_check_item(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Add Capture Reference(s)", "Paste capture references (one per line):"
        )
        if ok and text.strip():
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                try:
                    # פורמט צפוי: "Image: D:/dest/1_54/01.09.24, x=389, y=433"
                    parts = line.split(",")
                    if len(parts) < 3:
                        raise ValueError("Line does not contain all required parts.")
                    folder_part = parts[0].strip()  # "Image: D:/dest/1_54/01.09.24"
                    x_part = parts[1].strip()  # "x=389"
                    y_part = parts[2].strip()  # "y=433"

                    # הסרת "Image:" והפקת הנתיב
                    folder_path = folder_part.split("Image:")[1].strip()
                    x = int(x_part.split("=")[1].strip())
                    y = int(y_part.split("=")[1].strip())

                    print(f"Added capture reference: {folder_path}, x={x}, y={y}")

                    if os.path.exists(folder_path):
                        # הוספת ההפניה לרשימה כ-dictionary
                        self.capture_references.append(
                            {"folder": folder_path, "x": x, "y": y}
                        )
                        self.status_bar.showMessage(
                            f"Added capture reference: {folder_path}, x={x}, y={y}"
                        )
                    else:
                        QMessageBox.warning(
                            self, "Warning", f"Folder not found: {folder_path}"
                        )
                except Exception as e:
                    QMessageBox.warning(
                        self, "Warning", f"Failed to parse line:\n'{line}'\nError: {e}"
                    )

            if self.capture_references:
                self.current_capture_index = 0
                # ניתן לקרוא לפונקציה שמציגה את ההפניה הראשונה, למשל:
                print("iam hare")
                self.load_current_capture()

    def handle_previous_capture(self):
        if self.auto_sam_detections:
            if self.current_detection_index > 0:
                self.current_detection_index -= 1
                self.show_current_detection()
            else:
                QMessageBox.warning(self, "Warning", "This is the first segment.")
        else:
            if self.current_capture_index > 0:
                self.current_capture_index -= 1
                ref = self.capture_references[self.current_capture_index]
                print(
                    f"Added capture reference: {ref['folder']}, x={ref['x']}, y={ref['y']}"
                )
                self.load_current_capture()  # If needed, pass ref['folder'] to this function
            else:
                QMessageBox.warning(
                    self, "Warning", "This is the first capture reference in the list."
                )


    def any_user_action(self):
        if hasattr(self, "save_timer"):
            self.save_timer.stop()


    def auto_save_and_next(self):
        self.save_current_detection()
        self.current_detection_index += 1
        # warning if no more detections

        if self.current_detection_index >= len(self.auto_sam_detections):
            self.count_segment_saved = 0
            self.handle_next_folder()
            self.handle_auto_sam()
            return
        self.show_current_detection()


    def handle_next_capture(self):
        if self.auto_sam_detections:
            if self.current_detection_index < len(self.auto_sam_detections) - 1:
                self.current_detection_index += 1
                self.show_current_detection()
            else:
                QMessageBox.warning(self, "Warning", "This is the last segment.")
        else:
            if self.current_capture_index < len(self.capture_references) - 1:
                self.current_capture_index += 1
                ref = self.capture_references[self.current_capture_index]
                print(
                    f"Added capture reference: {ref['folder']}, x={ref['x']}, y={ref['y']}"
                )
                self.load_current_capture()  # If needed, pass ref['folder'] to this function
            else:
                QMessageBox.warning(
                    self, "Warning", "This is the last capture reference in the list."
                )

    def handle_previous_folder(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.load_current_folder()
        else:
            QMessageBox.warning(
                self, "Warning", "This is the first folder in the list."
            )

    def handle_next_folder(self):
        if self.current_folder_index < len(self.folder_list) - 1:
            self.current_folder_index += 1
            self.load_current_folder()
        else:
            QMessageBox.warning(self, "Warning", "This is the last folder in the list.")

    def load_current_folder(self):
        folder_path = self.folder_list[self.current_folder_index]
        self.folder_path_label.setText(f"Current Folder: {folder_path}")
        try:
            self.folder_main = os.path.basename(os.path.dirname(folder_path))
            self.folder_date = os.path.basename(folder_path)
            self.rgb_image, self.rgb_image_path = load_rgb_image_from_folder(
                folder_path
            )

            self.display_image(self.rgb_image, is_hsi=False)
            self.hsi_image = load_hsi_image_from_folder(folder_path)
            self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
            self.update_hsi_band()
            self.canon_rgb_image = load_canon_rgb_image(folder_path)
            self.display_image(
                self.canon_rgb_image, is_hsi=False, target_label=self.canon_rgb_label
            )
            self.status_bar.showMessage(f"Loaded folder: {folder_path}")
            self.sam_count = 1
        except Exception as e:
            self.status_bar.showMessage(f"Error loading folder: {e}")

    def load_current_capture(self):
        ref = self.capture_references[self.current_capture_index]
        folder_path = ref["folder"]
        self.folder_path_label.setText(f"Current Capture Folder: {folder_path}")

        try:
            self.folder_main = os.path.basename(os.path.dirname(folder_path))
            self.folder_date = os.path.basename(folder_path)

            # Load and display your RGB image
            print(f"load_rgb_image_from_folder {folder_path}")
            self.rgb_image, _ = load_rgb_image_from_folder(folder_path)
            self.display_image(self.rgb_image, is_hsi=False)

            # Load and display your HSI image
            print(f"load_hsi_image_from_folder {folder_path}")
            self.hsi_image = load_hsi_image_from_folder(folder_path)
            self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
            self.update_hsi_band()

            # Load and display your canon RGB image (if applicable)
            print(f"load_canon_rgb_image {folder_path}")
            self.canon_rgb_image = load_canon_rgb_image(folder_path)
            self.display_image(
                self.canon_rgb_image, is_hsi=False, target_label=self.canon_rgb_label
            )
            # self.mark_pixel_on_hsi_image(ref["x"], ref["y"])
            # self.mark_pixel_on_hsi_rgb_image(ref["x"], ref["y"])

            x = ref["x"]
            y = ref["y"]
            img_height, img_width, _ = self.rgb_image.shape
            intensity_values = self.hsi_image[img_width - x - 1, y, :]
            self.plot_pixel_signatures_in_ui(intensity_values)
            self.status_bar.showMessage(
                f"Selected pixel ({x}, {y}) from HSI. Plots displayed."
            )
            self.mark_pixel_on_hsi_image(x, y)
            self.mark_pixel_on_hsi_rgb_image(x, y)

            # Optionally, you can use ref['x'] and ref['y'] here if needed.
            self.status_bar.showMessage(
                f"Loaded capture: {folder_path}, x={ref['x']}, y={ref['y']}"
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error loading capture: {e}")

    def next_folder(self):
        if self.current_folder_index < len(self.folder_list) - 1:
            self.current_folder_index += 1
            self.load_current_folder()
        else:
            self.status_bar.showMessage("You are at the last folder.")

    def prev_folder(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.load_current_folder()
        else:
            self.status_bar.showMessage("You are at the first folder.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.save_pixel()
        elif event.key() == Qt.Key_D:
            self.prev_folder()
        elif event.key() == Qt.Key_A:
            self.next_folder()
        else:
            super().keyPressEvent(event)

    def plot_pixel_signatures_in_ui(self, intensity_values):
        # Clear previous plots
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Only use normalized and autoencoder signatures
        norm_sig = intensity_values
        if self.autoencoder_model is not None:
            reconstructed = self.autoencoder_model.predict(
                norm_sig.reshape(1, -1)
            ).flatten()
            mse = np.mean((norm_sig - reconstructed) ** 2)
        else:
            reconstructed = np.zeros_like(norm_sig)
            mse = 0

        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        x = np.arange(1, len(norm_sig) + 1)
        ax1.plot(x, norm_sig, marker="o")
        ax1.set_title("Normalized Spectral Signature")
        ax1.set_xlabel("Band")
        ax1.set_ylabel("Normalized Value")
        ax1.grid(True)

        ax2.plot(x, norm_sig, label="Original", marker="o")
        ax2.plot(x, reconstructed, label="Reconstructed", linestyle="--", marker="x")
        ax2.set_title(f"AutoEncoder (MSE={mse:.4f})")
        ax2.set_xlabel("Channel")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        fig.tight_layout()
        self.plot_layout.addWidget(canvas)
        canvas.draw()

    def handle_sam2(self):

        if self.rgb_image_path is None:
            print(f"if not self.rgb_image_path: {self.rgb_image_path}")
            QMessageBox.warning(self, "Warning", "load image first.")
            return

        # Verify pixel selection and folder loading
        if self.last_clicked_pixel is None:
            print(f"self.last_clicked_pixel: {self.last_clicked_pixel}")
            QMessageBox.warning(
                self, "Warning", "Select a pixel and load a folder first."
            )
            return

        x, y, intensity_values = self.last_clicked_pixel
        print(f"x: {x},y: {y} selected")
        print(f"self.rgb_image_path: {self.rgb_image_path}")
        print(f"self.sam_count: {self.sam_count}")

        x, y, _ = self.last_clicked_pixel
        img_path = self.rgb_image_path

        # Progress box
        progress = QProgressDialog("Segmenting...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        # Quick initialization of the SAM2 predictor
        if self.sam2_segmenter is None:
            initial_settings()
            predictor = initialize_sam2_predictor()
            self.sam2_segmenter = create_point_segmenter(predictor)

        # Running the segmentation
        image_rgb, mask_bool = self.sam2_segmenter.segment_object(img_path, [(x, y)])
        progress.close()

        # Overlay on RGB image
        overlay = image_rgb.copy()
        overlay[mask_bool] = overlay[mask_bool] * 0.7 + np.array([0, 0, 255]) * 0.3
        self.display_image(overlay, is_hsi=False, target_label=self.rgb_label)

        # Overlay על תמונת HSI
        band = self.current_band
        hsi_band = self.hsi_image[:, :, band]
        norm = cv2.normalize(hsi_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rot = cv2.rotate(norm, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB)
        # mask_rot = cv2.rotate(mask_bool.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
        mask_map = mask_bool.astype(bool)
        overlay_hsi = rgb.copy()
        # overlay_hsi[mask_rot] = overlay_hsi[mask_rot] * 0.7 + np.array([0, 255, 0]) * 0.3
        overlay_hsi[mask_map] = (
            overlay_hsi[mask_map] * 0.7 + np.array([0, 255, 0]) * 0.3
        )
        self.display_image(overlay_hsi, is_hsi=False, target_label=self.hsi_label)

        # Question whether to keep
        reply = QMessageBox.question(
            self, "Save?", "Save mask & JSON?", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            segment_class = self.ask_segment_class()
            if not segment_class:
                self.status_bar.showMessage(
                    "No class selected, segmentation not saved."
                )
                return

            JSON_DIR = r"/dataset_builder_grapes/Research_irrigation_methods/output/cracks/jsons"
            MASK_DIR = r"/dataset_builder_grapes/Research_irrigation_methods/output/cracks/masks"
            os.makedirs(JSON_DIR, exist_ok=True)
            os.makedirs(MASK_DIR, exist_ok=True)
            stem = pathlib.Path(img_path).stem
            mask_path = os.path.join(
                MASK_DIR, f"sample_{self.sam_count}_{stem}_mask.png"
            )
            json_path = os.path.join(JSON_DIR, f"sample_{self.sam_count}_{stem}.json")
            cv2.imwrite(mask_path, (mask_bool.astype(np.uint8) * 255))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"image_path": img_path, "mask_path": mask_path}, f, indent=2)
            self.status_bar.showMessage(f"Saved ➜ {mask_path}\nJSON ➜ {json_path}")
            self.sam_count += 1
        else:
            self.status_bar.showMessage("Segmentation not saved.")

    def handle_auto_sam(self):
        if self.rgb_image is None:
            QMessageBox.warning(self, "Warning", "Load image first.")
            return

        img = self.rgb_image
        img_resized = cv2.resize(img, (512, 512))

        if self.sam2_auto_generator is None:
            initial_settings()
            self.sam2_auto_generator = initialize_sam2()

        try:
            detections = generate_detections(
                self.sam2_auto_generator, img_resized, min_area=400, max_area=5000
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        if not detections:
            QMessageBox.warning(self, "Warning", "No segments found.")
            return

        # --- Run classifier, calculate area and confidence ---
        detection_infos = []
        for det in detections:
            mask = det.mask[0] if isinstance(det.mask, list) else det.mask
            area = int(mask.sum())
            confidence, _ = predict_grape_confidence(
                self.rgb_image.copy(),
                mask,
                self.grape_classifier,
                transform,
                self.device,
            )
            detection_infos.append(
                {
                    "det": det,
                    "area": area,
                    "confidence": confidence if confidence is not None else 0.0,
                }
            )

        # filtered = [d for d in detection_infos if d["confidence"] > 0.8]
        # filtered_sorted = sorted(
        #     filtered, key=lambda d: ( d["area"],d["confidence"]), reverse=True
        # )
        # detection_infos = filtered_sorted[:]




        MIN_AREA = 400


        center = np.array([256, 256])


        # Calculate distance from center for each detection
        for d in detection_infos:
            mask = d["det"].mask[0] if isinstance(d["det"].mask, list) else d["det"].mask
            centroid = mask_centroid(mask)
            d["distance_from_center"] = np.linalg.norm(centroid - center)

        # Filter detections based on confidence and area
        filtered = [
            d for d in detection_infos
            if d["confidence"] > 0.6 and d["area"] > MIN_AREA
        ]

        # Filter out detections with no valid mask
        filtered_sorted = sorted(
            filtered,
            key=lambda d: (d["distance_from_center"])
        )

        detection_infos = filtered_sorted[:] # Default to all
        # detections if no filtering is applied
        self.segments2saved  = min(len(detection_infos), 7)  # Limit to 7 segments for display

        # --- Store for later use ---
        self.auto_sam_detections = [d["det"] for d in detection_infos]
        self.auto_sam_meta = detection_infos  # Store meta for each detection
        self.current_detection_index = 0
        self.show_current_detection()


    def show_current_detection(self):
        idx = self.current_detection_index
        if not self.auto_sam_detections or idx < 0 or idx >= len(self.auto_sam_detections):
            self.status_bar.showMessage("No more detections to show in this image.")
            return

        det = self.auto_sam_detections[idx]
        mask = det.mask[0] if isinstance(det.mask, list) else det.mask

        # Overlay on RGB (blue)
        overlay = self.rgb_image.copy()
        overlay[mask.astype(bool)] = (
            overlay[mask.astype(bool)] * 0.7 + np.array([0, 0, 255]) * 0.3
        )
        self.display_image(overlay, is_hsi=False, target_label=self.rgb_label)

        # Overlay on HSI (green)
        band = self.current_band
        hsi_band = self.hsi_image[:, :, band]
        norm = cv2.normalize(hsi_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rot = cv2.rotate(norm, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB)
        mask_map = mask.astype(bool)
        overlay_hsi = rgb.copy()
        overlay_hsi[mask_map] = (
            overlay_hsi[mask_map] * 0.7 + np.array([0, 255, 0]) * 0.3
        )
        self.display_image(overlay_hsi, is_hsi=False, target_label=self.hsi_label)

        # --- Get meta info ---
        meta = self.auto_sam_meta[idx]
        confidence = meta["confidence"]
        area = meta["area"]

        center_pixel = get_center_pixel_from_mask(mask)
        if center_pixel is not None and self.hsi_image is not None:
            try:
                intensity_values = get_hsi_pixel_from_mask(center_pixel, self.hsi_image)
                segment_size = int(mask.sum())
                msg = (
                    f"Grape Confidence: {confidence:.2%} | Area: {area} px | "
                    f"Auto SAM - Segment {idx + 1} / {len(self.auto_sam_detections)}"
                    f" | Center Pixel: {center_pixel}"
                    f" | Segment Size: {segment_size} px"
                )
                self.status_bar.showMessage(msg)
                self.plot_pixel_signatures_in_ui(intensity_values)

                self.start_save_countdown(seconds=0.0)  # Start countdown for auto-save



            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")
        else:
            self.status_bar.showMessage(
                f"Auto SAM - Segment {idx + 1} / {len(self.auto_sam_detections)} | No valid center pixel."
            )

    def ask_save_current_detection(self):
        if not self.auto_sam_detections:
            QMessageBox.warning(self, "Warning", "No segment to save.")
            return

        reply = QMessageBox.question(
            self,
            "Save Segment?",
            "Do you want to save the last (current) segmentation?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.save_current_detection()
        else:
            self.status_bar.showMessage("Segment not saved.")

    def start_save_countdown(self, seconds=3):
        self._save_countdown = seconds
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.update_save_countdown)
        self.save_timer.start(100)  # Every 100 milliseconds

    def update_save_countdown(self):
        if self._save_countdown > 0:
            # Update the status bar with the countdown
            self.status_bar.showMessage(f"Auto save in... {self._save_countdown:.1f}s")
            self._save_countdown -= 0.1
        else:
            self.save_timer.stop()
            self.status_bar.clearMessage()
            self.auto_save_and_next()

    def save_current_detection(self, CLUSTER_IRRIGATION=None):
        try:
            print("DEBUG: save_current_detection called")

            if not self.auto_sam_detections:
                print("DEBUG: No auto_sam_detections!")
                QMessageBox.warning(self, "Warning", "No segment to save.")
                return False

            idx = self.current_detection_index
            print(f"DEBUG: current_detection_index={idx}")

            if idx < 0 or idx >= len(self.auto_sam_detections):
                print(
                    f"DEBUG: Index out of range! idx={idx}, len={len(self.auto_sam_detections)}"
                )
                QMessageBox.warning(
                    self, "Warning", f"Segment index out of range! ({idx})"
                )
                return False

            det = self.auto_sam_detections[idx]
            print(f"DEBUG: Got detection: {det}")
            mask = det.mask[0] if isinstance(det.mask, list) else det.mask
            print(f"DEBUG: Mask shape: {getattr(mask, 'shape', 'no shape')}")
            meta = self.auto_sam_meta[idx]
            print(f"DEBUG: Meta: {meta}")

            segment_class = "Grape"
            if not segment_class:
                print("DEBUG: No segment_class!")
                self.status_bar.showMessage(
                    "No class selected, segmentation not saved."
                )
                return False

            img_path = self.rgb_image_path
            print(f"DEBUG: img_path={img_path}")

            if not img_path:
                print("DEBUG: img_path is None or empty!")
                QMessageBox.warning(self, "Warning", "Image path is missing!")
                return False

            stem = pathlib.Path(img_path).stem
            seg_idx = idx + 1

            # --- Extract cluster_id and irrigation_color ---
            try:
                cluster_id = extract_cluster_id_from_path(img_path)
                print(f"DEBUG: cluster_id={cluster_id}")
            except Exception as e:
                print(f"DEBUG: Failed to extract cluster_id: {e}")
                cluster_id = None

            try:
                date = extract_date_from_path(img_path)
                print(f"DEBUG: date={date}")
            except Exception as e:
                print(f"DEBUG: Failed to extract date: {e}")
                date = None

            irrigation_color = get_irrigation_color_for_cluster(cluster_id)
            print(f"DEBUG: irrigation_color={irrigation_color}")


            # Directories
            MASK_DIR = rf"{BASE_PATH}/{os.getenv('OUTPUT_MASKS_PATH')}/masks"
            JSON_DIR = rf"{BASE_PATH}/{os.getenv('OUTPUT_MASKS_PATH')}/jsons"
            print(f"DEBUG: MASK_DIR={MASK_DIR}, JSON_DIR={JSON_DIR}")

            try:
                os.makedirs(JSON_DIR, exist_ok=True)
                os.makedirs(MASK_DIR, exist_ok=True)
                print("DEBUG: Directories created OK.")
            except Exception as e:
                print(f"DEBUG: Failed to create directories: {e}")
                QMessageBox.warning(
                    self, "Warning", f"Failed to create directories:\n{e}"
                )
                return False

            mask_path = os.path.join(MASK_DIR, f"sample_auto_{seg_idx}_{stem}_mask.png")
            json_path = os.path.join(JSON_DIR, f"sample_auto_{seg_idx}_{stem}.json")
            print(f"DEBUG: mask_path={mask_path}, json_path={json_path}")

            try:
                cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))
                print("DEBUG: Mask image written successfully.")
            except Exception as e:
                print(f"DEBUG: Failed to save mask image: {e}")
                QMessageBox.warning(self, "Warning", f"Failed to save mask image:\n{e}")
                return False

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "segment_class": segment_class,
                            "confidence": meta["confidence"],
                            "area": meta["area"],
                            "cluster_id": cluster_id,
                            "irrigation_color": irrigation_color,
                            "date": date,
                        },
                        f,
                        indent=2,
                    )
                print("DEBUG: JSON file written successfully.")
            except Exception as e:
                print(f"DEBUG: Failed to save JSON: {e}")
                QMessageBox.warning(self, "Warning", f"Failed to save JSON:\n{e}")
                return False

            self.status_bar.showMessage(f"Saved ➜ {mask_path}\nJSON ➜ {json_path}")
            print("DEBUG: All done, returning True")
            self.count_segment_saved += 1
            self.current_detection_index += 1
            self.show_current_detection()
            if self.count_segment_saved == self.segments2saved:
                self.save_segment_button.setEnabled(False)
                self.next_button.setEnabled(False)
                self.prev_button.setEnabled(False)

                self.count_segment_saved = 0
                self.handle_next_folder()
                self.handle_auto_sam()
                save_last_progress_to_env(cluster_id =cluster_id,  date = date, env_path=None)
                self.save_segment_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(True)
            return True

        except Exception as exc:
            print(f"DEBUG: Exception in save_current_detection: {exc}")
            QMessageBox.warning(self, "Error", f"Exception:\n{exc}")
            return False




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HSI_RGB_Viewer()
    viewer.show()
    sys.exit(app.exec_())
