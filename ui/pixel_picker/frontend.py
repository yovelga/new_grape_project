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
)
from PyQt5.QtCore import Qt
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
import sys, os, json, pathlib, re

# For EfficientNet-B0
from torchvision.models import efficientnet_b0
# Set project_path to the workspace root
project_path = Path(__file__).resolve().parents[2]
print(f"Project path set to: {project_path}")


from ui.pixel_picker.backend import (
    load_rgb_image_from_folder,
    load_hsi_image_from_folder,
    load_canon_rgb_image,
    compute_signatures_for_plot,
)



BASE_PATH = project_path
AUTOENCODER_MODEL_PATH = BASE_PATH / "src" / "models" / "training_classification_model_cnn_for_grapes_berry" / "model_weights" / "autoencoder_model.pkl"
CLASS_FILE = BASE_PATH / "data" / "processed" / "segment_classes.json"

sys.path.append(str(project_path))
from src.preprocessing.MaskGenerator.segment_object_module import create_point_segmenter
from src.preprocessing.MaskGenerator.mask_generator_module import (
    initial_settings,
    initialize_sam2_predictor,
    generate_detections,
    initialize_sam2,
)


# ===== Utility functions =====
def load_grape_classifier(device):
    # Build model architecture
    model = efficientnet_b0(
        weights=None
    )  # or weights="IMAGENET1K_V1" if you used pretrained
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2
    )  # 2 classes: grape/not grape

    # Load weights
    model_path = (
        project_path
        / "src"
        / "models"
        / "training_classification_model_cnn_for_grapes_berry"
        / "model_weights"
        / "best_model_all.pth"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_grape_confidence(rgb_img, mask, model, transform, device):
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    crop = rgb_img[y_min : y_max + 1, x_min : x_max + 1, :]

    # Optional: mask out background if you want (set to black)
    mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    crop[~mask_crop] = 0

    input_tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence = probs[0, 1].item()  # Assuming class 1 is "grape"
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


class HSI_RGB_Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rgb_image_path = None
        self.last_clicked_pixel = None
        self.sam2_segmenter = None
        self.sam2_auto_generator = None
        self.current_folder_path = None
        self.sam_count = None
        self.rgb_image = None
        self.auto_sam_detections = []
        self.current_detection_index = -1
        self.setWindowTitle("HSI and RGB Viewer")
        self.setGeometry(100, 100, 1600, 800)
        self.installEventFilter(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grape_classifier = load_grape_classifier(self.device)
        try:
            self.autoencoder_model = joblib.load(AUTOENCODER_MODEL_PATH)
        except Exception as e:
            self.autoencoder_model = None
        self.hsi_image = None
        self.current_band = 0
        self.saved_pixels = []
        self.folder_list = []
        self.current_folder_index = -1
        self.current_capture_index = -1
        self.capture_references = []
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        # Now call setup_ui
        self.setup_ui()
        # Load initial images from the specified folder
        initial_folder = r"C:\Users\yovel\Desktop\Grape_Project\data\raw\1_13\25.09.24"
        self.rgb_image, self.rgb_image_path = load_rgb_image_from_folder(initial_folder)
        self.hsi_image = load_hsi_image_from_folder(initial_folder)
        self.canon_rgb_image = load_canon_rgb_image(initial_folder)
        self.display_image(self.rgb_image, is_hsi=False)
        self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
        self.update_hsi_band()
        self.display_image(self.canon_rgb_image, is_hsi=False, target_label=self.canon_rgb_label)
        self.folder_path_label.setText(f"Current Folder: {initial_folder}")


    def setup_ui(self):

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

        self.load_fp_button = QPushButton("Load FP TXT")
        self.load_fp_button.clicked.connect(self.handle_load_fp_pixels)
        self.controls_layout.addWidget(self.load_fp_button)

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




    def handle_load_fp_pixels(self):
        try:
            txt_path, _ = QFileDialog.getOpenFileName(self, "Select FP TXT File", "", "Text Files (*.txt)")
            if not txt_path:
                print("No file selected.")
                return
            print("Selected file:", txt_path)
            pixels = []
            with open(txt_path, 'r') as f:
                for line in f:
                    match = re.match(r'(.+?) y=(\d+) x=(\d+)', line.strip())
                    if match:
                        img_path = match.group(1)
                        y = int(match.group(2))
                        x = int(match.group(3))
                        pixels.append((img_path, x, y))
                    else:
                        print("No match for line:", line.strip())
            print(f"Found {len(pixels)} pixels.")
            if not pixels:
                self.status_bar.showMessage("No pixels found in file.")
                return

            image_folder = pixels[0][0]
            print("Loading image from:", image_folder)
            self.rgb_image, self.rgb_image_path = load_rgb_image_from_folder(image_folder)
            self.hsi_image = load_hsi_image_from_folder(image_folder)
            self.display_image(self.rgb_image, is_hsi=False)
            self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
            self.update_hsi_band()
            self.folder_path_label.setText(f"Current Folder: {image_folder}")

            # Load and display Canon RGB image for the same cluster/folder
            self.canon_rgb_image = load_canon_rgb_image(image_folder)
            self.display_image(
                self.canon_rgb_image,
                is_hsi=False,
                target_label=self.canon_rgb_label,
            )

            rgb_copy = self.rgb_image.copy()
            h, w, _ = rgb_copy.shape
            count_in_bounds = 0
            for _, x, y in pixels:
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(rgb_copy, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
                    count_in_bounds += 1
            print(f"Plotted {count_in_bounds} pixels in image bounds.")
            self.display_image(rgb_copy, is_hsi=False, target_label=self.rgb_label)
            self.status_bar.showMessage(f"Loaded {count_in_bounds} FP pixels from {os.path.basename(txt_path)}")
        except Exception as e:
            print("Exception in handle_load_fp_pixels:", e)
            self.status_bar.showMessage("Error loading FP TXT file.")

        # נטען את התמונה הראשונה מהקובץ ונציג עליה את כל הפיקסלים (בהנחה שכל הפיקסלים מאותת התמונה)
        image_folder = pixels[0][0]
        self.rgb_image, self.rgb_image_path = load_rgb_image_from_folder(image_folder)
        self.hsi_image = load_hsi_image_from_folder(image_folder)
        self.display_image(self.rgb_image, is_hsi=False)
        self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
        self.update_hsi_band()
        self.folder_path_label.setText(f"Current Folder: {image_folder}")

        # צביעת כל הפיקסלים על ה-RGB
        rgb_copy = self.rgb_image.copy()
        for _, x, y in pixels:
            cv2.circle(rgb_copy, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        self.display_image(rgb_copy, is_hsi=False, target_label=self.rgb_label)

        self.status_bar.showMessage(f"Loaded {len(pixels)} FP pixels from {os.path.basename(txt_path)}")

    def load_classes(self):
        try:
            with open(CLASS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Default classes if file missing
            return ["Crack", "Regular", "branch", "leaf", "Plastic"]

    def save_classes(self, class_list):
        with open(CLASS_FILE, "w", encoding="utf-8") as f:
            json.dump(class_list, f, indent=2)

    def ask_segment_class(self):
        classes = self.load_classes()
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
            print(f"heyy iam hare : {self.rgb_image_path}")
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
        print("[SAM2] handle_sam2 called")
        if self.rgb_image_path is None:
            print(f"[SAM2] No RGB image path: {self.rgb_image_path}")
            QMessageBox.warning(self, "Warning", "load image first.")
            return
        if self.last_clicked_pixel is None:
            print(f"[SAM2] No pixel selected: {self.last_clicked_pixel}")
            QMessageBox.warning(self, "Warning", "Select a pixel and load a folder first.")
            return
        x, y, intensity_values = self.last_clicked_pixel
        print(f"[SAM2] Pixel selected: x={x}, y={y}")
        print(f"[SAM2] RGB image path: {self.rgb_image_path}")
        print(f"[SAM2] sam_count: {self.sam_count}")
        img_path = self.rgb_image_path
        progress = QProgressDialog("Segmenting...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        if self.sam2_segmenter is None:
            print("[SAM2] Initializing SAM2 predictor...")
            initial_settings()
            try:
                predictor = initialize_sam2_predictor()
                print("[SAM2] Predictor initialized.")
                self.sam2_segmenter = create_point_segmenter(predictor)
                print("[SAM2] Point segmenter created.")
            except Exception as e:
                print(f"[SAM2] ERROR during predictor initialization: {e}")
                progress.close()
                QMessageBox.warning(self, "Error", f"SAM2 initialization failed: {e}")
                return
        print("[SAM2] Running segmentation...")
        try:
            image_rgb, mask_bool = self.sam2_segmenter.segment_object(img_path, [(x, y)])
            print("[SAM2] Segmentation completed.")
        except Exception as e:
            print(f"[SAM2] ERROR during segmentation: {e}")
            progress.close()
            QMessageBox.warning(self, "Error", f"SAM2 segmentation failed: {e}")
            return
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

            # Save to per-class folder
            class_folder = segment_class.upper()
            BASE_RESULTS = r"C:\Users\yovel\Desktop\Grape_Project\ui\pixel_picker\sam2_results"
            MASK_DIR = os.path.join(BASE_RESULTS, class_folder)
            JSON_DIR = os.path.join(BASE_RESULTS, class_folder)
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
            print(f"JSON saved at: {os.path.abspath(json_path)}")  # <-- Added print
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
                self.sam2_auto_generator, img_resized, min_area=200, max_area=5000
            )
            print("TYPE in frontend:", type(detections[0]), detections[0])
        except Exception as e:
            print(f"Error in generate_detections: {e}")
            QMessageBox.warning(self, "Error", str(e))
            return

        if not detections:
            QMessageBox.warning(self, "Warning", "No segments found.")
            return

        # ------- NEW: Compute MSE for each detection --------
        mse_list = []
        for det in detections:
            mask = det.mask[0] if isinstance(det.mask, list) else det.mask
            center_pixel = get_center_pixel_from_mask(mask)
            if center_pixel is not None and self.hsi_image is not None:
                try:
                    intensity_values = get_hsi_pixel_from_mask(
                        center_pixel, self.hsi_image
                    )
                    reconstructed = self.autoencoder_model.predict(
                        intensity_values.reshape(1, -1)
                    ).flatten()
                    mse = np.mean((intensity_values - reconstructed) ** 2)
                except Exception as e:
                    print(f"Error calculating intensity/MSE: {e}")
                    mse = np.inf
            else:
                mse = np.inf
            mse_list.append(mse)
            print(f"Segment size: {mask.sum()} | MSE: {mse:.4f}")

        # Sort detections by MSE descending (largest MSE first)
        detections_sorted = [
            d for _, d in sorted(zip(mse_list, detections), key=lambda t: -t[0])
        ]

        self.auto_sam_detections = detections_sorted
        self.current_detection_index = 0
        self.show_current_detection()

    def show_current_detection(self):
        det = self.auto_sam_detections[self.current_detection_index]
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

        confidence, _ = predict_grape_confidence(
            self.rgb_image.copy(), mask, self.grape_classifier, transform, self.device
        )

        # ---- Find and plot the center pixel of the mask ----
        center_pixel = get_center_pixel_from_mask(mask)
        if center_pixel is not None and self.hsi_image is not None:
            try:
                intensity_values = get_hsi_pixel_from_mask(center_pixel, self.hsi_image)
                segment_size = int(mask.sum())

                # Combine confidence and segment info in a single message
                if confidence is not None:
                    msg = f"Grape Confidence: {confidence:.2%} | "
                else:
                    msg = "Could not compute grape confidence for this mask. | "
                msg += (
                    f"Auto SAM - Segment {self.current_detection_index + 1} / {len(self.auto_sam_detections)}"
                    f" | Center Pixel: {center_pixel}"
                    f" | Segment Size: {segment_size} px"
                )
                self.status_bar.showMessage(msg)

                self.plot_pixel_signatures_in_ui(intensity_values)

            except Exception as e:
                self.status_bar.showMessage(f"Error: {e}")
        else:
            self.status_bar.showMessage(
                f"Auto SAM - Segment {self.current_detection_index + 1} / {len(self.auto_sam_detections)} | No valid center pixel."
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

    def save_current_detection(self):
        if not self.auto_sam_detections:
            QMessageBox.warning(self, "Warning", "No segment to save.")
            return

        det = self.auto_sam_detections[self.current_detection_index]
        mask = det.mask[0]
        segment_class = self.ask_segment_class()
        if not segment_class:
            self.status_bar.showMessage("No class selected, segmentation not saved.")
            return

        # Save to per-class folder
        class_folder = segment_class.upper()
        BASE_RESULTS = r"C:\Users\yovel\Desktop\Grape_Project\pixel_picker\sam2_results"
        MASK_DIR = os.path.join(BASE_RESULTS, class_folder)
        JSON_DIR = os.path.join(BASE_RESULTS, class_folder)
        os.makedirs(JSON_DIR, exist_ok=True)
        os.makedirs(MASK_DIR, exist_ok=True)
        img_path = self.rgb_image_path
        stem = pathlib.Path(img_path).stem
        seg_idx = self.current_detection_index + 1
        mask_path = os.path.join(MASK_DIR, f"sample_auto_{seg_idx}_{stem}_mask.png")
        json_path = os.path.join(JSON_DIR, f"sample_auto_{seg_idx}_{stem}.json")
        cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "segment_class": segment_class,
                },
                f,
                indent=2,
            )
        self.status_bar.showMessage(f"Saved ➜ {mask_path}\nJSON ➜ {json_path}")
        print(f"JSON saved at: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HSI_RGB_Viewer()
    viewer.show()
    sys.exit(app.exec_())
