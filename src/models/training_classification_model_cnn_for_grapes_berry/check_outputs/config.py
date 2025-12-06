import os

# מיקום הקובץ הנוכחי (config.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# נניח שאנחנו צריכים לעלות שתי רמות כדי להגיע ל-Grape (repo root):
#  1) מ-check_dataset ל-training_classification_model_cnn_for_grapes_berry
#  2) מ-training_classification_model_cnn_for_grapes_berry ל-Grape
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# נתיבים לתיקיות
TRAIN_DIR = os.path.join(REPO_ROOT, "items_for_cnn_train", "Data_for_train_and_val_cnn", "Train")
TEST_DIR = os.path.join(REPO_ROOT, "items_for_cnn_train", "Data_for_train_and_val_cnn", "Val")
IMAGES_DIR = os.path.join(REPO_ROOT, "items_for_cnn_train", "used")
TIF_DIR = os.path.join(REPO_ROOT, "items_for_cnn_train", "masks")

# נתיב למשקולות המודל (שנשמרו בעבר)
MODEL_SAVE_PATH = os.path.join(
    REPO_ROOT,
    "training_classification_model_cnn_for_grapes_berry",
    "model_weights",
    "efficientnet_classifier_weights.pth",
)
