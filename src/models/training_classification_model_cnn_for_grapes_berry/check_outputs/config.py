import os

# מיקום הקובץ הנוכחי (config.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# נניח שאנחנו צריכים לעלות שתי רמות כדי להגיע ל-Grape (repo root):
#  1) מ-check_dataset ל-training_classification_model_cnn_for_grapes_berry
#  2) מ-training_classification_model_cnn_for_grapes_berry ל-Grape
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# נתיבים לתיקיות
TRAIN_DIR = os.path.join(REPO_ROOT, "items", "Data_for_train_and_val", "Train")
TEST_DIR = os.path.join(REPO_ROOT, "items", "Data_for_train_and_val", "Val")
IMAGES_DIR = os.path.join(REPO_ROOT, "items", "used")
TIF_DIR = os.path.join(REPO_ROOT, "items", "masks")

# נתיב למשקולות המודל (שנשמרו בעבר)
MODEL_SAVE_PATH = os.path.join(
    REPO_ROOT,
    "training_classification_model_cnn_for_grapes_berry",
    "model_weights",
    "efficientnet_classifier_weights.pth",
)
