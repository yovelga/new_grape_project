import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# We need to go up two levels to reach Grape (repo root):
#  1) from check_dataset to training_classification_model_cnn_for_grapes_berry
#  2) from training_classification_model_cnn_for_grapes_berry to Grape
REPO_ROOT = r"C:\Users\yovel\Desktop\Grape_Project"
DATA_ROOT = os.path.join(REPO_ROOT,"src","preprocessing","items_for_cnn_train","Data_for_train_and_val_cnn")

# Directory paths
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR = os.path.join(DATA_ROOT, "Val")
IMAGES_DIR = os.path.join(REPO_ROOT, "src", "preprocessing", "items_for_cnn_train", "used")
TIF_DIR = os.path.join(REPO_ROOT, "src", "preprocessing", "items_for_cnn_train", "masks")


# Directory paths
# TRAIN_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Train"
# TEST_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Val"
# IMAGES_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/used"

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# Path to save the trained model
MODEL_SAVE_PATH = r"C:\Users\yovel\Desktop\Grape_Project\training_classification_model_cnn_for_grapes_berry\missclassified\efficientnet_classifier_weights.pth"
MISCLASSIFIED_PATH = r"C:\Users\yovel\Desktop\Grape_Project\training_classification_model_cnn_for_grapes_berry\missclassified"
