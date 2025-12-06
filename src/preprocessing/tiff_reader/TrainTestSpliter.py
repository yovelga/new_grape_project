import os
import shutil
import random

# Paths for labeled files
SOURCE_GRAPE_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Grape"
SOURCE_NOT_GRAPE_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Not_Grape"

# Paths for Train and Test folders
DEST_TRAIN_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Train"
DEST_TEST_DIR = r"/storage/yovelg/Grape/items_for_cnn_train/Data_for_train_and_val_cnn/Val"

# Create Train and Test folders if they don't exist
os.makedirs(os.path.join(DEST_TRAIN_DIR, "Grape"), exist_ok=True)
os.makedirs(os.path.join(DEST_TRAIN_DIR, "Not Grape"), exist_ok=True)
os.makedirs(os.path.join(DEST_TEST_DIR, "Grape"), exist_ok=True)
os.makedirs(os.path.join(DEST_TEST_DIR, "Not Grape"), exist_ok=True)


# Function to split and copy files
def split_and_copy_files(source_dir, dest_train_dir, dest_test_dir, split_ratio=0.7):
    # Read all files in the directory
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".tif")]

    # Shuffle files randomly
    random.shuffle(files)

    # Calculate the number of files for Train and Test
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # Copy Train files
    for file_name in train_files:
        shutil.copy(os.path.join(source_dir, file_name), dest_train_dir)

    # Copy Test files
    for file_name in test_files:
        shutil.copy(os.path.join(source_dir, file_name), dest_test_dir)

    print(f"Copied {len(train_files)} files to {dest_train_dir}.")
    print(f"Copied {len(test_files)} files to {dest_test_dir}.")


# Split files for Grape
split_and_copy_files(
    SOURCE_GRAPE_DIR,
    os.path.join(DEST_TRAIN_DIR, "Grape"),
    os.path.join(DEST_TEST_DIR, "Grape"),
)

# Split files for Not_Grape
split_and_copy_files(
    SOURCE_NOT_GRAPE_DIR,
    os.path.join(DEST_TRAIN_DIR, "Not Grape"),
    os.path.join(DEST_TEST_DIR, "Not Grape"),
)
