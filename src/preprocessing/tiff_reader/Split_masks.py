import os
import shutil
import tifffile as tiff
import json

# dir for tif mask
SOURCE_DIR = r"/storage/yovelg/Grape/items/masks"

# dir for destination
DEST_DIR_GRAPE = r"/storage/yovelg/Grape/items/Data_for_train_and_val/Grape"
DEST_DIR_NOT_GRAPE = r"/storage/yovelg/Grape/items/Data_for_train_and_val/Not_Grape"

# create folders
os.makedirs(DEST_DIR_GRAPE, exist_ok=True)
os.makedirs(DEST_DIR_NOT_GRAPE, exist_ok=True)

# iterate on all the masks in the direction
for file_name in os.listdir(SOURCE_DIR):
    if file_name.lower().endswith(".tif"):
        tif_path = os.path.join(SOURCE_DIR, file_name)

        try:
            # reading meta data from tif file
            with tiff.TiffFile(tif_path) as tif:
                tags = tif.pages[0].tags
                description = tags.get("ImageDescription")

                if description and description.value:
                    metadata = json.loads(description.value)
                    tag = metadata.get("tag")

                    if tag == "Grape":
                        shutil.copy(tif_path, DEST_DIR_GRAPE)
                        print(f"Copied {file_name} to Grape folder.")
                    elif tag == "Not Grape":
                        shutil.copy(tif_path, DEST_DIR_NOT_GRAPE)
                        print(f"Copied {file_name} to Not_Grape folder.")
                    else:
                        print(f"Skipped {file_name}, tag is NONE or unknown.")
                else:
                    print(f"No metadata found in {file_name}.")

        except Exception as e:
            print(f"An error occurred with {file_name}: {e}")
