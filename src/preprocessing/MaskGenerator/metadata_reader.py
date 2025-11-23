import os
import tifffile as tiff
import json

# Path to the TIFF file
HOME = os.getcwd()
ITEMS_DIR = os.path.join(os.path.dirname(HOME), "items", "masks")
tif_path = os.path.join(ITEMS_DIR, "2024-08-01_117_mask_139.tif")
# .

# Check if the file exists
if not os.path.exists(tif_path):
    print(f"File not found: {tif_path}")
else:
    try:
        # Read the metadata
        with tiff.TiffFile(tif_path) as tif:
            # Access tags and metadata
            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")  # Read the image description

            if description and description.value:
                print("Metadata from TIFF file:")

                # Convert the description to a JSON dictionary
                metadata = json.loads(description.value)

                # Print all metadata
                for key, value in metadata.items():
                    print(f"{key}: {value}")

                # Access specific values
                print("\nAccessing specific values:")
                print(f"Image Name: {metadata.get('image_name')}")
                print(f"Original BBox: {metadata.get('original_bbox')}")
                print(f"Padded BBox: {metadata.get('padded_bbox')}")
                print(f"Tag: {metadata.get('tag')}")
            else:
                print("No metadata found in the file.")

    except Exception as e:
        print(f"An error occurred: {e}")
