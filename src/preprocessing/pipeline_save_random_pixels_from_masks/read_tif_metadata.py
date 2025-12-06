import tifffile as tiff
import json


def read_metadata(tif_path):
    with tiff.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        print("Reading metadata from:", tif_path)
        print("-" * 40)
        for tag in page.tags.values():
            tag_name = tag.name
            tag_value = tag.value
            print(f"{tag_name}: {tag_value}")
            # אם מדובר ב־ImageDescription ננסה לפרש אותו כ־JSON
            if tag_name == "ImageDescription":
                try:
                    metadata_json = json.loads(tag_value)
                    print("Parsed JSON metadata:")
                    print(json.dumps(metadata_json, indent=4))
                except Exception as e:
                    print("Could not parse ImageDescription as JSON:", e)


def main():
    # נתיב לקובץ TIF
    tif_path = "/storage/yovelg/Grape/items_for_cnn_train/masks/2024-08-01_003_mask_5.tif"
    read_metadata(tif_path)


if __name__ == "__main__":
    main()
