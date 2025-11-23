import os
import json
import tifffile as tiff
import streamlit as st
from config import TIF_DIR, IMAGES_DIR

# רשימת קבצי TIF ומצב ניווט גלובלי
tif_files = []
current_tif_index = 0


def load_tif_files():
    """טוען את רשימת קבצי ה-TIF מתיקיית המסכות."""
    global tif_files
    tif_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(TIF_DIR)
        for file in files
        if file.lower().endswith(".tif")
    ]
    if not tif_files:
        st.error(f"No TIF files found in {TIF_DIR}")
    return tif_files


def load_tif_metadata(tif_path):
    """טוען מטא-דאטה מקובץ TIF (מהתיאור)."""
    try:
        with tiff.TiffFile(tif_path) as tif:
            if not tif.pages:
                raise ValueError("No pages found")
            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")
            if description and description.value:
                metadata = json.loads(description.value)
                return metadata
            else:
                return {}
    except Exception as e:
        st.error(f"Error loading metadata from {tif_path}: {e}")
        return {}


def decode_mask(tif_path):
    """מקבל את המסכה (סגמנטציה) מה-TIF."""
    try:
        with tiff.TiffFile(tif_path) as tif:
            mask = tif.pages[0].asarray()
            return mask
    except Exception as e:
        st.error(f"Error decoding mask from {tif_path}: {e}")
        return None


def find_image_with_extension(
    image_name, directory, extensions=[".png", ".jpg", ".jpeg"]
):
    """
    מחפש תמונה עם שם בסיסי בתיקייה ובין הסיומות הנתמכות.
    כאן, אנו מניחים שהתמונה המקורית היא ב-PNG, אך ניתן להוסיף גם פורמטים אחרים.
    """
    for ext in extensions:
        candidate = os.path.join(directory, image_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def save_metadata_to_tif(tif_path, metadata):
    """שומר את המטא-דאטה המעודכן חזרה לקובץ ה-TIF."""
    try:
        metadata_json = json.dumps(metadata)
        with tiff.TiffFile(tif_path) as tif:
            image_data = tif.pages[0].asarray()
        temp_path = tif_path + ".tmp"
        with tiff.TiffWriter(temp_path) as writer:
            writer.write(image_data, description=metadata_json)
        os.replace(temp_path, tif_path)
    except Exception as e:
        st.error(f"Error saving metadata to {tif_path}: {e}")


def get_current_tif():
    """מחזיר את הקובץ הנוכחי (נתיב + מטא-דאטה)."""
    global current_tif_index, tif_files
    if not tif_files:
        load_tif_files()
    if not tif_files:
        return None
    tif_path = tif_files[current_tif_index]
    metadata = load_tif_metadata(tif_path)
    return {"path": tif_path, "metadata": metadata}


def next_tif():
    """מעביר לקובץ ה-TIF הבא."""
    global current_tif_index, tif_files
    if current_tif_index < len(tif_files) - 1:
        current_tif_index += 1
    else:
        st.info("Reached last TIF file.")


def prev_tif():
    """מעביר לקובץ ה-TIF הקודם."""
    global current_tif_index, tif_files
    if current_tif_index > 0:
        current_tif_index -= 1
    else:
        st.info("Reached first TIF file.")


def next_untagged_tif():
    """
    מעביר לקובץ TIF הבא שבו התיוג במטא-דאטה הוא 'none' (לא מסומן).
    במידה ולא נמצא – מראה אזהרה.
    """
    global tif_files, current_tif_index
    start_index = current_tif_index
    while True:
        next_tif()
        current = get_current_tif()
        if current is None:
            break
        metadata = current.get("metadata", {})
        if metadata.get("tag", "none") == "none":
            break
        if current_tif_index == start_index:  # סיבוב מלא ברשימה
            st.warning("No untagged TIF files found.")
            break
