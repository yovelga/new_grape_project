# hsi_signature.py
import tifffile as tiff
import numpy as np


def load_hsi_image(path):
    with tiff.TiffFile(path) as tif:
        hsi_array = tif.pages[0].asarray()
    return hsi_array


def compute_average_signature(hsi_crop, mask):
    # hsi_crop: numpy array בגודל (H, W, bands)
    # mask: numpy array בגודל (H, W) עם ערכים בינאריים (0,1)
    bool_mask = mask.astype(bool)
    if bool_mask.sum() == 0:
        return None
    avg_signature = np.mean(hsi_crop[bool_mask], axis=0)
    return avg_signature
