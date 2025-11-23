import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib  # Library for saving/loading models
import spectral as spy
import cv2
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]


######################################
# Existing functions for loading images
######################################
def load_rgb_image_from_folder(folder_path):
    """
    Load the RGB image from the specified folder.
    The RGB image is expected to have a .png extension in the main folder.
    """
    HS_folder_path = os.path.join(folder_path, "HS")
    rgb_files = [file for file in os.listdir(HS_folder_path) if file.endswith(".png")]
    if rgb_files:
        rgb_path = os.path.join(HS_folder_path, rgb_files[0])
        image = cv2.imread(rgb_path)
        print(rgb_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb, rgb_path
    else:
        raise FileNotFoundError("No RGB image found in the folder.")


def load_hsi_image_from_folder(folder_path):
    """
    Load the HSI image from the `results` subfolder.
    The HSI image is expected to have a .hdr file in the `results` subfolder.
    """
    HS_folder_path = os.path.join(folder_path, "HS")
    results_path = os.path.join(HS_folder_path, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError("No `results` folder found in the specified path.")

    hdr_files = [file for file in os.listdir(results_path) if file.endswith(".hdr")]
    if hdr_files:
        header_path = os.path.join(results_path, hdr_files[0])
        data_path = header_path.replace(".hdr", ".dat")
        if os.path.exists(data_path):
            hsi_obj = spy.envi.open(header_path, data_path)
            return np.array(hsi_obj.load())
        else:
            raise FileNotFoundError("Matching data file not found for the HSI header.")
    else:
        raise FileNotFoundError("No HSI header file found in the `results` folder.")


def load_rgb_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_hsi_image(header_path):
    data_path = header_path.replace(".hdr", ".dat")
    if os.path.exists(data_path):
        hsi_obj = spy.envi.open(header_path, data_path)
        return np.array(hsi_obj.load())
    else:
        raise FileNotFoundError("Matching data file not found for the HSI header.")


def load_canon_rgb_image(folder_path):
    """
    Load the Canon RGB image from the specified folder.
    The Canon image is expected to have a .jpg extension.
    """
    rgb_path = os.path.join(folder_path, "RGB")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError("No `RGB` folder found in the specified path.")

    jpg_files = [file for file in os.listdir(rgb_path) if file.lower().endswith(".jpg")]
    if jpg_files:
        image_path = os.path.join(rgb_path, jpg_files[0])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        if image is None:
            raise FileNotFoundError(f"Failed to load RGB image from: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError("No JPG files found in the `RGB` folder.")


######################################
# New functions for pixel signature processing
######################################
def get_normalized_signature(pixel):
    """
    Normalize a 1D spectral signature using min-max normalization.

    :param pixel: 1D numpy array of spectral values.
    :return: Normalized spectral signature.
    """
    min_val = np.min(pixel)
    max_val = np.max(pixel)
    diff = max_val - min_val if max_val != min_val else 1
    return (pixel - min_val) / diff


def get_pca_signature(normalized_signature, pca_model):
    """
    Apply the PCA transformation to a normalized spectral signature.

    :param normalized_signature: 1D normalized spectral signature.
    :param pca_model: Pre-trained PCA model.
    :return: PCA-transformed signature (flattened).
    """
    pca_sig = pca_model.transform(normalized_signature.reshape(1, -1))
    return pca_sig.flatten()


def plot_normalized_signature(signature, title="Normalized Spectral Signature"):
    """
    Plot the normalized spectral signature.

    :param signature: 1D array of normalized values.
    :param title: Plot title.
    """
    bands = len(signature)
    plt.figure()
    plt.plot(np.arange(1, bands + 1), signature, marker="o")
    plt.title(title)
    plt.xlabel("Band")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    return plt.gcf()


def plot_pca_signature(pca_signature, title="PCA-transformed Signature"):
    """
    Plot the PCA-transformed spectral signature.

    :param pca_signature: 1D array of PCA coefficients.
    :param title: Plot title.
    """
    n_components = len(pca_signature)
    plt.figure()
    plt.plot(np.arange(1, n_components + 1), pca_signature, marker="o", color="orange")
    plt.title(title)
    plt.xlabel("PCA Component")
    plt.ylabel("Coefficient")
    plt.grid(True)
    return plt.gcf()


def plot_pixel_signatures(intensity_values, pca_model):
    """
    Create a figure with two subplots:
      - Left: Normalized spectral signature.
      - Right: PCA-transformed spectral signature.

    :param intensity_values: 1D spectral signature from the clicked pixel.
    :param pca_model: Pre-trained PCA model.
    """
    norm_sig = get_normalized_signature(intensity_values)
    pca_sig = get_pca_signature(norm_sig, pca_model)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot normalized signature
    axs[0].plot(np.arange(1, len(norm_sig) + 1), norm_sig, marker="o")
    axs[0].set_title("Normalized Spectral Signature")
    axs[0].set_xlabel("Band")
    axs[0].set_ylabel("Normalized Value")
    axs[0].grid(True)

    # Plot PCA-transformed signature
    axs[1].plot(np.arange(1, len(pca_sig) + 1), pca_sig, marker="o", color="orange")
    axs[1].set_title("PCA-transformed Signature")
    axs[1].set_xlabel("PCA Component")
    axs[1].set_ylabel("Coefficient")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def compute_signatures_for_plot(intensity_values, pca_model):
    """
    Compute the normalized and PCA-transformed spectral signatures for a given pixel.

    :param intensity_values: 1D spectral signature from a pixel.
    :param pca_model: Pre-trained PCA model.
    :return: Tuple (normalized_signature, pca_signature)
    """
    # norm_sig = get_normalized_signature(intensity_values)
    norm_sig = intensity_values
    smmothied_sig = get_normalized_signature(smooth_stamp_gaussian(intensity_values))
    pca_sig = get_normalized_signature(get_pca_signature(smmothied_sig, pca_model))
    return norm_sig, pca_sig, smmothied_sig


def plot_three_stamps(norm_stamp, pca_stamp, smoothed_stamp, title="Three Subplots"):
    """
    Create a figure showing three images side by side:
    1. Original (normalized) stamp
    2. PCA-transformed stamp
    3. Smoothed stamp
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the original stamp
    axes[0].imshow(norm_stamp, cmap="gray")
    axes[0].set_title("Original Stamp")
    axes[0].axis("off")

    # Plot the PCA stamp
    axes[1].imshow(pca_stamp, cmap="gray")
    axes[1].set_title("PCA Stamp")
    axes[1].axis("off")

    # Plot the smoothed stamp
    axes[2].imshow(smoothed_stamp, cmap="gray")
    axes[2].set_title("Smoothed Stamp")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def smooth_stamp_moving_average(norm_stamp, kernel_size=50):
    """
    Simple moving average example to produce a "smoothed" version
    of 'norm_stamp'. This is just an example; adapt to your data format.
    """
    # For a 2D "stamp", you might do something like a simple box blur.
    # Here is a naive approach for demonstration:
    from scipy.ndimage import uniform_filter

    smoothed = uniform_filter(norm_stamp, size=kernel_size)
    return smoothed


def smooth_stamp_gaussian(norm_stamp, sigma=6):
    """
    Simple Gaussian smoothing example using scipy.
    """
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(norm_stamp, sigma=sigma)
    return smoothed
