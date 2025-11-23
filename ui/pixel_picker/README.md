# HSI and RGB Viewer

This application is a graphical user interface (GUI) tool built in Python using PyQt5. It is designed to load and display hyperspectral (HSI) and RGB images from specified folders, allowing users to interactively select pixels and view their corresponding reflectance spectra. The software also includes a wavelength calibration feature that maps each HSI band to its corresponding wavelength, using a dedicated dictionary stored in a separate file.

## Features

- **Folder Loading:**  
  Load a folder containing both HSI and RGB images. The application extracts and displays:
  - The RGB image from the hyperspectral (HS) folder.
  - The hyperspectral (HSI) image.
  - The Canon RGB image (if available).

- **Interactive Image Display:**  
  Display images side by side in the GUI and allow the user to click on any image to select a pixel.

- **Spectral Plotting:**  
  When a pixel is selected, plot the reflectance spectrum (intensity values) for that pixel.  
  If the image contains 204 spectral bands, the plot uses the wavelengths (in nanometers) provided in a separate dictionary to accurately label the x-axis.

- **Wavelength Calibration:**  
  A dedicated file (`wavelengths.py`) contains the `WAVELENGTHS` dictionary mapping band indices (1 to 204) to their corresponding wavelengths (nm).

- **Pixel Management:**  
  Save selected pixels along with their reflectance data and export this data to a CSV file for further analysis.

- **Histogram Export:**  
  Export histograms of the pixel intensity values for either the entire image or a specific HSI band.

- **Folder Navigation:**  
  Navigate through a list of folders using "Previous" and "Next" buttons or by adding a list of folder paths.

## Requirements

- **Python:** Version 3.6 or above  
- **Dependencies:**  
  - PyQt5  
  - OpenCV (`cv2`)  
  - NumPy  
  - Matplotlib  

You can install the required packages via pip:

```bash
pip install PyQt5 opencv-python numpy matplotlib
