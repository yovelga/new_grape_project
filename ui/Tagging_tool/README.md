# TIF Tagging Tool

## Overview
The TIF Tagging Tool is a Python-based application that enables users to process and annotate TIFF files containing masks and metadata. It is designed to facilitate tagging and reviewing of images for machine learning and computer vision tasks.

## Features
- Load TIFF files with metadata.
- Process and display associated images and masks.
- Tag images as `Grape` or `Not Grape`.
- Navigate through images with options to move to the next, previous, or the next untagged image.
- Save updated tags directly into the TIFF metadata.

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
Install the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

**Required libraries:**
- `streamlit`
- `numpy`
- `opencv-python`
- `Pillow`
- `tifffile`
- `imagecodecs` (for compressed TIFF files)

## File Structure
```
project_root/
├── process_tif_only.py     # Backend processing code
├── app.py                  # Streamlit frontend application
├── requirements.txt        # Python dependencies
├── ObjectJSONizer/         # Directory containing TIFF masks and related files
│   ├── masks/              # Directory with TIFF mask files
│   ├── used/               # Directory with corresponding images
├── README.md               # Project documentation
```

## Usage

### 1. Prepare Your Data
- Place your TIFF files (masks) in the `ObjectJSONizer/masks` directory.
- Ensure the images referenced in the metadata are in the `ObjectJSONizer/used` directory.

### 2. Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

### 3. Tagging and Navigation
- Use the **Navigation** section in the sidebar to move between images.
- Use the tagging buttons (`Tag as Grape`, `Tag as Not Grape`) to annotate the current image.
- Use the **Next Untagged TIF** button to jump to the next untagged image.

## Development

### Backend
The backend handles:
- Loading TIFF files and metadata.
- Processing masks and bounding boxes.
- Saving updated metadata back to the TIFF files.

### Frontend
The frontend is built with Streamlit and includes:
- Image visualization.
- Navigation and tagging options.
- Dynamic updates of the current image and its metadata.

## Contributing
Feel free to contribute to this project by submitting pull requests or reporting issues.

### How to Contribute
1. Fork the repository.
2. Clone your forked repository:
   ```bash
   git clone https://github.com/your-username/tif-tagging-tool.git
   ```
3. Create a new branch for your changes:
   ```bash
   git checkout -b feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m "Description of changes"
   ```
5. Push your branch to GitHub:
   ```bash
   git push origin feature-name
   ```
6. Open a pull request on the original repository.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or feedback, please contact:
- **Name**: Yovel Gani
- **Email**: yovelg@volcani.agri.gov.il

