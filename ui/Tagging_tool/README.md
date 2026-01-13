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
Grape_Project/
├── ui/
│   └── Tagging_tool/
│       ├── app.py                  # Streamlit frontend application
│       ├── back.py                 # Backend processing code
│       ├── requirements.txt        # Python dependencies
│       └── README.md               # This file
├── src/
│   └── preprocessing/
│       └── items_for_cnn_train/    # Directory containing TIFF masks and images
│           ├── masks/              # Directory with TIFF mask files
│           ├── used/               # Directory with corresponding images
│           └── images/             # Additional images
└── data/                           # Raw and processed data
```

## Usage

### 1. Prepare Your Data
- Place your TIFF files (masks) in the `src/preprocessing/items_for_cnn_train/masks` directory.
- Ensure the images referenced in the metadata are in the `src/preprocessing/items_for_cnn_train/used` directory.
- The tool automatically resolves paths relative to the project root.

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Validate Setup (Optional)
Run the validation script to verify all paths are correct:
```bash
python validate_setup.py
```

This will check that:
- The masks directory exists and contains TIF files
- The images directory exists and contains image files
- All required files are in place

### 4. Run the Application

**Option A: Quick Start (Windows)**
Simply double-click `start.bat` or run from command line:
```bash
cd ui/Tagging_tool
start.bat
```

**Option B: Manual Start**
Navigate to the Tagging_tool directory and start the Streamlit app:
```bash
cd ui/Tagging_tool
streamlit run app_NEW.py
```

Or from the project root:
```bash
streamlit run ui/Tagging_tool/app_NEW.py
```

### 5. Tagging and Navigation
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

## Troubleshooting

### Common Issues

**"No TIF files found" error:**
- Run `python validate_setup.py` to check if directories exist
- Verify that `src/preprocessing/items_for_cnn_train/masks/` contains .tif files
- Check that you're running from the correct directory

**"Image not found" error:**
- Ensure images referenced in TIF metadata exist in `src/preprocessing/items_for_cnn_train/used/`
- Check that image filenames match the metadata (without extension)

**Path resolution issues:**
- The application automatically resolves paths from the project root
- Make sure the project structure follows the layout in the File Structure section
- Don't move the Tagging_tool folder outside of `ui/`

**Import errors:**
- Run `pip install -r requirements.txt` to install dependencies
- Verify you're using Python 3.8 or higher

### Debug Mode
To see detailed path information, check the terminal output when starting the application. It will show:
- Project root directory
- Mask directory path
- Image directory path

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or feedback, please contact:
- **Name**: Yovel Gani
- **Email**: yovelg@volcani.agri.gov.il

