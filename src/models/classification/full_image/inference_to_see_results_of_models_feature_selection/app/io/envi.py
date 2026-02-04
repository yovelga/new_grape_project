"""
ENVI file format reader for hyperspectral images.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


class ENVIReader:
    """Reader for ENVI format hyperspectral images."""

    def __init__(self, hdr_path: str):
        """
        Initialize ENVI reader.

        Args:
            hdr_path: Path to .hdr header file
        """
        self.hdr_path = Path(hdr_path)
        self.img_path = self._find_image_file()
        self.metadata = self._parse_header()

    def _find_image_file(self) -> Path:
        """Find the corresponding image data file."""
        # Try common extensions
        base = self.hdr_path.with_suffix('')
        for ext in ['', '.img', '.dat', '.raw']:
            img_path = Path(str(base) + ext)
            if img_path.exists() and img_path != self.hdr_path:
                return img_path
        raise FileNotFoundError(f"Image file not found for {self.hdr_path}")

    def _parse_header(self) -> Dict:
        """Parse ENVI header file."""
        metadata = {}
        with open(self.hdr_path, 'r') as f:
            content = f.read()

        # Handle multi-line values (e.g., wavelength arrays)
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()

                # Check if value continues on next lines (contains '{' but not '}')
                if '{' in value and '}' not in value:
                    # Multi-line value - collect until '}'
                    while i < len(lines) - 1 and '}' not in value:
                        i += 1
                        value += ' ' + lines[i].strip()

                # Parse numeric values
                if key in ['samples', 'lines', 'bands']:
                    metadata[key] = int(value)
                elif key == 'data type':
                    metadata[key] = int(value)
                elif key == 'byte order':
                    metadata[key] = int(value)
                elif key == 'interleave':
                    metadata[key] = value
                elif key == 'wavelength':
                    # Parse wavelength array
                    metadata[key] = self._parse_array(value)
                else:
                    metadata[key] = value

            i += 1

        return metadata

    def _parse_array(self, value: str) -> np.ndarray:
        """Parse array values from ENVI header (e.g., wavelengths)."""
        # Remove braces and split by comma
        value = value.replace('{', '').replace('}', '')
        try:
            values = [float(v.strip()) for v in value.split(',') if v.strip()]
            return np.array(values)
        except ValueError:
            return np.array([])

    def read(self) -> np.ndarray:
        """
        Read ENVI image.

        Returns:
            Image array with shape (lines, samples, bands)
        """
        samples = self.metadata.get('samples', 0)
        lines = self.metadata.get('lines', 0)
        bands = self.metadata.get('bands', 0)
        data_type = self.metadata.get('data type', 4)
        interleave = self.metadata.get('interleave', 'bsq')

        # Map ENVI data types to numpy dtypes
        dtype_map = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            12: np.uint16,
        }
        dtype = dtype_map.get(data_type, np.float32)

        # Read binary data
        data = np.fromfile(self.img_path, dtype=dtype)

        # Reshape based on interleave
        if interleave.lower() == 'bsq':
            # Band sequential
            data = data.reshape((bands, lines, samples))
            data = np.transpose(data, (1, 2, 0))  # (lines, samples, bands)
        elif interleave.lower() == 'bil':
            # Band interleaved by line
            data = data.reshape((lines, bands, samples))
            data = np.transpose(data, (0, 2, 1))  # (lines, samples, bands)
        elif interleave.lower() == 'bip':
            # Band interleaved by pixel
            data = data.reshape((lines, samples, bands))

        return data

    def get_wavelengths(self) -> Optional[np.ndarray]:
        """Extract wavelength information if available."""
        return self.metadata.get('wavelength', None)

    def find_band_index(self, target_wavelength: float) -> int:
        """
        Find band index closest to target wavelength.

        Args:
            target_wavelength: Target wavelength in nm

        Returns:
            Band index (0-based)
        """
        wavelengths = self.get_wavelengths()
        if wavelengths is None or len(wavelengths) == 0:
            raise ValueError("No wavelength information available in header")

        idx = np.argmin(np.abs(wavelengths - target_wavelength))
        return int(idx)
