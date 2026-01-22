"""
Application settings and configuration management.

Loads environment variables and provides configuration objects.
Uses dataclass for structured configuration.
"""

import os
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field


def _load_env_file():
    """Load .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        # Manual parsing (avoiding python-dotenv dependency)
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value


# Load .env file before creating settings
_load_env_file()


def _get_base_dir() -> Path:
    """Get base directory (where this file is located)."""
    return Path(__file__).parent.parent.parent


def _resolve_path(path_str: str) -> Path:
    """
    Resolve path string to absolute path.

    Handles relative paths from base directory.
    """
    if not path_str:
        return Path()

    path = Path(path_str)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Otherwise, resolve relative to base directory
    return (_get_base_dir() / path).resolve()


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse boolean from string."""
    if not value:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _parse_image_size(size_str: str, default: Tuple[int, int] = (224, 224)) -> Tuple[int, int]:
    """Parse image size string into tuple."""
    if not size_str:
        return default
    try:
        width, height = map(int, size_str.split(','))
        return (width, height)
    except (ValueError, AttributeError):
        return default


def _parse_color_buckets(buckets_str: str) -> List[Tuple[float, Tuple[int, int, int]]]:
    """
    Parse color buckets string.

    Format: "50:0,0,180;40:0,30,200;30:0,120,255;20:0,200,255"
    Returns: [(50, (0, 0, 180)), (40, (0, 30, 200)), ...]
    """
    default_buckets = [
        (50, (0, 0, 180)),    # Dark red
        (40, (0, 30, 200)),   # Red
        (30, (0, 120, 255)),  # Orange
        (20, (0, 200, 255)),  # Yellow
    ]

    if not buckets_str:
        return default_buckets

    try:
        buckets = []
        for bucket in buckets_str.split(';'):
            threshold_str, color_str = bucket.split(':')
            threshold = float(threshold_str)
            r, g, b = map(int, color_str.split(','))
            buckets.append((threshold, (r, g, b)))
        return buckets
    except (ValueError, AttributeError):
        return default_buckets


def _parse_models_dict(models_str: str) -> Dict[str, str]:
    """
    Parse available models string.

    Format: "Model1:path1.joblib;Model2:path2.joblib"
    """
    if not models_str:
        return {}

    try:
        models = {}
        for model_entry in models_str.split(';'):
            if ':' in model_entry:
                name, path = model_entry.split(':', 1)
                models[name.strip()] = path.strip()
        return models
    except (ValueError, AttributeError):
        return {}


@dataclass
class Settings:
    """Application configuration settings."""

    # ===== Paths =====
    models_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("MODELS_DIR", "./models")))
    results_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("RESULTS_DIR", "./results")))
    default_trainval_csv: Path = field(default_factory=lambda: _resolve_path(os.getenv("DEFAULT_TRAINVAL_CSV", "./data/trainval.csv")))
    default_test_csv: Path = field(default_factory=lambda: _resolve_path(os.getenv("DEFAULT_TEST_CSV", "./data/test.csv")))
    default_search_folder: Path = field(default_factory=lambda: _resolve_path(os.getenv("DEFAULT_SEARCH_FOLDER", "./data/raw")))
    log_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("LOG_DIR", "./logs")))

    # ===== Dataset Configuration =====
    val_split_size: float = field(default_factory=lambda: float(os.getenv("VAL_SPLIT_SIZE", "0.30")))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))

    # ===== Device Configuration =====
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    # ===== Preprocessing =====
    wl_min: float = field(default_factory=lambda: float(os.getenv("WL_MIN", "450")))
    wl_max: float = field(default_factory=lambda: float(os.getenv("WL_MAX", "925")))
    apply_snv: bool = field(default_factory=lambda: _parse_bool(os.getenv("APPLY_SNV", "true")))

    # ===== Inference =====
    default_prob_threshold: float = field(default_factory=lambda: float(os.getenv("DEFAULT_PROB_THRESHOLD", "0.5")))
    inference_batch_size: int = field(default_factory=lambda: int(os.getenv("INFERENCE_BATCH_SIZE", "10000")))

    # ===== Grid Analysis =====
    grid_cell_size: int = field(default_factory=lambda: int(os.getenv("GRID_CELL_SIZE", "64")))
    grid_crack_ratio: float = field(default_factory=lambda: float(os.getenv("GRID_CRACK_RATIO", "0.1")))
    cluster_crack_ratio: float = field(default_factory=lambda: float(os.getenv("CLUSTER_CRACK_RATIO", "0.05")))

    # ===== CNN Configuration =====
    cnn_checkpoint: Optional[Path] = field(default_factory=lambda: _resolve_path(os.getenv("CNN_CHECKPOINT", "")) if os.getenv("CNN_CHECKPOINT") else None)
    cnn_image_size: Tuple[int, int] = field(default_factory=lambda: _parse_image_size(os.getenv("CNN_IMAGE_SIZE", "224,224")))
    cnn_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CNN_CONFIDENCE_THRESHOLD", "0.5")))

    # ===== SAM2 Configuration =====
    sam2_checkpoint: Optional[Path] = field(default_factory=lambda: _resolve_path(os.getenv("SAM2_CHECKPOINT", "")) if os.getenv("SAM2_CHECKPOINT") else None)
    sam2_model_cfg: str = field(default_factory=lambda: os.getenv("SAM2_MODEL_CFG", "sam2_hiera_l.yaml"))
    sam2_device: str = field(default_factory=lambda: os.getenv("SAM2_DEVICE", "cpu"))

    # ===== Visualization =====
    default_band_index: int = field(default_factory=lambda: int(os.getenv("DEFAULT_BAND_INDEX", "138")))
    overlay_alpha: float = field(default_factory=lambda: float(os.getenv("OVERLAY_ALPHA", "0.35")))
    grid_color_buckets: List[Tuple[float, Tuple[int, int, int]]] = field(
        default_factory=lambda: _parse_color_buckets(os.getenv("GRID_COLOR_BUCKETS", ""))
    )

    # ===== HSI Display Orientation =====
    hsi_rotate_deg: int = field(default_factory=lambda: int(os.getenv("HSI_ROTATE_DEG", "0")))
    hsi_flip_h: bool = field(default_factory=lambda: _parse_bool(os.getenv("HSI_FLIP_H", "false")))
    hsi_flip_v: bool = field(default_factory=lambda: _parse_bool(os.getenv("HSI_FLIP_V", "false")))
    hsi_use_colormap: bool = field(default_factory=lambda: _parse_bool(os.getenv("HSI_USE_COLORMAP", "false")))

    # ===== Logging =====
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    debug: bool = field(default_factory=lambda: _parse_bool(os.getenv("DEBUG", "false")))

    # ===== Model Selection =====
    available_models: Dict[str, str] = field(default_factory=lambda: _parse_models_dict(os.getenv("AVAILABLE_MODELS", "")))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", ""))

    def __post_init__(self):
        """Post-initialization processing."""
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.results_dir, self.log_dir]:
            if dir_path and not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """
        Validate settings and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check device
        if self.device not in ["cpu", "cuda", "mps"]:
            errors.append(f"Invalid device: {self.device}. Must be cpu, cuda, or mps")

        # Check thresholds
        if not 0 <= self.default_prob_threshold <= 1:
            errors.append(f"Invalid probability threshold: {self.default_prob_threshold}. Must be in [0, 1]")

        if not 0 <= self.grid_crack_ratio <= 1:
            errors.append(f"Invalid grid crack ratio: {self.grid_crack_ratio}. Must be in [0, 1]")

        if not 0 <= self.cluster_crack_ratio <= 1:
            errors.append(f"Invalid cluster crack ratio: {self.cluster_crack_ratio}. Must be in [0, 1]")

        if not 0 <= self.overlay_alpha <= 1:
            errors.append(f"Invalid overlay alpha: {self.overlay_alpha}. Must be in [0, 1]")

        # Check validation split size
        if not 0 < self.val_split_size < 1:
            errors.append(f"Invalid val_split_size: {self.val_split_size}. Must be in (0, 1)")

        # Check wavelengths
        if self.wl_min >= self.wl_max:
            errors.append(f"Invalid wavelength range: {self.wl_min}-{self.wl_max}. Min must be < max")

        # Check grid cell size
        if self.grid_cell_size < 1:
            errors.append(f"Invalid grid cell size: {self.grid_cell_size}. Must be > 0")

        # Check batch size
        if self.inference_batch_size < 1:
            errors.append(f"Invalid batch size: {self.inference_batch_size}. Must be > 0")

        # Warn about missing optional paths (not errors)
        if self.cnn_checkpoint and not self.cnn_checkpoint.exists():
            errors.append(f"Warning: CNN checkpoint not found: {self.cnn_checkpoint}")

        if self.sam2_checkpoint and not self.sam2_checkpoint.exists():
            errors.append(f"Warning: SAM2 checkpoint not found: {self.sam2_checkpoint}")

        return errors

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get path for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Resolved path or None if not found
        """
        if model_name in self.available_models:
            return _resolve_path(self.available_models[model_name])

        # Try to find in models directory
        potential_path = self.models_dir / f"{model_name}.joblib"
        if potential_path.exists():
            return potential_path

        potential_path = self.models_dir / f"{model_name}.pkl"
        if potential_path.exists():
            return potential_path

        return None

    def auto_discover_models(self) -> Dict[str, Path]:
        """
        Auto-discover models in models directory.

        Returns:
            Dictionary of {name: path}
        """
        if not self.models_dir.exists():
            return {}

        models = {}
        for ext in ['.joblib', '.pkl', '.pth']:
            for model_file in self.models_dir.glob(f'*{ext}'):
                name = model_file.stem
                models[name] = model_file

        return models


# Global settings instance
settings = Settings()
