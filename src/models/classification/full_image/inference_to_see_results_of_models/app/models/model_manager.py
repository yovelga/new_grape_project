"""
Model Manager - Shared model loading and management service.

Provides centralized model loading for both Visual Debug and Optuna tabs.
"""

import joblib
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
import numpy as np

from app.config.types import PreprocessConfig
from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model metadata."""
    path: str
    name: str
    model_type: str
    n_classes: int
    preprocess_cfg: Optional["PreprocessConfig"] = None


class ModelManager:
    """
    Centralized model loading and inference management.
    
    Provides a single source of truth for loaded models, shared across UI tabs.
    """
    
    def __init__(self):
        """Initialize model manager."""
        self.model: Optional[Any] = None
        self.model_info: Optional[ModelInfo] = None
        self.preprocess_cfg: Optional[PreprocessConfig] = None
        self._inference_cache_key: Optional[str] = None
    
    def load_model(self, model_path: str, preprocess_cfg: Optional[PreprocessConfig] = None) -> ModelInfo:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file (.joblib, .pkl, .pth)
            preprocess_cfg: Optional preprocessing configuration. If None, uses default SNV + wavelength filtering.
            
        Returns:
            ModelInfo with metadata
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create default preprocessing config if not provided
        # All models require SNV + wavelength filtering [450-925nm]
        if preprocess_cfg is None:
            preprocess_cfg = PreprocessConfig(
                use_snv=settings.apply_snv,  # Default: True
                wl_min=settings.wl_min,      # Default: 450nm
                wl_max=settings.wl_max,      # Default: 925nm
                wavelengths=None,  # Will be loaded from .hdr file during inference
                band_indices=None,
                use_l2_norm=False
            )
            logger.info(f"Using default preprocessing: SNV={preprocess_cfg.use_snv}, "
                        f"wavelength range=[{preprocess_cfg.wl_min}-{preprocess_cfg.wl_max}]nm")
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load raw model
            raw_model = joblib.load(str(model_path))
            model_type = type(raw_model).__name__
            
            # Wrap in adapter for consistent interface
            from app.models.adapters_new import SklearnAdapter
            model = SklearnAdapter(raw_model, name=model_path.stem)
            
            # Get number of classes from adapter
            n_classes = model.n_classes
            
            # Store wrapped model and info
            self.model = model
            self.model_info = ModelInfo(
                path=str(model_path),
                name=model_path.name,
                model_type=model_type,
                n_classes=n_classes,
                preprocess_cfg=preprocess_cfg
            )
            self.preprocess_cfg = preprocess_cfg
            
            # Update cache key (invalidates old cache if model changed)
            self._inference_cache_key = f"{model_path}_{id(model)}"
            
            logger.info(f"Model loaded successfully: {model_type}, {n_classes} classes")
            
            return self.model_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None and self.model_info is not None
    
    def get_model(self) -> Optional[Any]:
        """Get the loaded model object."""
        return self.model
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get model metadata."""
        return self.model_info
    
    def get_preprocess_cfg(self) -> Optional[PreprocessConfig]:
        """Get preprocessing configuration."""
        return self.preprocess_cfg
    
    def get_cache_key(self) -> Optional[str]:
        """
        Get unique cache key for current model.
        
        Used by inference cache to invalidate when model changes.
        """
        return self._inference_cache_key
    
    def create_inference_fn(self, target_class_index: int = 1) -> Callable[[str], np.ndarray]:
        """
        Create an inference function for the loaded model.
        
        Args:
            target_class_index: Class index to extract probabilities for (default: 1 for CRACK)
            
        Returns:
            Inference function: path -> probability map
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        def inference_fn(path: str) -> np.ndarray:
            """Run inference on HSI cube and return probability map."""
            from app.io import ENVIReader
            from app.inference.prob_map import build_prob_map
            from pathlib import Path as PathLib
            
            # Convert to Path object
            folder = PathLib(path)
            
            # If it's a file, use directly
            if folder.is_file():
                hdr_file = folder
            else:
                # Search for .hdr file in folder (same logic as Visual Debug tab)
                hdr_files = list(folder.glob("*.hdr"))
                if not hdr_files:
                    hdr_files = list(folder.glob("HS/results/*.hdr"))
                if not hdr_files:
                    hdr_files = list(folder.glob("**/*.hdr"))
                
                # Filter to REFLECTANCE files if multiple found
                if len(hdr_files) > 1:
                    reflectance_files = [f for f in hdr_files if 'REFLECTANCE' in f.name.upper()]
                    if reflectance_files:
                        hdr_files = reflectance_files
                
                if not hdr_files:
                    raise FileNotFoundError(f"No .hdr file found in {folder}")
                
                hdr_file = hdr_files[0]
            
            # Load cube and wavelengths
            reader = ENVIReader(str(hdr_file))
            cube = reader.read()  # Returns (H, W, C) array
            wavelengths = reader.get_wavelengths()  # Extract wavelengths from .hdr
            
            # Update preprocessing config with wavelengths from this specific .hdr file
            # This is necessary because wavelengths are not known at model load time
            preprocess_cfg_with_wl = PreprocessConfig(
                use_snv=self.preprocess_cfg.use_snv,
                wl_min=self.preprocess_cfg.wl_min,
                wl_max=self.preprocess_cfg.wl_max,
                wavelengths=wavelengths,  # Now populated from .hdr file
                band_indices=self.preprocess_cfg.band_indices,
                use_l2_norm=self.preprocess_cfg.use_l2_norm
            )
            
            # Run inference with updated config
            prob_map = build_prob_map(
                cube, 
                self.model, 
                preprocess_cfg_with_wl,  # Use config with wavelengths
                target_class_index=target_class_index,
                chunk_size=100_000
            )
            
            return prob_map
        
        return inference_fn
    
    def get_summary(self) -> str:
        """Get human-readable summary of loaded model."""
        if not self.is_loaded():
            return "No model loaded"
        
        info = self.model_info
        lines = [
            f"Model: {info.name}",
            f"Type: {info.model_type}",
            f"Classes: {info.n_classes}",
            f"Path: {info.path}"
        ]
        
        return "\n".join(lines)
    
    def unload(self):
        """Unload current model and clear state."""
        self.model = None
        self.model_info = None
        self.preprocess_cfg = None
        self._inference_cache_key = None
        logger.info("Model unloaded")


# Singleton instance (optional - can also be passed as dependency)
_global_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager
