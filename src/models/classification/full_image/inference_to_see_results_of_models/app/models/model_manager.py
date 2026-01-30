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


# =====================================================================
# BoosterWrapper - Required for loading XGBoost models trained with
# native API and custom PR-AUC early stopping
# =====================================================================
class BoosterWrapper:
    """
    Wrapper to make xgb.Booster work with sklearn-like interface.
    
    This class is used when training XGBoost with the native API (xgb.train)
    for custom metric early stopping (e.g., CRACK PR-AUC). It provides
    sklearn-compatible predict() and predict_proba() methods.
    """
    def __init__(self, booster, n_classes, best_iteration, best_score):
        self._Booster = booster
        self.n_classes_ = n_classes
        self._classes = np.arange(n_classes)
        self.best_iteration = best_iteration
        self.best_score = best_score
        
    @property
    def classes_(self):
        return self._classes
    
    def predict(self, X):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        probs = self._Booster.predict(dmatrix)
        if probs.ndim == 1:
            probs = probs.reshape(-1, self.n_classes_)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        probs = self._Booster.predict(dmatrix)
        if probs.ndim == 1:
            probs = probs.reshape(-1, self.n_classes_)
        return probs
    
    def get_booster(self):
        return self._Booster


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
    
    def _is_autoencoder_folder(self, path: Path) -> bool:
        """Check if path is an autoencoder model folder."""
        if path.is_dir():
            return (path / 'autoencoder_best_model.pt').exists() and \
                   (path / 'model_config.json').exists() and \
                   (path / 'scaler.joblib').exists()
        return False
    
    def _load_autoencoder(self, folder_path: Path, preprocess_cfg: PreprocessConfig) -> ModelInfo:
        """Load autoencoder model from folder."""
        import torch
        import torch.nn as nn
        import json
        
        model_file = folder_path / 'autoencoder_best_model.pt'
        config_file = folder_path / 'model_config.json'
        scaler_file = folder_path / 'scaler.joblib'
        
        logger.info(f"Loading autoencoder from: {folder_path}")
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        hidden_dims = tuple(config['hidden_dims'])
        input_dim = config['input_dim']
        threshold = config['threshold']
        training_class = config.get('training_class', 'CRACK')
        
        logger.info(f"Autoencoder config: input_dim={input_dim}, hidden_dims={hidden_dims}, "
                    f"threshold={threshold:.6f}, training_class={training_class}")
        
        # Load scaler
        scaler = joblib.load(str(scaler_file))
        
        # Define autoencoder architecture
        class SpectralAutoencoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims):
                super().__init__()
                h1, h2, h3 = hidden_dims
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(0.2),
                    nn.Linear(h1, h2), nn.ReLU(), nn.BatchNorm1d(h2), nn.Dropout(0.2),
                    nn.Linear(h2, h3), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(h3, h2), nn.ReLU(), nn.BatchNorm1d(h2), nn.Dropout(0.2),
                    nn.Linear(h2, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(0.2),
                    nn.Linear(h1, input_dim),
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        # Create model and load weights
        raw_model = SpectralAutoencoder(input_dim, hidden_dims)
        checkpoint = torch.load(str(model_file), map_location='cpu')
        
        # Handle both formats: raw state_dict or checkpoint with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        raw_model.load_state_dict(state_dict)
        raw_model.eval()
        
        # Wrap in AutoencoderAdapter
        from app.models.adapters_new import AutoencoderAdapter
        model = AutoencoderAdapter(
            raw_model, scaler=scaler, threshold=threshold,
            training_class=training_class, name=folder_path.name
        )
        
        # Store model and info
        self.model = model
        self.model_info = ModelInfo(
            path=str(folder_path),
            name=f"Autoencoder ({training_class})",
            model_type="Autoencoder",
            n_classes=2,
            preprocess_cfg=preprocess_cfg
        )
        self.preprocess_cfg = preprocess_cfg
        self._inference_cache_key = f"{folder_path}_{id(model)}"
        
        logger.info(f"Autoencoder loaded successfully: threshold={threshold:.6f}")
        return self.model_info

    def load_model(self, model_path: str, preprocess_cfg: Optional[PreprocessConfig] = None) -> ModelInfo:
        """
        Load model from file or folder.
        
        Args:
            model_path: Path to model file (.joblib, .pkl) or autoencoder folder
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
        if preprocess_cfg is None:
            preprocess_cfg = PreprocessConfig(
                use_snv=settings.apply_snv,
                wl_min=settings.wl_min,
                wl_max=settings.wl_max,
                wavelengths=None,
                band_indices=None,
                use_l2_norm=False
            )
            logger.info(f"Using default preprocessing: SNV={preprocess_cfg.use_snv}, "
                        f"wavelength range=[{preprocess_cfg.wl_min}-{preprocess_cfg.wl_max}]nm")
        
        # Check if this is an autoencoder folder
        if self._is_autoencoder_folder(model_path):
            return self._load_autoencoder(model_path, preprocess_cfg)
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Register BoosterWrapper for pickle compatibility
            import sys
            if 'train_xgboost_row1_segments' not in sys.modules:
                import types
                mock_module = types.ModuleType('train_xgboost_row1_segments')
                mock_module.BoosterWrapper = BoosterWrapper
                sys.modules['train_xgboost_row1_segments'] = mock_module
            
            import __main__
            if not hasattr(__main__, 'BoosterWrapper'):
                __main__.BoosterWrapper = BoosterWrapper
            
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
