"""
Model Manager - Shared model loading and management service.

Provides centralized model loading for both Visual Debug and Optuna tabs.
"""

import joblib
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
import numpy as np

from app.config.types import PreprocessConfig
from app.config.settings import settings

logger = logging.getLogger(__name__)


# =====================================================================
# Model Category Constants
# =====================================================================
MODEL_CATEGORY_FULL = "full"          # Full-feature model (all spectral bands)
MODEL_CATEGORY_REDUCED = "reduced"    # Reduced-feature model (BFS subset)
MODEL_CATEGORY_AUTOENCODER = "autoencoder"  # Autoencoder model


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
    """
    Model metadata with feature contract information.
    
    Attributes:
        path: Path to model file or package folder
        name: Human-readable model name
        model_type: Type of model (e.g., "XGBClassifier", "Autoencoder")
        n_classes: Number of output classes
        preprocess_cfg: Preprocessing configuration
        model_category: Category - "full", "reduced", or "autoencoder"
        required_feature_names: List of required feature names for reduced models.
            None for full-feature models. Order MUST match training order.
    """
    path: str
    name: str
    model_type: str
    n_classes: int
    preprocess_cfg: Optional["PreprocessConfig"] = None
    model_category: str = MODEL_CATEGORY_FULL
    required_feature_names: Optional[List[str]] = None
    
    @property
    def is_reduced(self) -> bool:
        """Check if this is a reduced-feature model."""
        return self.model_category == MODEL_CATEGORY_REDUCED
    
    @property
    def is_full(self) -> bool:
        """Check if this is a full-feature model."""
        return self.model_category == MODEL_CATEGORY_FULL
    
    @property
    def is_autoencoder(self) -> bool:
        """Check if this is an autoencoder model."""
        return self.model_category == MODEL_CATEGORY_AUTOENCODER
    
    @property
    def expected_feature_count(self) -> Optional[int]:
        """Get expected number of features for reduced models."""
        if self.required_feature_names:
            return len(self.required_feature_names)
        return None


class ModelManager:
    """
    Centralized model loading and inference management.
    
    Provides a single source of truth for loaded models, shared across UI tabs.
    
    Supports two model types:
    - Full-feature models: Single .pkl/.joblib files expecting all spectral bands
    - Reduced-feature models: Package folders containing model + feature_names.json
    """
    
    def __init__(self):
        """Initialize model manager."""
        self.model: Optional[Any] = None
        self.model_info: Optional[ModelInfo] = None
        self.preprocess_cfg: Optional[PreprocessConfig] = None
        self._inference_cache_key: Optional[str] = None
        # Required feature names for reduced models (None for full-feature models)
        self.required_feature_names: Optional[List[str]] = None
    
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
            preprocess_cfg=preprocess_cfg,
            model_category=MODEL_CATEGORY_AUTOENCODER,
            required_feature_names=None
        )
        self.preprocess_cfg = preprocess_cfg
        self.required_feature_names = None  # Autoencoders use full features
        self._inference_cache_key = f"{folder_path}_{id(model)}"
        
        logger.info(f"Autoencoder loaded successfully: threshold={threshold:.6f}")
        return self.model_info
    
    def _is_reduced_model_package(self, path: Path) -> bool:
        """
        Check if path is a reduced model package folder.
        
        A reduced model package is a directory that:
        - Is NOT an autoencoder folder
        - Contains (directly or nested) both a model file and feature_names.json
        """
        if not path.is_dir():
            return False
        if self._is_autoencoder_folder(path):
            return False
        
        # Try to resolve as reduced package
        try:
            from app.models.feature_alignment import resolve_reduced_package
            resolve_reduced_package(path)
            return True
        except (ValueError, ImportError):
            return False
    
    def _load_reduced_package(self, folder_path: Path, preprocess_cfg: PreprocessConfig) -> ModelInfo:
        """
        Load a reduced-feature model from a package folder.
        
        Package folder must contain:
        - Model file (*.pkl or *.joblib)
        - feature_names.json (list of required feature names in training order)
        
        The package may have nested structure with timestamp folders and Balanced/Unbalanced subdirs.
        
        Args:
            folder_path: Path to the package folder
            preprocess_cfg: Preprocessing configuration
            
        Returns:
            ModelInfo with model_category="reduced" and required_feature_names set
        """
        from app.models.feature_alignment import resolve_reduced_package, log_feature_alignment_summary
        
        logger.info(f"Loading reduced model package from: {folder_path}")
        
        # Resolve package structure
        pkg_info = resolve_reduced_package(folder_path)
        
        logger.info(
            f"Resolved package:\n"
            f"  Model file: {pkg_info.model_path}\n"
            f"  Feature names: {pkg_info.feature_names_path}\n"
            f"  Balance type: {pkg_info.balance_type or 'Unknown'}\n"
            f"  Required features: {len(pkg_info.feature_names)}"
        )
        
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
        raw_model = joblib.load(str(pkg_info.model_path))
        model_type = type(raw_model).__name__
        
        # Wrap in adapter for consistent interface
        from app.models.adapters_new import SklearnAdapter
        model = SklearnAdapter(raw_model, name=folder_path.name)
        
        # Get number of classes from adapter
        n_classes = model.n_classes
        
        # Build display name
        display_name = folder_path.name
        if pkg_info.balance_type:
            display_name += f" ({pkg_info.balance_type})"
        
        # Store model and info
        self.model = model
        self.required_feature_names = pkg_info.feature_names
        self.model_info = ModelInfo(
            path=str(folder_path),
            name=display_name,
            model_type=model_type,
            n_classes=n_classes,
            preprocess_cfg=preprocess_cfg,
            model_category=MODEL_CATEGORY_REDUCED,
            required_feature_names=pkg_info.feature_names
        )
        self.preprocess_cfg = preprocess_cfg
        self._inference_cache_key = f"{folder_path}_{id(model)}"
        
        # Log summary for sanity check
        log_feature_alignment_summary(
            full_feature_count=0,  # Will be known at inference time
            required_feature_names=pkg_info.feature_names,
            model_name=display_name
        )
        
        logger.info(
            f"✓ REDUCED MODEL loaded successfully:\n"
            f"  Type: {model_type}\n"
            f"  Classes: {n_classes}\n"
            f"  Required features: {len(pkg_info.feature_names)}\n"
            f"  First 5: {pkg_info.feature_names[:5]}\n"
            f"  Last 2: {pkg_info.feature_names[-2:]}"
        )
        
        return self.model_info
    
    def _find_feature_names_json_nearby(self, model_file: Path) -> Optional[Path]:
        """
        Check if a model file is inside a reduced model package.
        
        Looks for feature_names.json in:
        1. Same directory as model file
        2. Parent directories (up to 5 levels)
        
        Args:
            model_file: Path to .pkl/.joblib model file
            
        Returns:
            Path to feature_names.json if found, None otherwise
        """
        # Check same directory
        same_dir = model_file.parent / "feature_names.json"
        if same_dir.exists():
            return same_dir
        
        # Check parent directories (up to 5 levels)
        current = model_file.parent
        for _ in range(5):
            parent = current.parent
            if parent == current:
                break  # Reached root
            
            feature_file = parent / "feature_names.json"
            if feature_file.exists():
                return feature_file
            
            # Also check subdirs like Balanced/, Unbalanced/
            for subdir in parent.iterdir():
                if subdir.is_dir():
                    feature_file = subdir / "feature_names.json"
                    if feature_file.exists():
                        return feature_file
            
            current = parent
        
        return None
    
    def _load_reduced_model_file_with_features(
        self, 
        model_path: Path, 
        feature_names_path: Path,
        preprocess_cfg: PreprocessConfig
    ) -> ModelInfo:
        """
        Load a model file that's part of a reduced model package.
        
        Used when user selects .pkl file directly instead of the package folder.
        
        Args:
            model_path: Path to .pkl/.joblib model file
            feature_names_path: Path to feature_names.json
            preprocess_cfg: Preprocessing configuration
            
        Returns:
            ModelInfo with model_category="reduced"
        """
        import json
        from app.models.feature_alignment import normalize_feature_name, log_feature_alignment_summary
        
        logger.info(f"Loading REDUCED model from file: {model_path}")
        
        # Load feature names
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        # Normalize feature names
        feature_names = [normalize_feature_name(name) for name in feature_names]
        
        # Determine balance type from path
        balance_type = None
        for part in model_path.parts:
            if part in ['Balanced', 'Unbalanced']:
                balance_type = part
                break
        
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
        
        # Build display name
        display_name = model_path.stem
        if balance_type:
            display_name += f" ({balance_type})"
        
        # Store model and info
        self.model = model
        self.required_feature_names = feature_names
        self.model_info = ModelInfo(
            path=str(model_path),
            name=display_name,
            model_type=model_type,
            n_classes=n_classes,
            preprocess_cfg=preprocess_cfg,
            model_category=MODEL_CATEGORY_REDUCED,
            required_feature_names=feature_names
        )
        self.preprocess_cfg = preprocess_cfg
        self._inference_cache_key = f"{model_path}_{id(model)}"
        
        # Log summary for sanity check
        log_feature_alignment_summary(
            full_feature_count=0,  # Will be known at inference time
            required_feature_names=feature_names,
            model_name=display_name
        )
        
        logger.info(
            f"✓ REDUCED MODEL (from file) loaded successfully:\n"
            f"  Type: {model_type}\n"
            f"  Classes: {n_classes}\n"
            f"  Required features: {len(feature_names)}\n"
            f"  First 5: {feature_names[:5]}\n"
            f"  Last 2: {feature_names[-2:]}\n"
            f"  Feature contract: {feature_names_path}"
        )
        
        return self.model_info

    def load_model(self, model_path: str, preprocess_cfg: Optional[PreprocessConfig] = None) -> ModelInfo:
        """
        Load model from file or folder.
        
        Supports three model types:
        1. Full-feature models: Single .pkl/.joblib file expecting all spectral bands
        2. Reduced-feature models: Package folders containing model + feature_names.json
        3. Autoencoder models: Folders with autoencoder_best_model.pt + model_config.json + scaler.joblib
        
        Args:
            model_path: Path to model file (.joblib, .pkl) or model folder (autoencoder/reduced)
            preprocess_cfg: Optional preprocessing configuration. If None, uses default SNV + wavelength filtering.
            
        Returns:
            ModelInfo with metadata including model_category and required_feature_names
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            ValueError: If reduced model package is invalid
            Exception: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
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
        
        # Check if this is a reduced model package folder
        if model_path.is_dir():
            if self._is_reduced_model_package(model_path):
                return self._load_reduced_package(model_path, preprocess_cfg)
            else:
                raise ValueError(
                    f"Directory is not a valid model package: {model_path}\n"
                    f"For reduced models, ensure the folder contains:\n"
                    f"  - Model file (*.pkl or *.joblib)\n"
                    f"  - feature_names.json (list of feature names in training order)\n"
                    f"For autoencoders, ensure the folder contains:\n"
                    f"  - autoencoder_best_model.pt\n"
                    f"  - model_config.json\n"
                    f"  - scaler.joblib"
                )
        
        # File-based model: check if it's inside a reduced model package
        # (e.g., user selected .pkl file directly instead of the package folder)
        feature_names_nearby = self._find_feature_names_json_nearby(model_path)
        
        if feature_names_nearby is not None:
            # This .pkl file is inside a reduced model package!
            logger.info(
                f"Detected reduced model package (feature_names.json found nearby):\n"
                f"  Model file: {model_path}\n"
                f"  Feature names: {feature_names_nearby}"
            )
            return self._load_reduced_model_file_with_features(
                model_path, feature_names_nearby, preprocess_cfg
            )
        
        # Regular full-feature model
        logger.info(f"Loading FULL-FEATURE model from: {model_path}")
        
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
            self.required_feature_names = None  # Full-feature model: use all features
            self.model_info = ModelInfo(
                path=str(model_path),
                name=model_path.name,
                model_type=model_type,
                n_classes=n_classes,
                preprocess_cfg=preprocess_cfg,
                model_category=MODEL_CATEGORY_FULL,
                required_feature_names=None
            )
            self.preprocess_cfg = preprocess_cfg
            
            # Update cache key (invalidates old cache if model changed)
            self._inference_cache_key = f"{model_path}_{id(model)}"
            
            logger.info(
                f"✓ FULL-FEATURE MODEL loaded successfully:\n"
                f"  Type: {model_type}\n"
                f"  Classes: {n_classes}\n"
                f"  Model category: full (all features)"
            )
            
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
            # Pass required_feature_names for reduced models
            prob_map = build_prob_map(
                cube, 
                self.model, 
                preprocess_cfg_with_wl,  # Use config with wavelengths
                target_class_index=target_class_index,
                chunk_size=100_000,
                required_feature_names=self.required_feature_names
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
            f"Category: {info.model_category}",
            f"Classes: {info.n_classes}",
            f"Path: {info.path}"
        ]
        
        if info.is_reduced and info.required_feature_names:
            lines.append(f"Required features: {len(info.required_feature_names)}")
        
        return "\n".join(lines)
    
    def unload(self):
        """Unload current model and clear state."""
        self.model = None
        self.model_info = None
        self.preprocess_cfg = None
        self.required_feature_names = None
        self._inference_cache_key = None
        logger.info("Model unloaded")
    
    def get_required_feature_names(self) -> Optional[List[str]]:
        """
        Get required feature names for the loaded model.
        
        Returns:
            List of feature names for reduced models, None for full-feature models.
        """
        return self.required_feature_names


# Singleton instance (optional - can also be passed as dependency)
_global_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager
