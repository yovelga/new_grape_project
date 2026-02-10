"""
Model adapters for unified inference interface.

Provides ModelAdapter base class and implementations for sklearn and PyTorch models.
"""

import numpy as np
from typing import Optional, List, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Base interface for model adapters.

    Provides unified interface for different model types (sklearn, PyTorch, etc.).
    """

    def __init__(self, model: Any, name: str = "model"):
        """
        Initialize adapter.

        Args:
            model: The underlying model object
            name: Human-readable name for the model
        """
        self.model = model
        self.name = name
        self._validate_model()

    @abstractmethod
    def _validate_model(self):
        """Validate that the model is compatible. Raise ValueError if not."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features (N, D) where N=samples, D=features

        Returns:
            Probabilities (N, K) where K=number of classes

        Note:
            Output must be normalized probabilities (rows sum to 1.0)
        """
        pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of classes the model predicts."""
        pass

    @property
    def is_binary(self) -> bool:
        """Whether this is a binary classification model."""
        return self.n_classes == 2

    @property
    def classes_(self) -> Optional[List[Any]]:
        """Class labels if available."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', n_classes={self.n_classes})"


class SklearnAdapter(ModelAdapter):
    """
    Adapter for sklearn models and pipelines.

    Handles:
    - Standard sklearn classifiers
    - sklearn Pipelines
    - Models with or without predict_proba
    """

    def _validate_model(self):
        """Validate sklearn model."""
        if not hasattr(self.model, 'predict'):
            raise ValueError(
                f"Model does not have 'predict' method. "
                f"Expected sklearn-compatible model, got {type(self.model).__name__}"
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using sklearn model.

        Args:
            X: Input features (N, D)

        Returns:
            Probabilities (N, K)
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (N, D), got shape {X.shape}")

        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
        else:
            # Fallback: use predict and create one-hot encoding
            logger.warning(f"Model has no predict_proba, using predict() with one-hot encoding")
            predictions = self.model.predict(X)
            unique_classes = np.unique(predictions)
            n_classes = len(unique_classes)
            n_samples = len(predictions)

            proba = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(unique_classes):
                proba[predictions == cls, i] = 1.0

        # Validate output shape
        if proba.ndim != 2:
            raise ValueError(
                f"predict_proba must return 2D array (N, K), got {proba.ndim}D with shape {proba.shape}"
            )

        if proba.shape[0] != X.shape[0]:
            raise ValueError(
                f"predict_proba returned wrong number of samples: "
                f"expected {X.shape[0]}, got {proba.shape[0]}"
            )

        # Validate probabilities sum to 1 (within tolerance)
        row_sums = np.sum(proba, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            logger.warning(
                f"Probabilities don't sum to 1.0: "
                f"min={row_sums.min():.4f}, max={row_sums.max():.4f}"
            )

        # Validate probability range [0, 1]
        if np.any(proba < 0) or np.any(proba > 1):
            raise ValueError(
                f"Probabilities must be in [0, 1], "
                f"got min={proba.min():.4f}, max={proba.max():.4f}"
            )

        logger.debug(f"predict_proba: input shape={X.shape}, output shape={proba.shape}")

        return proba

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        # Try classes_ first (standard sklearn attribute)
        if hasattr(self.model, 'classes_'):
            try:
                classes = self.model.classes_
                if classes is not None:
                    return len(classes)
            except (AttributeError, ValueError):
                pass  # Some models have classes_ but it can raise errors

        # Try n_classes_ (some sklearn models like LDA)
        if hasattr(self.model, 'n_classes_'):
            try:
                n_classes = self.model.n_classes_
                if n_classes is not None:
                    return int(n_classes)
            except (AttributeError, ValueError):
                pass

        # Try _n_classes for some internal sklearn attributes
        if hasattr(self.model, '_n_classes'):
            try:
                return int(self.model._n_classes)
            except (AttributeError, ValueError):
                pass

        # For XGBoost: check objective to determine binary vs multi-class
        if hasattr(self.model, 'objective'):
            obj = self.model.objective
            if obj in ['binary:logistic', 'binary:hinge']:
                return 2

        # Try to infer from a dummy prediction
        try:
            dummy_X = np.zeros((1, self._get_n_features()))
            proba = self.model.predict_proba(dummy_X)
            return proba.shape[1]
        except Exception:
            pass

        # Default fallback: assume binary classification
        logger.warning(f"Cannot determine n_classes for {type(self.model).__name__}, assuming 2")
        return 2

    @property
    def classes_(self) -> Optional[List[Any]]:
        """Get class labels."""
        if hasattr(self.model, 'classes_'):
            return list(self.model.classes_)
        return None

    def _get_n_features(self) -> int:
        """Get number of input features expected by model."""
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        elif hasattr(self.model, 'n_features_'):
            return self.model.n_features_
        else:
            return 10  # Fallback guess


class TorchAdapter(ModelAdapter):
    """
    Adapter for PyTorch models.

    Handles:
    - PyTorch nn.Module models
    - Both binary (sigmoid) and multi-class (softmax) outputs
    - Automatic device management from settings
    """

    def __init__(self,
                 model: Any,
                 n_classes: int,
                 device: str = None,
                 name: str = "torch_model"):
        """
        Initialize PyTorch adapter.

        Args:
            model: PyTorch nn.Module
            n_classes: Number of output classes
            device: Device to run inference on (None = use settings)
            name: Model name
        """
        self._n_classes = n_classes
        self._device = device

        # Import torch (fail gracefully if not installed)
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. "
                "Install with: pip install torch torchvision"
            )

        # Set device
        if self._device is None:
            from ..config.settings import settings
            self._device = settings.device

        # Initialize base class
        super().__init__(model, name)

        # Move model to device and set to eval mode
        self.model = self.model.to(self._device)
        self.model.eval()

        logger.info(f"PyTorch model on device: {self._device}")

    def _validate_model(self):
        """Validate PyTorch model."""
        if not isinstance(self.model, self.torch.nn.Module):
            raise ValueError(
                f"Model must be torch.nn.Module, got {type(self.model).__name__}"
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using PyTorch model.

        Args:
            X: Input features (N, D)

        Returns:
            Probabilities (N, K)
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (N, D), got shape {X.shape}")

        # Convert to torch tensor
        X_tensor = self.torch.from_numpy(X).float().to(self._device)

        # Run inference
        with self.torch.no_grad():
            logits = self.model(X_tensor)

            # Apply activation based on number of classes
            if self._n_classes == 2:
                # Binary: use sigmoid
                proba_pos = self.torch.sigmoid(logits).cpu().numpy()

                # Handle both (N,) and (N, 1) outputs
                if proba_pos.ndim == 1:
                    proba_pos = proba_pos[:, np.newaxis]

                # Create (N, 2) array
                proba = np.hstack([1 - proba_pos, proba_pos])
            else:
                # Multi-class: use softmax
                proba = self.torch.softmax(logits, dim=1).cpu().numpy()

        # Validate output shape
        assert proba.ndim == 2, f"Output must be 2D, got {proba.ndim}D"
        assert proba.shape[0] == X.shape[0], f"Sample count mismatch"
        assert proba.shape[1] == self._n_classes, f"Class count mismatch"

        # Validate probabilities
        row_sums = np.sum(proba, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1"
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of [0, 1] range"

        logger.debug(f"predict_proba: input shape={X.shape}, output shape={proba.shape}")

        return proba

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return self._n_classes


class AutoencoderAdapter(ModelAdapter):
    """
    Adapter for Autoencoder-based anomaly detection models.

    Handles:
    - Autoencoder models trained to detect CRACK class
    - Converts reconstruction error to probability-like scores
    - Uses threshold from model_config.json for anomaly detection
    
    The autoencoder is trained on CRACK samples only. Low reconstruction error
    means the sample is similar to CRACK (high probability of CRACK).
    High reconstruction error means the sample is dissimilar (low CRACK probability).
    """

    def __init__(self,
                 model: Any,
                 scaler: Any,
                 threshold: float,
                 training_class: str = "CRACK",
                 device: str = None,
                 name: str = "autoencoder"):
        """
        Initialize Autoencoder adapter.

        Args:
            model: PyTorch autoencoder model
            scaler: StandardScaler from sklearn for input normalization
            threshold: Reconstruction error threshold for anomaly detection
            training_class: The class the autoencoder was trained on (default: CRACK)
            device: Device to run inference on (None = use settings)
            name: Model name
        """
        self.scaler = scaler
        self.threshold = threshold
        self.training_class = training_class
        self._device = device
        self._n_classes = 2  # Binary: CRACK vs non-CRACK

        # Import torch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError("PyTorch is not installed.")

        # Set device
        if self._device is None:
            from ..config.settings import settings
            self._device = settings.device

        # Initialize base class
        super().__init__(model, name)

        # Move model to device and set to eval mode
        self.model = self.model.to(self._device)
        self.model.eval()

        logger.info(f"Autoencoder model on device: {self._device}, threshold: {threshold:.6f}")

    def _validate_model(self):
        """Validate autoencoder model."""
        if not isinstance(self.model, self.torch.nn.Module):
            raise ValueError(f"Model must be torch.nn.Module, got {type(self.model).__name__}")

    def _get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error for each sample."""
        X_tensor = self.torch.from_numpy(X).float().to(self._device)
        
        with self.torch.no_grad():
            self.model.eval()
            reconstructed = self.model(X_tensor)
            mse = ((X_tensor - reconstructed) ** 2).mean(dim=1)
        
        return mse.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using autoencoder reconstruction error.
        
        Conversion: MSE → probability using sigmoid function.
        
        The autoencoder is trained on CRACK samples, so:
        - Low MSE → sample looks like CRACK → P(CRACK) is HIGH
        - High MSE → sample is anomalous (not CRACK) → P(CRACK) is LOW
        
        We use: P(CRACK) = sigmoid(-(MSE - threshold) / temperature)
                         = 1 / (1 + exp((MSE - threshold) / temperature))
        
        This gives a smooth S-curve centered at the threshold:
        - MSE << threshold → P(CRACK) ≈ 1.0
        - MSE == threshold → P(CRACK) = 0.5
        - MSE >> threshold → P(CRACK) ≈ 0.0

        Args:
            X: Input features (N, D) - NOT scaled yet

        Returns:
            Probabilities (N, 2) where [:, 1] is P(CRACK)
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (N, D), got shape {X.shape}")

        # Scale input using the trained scaler
        X_scaled = self.scaler.transform(X)

        # Compute reconstruction error
        errors = self._get_reconstruction_error(X_scaled.astype(np.float32))

        # Convert MSE to probability using sigmoid
        # temperature controls the steepness of the transition (smaller = sharper)
        temperature = self.threshold * 0.3  # ~30% of threshold as temperature
        prob_crack = 1.0 / (1.0 + np.exp((errors - self.threshold) / temperature))
        prob_crack = np.clip(prob_crack, 0.0, 1.0)
        
        # Create (N, 2) probability array: [NOT_CRACK, CRACK]
        proba = np.column_stack([1 - prob_crack, prob_crack])

        logger.debug(f"predict_proba (autoencoder): input shape={X.shape}, "
                     f"error range=[{errors.min():.4f}, {errors.max():.4f}]")

        return proba

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def classes_(self) -> Optional[List[Any]]:
        return ["NOT_CRACK", self.training_class]


def create_adapter(model: Any,
                   model_type: str = "auto",
                   **kwargs) -> ModelAdapter:
    """
    Create appropriate adapter for model.

    Args:
        model: Loaded model object
        model_type: Type hint ("sklearn", "torch", "autoencoder", or "auto")
        **kwargs: Additional arguments for adapter

    Returns:
        ModelAdapter instance
    """
    if model_type == "sklearn" or (model_type == "auto" and hasattr(model, 'predict')):
        return SklearnAdapter(model, **kwargs)

    elif model_type == "torch":
        # Requires n_classes to be specified
        if 'n_classes' not in kwargs:
            raise ValueError("TorchAdapter requires 'n_classes' argument")
        return TorchAdapter(model, **kwargs)

    elif model_type == "autoencoder":
        # Requires scaler and threshold
        if 'scaler' not in kwargs or 'threshold' not in kwargs:
            raise ValueError("AutoencoderAdapter requires 'scaler' and 'threshold' arguments")
        return AutoencoderAdapter(model, **kwargs)

    else:
        raise ValueError(
            f"Cannot auto-detect model type for {type(model).__name__}. "
            f"Please specify model_type='sklearn', 'torch', or 'autoencoder'"
        )
