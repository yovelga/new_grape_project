"""
Traditional ML model wrappers (sklearn-based models).

Provides wrappers for LDA, XGBoost, Random Forest, SVM, etc.
"""

import numpy as np
import joblib
from typing import Optional, Any
from pathlib import Path


class SklearnModelWrapper:
    """
    Wrapper for sklearn-based models to provide unified interface.

    Handles LDA, Random Forest, XGBoost, SVM, Logistic Regression, etc.
    """

    def __init__(self, model: Any, model_type: str = "auto"):
        """
        Initialize sklearn model wrapper.

        Args:
            model: Sklearn-compatible model
            model_type: Type hint for model ('lda', 'xgboost', 'rf', 'svm', 'auto')
        """
        self.model = model
        self.model_type = model_type

        # Detect model type if auto
        if model_type == "auto":
            self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect model type from class name."""
        class_name = self.model.__class__.__name__.lower()

        if 'lda' in class_name or 'lineardiscriminant' in class_name:
            return 'lda'
        elif 'xgb' in class_name:
            return 'xgboost'
        elif 'randomforest' in class_name or 'forest' in class_name:
            return 'random_forest'
        elif 'svm' in class_name or 'svc' in class_name:
            return 'svm'
        elif 'logistic' in class_name:
            return 'logistic_regression'
        elif 'pls' in class_name:
            return 'pls'
        else:
            return 'unknown'

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features (N, D)

        Returns:
            Predicted labels (N,)
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features (N, D)

        Returns:
            Class probabilities (N, C)
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            n_samples = len(predictions)
            n_classes = len(np.unique(predictions))

            proba = np.zeros((n_samples, n_classes))
            proba[np.arange(n_samples), predictions] = 1.0
            return proba

    @classmethod
    def from_file(cls, path: str, model_type: str = "auto") -> 'SklearnModelWrapper':
        """
        Load model from joblib file.

        Args:
            path: Path to .joblib or .pkl file
            model_type: Model type hint

        Returns:
            SklearnModelWrapper instance
        """
        model = joblib.load(path)
        return cls(model, model_type)


class LDAModelShim:
    """
    Shim for legacy LDA pickle files that may have custom class wrappers.

    Handles models saved with custom __main__.LDAModel classes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize shim."""
        pass

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)

    def _get_inner_model(self):
        """Find the actual sklearn model inside the wrapper."""
        for attr in ('model', 'clf', 'estimator_', 'lda', 'classifier', 'classifier_'):
            estimator = getattr(self, attr, None)
            if estimator is not None:
                return estimator
        return None

    def predict(self, X):
        """Predict using inner model."""
        inner = self._get_inner_model()
        if inner is None or not hasattr(inner, 'predict'):
            raise AttributeError("LDAModelShim: no inner model with predict")
        return inner.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using inner model."""
        inner = self._get_inner_model()
        if inner is None or not hasattr(inner, 'predict_proba'):
            raise AttributeError("LDAModelShim: no inner model with predict_proba")
        return inner.predict_proba(X)


def load_sklearn_model(path: str, model_type: str = "auto") -> 'SklearnModelWrapper':
    """
    Load sklearn model from file with automatic shim handling.

    Args:
        path: Path to model file
        model_type: Model type hint

    Returns:
        SklearnModelWrapper instance
    """
    # Register shims for legacy models
    import sys
    import __main__

    # Add shims to __main__ namespace for unpickling
    if not hasattr(__main__, 'LDAModel'):
        __main__.LDAModel = LDAModelShim

    try:
        model = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

    # If model is a shim, extract inner model
    if isinstance(model, LDAModelShim):
        inner = model._get_inner_model()
        if inner is not None:
            model = inner

    return SklearnModelWrapper(model, model_type)


def find_scaler_for_model(model_path: str) -> Optional[str]:
    """
    Find associated scaler file for a model.

    Looks for scaler files in same directory as model:
    - {model_name}_scaler.joblib
    - scaler.joblib
    - StandardScaler.joblib

    Args:
        model_path: Path to model file

    Returns:
        Path to scaler file if found, None otherwise
    """
    model_path = Path(model_path)
    model_dir = model_path.parent
    model_stem = model_path.stem

    # Try different scaler naming conventions
    scaler_candidates = [
        model_dir / f"{model_stem}_scaler.joblib",
        model_dir / f"{model_stem}_scaler.pkl",
        model_dir / "scaler.joblib",
        model_dir / "scaler.pkl",
        model_dir / "StandardScaler.joblib",
        model_dir / "StandardScaler.pkl",
    ]

    for candidate in scaler_candidates:
        if candidate.exists():
            return str(candidate)

    return None


def load_scaler(path: str) -> Any:
    """
    Load sklearn scaler from file.

    Args:
        path: Path to scaler file

    Returns:
        Scaler object
    """
    return joblib.load(path)
