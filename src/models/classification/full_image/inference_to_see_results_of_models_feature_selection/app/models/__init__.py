"""Model loading and adapter module"""

from .model_manager import (
    ModelManager, 
    ModelInfo, 
    get_model_manager,
    MODEL_CATEGORY_FULL,
    MODEL_CATEGORY_REDUCED,
    MODEL_CATEGORY_AUTOENCODER,
)
from .feature_alignment import (
    align_features_for_model,
    wavelengths_to_feature_names,
    normalize_feature_name,
    resolve_reduced_package,
    is_reduced_model_package,
    list_reduced_packages,
    ReducedModelPackageInfo,
)

__all__ = [
    "ModelManager", 
    "ModelInfo", 
    "get_model_manager",
    "MODEL_CATEGORY_FULL",
    "MODEL_CATEGORY_REDUCED",
    "MODEL_CATEGORY_AUTOENCODER",
    "align_features_for_model",
    "wavelengths_to_feature_names",
    "normalize_feature_name",
    "resolve_reduced_package",
    "is_reduced_model_package",
    "list_reduced_packages",
    "ReducedModelPackageInfo",
]
