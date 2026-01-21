"""
Postprocessing module for full-image inference results.

Provides:
- PostprocessConfig: Configuration dataclass
- PostprocessPipeline: Main pipeline class
- Morphological operations (closing, opening, etc.)
- Blob filtering utilities
"""

from .pipeline import (
    PostprocessConfig,
    PostprocessPipeline,
    LegacyPostprocessPipeline,
    visualize_predictions,
)

from .morphology import (
    morphological_close,
    morphological_open,
    dilate,
    erode,
    get_backend_info,
)

from .blob_filters import (
    BlobFeatures,
    extract_connected_components,
    compute_blob_features,
    extract_all_blob_features,
    filter_blobs,
    create_filtered_mask,
)

__all__ = [
    # Pipeline
    "PostprocessConfig",
    "PostprocessPipeline",
    "LegacyPostprocessPipeline",
    "visualize_predictions",
    # Morphology
    "morphological_close",
    "morphological_open",
    "dilate",
    "erode",
    "get_backend_info",
    # Blob filters
    "BlobFeatures",
    "extract_connected_components",
    "compute_blob_features",
    "extract_all_blob_features",
    "filter_blobs",
    "create_filtered_mask",
]
