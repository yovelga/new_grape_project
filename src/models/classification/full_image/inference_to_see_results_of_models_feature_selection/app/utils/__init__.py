"""
Utilities module.

Provides logging, results I/O, and image processing utilities.
"""

from .logging import (
    Logger,
    setup_logger,
    logger,
    get_log_level,
    configure_global_logger,
    get_logger,
    setup_app_logger,
)
from .results_io import (
    save_json,
    load_json,
    save_csv,
    load_csv,
    save_text,
    save_best_params,
    save_trials_csv,
    save_per_sample_results,
    save_summary,
    create_experiment_dir,
    NumpyJSONEncoder,
)
from .image_ops import (
    normalize_to_uint8,
    apply_colormap,
)
from .orientation import (
    ensure_hwc,
    apply_display_transform,
    get_transformed_shape,
    DisplayTransform,
    validate_2d_for_display,
    contiguous_array,
)

__all__ = [
    # Logging
    "Logger",
    "setup_logger",
    "logger",
    "get_log_level",
    "configure_global_logger",
    "get_logger",
    "setup_app_logger",
    # Results I/O
    "save_json",
    "load_json",
    "save_csv",
    "load_csv",
    "save_text",
    "save_best_params",
    "save_trials_csv",
    "save_per_sample_results",
    "save_summary",
    "create_experiment_dir",
    "NumpyJSONEncoder",
    # Image Ops
    "normalize_to_uint8",
    "apply_colormap",
    # Orientation
    "ensure_hwc",
    "apply_display_transform",
    "get_transformed_shape",
    "DisplayTransform",
    "validate_2d_for_display",
    "contiguous_array",

    "save_summary",
    "create_experiment_dir",
    "NumpyJSONEncoder",
    # Image operations
    "normalize_to_uint8",
    "apply_colormap"
]
