"""
Hyperparameter tuning module.

Provides Optuna-based tuning and grid search for the inference pipeline.
"""

from .optuna_runner import (
    run_optuna,
    evaluate_final,
    run_full_tuning_pipeline,
    TuningSearchSpace,
    OptunaRunner,  # Legacy
)
from .grid_search_single import (
    run_grid_on_prob_map,
    get_default_param_grid,
    get_comprehensive_param_grid,
    save_grid_results,
    analyze_grid_results,
)

# New Optuna-based components
from .dataset_loader import DatasetCSVLoader, Sample
from .postprocess_patch import (
    PatchClassifierParams,
    PostprocessPatchClassifier,
    ClassificationResult,
)
from .metrics_report import (
    ConfusionMatrix,
    BinaryMetrics,
    MetricsCalculator,
    MetricsReport,
)
from .inference_cache import InferenceCache
from .optuna_tuner import OptunaTuner
from .optuna_worker import OptunaWorker
from .search_space import HyperparameterSpec, HyperparameterSearchSpace

__all__ = [
    # Optuna tuning
    "run_optuna",
    "evaluate_final",
    "run_full_tuning_pipeline",
    "TuningSearchSpace",
    "OptunaRunner",
    # Grid search
    "run_grid_on_prob_map",
    "get_default_param_grid",
    "get_comprehensive_param_grid",
    "save_grid_results",
    "analyze_grid_results",
    # New CSV-based Optuna components
    "DatasetCSVLoader",
    "Sample",
    "PatchClassifierParams",
    "PostprocessPatchClassifier",
    "ClassificationResult",
    "ConfusionMatrix",
    "BinaryMetrics",
    "MetricsCalculator",
    "MetricsReport",
    "InferenceCache",
    "OptunaTuner",
    "OptunaWorker",
    "HyperparameterSpec",
    "HyperparameterSearchSpace",
]
