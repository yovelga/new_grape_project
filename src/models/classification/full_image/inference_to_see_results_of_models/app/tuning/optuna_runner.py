"""
Optuna-based hyperparameter tuning for the inference pipeline.

Key design principles:
- Tuning uses ONLY train_df and val_df (test_df never touched during optimization)
- After best params are found, evaluate_final runs ONCE on test_df
- Uses shared modules (prob_map, postprocess pipeline) - no duplicated logic
- All results saved as artifacts (JSON, CSV, summary.txt)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Local imports
from ..metrics.classification import get_metric_value, compute_all_metrics
from ..utils.results_io import (
    save_json,
    save_summary,
    save_best_params,
    save_trials_csv,
    save_per_sample_results,
)
from ..postprocess.pipeline import PostprocessConfig, PostprocessPipeline

logger = logging.getLogger(__name__)


@dataclass
class TuningSearchSpace:
    """
    Search space configuration for hyperparameter tuning.

    Attributes:
        prob_threshold: (low, high) range for probability threshold
        morph_close_size: List of kernel sizes to try (0 = disabled)
        min_blob_area: (low, high) range for minimum blob area
        exclude_border: Whether to search over border exclusion
        border_margin_px: (low, high) range for border margin
    """
    prob_threshold: Tuple[float, float] = (0.3, 0.8)
    morph_close_size: List[int] = field(default_factory=lambda: [0, 3, 5, 7])
    min_blob_area: Tuple[int, int] = (0, 500)
    exclude_border: List[bool] = field(default_factory=lambda: [True, False])
    border_margin_px: Tuple[int, int] = (0, 20)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "prob_threshold": self.prob_threshold,
            "morph_close_size": self.morph_close_size,
            "min_blob_area": self.min_blob_area,
            "exclude_border": self.exclude_border,
            "border_margin_px": self.border_margin_px,
        }


def _sample_params_from_trial(trial, search_space: TuningSearchSpace) -> Dict[str, Any]:
    """
    Sample hyperparameters from Optuna trial.

    Args:
        trial: Optuna trial object
        search_space: Search space configuration

    Returns:
        Dictionary of sampled parameters
    """
    params = {}

    # Probability threshold
    params["prob_threshold"] = trial.suggest_float(
        "prob_threshold",
        search_space.prob_threshold[0],
        search_space.prob_threshold[1]
    )

    # Morphological closing size
    params["morph_close_size"] = trial.suggest_categorical(
        "morph_close_size",
        search_space.morph_close_size
    )

    # Minimum blob area
    params["min_blob_area"] = trial.suggest_int(
        "min_blob_area",
        search_space.min_blob_area[0],
        search_space.min_blob_area[1]
    )

    # Border exclusion
    params["exclude_border"] = trial.suggest_categorical(
        "exclude_border",
        search_space.exclude_border
    )

    # Border margin (only relevant if exclude_border is True)
    if params["exclude_border"]:
        params["border_margin_px"] = trial.suggest_int(
            "border_margin_px",
            search_space.border_margin_px[0],
            search_space.border_margin_px[1]
        )
    else:
        params["border_margin_px"] = 0

    return params


def _create_postprocess_config(params: Dict[str, Any]) -> PostprocessConfig:
    """
    Create PostprocessConfig from parameter dictionary.

    Args:
        params: Dictionary of parameters

    Returns:
        PostprocessConfig instance
    """
    return PostprocessConfig(
        prob_threshold=params.get("prob_threshold", 0.5),
        morph_close_size=params.get("morph_close_size", 0),
        min_blob_area=params.get("min_blob_area", 0),
        exclude_border=params.get("exclude_border", False),
        border_margin_px=params.get("border_margin_px", 0),
        circularity_min=params.get("circularity_min"),
        solidity_min=params.get("solidity_min"),
        aspect_ratio_range=params.get("aspect_ratio_range"),
    )


def _evaluate_on_split(
    df: pd.DataFrame,
    prob_map_fn: Callable[[str], np.ndarray],
    params: Dict[str, Any],
    label_col: str = "label",
    pos_label: Any = 1,
    metric: str = "f2"
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate parameters on a data split.

    Args:
        df: DataFrame with samples
        prob_map_fn: Function that takes image_path and returns probability map
        params: Hyperparameters for postprocessing
        label_col: Column name for labels
        pos_label: Positive class label
        metric: Metric to compute

    Returns:
        Tuple of (metric_value, per_sample_results_df)
    """
    # Create postprocess pipeline
    config = _create_postprocess_config(params)
    pipeline = PostprocessPipeline(config)

    y_true = []
    y_pred = []
    results_rows = []

    for idx, row in df.iterrows():
        sample_id = row.get("grape_id", idx)
        image_path = row["image_path"]
        true_label = row[label_col]

        try:
            # Get probability map
            prob_map = prob_map_fn(image_path)

            # Apply postprocessing
            mask, stats = pipeline.run(prob_map)

            # Determine prediction based on crack ratio
            # For binary: if any crack detected, predict positive
            pred_label = pos_label if stats["total_positive_pixels"] > 0 else (1 - pos_label if isinstance(pos_label, int) else 0)

            y_true.append(true_label)
            y_pred.append(pred_label)

            results_rows.append({
                "sample_id": sample_id,
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "crack_ratio": stats["crack_ratio"],
                "num_blobs": stats["num_blobs_after"],
                "positive_pixels": stats["total_positive_pixels"],
            })

        except Exception as e:
            logger.warning(f"Failed to process sample {sample_id}: {e}")
            # Skip failed samples
            continue

    if len(y_true) == 0:
        return 0.0, pd.DataFrame()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute metric
    metric_value = get_metric_value(y_true, y_pred, metric=metric, pos_label=pos_label)

    # Create results DataFrame
    results_df = pd.DataFrame(results_rows)

    return metric_value, results_df


def run_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    prob_map_fn: Callable[[str], np.ndarray],
    n_trials: int = 50,
    seed: int = 42,
    output_dir: str = "./optuna_results",
    metric: str = "f2",
    label_col: str = "label",
    pos_label: Any = 1,
    search_space: Optional[TuningSearchSpace] = None,
    model_name: str = "unknown_model"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run Optuna hyperparameter tuning using train/val split.

    IMPORTANT: This function ONLY uses val_df for scoring.
    test_df should NEVER be passed here.

    Args:
        train_df: Training DataFrame (used for reference only in this implementation)
        val_df: Validation DataFrame for computing objective score
        prob_map_fn: Function(image_path) -> prob_map (H, W) float32
        n_trials: Number of Optuna trials
        seed: Random seed for reproducibility
        output_dir: Directory to save artifacts
        metric: Optimization metric ('f2', 'f1', 'accuracy', 'macro_f1')
        label_col: Column name for labels
        pos_label: Positive class label for binary classification
        search_space: Optional custom search space
        model_name: Model name for logging

    Returns:
        Tuple of:
            - best_params: Dictionary of best hyperparameters
            - trials_df: DataFrame with all trial results

    Raises:
        ImportError: If Optuna is not installed
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning.\n"
            "Install with: pip install optuna"
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default search space
    if search_space is None:
        search_space = TuningSearchSpace()

    logger.info("=" * 60)
    logger.info("Starting Optuna hyperparameter tuning")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Val samples: {len(val_df)}")
    logger.info(f"  N trials: {n_trials}")
    logger.info(f"  Metric: {metric}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Track best validation results
    best_val_score = -float('inf')
    best_val_results_df = None

    def objective(trial):
        nonlocal best_val_score, best_val_results_df

        # Sample parameters
        params = _sample_params_from_trial(trial, search_space)

        # Evaluate on validation set ONLY
        val_score, val_results_df = _evaluate_on_split(
            val_df,
            prob_map_fn,
            params,
            label_col=label_col,
            pos_label=pos_label,
            metric=metric
        )

        # Track best results
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_results_df = val_results_df

        logger.info(f"Trial {trial.number}: {metric}={val_score:.4f}, params={params}")

        return val_score

    # Create study with fixed seed
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"tuning_{model_name}_{metric}"
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info("=" * 60)
    logger.info(f"Optimization complete!")
    logger.info(f"  Best {metric}: {best_value:.4f}")
    logger.info(f"  Best params: {best_params}")
    logger.info("=" * 60)

    # Create trials DataFrame
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        trial_data.update(trial.params)
        trials_data.append(trial_data)

    trials_df = pd.DataFrame(trials_data)

    # Save artifacts
    save_best_params(best_params, output_dir)
    save_trials_csv(trials_df, output_dir)

    if best_val_results_df is not None:
        save_per_sample_results(best_val_results_df, output_dir, "val_best_results.csv")

    # Save search space configuration
    save_json(search_space.to_dict(), str(output_path / "search_space.json"))

    return best_params, trials_df


def evaluate_final(
    test_df: pd.DataFrame,
    prob_map_fn: Callable[[str], np.ndarray],
    best_params: Dict[str, Any],
    output_dir: str,
    label_col: str = "label",
    pos_label: Any = 1,
    model_name: str = "unknown_model",
    seed: int = 42,
    metric: str = "f2",
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run final evaluation ONCE on test set with best parameters.

    This should be called AFTER tuning is complete.

    Args:
        test_df: Test DataFrame for final evaluation
        prob_map_fn: Function(image_path) -> prob_map
        best_params: Best hyperparameters from tuning
        output_dir: Directory to save results
        label_col: Column name for labels
        pos_label: Positive class label
        model_name: Model name for logging
        seed: Seed used during tuning (for logging)
        metric: Metric used during tuning (for logging)
        train_df: Optional train DataFrame (for summary only)
        val_df: Optional val DataFrame (for summary only)

    Returns:
        Tuple of:
            - metrics: Dictionary of all computed metrics
            - per_sample_df: DataFrame with per-sample results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running FINAL evaluation on TEST set")
    logger.info(f"  Test samples: {len(test_df)}")
    logger.info(f"  Best params: {best_params}")
    logger.info("=" * 60)

    # Create postprocess pipeline
    config = _create_postprocess_config(best_params)
    pipeline = PostprocessPipeline(config)

    y_true = []
    y_pred = []
    results_rows = []

    for idx, row in test_df.iterrows():
        sample_id = row.get("grape_id", idx)
        image_path = row["image_path"]
        true_label = row[label_col]

        try:
            # Get probability map
            prob_map = prob_map_fn(image_path)

            # Apply postprocessing
            mask, stats = pipeline.run(prob_map)

            # Determine prediction
            pred_label = pos_label if stats["total_positive_pixels"] > 0 else (1 - pos_label if isinstance(pos_label, int) else 0)

            y_true.append(true_label)
            y_pred.append(pred_label)

            results_rows.append({
                "sample_id": sample_id,
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": true_label == pred_label,
                "crack_ratio": stats["crack_ratio"],
                "num_blobs": stats["num_blobs_after"],
                "positive_pixels": stats["total_positive_pixels"],
            })

        except Exception as e:
            logger.error(f"Failed to process test sample {sample_id}: {e}")
            continue

    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    all_metrics = compute_all_metrics(y_true, y_pred, pos_label=pos_label)

    metrics_dict = {
        "accuracy": all_metrics.accuracy,
        "precision": all_metrics.precision,
        "recall": all_metrics.recall,
        "f1": all_metrics.f1,
        "f2": all_metrics.f2,
        "support": all_metrics.support,
        "n_samples": len(y_true),
        "n_correct": int(np.sum(y_true == y_pred)),
    }

    per_sample_df = pd.DataFrame(results_rows)

    # Log results
    logger.info("TEST RESULTS:")
    logger.info(f"  Accuracy: {metrics_dict['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics_dict['precision']:.4f}")
    logger.info(f"  Recall: {metrics_dict['recall']:.4f}")
    logger.info(f"  F1: {metrics_dict['f1']:.4f}")
    logger.info(f"  F2: {metrics_dict['f2']:.4f}")

    # Save test results
    save_per_sample_results(per_sample_df, output_dir, "test_results.csv")
    save_json(metrics_dict, str(output_path / "test_metrics.json"))

    # Create comprehensive summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "train_size": len(train_df) if train_df is not None else "N/A",
            "val_size": len(val_df) if val_df is not None else "N/A",
            "test_size": len(test_df),
        },
        "config": {
            "model_name": model_name,
            "seed": seed,
            "metric": metric,
        },
        "best_params": best_params,
        "test_metrics": metrics_dict,
    }

    # Add label distributions if available
    if train_df is not None and label_col in train_df.columns:
        summary["dataset"]["train_distribution"] = train_df[label_col].value_counts().to_dict()
    if val_df is not None and label_col in val_df.columns:
        summary["dataset"]["val_distribution"] = val_df[label_col].value_counts().to_dict()
    if label_col in test_df.columns:
        summary["dataset"]["test_distribution"] = test_df[label_col].value_counts().to_dict()

    save_summary(summary, output_dir)

    logger.info(f"Results saved to: {output_dir}")

    return metrics_dict, per_sample_df


def run_full_tuning_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prob_map_fn: Callable[[str], np.ndarray],
    n_trials: int = 50,
    seed: int = 42,
    output_dir: str = "./tuning_results",
    metric: str = "f2",
    label_col: str = "label",
    pos_label: Any = 1,
    model_name: str = "unknown_model",
    search_space: Optional[TuningSearchSpace] = None,
) -> Dict[str, Any]:
    """
    Run complete tuning pipeline: optimize on val, evaluate on test.

    This is a convenience function that:
    1. Runs Optuna tuning using train/val data
    2. Evaluates best params ONCE on test data
    3. Saves all artifacts

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (for tuning objective)
        test_df: Test DataFrame (for final evaluation ONLY)
        prob_map_fn: Function(image_path) -> prob_map
        n_trials: Number of Optuna trials
        seed: Random seed
        output_dir: Output directory
        metric: Optimization metric
        label_col: Label column name
        pos_label: Positive class label
        model_name: Model name
        search_space: Optional custom search space

    Returns:
        Dictionary with:
            - best_params: Best hyperparameters
            - trials_df: All trials
            - test_metrics: Final test metrics
            - test_results_df: Per-sample test results
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_output_dir = Path(output_dir) / f"tuning_{model_name}_{timestamp}"
    full_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FULL TUNING PIPELINE")
    logger.info(f"Output directory: {full_output_dir}")
    logger.info("=" * 60)

    # Step 1: Run Optuna tuning (uses val_df only for scoring)
    best_params, trials_df = run_optuna(
        train_df=train_df,
        val_df=val_df,
        prob_map_fn=prob_map_fn,
        n_trials=n_trials,
        seed=seed,
        output_dir=str(full_output_dir),
        metric=metric,
        label_col=label_col,
        pos_label=pos_label,
        search_space=search_space,
        model_name=model_name,
    )

    # Step 2: Final evaluation on test set (ONCE)
    test_metrics, test_results_df = evaluate_final(
        test_df=test_df,
        prob_map_fn=prob_map_fn,
        best_params=best_params,
        output_dir=str(full_output_dir),
        label_col=label_col,
        pos_label=pos_label,
        model_name=model_name,
        seed=seed,
        metric=metric,
        train_df=train_df,
        val_df=val_df,
    )

    return {
        "best_params": best_params,
        "trials_df": trials_df,
        "test_metrics": test_metrics,
        "test_results_df": test_results_df,
        "output_dir": str(full_output_dir),
    }


# ============================================================================
# Legacy compatibility (keep old class for backward compatibility)
# ============================================================================

class OptunaRunner:
    """
    DEPRECATED: Legacy runner class.

    Use run_optuna() and evaluate_final() functions instead.
    """

    def __init__(self,
                 objective_fn: Callable,
                 n_trials: int = 100,
                 direction: str = "maximize"):
        """Initialize Optuna runner."""
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.direction = direction
        self.study = None

        import warnings
        warnings.warn(
            "OptunaRunner class is deprecated. "
            "Use run_optuna() and evaluate_final() functions instead.",
            DeprecationWarning
        )

    def run(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        try:
            import optuna

            def objective(trial):
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                return self.objective_fn(params)

            self.study = optuna.create_study(direction=self.direction)
            self.study.optimize(objective, n_trials=self.n_trials)
            return self.study.best_params

        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on tuning functions."""
    print("Running optuna_runner sanity checks...")

    # Test 1: TuningSearchSpace
    space = TuningSearchSpace()
    space_dict = space.to_dict()

    assert "prob_threshold" in space_dict
    assert "morph_close_size" in space_dict
    print("  ✓ TuningSearchSpace works")

    # Test 2: _create_postprocess_config
    params = {
        "prob_threshold": 0.5,
        "morph_close_size": 5,
        "min_blob_area": 100,
        "exclude_border": True,
        "border_margin_px": 10,
    }

    config = _create_postprocess_config(params)

    assert config.prob_threshold == 0.5
    assert config.morph_close_size == 5
    assert config.min_blob_area == 100
    assert config.exclude_border == True
    print("  ✓ _create_postprocess_config works")

    # Test 3: Mock evaluation (without actual images)
    # Just verify the function signatures are correct
    import inspect

    sig_optuna = inspect.signature(run_optuna)
    assert "train_df" in sig_optuna.parameters
    assert "val_df" in sig_optuna.parameters
    assert "n_trials" in sig_optuna.parameters
    print("  ✓ run_optuna has correct signature")

    sig_eval = inspect.signature(evaluate_final)
    assert "test_df" in sig_eval.parameters
    assert "best_params" in sig_eval.parameters
    print("  ✓ evaluate_final has correct signature")

    print("\n✅ All optuna_runner sanity checks passed!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()

