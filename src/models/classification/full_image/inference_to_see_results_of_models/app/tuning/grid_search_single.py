"""
Grid Search for Single Sample Postprocessing

Explores postprocessing parameter combinations on a single probability map
to find optimal settings for blob detection and filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from itertools import product
from pathlib import Path
from datetime import datetime

from ..postprocess import PostprocessPipeline, PostprocessConfig


def run_grid_on_prob_map(
    prob_map: np.ndarray,
    param_grid: Dict[str, List[Any]],
    metric: str = "crack_ratio"
) -> pd.DataFrame:
    """
    Run grid search over postprocessing parameters on a single probability map.

    Evaluates all combinations of parameters and returns a sorted DataFrame
    with results.

    Args:
        prob_map: Probability map (H, W) with values in [0, 1]
        param_grid: Dictionary mapping parameter names to lists of values to try.
            Supported parameters:
            - prob_threshold: List[float]
            - morph_close_size: List[int]
            - min_blob_area: List[int]
            - exclude_border: List[bool]
            - border_margin_px: List[int]
            - circularity_min: List[Optional[float]]
            - solidity_min: List[Optional[float]]
        metric: Metric to sort by. Options:
            - "crack_ratio": Total positive pixels / total pixels
            - "num_blobs": Number of blobs after filtering
            - "total_pixels": Total positive pixels

    Returns:
        DataFrame with columns for each parameter plus stats columns:
        - All parameter columns from param_grid
        - num_blobs_before: Blobs before filtering
        - num_blobs_after: Blobs after filtering
        - total_positive_pixels: Sum of True pixels
        - crack_ratio: Ratio of positive pixels
        - Sorted by specified metric (descending for crack_ratio, ascending for num_blobs)

    Example:
        >>> param_grid = {
        ...     'prob_threshold': [0.3, 0.5, 0.7],
        ...     'morph_close_size': [0, 3, 5],
        ...     'min_blob_area': [0, 50, 100]
        ... }
        >>> results = run_grid_on_prob_map(prob_map, param_grid)
        >>> print(results.head())
    """
    if prob_map.ndim != 2:
        raise ValueError(f"prob_map must be 2D, got {prob_map.ndim}D")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values))

    results = []

    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            # Ensure morph_close_size is odd or 0
            morph_size = params.get('morph_close_size', 0)
            if morph_size > 0 and morph_size % 2 == 0:
                morph_size += 1
                params['morph_close_size'] = morph_size

            # Build config
            config = PostprocessConfig(
                prob_threshold=params.get('prob_threshold', 0.5),
                morph_close_size=params.get('morph_close_size', 0),
                min_blob_area=params.get('min_blob_area', 0),
                exclude_border=params.get('exclude_border', False),
                border_margin_px=params.get('border_margin_px', 0),
                circularity_min=params.get('circularity_min', None),
                solidity_min=params.get('solidity_min', None),
            )

            # Run postprocessing
            pipeline = PostprocessPipeline(config)
            _, stats = pipeline.run(prob_map)

            # Combine params and stats
            result = {**params, **stats}

            # Remove blob lists (too large for table)
            result.pop('accepted_blobs', None)
            result.pop('rejected_blobs', None)

            results.append(result)

        except Exception as e:
            # Skip invalid parameter combinations
            print(f"Skipping invalid combo {params}: {e}")
            continue

    if not results:
        raise ValueError("No valid parameter combinations found")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by metric
    if metric == "crack_ratio":
        df = df.sort_values("crack_ratio", ascending=False)
    elif metric == "num_blobs":
        df = df.sort_values("num_blobs_after", ascending=True)
    elif metric == "total_pixels":
        df = df.sort_values("total_positive_pixels", ascending=False)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def get_default_param_grid() -> Dict[str, List[Any]]:
    """
    Get default parameter grid for quick exploration.

    Returns a sensible set of parameter values that can be run
    immediately without configuration.

    Returns:
        Dictionary with default parameter ranges

    Example:
        >>> grid = get_default_param_grid()
        >>> results = run_grid_on_prob_map(prob_map, grid)
    """
    return {
        'prob_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
        'morph_close_size': [0, 3, 5, 7],
        'min_blob_area': [0, 25, 50, 100, 200],
        'exclude_border': [False, True],
    }


def get_comprehensive_param_grid() -> Dict[str, List[Any]]:
    """
    Get comprehensive parameter grid for thorough exploration.

    WARNING: This will generate many combinations and may take time.

    Returns:
        Dictionary with comprehensive parameter ranges
    """
    return {
        'prob_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'morph_close_size': [0, 3, 5, 7, 9],
        'min_blob_area': [0, 10, 25, 50, 100, 150, 200, 300],
        'exclude_border': [False, True],
        'border_margin_px': [0, 5, 10],
        'circularity_min': [None, 0.3, 0.5, 0.7],
        'solidity_min': [None, 0.5, 0.7, 0.9],
    }


def save_grid_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save grid search results to CSV with metadata.

    Args:
        results_df: DataFrame from run_grid_on_prob_map
        output_dir: Output directory (will be created if needed)
        sample_id: Identifier for the sample (e.g., grape_id or folder name)
        metadata: Optional metadata dict to save as JSON

    Returns:
        Path to saved CSV file

    Example:
        >>> results = run_grid_on_prob_map(prob_map, param_grid)
        >>> save_path = save_grid_results(
        ...     results,
        ...     Path("results/grids"),
        ...     sample_id="1_01_16.08.24"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"grid_{sample_id}_{timestamp}.csv"

    # Save CSV
    results_df.to_csv(csv_path, index=False)

    # Save metadata if provided
    if metadata:
        import json
        meta_path = output_dir / f"grid_{sample_id}_{timestamp}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return csv_path


def analyze_grid_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze grid search results to find trends and optimal parameters.

    Args:
        results_df: DataFrame from run_grid_on_prob_map

    Returns:
        Dictionary with analysis insights:
        - best_params: Parameters of best row
        - top_5: Top 5 parameter combinations
        - param_importance: Which parameters have most impact

    Example:
        >>> analysis = analyze_grid_results(results_df)
        >>> print(f"Best threshold: {analysis['best_params']['prob_threshold']}")
    """
    if len(results_df) == 0:
        return {"error": "No results to analyze"}

    # Best parameters (first row after sorting)
    best_row = results_df.iloc[0]
    best_params = {
        col: best_row[col]
        for col in results_df.columns
        if col not in ['num_blobs_before', 'num_blobs_after',
                       'total_positive_pixels', 'crack_ratio']
    }

    # Top 5 combinations
    top_5 = results_df.head(5).to_dict('records')

    # Parameter importance (correlation with crack_ratio)
    param_cols = [col for col in results_df.columns
                  if col not in ['num_blobs_before', 'num_blobs_after',
                                 'total_positive_pixels', 'crack_ratio']]

    importance = {}
    for col in param_cols:
        # For numeric columns, compute correlation
        if results_df[col].dtype in [np.float64, np.int64]:
            corr = results_df[col].corr(results_df['crack_ratio'])
            if not np.isnan(corr):
                importance[col] = abs(corr)

    return {
        'best_params': best_params,
        'best_crack_ratio': float(best_row['crack_ratio']),
        'best_num_blobs': int(best_row['num_blobs_after']),
        'top_5': top_5,
        'param_importance': importance,
        'total_combinations': len(results_df),
    }
