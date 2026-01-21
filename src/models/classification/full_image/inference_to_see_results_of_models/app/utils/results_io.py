"""
Results I/O utilities for saving and loading experiment artifacts.

Provides consistent file formats for:
- JSON configuration and parameters
- CSV results tables
- Text summaries
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyJSONEncoder)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        path: Output file path
        index: Whether to include row index
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=index)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.

    Args:
        path: Input file path

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path)


def save_text(content: str, path: str) -> None:
    """
    Save text content to file.

    Args:
        content: Text content
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def create_experiment_dir(
    base_dir: str,
    prefix: str = "experiment"
) -> Path:
    """
    Create timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments
        prefix: Prefix for directory name

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_best_params(
    params: Dict[str, Any],
    output_dir: str,
    filename: str = "best_params.json"
) -> str:
    """
    Save best hyperparameters to JSON.

    Args:
        params: Best parameters dictionary
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    path = Path(output_dir) / filename
    save_json(params, str(path))
    return str(path)


def save_trials_csv(
    trials_df: pd.DataFrame,
    output_dir: str,
    filename: str = "trials.csv"
) -> str:
    """
    Save Optuna trials to CSV.

    Args:
        trials_df: DataFrame with trial results
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    path = Path(output_dir) / filename
    save_csv(trials_df, str(path))
    return str(path)


def save_per_sample_results(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = "per_sample_results.csv"
) -> str:
    """
    Save per-sample prediction results to CSV.

    Args:
        results_df: DataFrame with per-sample results
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    path = Path(output_dir) / filename
    save_csv(results_df, str(path))
    return str(path)


def save_summary(
    summary: Dict[str, Any],
    output_dir: str,
    filename: str = "summary.txt"
) -> str:
    """
    Save experiment summary as formatted text.

    Args:
        summary: Summary dictionary
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    path = Path(output_dir) / filename

    lines = [
        "=" * 60,
        "EXPERIMENT SUMMARY",
        "=" * 60,
        f"Timestamp: {summary.get('timestamp', 'N/A')}",
        "",
    ]

    # Dataset info
    if 'dataset' in summary:
        lines.append("DATASET:")
        ds = summary['dataset']
        lines.append(f"  Train samples: {ds.get('train_size', 'N/A')}")
        lines.append(f"  Val samples: {ds.get('val_size', 'N/A')}")
        lines.append(f"  Test samples: {ds.get('test_size', 'N/A')}")
        if 'train_distribution' in ds:
            lines.append(f"  Train distribution: {ds['train_distribution']}")
        if 'val_distribution' in ds:
            lines.append(f"  Val distribution: {ds['val_distribution']}")
        if 'test_distribution' in ds:
            lines.append(f"  Test distribution: {ds['test_distribution']}")
        lines.append("")

    # Configuration
    if 'config' in summary:
        lines.append("CONFIGURATION:")
        cfg = summary['config']
        lines.append(f"  Model: {cfg.get('model_name', 'N/A')}")
        lines.append(f"  Seed: {cfg.get('seed', 'N/A')}")
        lines.append(f"  Metric: {cfg.get('metric', 'N/A')}")
        lines.append(f"  N trials: {cfg.get('n_trials', 'N/A')}")
        lines.append("")

    # Best parameters
    if 'best_params' in summary:
        lines.append("BEST PARAMETERS:")
        for key, value in summary['best_params'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Validation results
    if 'val_metrics' in summary:
        lines.append("VALIDATION RESULTS (best trial):")
        for key, value in summary['val_metrics'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    # Test results
    if 'test_metrics' in summary:
        lines.append("TEST RESULTS (final evaluation):")
        for key, value in summary['test_metrics'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    lines.append("=" * 60)

    content = "\n".join(lines)
    save_text(content, str(path))

    # Also save as JSON for programmatic access
    json_path = path.with_suffix('.json')
    save_json(summary, str(json_path))

    return str(path)


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """
    Format metrics dictionary as text table.

    Args:
        metrics: Dictionary of metric names to values
        title: Table title

    Returns:
        Formatted table string
    """
    lines = [title, "-" * len(title)]

    max_key_len = max(len(k) for k in metrics.keys())

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:<{max_key_len}}: {value:.4f}")
        else:
            lines.append(f"{key:<{max_key_len}}: {value}")

    return "\n".join(lines)


# ============================================================================
# Sanity Checks
# ============================================================================

def _run_sanity_checks() -> bool:
    """Run sanity checks on results I/O utilities."""
    import tempfile
    import shutil

    print("Running results_io sanity checks...")

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: JSON save/load with numpy types
        data = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
        }

        json_path = f"{temp_dir}/test.json"
        save_json(data, json_path)
        loaded = load_json(json_path)

        assert loaded["int_val"] == 42
        assert abs(loaded["float_val"] - 3.14) < 0.01
        assert loaded["array"] == [1, 2, 3]
        print("  ✓ JSON save/load with numpy types works")

        # Test 2: CSV save/load
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [0.1, 0.2, 0.3],
            "label": ["a", "b", "c"]
        })

        csv_path = f"{temp_dir}/test.csv"
        save_csv(df, csv_path)
        loaded_df = load_csv(csv_path)

        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["id", "value", "label"]
        print("  ✓ CSV save/load works")

        # Test 3: Text save
        save_text("Hello, World!", f"{temp_dir}/test.txt")
        with open(f"{temp_dir}/test.txt") as f:
            assert f.read() == "Hello, World!"
        print("  ✓ Text save works")

        # Test 4: Experiment directory creation
        exp_dir = create_experiment_dir(temp_dir, "test_exp")
        assert exp_dir.exists()
        assert "test_exp_" in str(exp_dir)
        print("  ✓ Experiment directory creation works")

        # Test 5: Summary save
        summary = {
            "timestamp": "2026-01-21",
            "dataset": {"train_size": 100, "val_size": 30},
            "config": {"model_name": "test_model", "seed": 42},
            "best_params": {"threshold": 0.5},
            "test_metrics": {"accuracy": 0.95, "f1": 0.92},
        }

        summary_path = save_summary(summary, temp_dir)
        assert Path(summary_path).exists()
        assert Path(summary_path).with_suffix('.json').exists()
        print("  ✓ Summary save works")

        print("\n✅ All results_io sanity checks passed!")
        return True

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    _run_sanity_checks()
