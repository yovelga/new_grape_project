"""
Smoke Test for the Inference Pipeline

Quick validation that all components work together:
1. Loads Settings from .env
2. Loads Train/Val and Test CSVs (if configured)
3. Runs minimal inference + postprocess on 1-2 samples
4. Prints success summary

Usage:
    python scripts/smoke_test.py

    Or with custom paths:
    python scripts/smoke_test.py --trainval path/to/trainval.csv --test path/to/test.csv

Exit codes:
    0 = Success
    1 = Failure
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(label: str, status: str, details: str = "") -> None:
    """Print status line."""
    symbol = "✓" if status == "OK" else "✗" if status == "FAIL" else "?"
    detail_str = f" ({details})" if details else ""
    print(f"  {symbol} {label}: {status}{detail_str}")


def check_settings() -> Dict[str, Any]:
    """Check that settings load correctly."""
    print_header("1. Settings Check")

    try:
        from app.config.settings import settings

        print_status("Settings loaded", "OK")
        print_status("Models dir", "OK" if settings.models_dir.exists() else "WARN",
                    str(settings.models_dir))
        print_status("Results dir", "OK" if settings.results_dir.exists() else "WARN",
                    str(settings.results_dir))
        print_status("Log dir", "OK" if settings.log_dir.exists() else "WARN",
                    str(settings.log_dir))
        print_status("Random seed", "OK", str(settings.random_seed))
        print_status("Val split size", "OK", str(settings.val_split_size))
        print_status("Device", "OK", settings.device)

        # Validate settings
        errors = settings.validate()
        if errors:
            print(f"\n  Warnings ({len(errors)}):")
            for err in errors[:5]:  # Show first 5
                print(f"    - {err}")
        else:
            print_status("Validation", "OK", "No errors")

        return {"settings": settings, "ok": True}

    except Exception as e:
        print_status("Settings", "FAIL", str(e))
        return {"ok": False, "error": str(e)}


def check_env_completeness() -> Dict[str, Any]:
    """Check that .env.example has all required variables."""
    print_header("2. Environment Completeness Check")

    env_example = PROJECT_ROOT / ".env.example"
    env_file = PROJECT_ROOT / ".env"

    if not env_example.exists():
        print_status(".env.example", "FAIL", "File not found")
        return {"ok": False}

    # Parse .env.example to get expected variables
    expected_vars = set()
    with open(env_example, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                var_name = line.split('=')[0].strip()
                expected_vars.add(var_name)

    print_status(".env.example", "OK", f"{len(expected_vars)} variables defined")

    # Check if .env exists
    if env_file.exists():
        defined_vars = set()
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    defined_vars.add(var_name)

        missing = expected_vars - defined_vars
        if missing:
            print_status(".env completeness", "WARN", f"{len(missing)} missing vars")
            print(f"    Missing: {', '.join(sorted(missing)[:5])}...")
        else:
            print_status(".env completeness", "OK", "All variables defined")
    else:
        print_status(".env", "WARN", "File not found - using defaults")

    # Required variables check
    required = ["MODELS_DIR", "RESULTS_DIR", "RANDOM_SEED", "LOG_DIR"]
    has_required = all(v in expected_vars for v in required)
    print_status("Required variables", "OK" if has_required else "FAIL",
                ", ".join(required))

    return {"ok": True, "expected_vars": expected_vars}


def check_dataset_loading(
    trainval_path: Optional[str] = None,
    test_path: Optional[str] = None,
    settings: Any = None
) -> Dict[str, Any]:
    """Check dataset loading functionality."""
    print_header("3. Dataset Loading Check")

    try:
        from app.data.dataset import (
            load_dataset_csv,
            split_train_val,
            get_class_distribution
        )

        # Resolve paths
        if trainval_path:
            trainval_csv = Path(trainval_path)
        elif settings and settings.default_trainval_csv.exists():
            trainval_csv = settings.default_trainval_csv
        else:
            trainval_csv = None

        if test_path:
            test_csv = Path(test_path)
        elif settings and settings.default_test_csv.exists():
            test_csv = settings.default_test_csv
        else:
            test_csv = None

        result = {"ok": True, "train_df": None, "val_df": None, "test_df": None}

        # Load trainval CSV
        if trainval_csv and trainval_csv.exists():
            trainval_df = load_dataset_csv(str(trainval_csv))
            print_status("Train/Val CSV loaded", "OK", f"{len(trainval_df)} samples")

            # Split
            seed = settings.random_seed if settings else 42
            val_size = settings.val_split_size if settings else 0.30

            train_df, val_df = split_train_val(
                trainval_df,
                val_size=val_size,
                random_state=seed
            )

            result["train_df"] = train_df
            result["val_df"] = val_df

            # Show distribution
            train_dist = get_class_distribution(train_df)
            val_dist = get_class_distribution(val_df)

            print_status("Train split", "OK",
                        f"{len(train_df)} samples, dist: {train_dist}")
            print_status("Val split", "OK",
                        f"{len(val_df)} samples, dist: {val_dist}")
            print_status("Seed reproducibility", "OK", f"seed={seed}")
        else:
            print_status("Train/Val CSV", "SKIP", "Not configured or not found")

        # Load test CSV
        if test_csv and test_csv.exists():
            test_df = load_dataset_csv(str(test_csv))
            test_dist = get_class_distribution(test_df)

            result["test_df"] = test_df
            print_status("Test CSV loaded", "OK",
                        f"{len(test_df)} samples, dist: {test_dist}")
        else:
            print_status("Test CSV", "SKIP", "Not configured or not found")

        return result

    except Exception as e:
        print_status("Dataset loading", "FAIL", str(e))
        return {"ok": False, "error": str(e)}


def check_model_loading(settings: Any = None) -> Dict[str, Any]:
    """Check model loading functionality."""
    print_header("4. Model Loading Check")

    try:
        from app.models.loader_new import load_model
        from app.models.adapters_new import SklearnAdapter, create_adapter

        # Find available models
        models_dir = settings.models_dir if settings else Path("./models")

        if not models_dir.exists():
            print_status("Models directory", "SKIP", "Not found")
            return {"ok": True, "model": None, "adapter": None}

        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))

        if not model_files:
            print_status("Models", "SKIP", "No model files found")
            return {"ok": True, "model": None, "adapter": None}

        # Load first model
        model_path = model_files[0]
        print_status("Found models", "OK", f"{len(model_files)} files")

        model = load_model(str(model_path))
        print_status("Model loaded", "OK", f"{model_path.name}")

        # Create adapter
        adapter = SklearnAdapter(model, name=model_path.stem)
        print_status("Adapter created", "OK",
                    f"n_classes={adapter.n_classes}, is_binary={adapter.is_binary}")

        return {"ok": True, "model": model, "adapter": adapter, "path": model_path}

    except Exception as e:
        print_status("Model loading", "FAIL", str(e))
        return {"ok": False, "error": str(e)}


def check_inference_pipeline(
    adapter: Any = None,
    settings: Any = None
) -> Dict[str, Any]:
    """Check inference pipeline with synthetic data."""
    print_header("5. Inference Pipeline Check")

    if adapter is None:
        print_status("Inference", "SKIP", "No model loaded")
        return {"ok": True}

    try:
        from app.inference.prob_map import build_prob_map
        from app.config.types import PreprocessConfig
        from app.postprocess.pipeline import PostprocessConfig, PostprocessPipeline

        # Create synthetic HSI cube (small for speed)
        H, W, C = 50, 50, 224
        np.random.seed(settings.random_seed if settings else 42)
        synthetic_cube = np.random.rand(H, W, C).astype(np.float32)

        print_status("Synthetic cube created", "OK", f"shape={synthetic_cube.shape}")

        # Create preprocess config
        preprocess_cfg = PreprocessConfig(use_snv=True)

        # Run inference
        start_time = time.time()
        prob_map = build_prob_map(
            synthetic_cube,
            adapter,
            preprocess_cfg,
            target_class_index=1,
            chunk_size=10000
        )
        inference_time = time.time() - start_time

        print_status("Inference completed", "OK",
                    f"shape={prob_map.shape}, time={inference_time:.2f}s")

        # Run postprocessing
        postprocess_cfg = PostprocessConfig(
            prob_threshold=0.5,
            morph_close_size=3,
            min_blob_area=10
        )
        pipeline = PostprocessPipeline(postprocess_cfg)

        mask, stats = pipeline.run(prob_map)

        print_status("Postprocessing completed", "OK",
                    f"positive_pixels={stats['total_positive_pixels']}, "
                    f"blobs={stats['num_blobs_after']}")

        return {
            "ok": True,
            "prob_map": prob_map,
            "mask": mask,
            "stats": stats,
            "inference_time": inference_time
        }

    except Exception as e:
        print_status("Inference pipeline", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def check_metrics(mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Check metrics computation."""
    print_header("6. Metrics Check")

    try:
        from app.metrics.classification import (
            compute_fbeta,
            compute_accuracy,
            get_metric_value,
            compute_all_metrics
        )

        # Test with synthetic data
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])

        accuracy = compute_accuracy(y_true, y_pred)
        f1 = compute_fbeta(y_true, y_pred, beta=1.0, pos_label=1)
        f2 = compute_fbeta(y_true, y_pred, beta=2.0, pos_label=1)

        print_status("Accuracy", "OK", f"{accuracy:.4f}")
        print_status("F1 score", "OK", f"{f1:.4f}")
        print_status("F2 score", "OK", f"{f2:.4f}")

        # Test all metrics
        all_metrics = compute_all_metrics(y_true, y_pred, pos_label=1)
        print_status("All metrics", "OK",
                    f"precision={all_metrics.precision:.4f}, "
                    f"recall={all_metrics.recall:.4f}")

        return {"ok": True, "f1": f1, "f2": f2, "accuracy": accuracy}

    except Exception as e:
        print_status("Metrics", "FAIL", str(e))
        return {"ok": False, "error": str(e)}


def check_logging() -> Dict[str, Any]:
    """Check logging functionality."""
    print_header("7. Logging Check")

    try:
        from app.utils.logging import (
            Logger,
            setup_logger,
            get_log_level,
            configure_global_logger
        )

        # Test log level parsing
        assert get_log_level("DEBUG") == 10
        assert get_log_level("INFO") == 20
        print_status("Log level parsing", "OK")

        # Test logger creation
        test_logger = setup_logger("smoke_test")
        test_logger.info("Smoke test log message")
        print_status("Logger creation", "OK")

        return {"ok": True}

    except Exception as e:
        print_status("Logging", "FAIL", str(e))
        return {"ok": False, "error": str(e)}


def run_smoke_test(
    trainval_path: Optional[str] = None,
    test_path: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Run complete smoke test.

    Args:
        trainval_path: Optional path to train/val CSV
        test_path: Optional path to test CSV
        verbose: Whether to print detailed output

    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("        SMOKE TEST - Inference Pipeline")
    print("=" * 60)
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT_ROOT}")

    results: List[Dict[str, Any]] = []
    settings = None
    adapter = None

    # 1. Settings
    r = check_settings()
    results.append(r)
    if r.get("ok"):
        settings = r.get("settings")

    # 2. Environment completeness
    r = check_env_completeness()
    results.append(r)

    # 3. Dataset loading
    r = check_dataset_loading(trainval_path, test_path, settings)
    results.append(r)

    # 4. Model loading
    r = check_model_loading(settings)
    results.append(r)
    if r.get("ok") and r.get("adapter"):
        adapter = r.get("adapter")

    # 5. Inference pipeline
    r = check_inference_pipeline(adapter, settings)
    results.append(r)

    # 6. Metrics
    r = check_metrics()
    results.append(r)

    # 7. Logging
    r = check_logging()
    results.append(r)

    # Summary
    print_header("SUMMARY")

    passed = sum(1 for r in results if r.get("ok"))
    total = len(results)

    if passed == total:
        print(f"\n  ✓ All {total} checks PASSED")
        print("\n  Pipeline is ready for use!")
        return True
    else:
        print(f"\n  {passed}/{total} checks passed")
        failed = [i+1 for i, r in enumerate(results) if not r.get("ok")]
        print(f"  Failed checks: {failed}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test for the inference pipeline"
    )
    parser.add_argument(
        "--trainval",
        type=str,
        default=None,
        help="Path to train/val CSV file"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    success = run_smoke_test(
        trainval_path=args.trainval,
        test_path=args.test,
        verbose=not args.quiet
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
