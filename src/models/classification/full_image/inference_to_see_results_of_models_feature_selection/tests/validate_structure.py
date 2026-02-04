"""
Structure validation script.

Verifies that all required modules are importable and functional.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))


def validate_imports():
    """Validate all module imports."""
    print("Validating module imports...")
    errors = []
    
    try:
        from app.config.settings import settings
        print("  ✓ app.config.settings")
    except Exception as e:
        errors.append(f"app.config.settings: {e}")
    
    try:
        from app.data.dataset import InferenceDataset, RGBDataset
        print("  ✓ app.data.dataset")
    except Exception as e:
        errors.append(f"app.data.dataset: {e}")
    
    try:
        from app.io.envi import ENVIReader
        print("  ✓ app.io.envi")
    except Exception as e:
        errors.append(f"app.io.envi: {e}")
    
    try:
        from app.io.rgb import RGBReader
        print("  ✓ app.io.rgb")
    except Exception as e:
        errors.append(f"app.io.rgb: {e}")
    
    try:
        from app.preprocess.spectral import SpectralPreprocessor, snv_normalize, filter_wavelengths
        print("  ✓ app.preprocess.spectral")
    except Exception as e:
        errors.append(f"app.preprocess.spectral: {e}")
    
    try:
        from app.models.loader import ModelLoader
        print("  ✓ app.models.loader")
    except Exception as e:
        errors.append(f"app.models.loader: {e}")
    
    try:
        from app.models.adapters import ModelAdapter, CNNAdapter
        print("  ✓ app.models.adapters")
    except Exception as e:
        errors.append(f"app.models.adapters: {e}")
    
    try:
        from app.models.sklearn_models import SklearnModelWrapper, load_sklearn_model
        print("  ✓ app.models.sklearn_models")
    except Exception as e:
        errors.append(f"app.models.sklearn_models: {e}")

    try:
        from app.models.architectures import EfficientNetBinary, SimpleCNN
        print("  ✓ app.models.architectures")
    except Exception as e:
        errors.append(f"app.models.architectures: {e}")

    try:
        from app.inference.prob_map import ProbabilityMapGenerator
        print("  ✓ app.inference.prob_map")
    except Exception as e:
        errors.append(f"app.inference.prob_map: {e}")
    
    try:
        from app.inference.engine import HyperspectralInferenceEngine, GridAnalyzer
        print("  ✓ app.inference.engine")
    except Exception as e:
        errors.append(f"app.inference.engine: {e}")

    try:
        from app.postprocess.pipeline import PostprocessPipeline
        print("  ✓ app.postprocess.pipeline")
    except Exception as e:
        errors.append(f"app.postprocess.pipeline: {e}")
    
    try:
        from app.tuning.optuna_runner import OptunaRunner
        print("  ✓ app.tuning.optuna_runner")
    except Exception as e:
        errors.append(f"app.tuning.optuna_runner: {e}")
    
    try:
        from app.utils.logging import logger
        print("  ✓ app.utils.logging")
    except Exception as e:
        errors.append(f"app.utils.logging: {e}")
    
    try:
        from app.visualization.overlays import create_binary_overlay, create_grid_overlay
        print("  ✓ app.visualization.overlays")
    except Exception as e:
        errors.append(f"app.visualization.overlays: {e}")

    return errors


def validate_structure():
    """Validate folder structure."""
    print("\nValidating folder structure...")
    
    base = Path(__file__).parent
    required_files = [
        "binary_class_inference_ui.py",
        "SCOPE.md",
        "README.md",
        ".env.example",
        "app/__init__.py",
        "app/config/__init__.py",
        "app/config/settings.py",
        "app/data/__init__.py",
        "app/data/dataset.py",
        "app/io/__init__.py",
        "app/io/envi.py",
        "app/io/rgb.py",
        "app/preprocess/__init__.py",
        "app/preprocess/spectral.py",
        "app/models/__init__.py",
        "app/models/loader.py",
        "app/models/adapters.py",
        "app/models/sklearn_models.py",
        "app/models/architectures.py",
        "app/inference/__init__.py",
        "app/inference/prob_map.py",
        "app/inference/engine.py",
        "app/postprocess/__init__.py",
        "app/postprocess/pipeline.py",
        "app/tuning/__init__.py",
        "app/tuning/optuna_runner.py",
        "app/utils/__init__.py",
        "app/utils/logging.py",
        "app/visualization/__init__.py",
        "app/visualization/overlays.py",
    ]
    
    missing = []
    for file in required_files:
        path = base / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            missing.append(file)
            print(f"  ✗ {file} - MISSING")
    
    return missing


def main():
    """Run validation."""
    print("=" * 60)
    print("Structure Validation")
    print("=" * 60)
    
    # Validate structure
    missing = validate_structure()
    
    # Validate imports
    print()
    import_errors = validate_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    if missing:
        print(f"\n❌ Missing files: {len(missing)}")
        for file in missing:
            print(f"  - {file}")
    else:
        print("\n✓ All required files present")
    
    if import_errors:
        print(f"\n❌ Import errors: {len(import_errors)}")
        for error in import_errors:
            print(f"  - {error}")
    else:
        print("✓ All modules importable")
    
    if not missing and not import_errors:
        print("\n" + "🎉 Structure validation PASSED!")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env")
        print("  2. Configure MODEL_PATH and other settings")
        print("  3. Run: python binary_class_inference_ui.py")
        return 0
    else:
        print("\n❌ Validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
