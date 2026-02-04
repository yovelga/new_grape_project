"""Test settings loading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings import settings

print("=" * 60)
print("Settings Loaded Successfully")
print("=" * 60)
print(f"Models Dir: {settings.models_dir}")
print(f"Results Dir: {settings.results_dir}")
print(f"Device: {settings.device}")
print(f"WL Min: {settings.wl_min}")
print(f"WL Max: {settings.wl_max}")
print(f"Apply SNV: {settings.apply_snv}")
print(f"Default Prob Threshold: {settings.default_prob_threshold}")
print(f"Inference Batch Size: {settings.inference_batch_size}")
print(f"Grid Cell Size: {settings.grid_cell_size}")
print(f"Grid Crack Ratio: {settings.grid_crack_ratio}")
print(f"Default Band Index: {settings.default_band_index}")
print(f"Overlay Alpha: {settings.overlay_alpha}")
print(f"Log Level: {settings.log_level}")
print("=" * 60)

# Test validation
errors = settings.validate()
if errors:
    print("\nValidation Errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("\n✓ All settings valid!")

print("=" * 60)
