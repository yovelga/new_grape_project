"""Test loading the LDA Best F1 Weighted model"""
import joblib
import os

model_path = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_reduce_bands\lda_model_best_f1_weighted.joblib'

print(f"Testing model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

try:
    model = joblib.load(model_path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(model)}")
    print(f"  Has classes_: {hasattr(model, 'classes_')}")
    if hasattr(model, 'classes_'):
        print(f"  Classes: {model.classes_}")
    print(f"  Has predict: {hasattr(model, 'predict')}")
    print(f"  Has predict_proba: {hasattr(model, 'predict_proba')}")

    # Check model attributes
    if hasattr(model, 'n_features_in_'):
        print(f"  Expected input features: {model.n_features_in_}")

    print("\n✓ Model is ready to use!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

