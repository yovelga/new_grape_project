"""Inspect the LDA Best F1 Weighted model structure"""
import joblib
import os

model_path = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\full_image\infernce_with_new_model_with_sam2_reduce_bands\lda_model_best_f1_weighted.joblib'

print(f"Loading model from: {model_path}\n")

model_data = joblib.load(model_path)
print(f"Model type: {type(model_data)}")

if isinstance(model_data, dict):
    print(f"\nKeys in model data:")
    for key in model_data.keys():
        print(f"  - {key}: {type(model_data[key])}")

    # Check for the actual model
    if 'model' in model_data:
        actual_model = model_data['model']
        print(f"\n✓ Found 'model' key!")
        print(f"  Model type: {type(actual_model)}")
        print(f"  Has classes_: {hasattr(actual_model, 'classes_')}")
        if hasattr(actual_model, 'classes_'):
            print(f"  Classes: {actual_model.classes_}")
        if hasattr(actual_model, 'n_features_in_'):
            print(f"  Expected input features: {actual_model.n_features_in_}")

    # Check for scaler
    if 'scaler' in model_data:
        scaler = model_data['scaler']
        print(f"\n✓ Found 'scaler' key!")
        print(f"  Scaler type: {type(scaler)}")
        print(f"  Has transform: {hasattr(scaler, 'transform')}")

    # Check for selected features
    if 'selected_features' in model_data:
        print(f"\n✓ Found 'selected_features' key!")
        print(f"  Type: {type(model_data['selected_features'])}")
        if isinstance(model_data['selected_features'], (list, tuple)):
            print(f"  Number of features: {len(model_data['selected_features'])}")
            print(f"  First 10 features: {model_data['selected_features'][:10]}")
        else:
            print(f"  Content: {model_data['selected_features']}")

    # Check for feature names
    if 'feature_names' in model_data:
        print(f"\n✓ Found 'feature_names' key!")
        print(f"  Number of feature names: {len(model_data['feature_names'])}")

