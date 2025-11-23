import joblib
import sys
import os

# Add shim for LDAModel
class LDAModel:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)

sys.modules['__main__'].LDAModel = LDAModel

# Load models
m1_path = r'C:\Users\yovel\Desktop\Grape_Project\src\models\classification\pixel_level\simple_classification_leave_one_out\comare_all_models\models\LDA_Balanced.pkl'
m2_path = r'src/models/classification/full_image/Train\LDA\lda_model_2_class.joblib'

print("Loading LDA_Balanced.pkl...")
m1 = joblib.load(m1_path)
print(f"  Type: {type(m1)}")
print(f"  Classes: {m1.classes_ if hasattr(m1, 'classes_') else 'No classes attr'}")

# Check if it's wrapped
if hasattr(m1, 'model'):
    print(f"  Inner model type: {type(m1.model)}")
    print(f"  Inner classes: {m1.model.classes_ if hasattr(m1.model, 'classes_') else 'No classes'}")

print("\nLoading lda_model_2_class.joblib...")
m2 = joblib.load(m2_path)
print(f"  Type: {type(m2)}")
print(f"  Classes: {m2.classes_ if hasattr(m2, 'classes_') else 'No classes attr'}")

# Test predictions
import numpy as np
test_data = np.random.rand(1, 204)

print("\nTesting predictions with sample data...")
try:
    if hasattr(m1, 'predict_proba'):
        probs1 = m1.predict_proba(test_data)
    elif hasattr(m1, 'model') and hasattr(m1.model, 'predict_proba'):
        probs1 = m1.model.predict_proba(test_data)
    else:
        probs1 = None
    print(f"LDA_Balanced probs shape: {probs1.shape if probs1 is not None else 'N/A'}")
    print(f"LDA_Balanced probs: {probs1}")
except Exception as e:
    print(f"LDA_Balanced prediction error: {e}")

try:
    probs2 = m2.predict_proba(test_data)
    print(f"lda_model_2_class probs shape: {probs2.shape}")
    print(f"lda_model_2_class probs: {probs2}")
except Exception as e:
    print(f"lda_model_2_class prediction error: {e}")

