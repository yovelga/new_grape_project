"""
Investigation: Does sklearn's LDA support sample_weight?
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import inspect

print("="*80)
print("Checking LinearDiscriminantAnalysis.fit() signature")
print("="*80)

lda = LinearDiscriminantAnalysis()
sig = inspect.signature(lda.fit)
print(f"\nSignature: {sig}")
print(f"\nParameters:")
for param_name, param in sig.parameters.items():
    print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}")

print("\n" + "="*80)
print("FINDING: LDA does NOT support sample_weight!")
print("="*80)
print("\nAlternatives:")
print("1. Manual weighting by replicating samples (like oversampling)")
print("2. Use weighted covariance matrices (advanced)")
print("3. Use a different classifier that supports sample_weight")
print("4. Stick with oversampling approach")
print("\nRECOMMENDATION:")
print("Since LDA doesn't support sample_weight, the user's request")
print("to 'use sample weights instead of oversampling' cannot be")
print("directly implemented with LDA.")
print("\nOPTIONS:")
print("A) Keep oversampling (original approach)")
print("B) Switch to a classifier that supports sample_weight")
print("   (e.g., LogisticRegression, RandomForest, GradientBoosting)")
print("C) Implement custom weighted LDA (complex)")

