import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = r"D:\Grape_Project\dataset_builder_grapes\dataset\combined_cleaned_signatures_16_12.csv"  # edit if needed
BAND_LOWER, BAND_UPPER = (
    30,
    180,
)  # keep bands indices 30..180 (inclusive) → 180 features
K_RANGE = range(2, 51)  # test k = 2..50
N_FOLDS = 5  # cross‑validation folds
SCORING = "roc_auc"  # metric: "accuracy", "roc_auc", etc.
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def load_features_labels(path: str):
    """Read CSV → X (per‑sample min‑max scaled) and y."""
    df = pd.read_csv(path)
    # Support both wavelength-based (ending with 'nm') and band-based column names
    feature_cols = [c for c in df.columns if c.endswith("nm")]
    if not feature_cols:
        # Fallback to band_ columns if wavelength columns not found
        feature_cols = [c for c in df.columns if c.startswith("band_")]
    if not feature_cols:
        raise ValueError("No spectral columns ending with 'nm' or starting with 'band_' were found.")
    # slice desired band indices
    feature_cols = feature_cols[BAND_LOWER : BAND_UPPER + 1]
    X_raw = df[feature_cols].values.astype(np.float32)
    # per‑sample min‑max scaling using np.ptp (range) – compatible with NumPy ≥2.0
    mins = X_raw.min(axis=1, keepdims=True)
    ranges = np.ptp(X_raw, axis=1, keepdims=True) + 1e-9  # avoid division by zero
    X = (X_raw - mins) / ranges
    y = df["label"].values
    return X, y


def evaluate_k_values(X, y, k_values, cv_folds=5, scoring="roc_auc"):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        results[k] = (scores.mean(), scores.std())
        print(f"k={k:2d} | {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return results


def plot_k_curve(results, metric_name="ROC AUC"):
    ks = list(results.keys())
    means = [results[k][0] for k in ks]
    stds = [results[k][1] for k in ks]
    plt.figure(figsize=(8, 4))
    plt.errorbar(ks, means, yerr=stds, fmt="-o")
    best_k = max(results, key=lambda k: results[k][0])
    plt.scatter(
        best_k,
        results[best_k][0],
        color="red",
        zorder=5,
        label=f"Best k={best_k} (mean={results[best_k][0]:.3f})",
    )
    plt.xlabel("k (number of neighbours)")
    plt.ylabel(metric_name)
    plt.title(f"K‑NN: {metric_name} vs. k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_features_labels(CSV_PATH)
    metrics = evaluate_k_values(X, y, K_RANGE, cv_folds=N_FOLDS, scoring=SCORING)
    plot_k_curve(metrics, metric_name="ROC AUC" if SCORING == "roc_auc" else SCORING)
    best_k = max(metrics, key=lambda k: metrics[k][0])
    print(f"\nBest k = {best_k}  →  mean {SCORING} = {metrics[best_k][0]:.4f}")
