from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Import the reusable preprocessing function
from src.preprocessing.spectral_preprocessing import preprocess_pixel_level_dataset

try:
    import joblib
except ImportError:
    joblib = None
    print("[WARN] joblib not available; model saving will be disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------- PLS-DA Wrapper -------------------
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS-DA classifier using PLSRegression with thresholding."""
    def __init__(self, n_components=10, threshold=0.5):
        self.n_components = n_components
        self.threshold = threshold
        # Note: Don't create self.pls here - sklearn clone() requires __init__ to only store params

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (self.pls_.predict(X).ravel() >= self.threshold).astype(int)

    def predict_proba(self, X):
        scores = self.pls_.predict(X).ravel()
        p1 = np.clip(scores, 0, 1)
        # sklearn convention: (n_samples, 2)
        return np.column_stack([1 - p1, p1])



# ------------------- LOGO Evaluation -------------------
def evaluate_logo(model, model_name: str, X, y, groups):
    """Performs Leave-One-Group-Out cross-validation."""
    logo = LeaveOneGroupOut()
    metrics = {'accs': [], 'rocs': [], 'prs': [], 'f1s': [], 'precs': [], 'recs': [],
               'train_times': [], 'infer_times': []}

    for train_idx, test_idx in tqdm(logo.split(X, y, groups), total=logo.get_n_splits(groups=groups),
                                    desc=f"Evaluating {model_name}", leave=False):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone the model for each fold to avoid state issues
        fold_model = clone(model)

        start_train = time.time()
        fold_model.fit(X_train, y_train)
        metrics['train_times'].append(time.time() - start_train)

        start_infer = time.time()
        y_pred = fold_model.predict(X_test)
        y_prob = fold_model.predict_proba(X_test)
        if hasattr(y_prob, 'ndim') and y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        metrics['infer_times'].append(time.time() - start_infer)

        metrics['accs'].append(accuracy_score(y_test, y_pred))
        metrics['prs'].append(average_precision_score(y_test, y_prob))
        metrics['f1s'].append(f1_score(y_test, y_pred, average="weighted"))
        metrics['precs'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        metrics['recs'].append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        metrics['rocs'].append(roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan)

    return {
        "Model Name": model_name,
        "Mean Accuracy": np.nanmean(metrics['accs']),
        "Std Accuracy": np.nanstd(metrics['accs']),
        "Mean ROC AUC": np.nanmean(metrics['rocs']),
        "Std ROC AUC": np.nanstd(metrics['rocs']),
        "Mean PR AUC": np.nanmean(metrics['prs']),
        "Mean F1 Weighted": np.nanmean(metrics['f1s']),
        "Mean Precision (Cracked)": np.nanmean(metrics['precs']),
        "Mean Recall (Cracked)": np.nanmean(metrics['recs']),
        "Mean Train Time (s)": np.mean(metrics['train_times']),
        "Mean Infer Time (s)": np.mean(metrics['infer_times']),
    }


def save_model(model, out_dir: Path, model_name: str, balance_mode: str):
    """Save model to disk."""
    if joblib is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name.replace(' ', '_')}_{balance_mode}.pkl"
    try:
        joblib.dump(model, path)
        print(f"[MODEL SAVED] {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Failed to save {model_name}: {e}")
        return None


# ------------------- Main -------------------
def main():
    # Paths
    CSV_PATH = Path(str(_PROJECT_ROOT / r"src/preprocessing/dataset_builder_grapes/detection/raw_exported_data/all_origin_signatures_results_2026-01-13.csv"))
    EXPERIMENT_DIR = Path(str(_PROJECT_ROOT / r"experiments/pixel_level_classifier_2_classes"))
    OUT_PATH = EXPERIMENT_DIR / "model_comparison_results.xlsx"
    MODELS_DIR = EXPERIMENT_DIR / "models"

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Define the 5 models to run
    def get_models():
        return [
            ("PLS-DA", Pipeline([("scaler", StandardScaler()), ("pls", PLSDAClassifier(n_components=10))])),
            ("Logistic Regression (L1)", Pipeline([("scaler", StandardScaler()),
                ("logreg", LogisticRegression(penalty="l1", solver="saga", max_iter=1000, random_state=42))])),
            ("SVM (RBF)", Pipeline([("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=1, probability=True, random_state=42))])),
            ("Random Forest", Pipeline([("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))])),
            ("XGBoost", XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                eval_metric="logloss", tree_method="hist", random_state=42)),
        ]

    all_results = {}

    for is_balanced in [False, True]:
        balance_mode = "Balanced" if is_balanced else "Unbalanced"
        print(f"\n{'='*60}\n--- Running Benchmark: {balance_mode} Dataset ---\n{'='*60}")

        # Load and preprocess data using the reusable preprocessing function
        print(f"[INFO] Loading data from {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
        X, y, groups, feature_names = preprocess_pixel_level_dataset(
            df,
            wl_min=450,
            wl_max=925,
            apply_snv=True,
            remove_outliers=False,
            balanced=is_balanced,
        )
        print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(groups))} groups.")

        # =========================
        # Save to Excel
        # =========================

        EXCEL_OUT_DIR = os.path.join(str(_PROJECT_ROOT), "experiments", "check")
        os.makedirs(EXCEL_OUT_DIR, exist_ok=True)

        XLSX_PATH = os.path.join(EXCEL_OUT_DIR, f"check_preprocessed_dataset_is_balanced_{is_balanced}.xlsx")

        # 1) Build a single dataframe: features + labels
        df_X = pd.DataFrame(X, columns=feature_names)
        df_X.insert(0, "group", groups)
        df_X.insert(0, "y", y)

        # 2) Extra helpful sheets (optional but useful)
        df_y = pd.DataFrame({"y": y})
        df_groups = pd.DataFrame({"group": groups})
        df_features = pd.DataFrame({"feature_name": feature_names})

        with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
            df_X.to_excel(writer, sheet_name="dataset", index=False)
            df_features.to_excel(writer, sheet_name="feature_names", index=False)
            df_y.to_excel(writer, sheet_name="y", index=False)
            df_groups.to_excel(writer, sheet_name="groups", index=False)

        print(f"[INFO] Saved preprocessed dataset to: {XLSX_PATH}")
        print(f"[INFO] Sheet 'dataset' columns: y, group, + {len(feature_names)} features")


if __name__ == "__main__":
    main()
