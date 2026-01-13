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

try:
    import joblib
except ImportError:
    joblib = None
    print("[WARN] joblib not available; model saving will be disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------- PLS-DA Wrapper -------------------
class PLSDAClassifier:
    """PLS-DA classifier using PLSRegression with thresholding."""
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.threshold = 0.5

    def fit(self, X, y):
        self.pls.fit(X, y)
        return self

    def predict(self, X):
        return (self.pls.predict(X).ravel() >= self.threshold).astype(int)

    def predict_proba(self, X):
        scores = self.pls.predict(X).ravel()
        # Clip to [0, 1] for probability-like output
        return np.clip(scores, 0, 1)


# ------------------- Data Loader -------------------
def extract_cluster_id(hs_dir: str) -> str:
    """Return the cluster folder name (three levels up)."""
    parts = Path(hs_dir).parts
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_data(path: Path, balanced: bool = False):
    """Return X, y, groups, feature_names."""
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_csv(path)
    if "hs_dir" not in df.columns:
        raise ValueError("Column 'hs_dir' not found in dataset.")
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

    if balanced:
        print("[INFO] Balancing dataset via undersampling...")
        cracked = df[df.label == 1]
        regular = df[df.label == 0].sample(len(cracked), random_state=42)
        df = pd.concat([cracked, regular]).sample(frac=1, random_state=42)

    features = [c for c in df.columns if c.endswith("nm")]
    if not features:
        raise ValueError("No feature columns ending with 'nm' found in the CSV.")

    X = df[features].to_numpy(np.float32)
    y = df["label"].to_numpy(int)
    groups = df["cluster_id"].to_numpy(str)
    print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(groups))} groups.")
    return X, y, groups, features


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

        start_train = time.time()
        model.fit(X_train, y_train)
        metrics['train_times'].append(time.time() - start_train)

        start_infer = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
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
    CSV_PATH = Path(r"C:\Users\yovel\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_2026-01-13.csv")
    EXPERIMENT_DIR = Path(r"C:\Users\yovel\Desktop\Grape_Project\experiments\pixel_level_classifier_2_classes")
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

        X, y, groups, _ = load_data(CSV_PATH, balanced=is_balanced)
        results_table = []

        for model_name, model in get_models():
            print(f"\n[INFO] Evaluating: {model_name}")
            try:
                metrics = evaluate_logo(model, model_name, X, y, groups)
                results_table.append(metrics)
                print(f"[SUCCESS] {model_name} - Accuracy: {metrics['Mean Accuracy']:.4f}, ROC-AUC: {metrics['Mean ROC AUC']:.4f}")

                # Train on full data and save
                model.fit(X, y)
                save_model(model, MODELS_DIR, model_name, balance_mode)

            except Exception as e:
                print(f"[ERROR] {model_name} failed: {e}")
                results_table.append({"Model Name": model_name, "Mean Accuracy": "FAILED"})

        df_results = pd.DataFrame(results_table)
        for col in df_results.select_dtypes(include=np.number).columns:
            df_results[col] = df_results[col].round(4)
        all_results[balance_mode] = df_results

    # Save to Excel
    print(f"\n[INFO] Saving results to: {OUT_PATH}")
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
        all_results["Unbalanced"].to_excel(writer, sheet_name='Unbalanced_Results', index=False)
        all_results["Balanced"].to_excel(writer, sheet_name='Balanced_Results', index=False)

    print("\n--- Unbalanced Dataset Results ---")
    print(all_results["Unbalanced"].to_string())
    print("\n--- Balanced Dataset Results ---")
    print(all_results["Balanced"].to_string())
    print(f"\n[INFO] Benchmark complete. Results saved to: {OUT_PATH}")
    print(f"[INFO] Models saved under: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
