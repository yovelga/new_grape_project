import os
import time
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# --- Improvement: Add optional cuML (GPU) imports ---
try:
    from cuml.neural_network import MLPClassifier as cuMLPClassifier
    from cuml.preprocessing import StandardScaler as cuMLScaler
    from cuml.pipeline import Pipeline as cuMLPipeline

    cuml_available = True
except ImportError:
    cuml_available = False

# This assumes you have a local module named 'project_helps'
# with a function 'project_path'.
# from project_helps import project_path

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# --- NEW: add joblib for model persistence ---
try:
    import joblib
except Exception as e:
    joblib = None
    print("[WARN] joblib not available; model saving will be disabled.", e)


# ------------------- Base Classifier -------------------
class BaseClassifier(ABC):
    """Abstract base class for all model wrappers."""

    @abstractmethod
    def train(self, X, y): pass

    @abstractmethod
    def predict(self, X): pass

    @abstractmethod
    def predict_proba(self, X): pass

    @abstractmethod
    def get_params(self): pass

    @abstractmethod
    def get_model_name(self): pass


# ------------------- Model Wrappers -------------------
class XGBoostModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "XGBoost"


class LDAModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("lda", LinearDiscriminantAnalysis(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "LDA"


class QDAModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("qda", QuadraticDiscriminantAnalysis(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "QDA"


class LogisticRegressionModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "LogisticRegression"


# --- New Model Wrappers ---
class DecisionTreeModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("tree", DecisionTreeClassifier(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "DecisionTree"


class RandomForestModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "RandomForest"


class GradientBoostingModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("gb", GradientBoostingClassifier(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "GradientBoosting"


class SVCModel(BaseClassifier):
    def __init__(self, **kwargs):
        # Ensure probability is True for predict_proba
        kwargs['probability'] = True
        self.model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "SVC"


class MLPModel(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**kwargs))])
        self.params = kwargs

    def train(self, X, y): self.model.fit(X, y)

    def predict(self, X): return self.model.predict(X)

    def predict_proba(self, X): return self.model.predict_proba(X)[:, 1]

    def get_params(self): return self.params

    def get_model_name(self): return "MLP (CPU)"


# --- New GPU-accelerated MLP Wrapper ---
if cuml_available:
    class MLPModelGPU(BaseClassifier):
        def __init__(self, **kwargs):
            # Use cuML's pipeline to keep all data on the GPU
            self.model = cuMLPipeline([
                ("scaler", cuMLScaler()),
                ("mlp", cuMLPClassifier(**kwargs))
            ])
            self.params = kwargs

        def train(self, X, y):
            # cuML expects pandas or numpy arrays, which is what we have
            self.model.fit(X, y)

        def predict(self, X):
            # .to_numpy() is used to bring data back to CPU for sklearn metrics
            return self.model.predict(X).to_numpy()

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, 1].to_numpy()

        def get_params(self):
            return self.params

        def get_model_name(self):
            return "MLP (GPU)"


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
def evaluate_logo(model: BaseClassifier, X, y, groups):
    """Performs Leave-One-Group-Out cross-validation for a given model."""
    logo = LeaveOneGroupOut()
    metrics = {
        'accs': [], 'rocs': [], 'prs': [], 'f1s': [], 'precs': [], 'recs': [],
        'train_times': [], 'infer_times': [], 'train_sizes': [], 'test_sizes': []
    }

    for train_idx, test_idx in tqdm(logo.split(X, y, groups), total=logo.get_n_splits(groups=groups),
                                    desc=f"Evaluating {model.get_model_name()}", leave=False):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        metrics['train_sizes'].append(len(X_train))
        metrics['test_sizes'].append(len(X_test))

        start_train = time.time()
        model.train(X_train, y_train)
        metrics['train_times'].append(time.time() - start_train)

        start_infer = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics['infer_times'].append(time.time() - start_infer)

        metrics['accs'].append(accuracy_score(y_test, y_pred))
        metrics['prs'].append(average_precision_score(y_test, y_prob))
        metrics['f1s'].append(f1_score(y_test, y_pred, average="weighted"))
        metrics['precs'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        metrics['recs'].append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))

        if len(np.unique(y_test)) > 1:
            metrics['rocs'].append(roc_auc_score(y_test, y_prob))
        else:
            metrics['rocs'].append(np.nan)

    results = {
        "Mean Accuracy": np.nanmean(metrics['accs']), "Std Dev Accuracy": np.nanstd(metrics['accs']),
        "Mean ROC AUC": np.nanmean(metrics['rocs']), "Std Dev ROC AUC": np.nanstd(metrics['rocs']),
        "Mean PR AUC": np.nanmean(metrics['prs']), "Mean F1 Weighted": np.nanmean(metrics['f1s']),
        "Mean Precision (Cracked)": np.nanmean(metrics['precs']), "Mean Recall (Cracked)": np.nanmean(metrics['recs']),
        "Mean Training Time (s)": np.mean(metrics['train_times']),
        "Mean Inference Time (s)": np.mean(metrics['infer_times']),
        "Mean Train Samples": int(np.mean(metrics['train_sizes'])),
        "Mean Test Samples": int(np.mean(metrics['test_sizes'])),
    }
    return results


# ------------------- Helpers: Model Saving -------------------
def _safe_save_model(model_wrapper: BaseClassifier, out_dir: Path, model_name: str, balance_mode: str) -> Path | None:
    """
    Try to persist the model wrapper; if it fails, fall back to saving the underlying estimator.
    Returns the saved path or None if saving failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{model_name.replace(' ', '_')}_{balance_mode}.pkl"
    path_wrapper = out_dir / base_name
    if joblib is None:
        print(f"[WARN] joblib missing. Skipping save for {model_name} ({balance_mode}).")
        return None
    # Try saving the wrapper first
    try:
        joblib.dump(model_wrapper, path_wrapper)
        print(f"[MODEL SAVED] {path_wrapper}")
        return path_wrapper
    except Exception as e:
        print(f"[WARN] Saving wrapper failed for {model_name}: {e}. Trying underlying estimator...")
        # Try saving the underlying estimator/pipeline
        try:
            path_inner = out_dir / f"{model_name.replace(' ', '_')}_{balance_mode}_estimator.pkl"
            joblib.dump(getattr(model_wrapper, "model", model_wrapper), path_inner)
            print(f"[MODEL SAVED] {path_inner}")
            return path_inner
        except Exception as e2:
            print(f"[ERROR] Failed to save model {model_name} for {balance_mode}: {e2}")
            return None


# ------------------- Main Benchmark Runner -------------------
def main():


    """Main function to run the model benchmark using hardcoded paths."""
    CSV_PATH = Path(
        r"C:\Users\yovel\Desktop\Grape_Project\dataset_builder_grapes\detection\dataset\dataset_outlier_0.01.csv")
    OUT_PATH = Path(
        r"C:\Users\yovel\Desktop\Grape_Project\classification\pixel_level\simple_classification_leave_one_out\comare_all_models\model_comparison_results_2.xlsx")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_STRATEGY = "Leave-One-Group-Out"
    # --- NEW: control saving and models directory ---
    SAVE_MODELS = True
    MODELS_DIR = OUT_PATH.parent / "models"

    models_to_run = [
        XGBoostModel(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss",
                     tree_method='gpu_hist'),
        RandomForestModel(n_estimators=100, max_depth=10, random_state=42),
        GradientBoostingModel(n_estimators=100, max_depth=5, random_state=42),
        MLPModel(hidden_layer_sizes=(204, 100, 50), max_iter=50, random_state=42, early_stopping=True,
                 n_iter_no_change=5, learning_rate='adaptive'),
        LDAModel(solver="svd"),
        QDAModel(reg_param=0.1),
        LogisticRegressionModel(max_iter=1000, solver="lbfgs"),
        DecisionTreeModel(max_depth=10, random_state=42),
        SVCModel(kernel='rbf', C=1, random_state=42)
    ]

    # --- Add GPU MLP if cuML is available ---
    if cuml_available:
        print("[INFO] cuML found. Adding GPU-accelerated MLP to the benchmark.")
        models_to_run.append(
            MLPModelGPU(hidden_layer_sizes=(204, 100, 50), max_iter=50, random_state=42, tol=1e-4)
        )
    else:
        print("[INFO] cuML not found. GPU-accelerated MLP will be skipped.")

    all_results = {}

    for is_balanced in [False, True]:
        balance_mode = "Balanced" if is_balanced else "Unbalanced"
        print(f"\n--- Running Benchmark for {balance_mode} Dataset ---")

        X, y, groups, _ = load_data(CSV_PATH, balanced=is_balanced)

        results_table = []
        for model in models_to_run:
            print(f"\n[INFO] Starting evaluation for: {model.get_model_name()}")
            try:
                metrics = evaluate_logo(model, X, y, groups)
                results_table.append({
                    "Model Name": model.get_model_name(), "Validation Strategy": VALIDATION_STRATEGY,
                    **metrics, "Key Parameters": str(model.get_params())
                })
                print(f"[SUCCESS] {model.get_model_name()} evaluation complete.")

                # --- NEW: Train on full data for this balance mode and save ---
                if SAVE_MODELS:
                    try:
                        start_full_train = time.time()
                        model.train(X, y)
                        train_dur = time.time() - start_full_train
                        saved_path = _safe_save_model(model, MODELS_DIR, model.get_model_name(), balance_mode)
                        if saved_path:
                            print(f"[INFO] Final {model.get_model_name()} ({balance_mode}) trained on full data "
                                  f"in {train_dur:.2f}s and saved to: {saved_path}")
                        else:
                            print(f"[WARN] Save path not returned for {model.get_model_name()} ({balance_mode}).")
                    except Exception as se:
                        print(f"[ERROR] Failed to train/save final {model.get_model_name()} ({balance_mode}): {se}")

            except Exception as e:
                print(f"\n[ERROR] Evaluation failed for {model.get_model_name()}. Reason: {e}")
                results_table.append({
                    "Model Name": model.get_model_name(), "Validation Strategy": VALIDATION_STRATEGY,
                    "Mean Accuracy": "FAILED", "Mean ROC AUC": "FAILED",
                    "Key Parameters": str(model.get_params())
                })

        df_results = pd.DataFrame(results_table)
        for col in df_results.select_dtypes(include=np.number).columns:
            df_results[col] = df_results[col].round(4)

        all_results[balance_mode] = df_results

    # --- Save results to an Excel file with multiple sheets ---
    print(f"\n--- Saving results to Excel file: {OUT_PATH} ---")
    with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
        all_results["Unbalanced"].to_excel(writer, sheet_name='Unbalanced_Results', index=False)
        all_results["Balanced"].to_excel(writer, sheet_name='Balanced_Results', index=False)

    print("\n--- Unbalanced Dataset Results ---")
    print(all_results["Unbalanced"].to_string())
    print("\n--- Balanced Dataset Results ---")
    print(all_results["Balanced"].to_string())
    print(f"\n[INFO] Benchmark complete. Results saved to: {OUT_PATH}")
    if SAVE_MODELS:
        print(f"[INFO] Models saved under: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
