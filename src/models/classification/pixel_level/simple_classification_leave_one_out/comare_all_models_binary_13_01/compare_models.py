from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
import time
import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
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


# ------------------- Helper Functions -------------------
def slugify_model_name(model_name: str) -> str:
    """Convert model name to a filesystem-friendly slug."""
    slug = model_name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_curves_and_confusions(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray,
    out_dir: Path,
    model_name: str,
    balance_mode: str,
):
    """
    Save ROC curve, PR curve, confusion matrices, and predictions CSV.

    Parameters
    ----------
    y_true_all : np.ndarray
        Concatenated ground truth labels across all folds.
    y_pred_all : np.ndarray
        Concatenated hard predictions across all folds.
    y_prob_all : np.ndarray
        Concatenated probabilities for positive class across all folds.
    out_dir : Path
        Base output directory for results.
    model_name : str
        Name of the model.
    balance_mode : str
        'Balanced' or 'Unbalanced'.
    """
    model_slug = slugify_model_name(model_name)
    save_dir = ensure_dir(out_dir / "results" / "Binary_Pixel_Level_Classification" / balance_mode / model_slug)

    # --- Save predictions CSV ---
    predictions_df = pd.DataFrame({
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_prob': y_prob_all
    })
    predictions_df.to_csv(save_dir / "predictions.csv", index=False)
    print(f"    [SAVED] predictions.csv")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name} ({balance_mode})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"    [SAVED] roc_curve.png")

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    ap_score = average_precision_score(y_true_all, y_prob_all)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name} ({balance_mode})')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "pr_curve.png", dpi=150)
    plt.close(fig)
    print(f"    [SAVED] pr_curve.png")

    # --- Confusion Matrix (Normalized) ---
    cm_norm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Not Cracked', 'Cracked'])
    disp.plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True)
    ax.set_title(f'Confusion Matrix (Normalized) - {model_name} ({balance_mode})')
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"    [SAVED] confusion_matrix.png")

    # --- Confusion Matrix (Raw) ---
    cm_raw = confusion_matrix(y_true_all, y_pred_all, normalize=None)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=['Not Cracked', 'Cracked'])
    disp_raw.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    ax.set_title(f'Confusion Matrix (Raw) - {model_name} ({balance_mode})')
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix_raw.png", dpi=150)
    plt.close(fig)
    print(f"    [SAVED] confusion_matrix_raw.png")


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
def evaluate_logo(model, model_name: str, X, y, groups, return_predictions: bool = True):
    """
    Performs Leave-One-Group-Out cross-validation.

    Parameters
    ----------
    model : estimator
        The model to evaluate.
    model_name : str
        Name of the model for display.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    groups : np.ndarray
        Group labels for LOGO CV.
    return_predictions : bool
        If True, also return concatenated predictions across folds.

    Returns
    -------
    summary_dict : dict
        Summary metrics for the model.
    curves_dict : dict or None
        If return_predictions is True, contains y_true_all, y_pred_all, y_prob_all.
    """
    logo = LeaveOneGroupOut()
    metrics = {'accs': [], 'rocs': [], 'prs': [], 'f1s': [], 'precs': [], 'recs': [],
               'train_times': [], 'infer_times': []}

    # For aggregated curves
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

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

        # Collect predictions for aggregated curves
        if return_predictions:
            y_true_list.append(y_test)
            y_pred_list.append(y_pred)
            y_prob_list.append(y_prob)

    summary_dict = {
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

    if return_predictions:
        curves_dict = {
            'y_true_all': np.concatenate(y_true_list),
            'y_pred_all': np.concatenate(y_pred_list),
            'y_prob_all': np.concatenate(y_prob_list),
        }
        return summary_dict, curves_dict
    else:
        return summary_dict, None


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
        results_table = []

        for model_name, model in get_models():
            print(f"\n[INFO] Evaluating: {model_name}")
            try:
                metrics, curves_dict = evaluate_logo(model, model_name, X, y, groups, return_predictions=True)
                results_table.append(metrics)
                print(f"[SUCCESS] {model_name} - Accuracy: {metrics['Mean Accuracy']:.4f}, ROC-AUC: {metrics['Mean ROC AUC']:.4f}")

                # Save evaluation artifacts (curves, confusion matrix, predictions CSV)
                if curves_dict is not None:
                    save_curves_and_confusions(
                        y_true_all=curves_dict['y_true_all'],
                        y_pred_all=curves_dict['y_pred_all'],
                        y_prob_all=curves_dict['y_prob_all'],
                        out_dir=EXPERIMENT_DIR,
                        model_name=model_name,
                        balance_mode=balance_mode,
                    )

                # Clone and train on full data for saving (fresh instance)
                final_model = clone(model)
                final_model.fit(X, y)
                save_model(final_model, MODELS_DIR, model_name, balance_mode)

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
