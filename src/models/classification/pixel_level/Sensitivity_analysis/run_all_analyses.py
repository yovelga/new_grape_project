# run_all_analyses_v7_refactored_sfs.py
# ------------------------------------------------------------------------------------
# This version refactors the Sequential Feature Selector (SFS) logic into a
# reusable helper method to keep the code DRY. It re-introduces the critical
# validation for LOGO splits to ensure every fold is viable for both training
# and testing. Unused methods have been removed, and the main cross-validation
# process is now parallelized to improve performance.
# ------------------------------------------------------------------------------------

import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


# --- Utility Functions ---

def extract_cluster_id(hs_dir: str) -> str:
    parts = os.path.normpath(hs_dir).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def load_xyg(dataset_path: Path):
    """Loads X, y, and groups from the dataset."""
    df = pd.read_csv(dataset_path)
    feat_cols = [c for c in df.columns if str(c).endswith("nm")]
    groups = df["hs_dir"].apply(extract_cluster_id).values
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y, groups, feat_cols


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
warnings.filterwarnings('ignore', category=UserWarning)


# --- Main Analysis Class ---

class AnalysisRunner:
    """Encapsulates the entire analysis workflow."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        parent_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        self.results_dir = parent_dir / "results_v7_sfs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Runner initialized. Results will be saved to '{self.results_dir.name}'.")

    @staticmethod
    def _get_valid_logo_splits(y: np.ndarray, groups: np.ndarray):
        """Pre-calculates LOGO splits, filtering any split where train or test set has only one class."""
        logo = LeaveOneGroupOut()
        valid_splits = []
        n_original = logo.get_n_splits(groups=groups)
        for train_idx, test_idx in logo.split(X=np.zeros(len(y)), y=y, groups=groups):
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
                valid_splits.append((train_idx, test_idx))
        n_valid = len(valid_splits)
        if n_valid < n_original:
            logging.warning(
                f"Removed {n_original - n_valid} LOGO splits due to single-class train/test sets. Using {n_valid} valid splits.")
        else:
            logging.info("All LOGO splits are valid.")
        if n_valid == 0:
            raise RuntimeError("No valid LOGO splits found!")
        return valid_splits

    def _create_sfs_pipeline(self, model, k: int, balance_per_fold: bool):
        """Helper method to create a Sequential Feature Selector pipeline."""
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=k,
            direction="forward",
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=1  # Important for nested parallelism: inner loop is sequential
        )
        steps = []
        if balance_per_fold:
            steps.append(("sampler", RandomUnderSampler(random_state=42)))
        steps.extend([
            ("scaler", StandardScaler()),
            ("selector", sfs),
            ("clf", model),
        ])
        return (ImbPipeline if balance_per_fold else Pipeline)(steps=steps)

    def run_k_tables_balanced(self, out_dir: Path):
        """Builds LOGO tables for k=1..20 using SFS and validated folds."""
        MODELS_K = {
            "LDA": LinearDiscriminantAnalysis(solver="svd"),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        }
        SCORING = {"accuracy": "accuracy", "roc_auc": "roc_auc", "pr_auc": "average_precision",
                   "f1": "f1", "precision": "precision", "recall": "recall"}

        X, y, groups, _ = load_xyg(self.dataset_path)
        valid_logo_cv = self._get_valid_logo_splits(y, groups)
        out_dir.mkdir(parents=True, exist_ok=True)
        master_rows = []

        for k in tqdm(range(1, 21), desc="Processing k"):
            rows = []
            for model_name, model in MODELS_K.items():
                pipe = self._create_sfs_pipeline(model, k, balance_per_fold=True)
                cvres = cross_validate(
                    pipe, X, y,
                    cv=valid_logo_cv,
                    scoring=SCORING,
                    n_jobs=-1  # Use all cores for the outer LOGO loop
                )
                row = {
                    "Model Name": model_name,
                    "Mean Acc.": np.mean(cvres["test_accuracy"]),
                    "Mean ROC AUC": np.mean(cvres["test_roc_auc"]),
                    "Mean PR AUC": np.mean(cvres["test_pr_auc"]),
                    "Mean F1": np.mean(cvres["test_f1"]),
                    "Prec. (Cracked)": np.mean(cvres["test_precision"]),
                    "Recall (Cracked)": np.mean(cvres["test_recall"]),
                    "Train Time (s)": float(np.mean(cvres["fit_time"])),
                }
                rows.append(row)
                master_rows.append({"k": k, **row})

            kdf = pd.DataFrame(rows)
            k_dir = out_dir / f"k_{k:02d}_bands"
            k_dir.mkdir(exist_ok=True)
            kdf.to_csv(k_dir / "table_k.csv", index=False)
            kdf.to_latex(k_dir / "table_k.tex", index=False, float_format="%.4f",
                         caption=f"SFS Performance (k={k}, Balanced LOGO).", label=f"tab:k{k}_sfs", position="ht!")

        master_df = pd.DataFrame(master_rows)
        master_df.to_csv(out_dir / "all_k_master_balanced.csv", index=False)
        logging.info(f"✅ Balanced k-tables saved to {out_dir}")

    def run_k_marginal_logo(self, out_dir: Path, balance_per_fold: bool = True):
        """Runs LOGO with SFS for marginal gain analysis using validated folds."""
        MODELS = {
            "LDA": LinearDiscriminantAnalysis(solver="svd"),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        }
        SCORING = {"accuracy": "accuracy", "roc_auc": "roc_auc", "pr_auc": "average_precision",
                   "f1": "f1", "precision": "precision", "recall": "recall"}

        X, y, groups, feat_cols = load_xyg(self.dataset_path)
        valid_logo_cv = self._get_valid_logo_splits(y, groups)
        out_dir.mkdir(parents=True, exist_ok=True)
        wl_freq_dir = out_dir / "wavelengths_selected"
        wl_freq_dir.mkdir(exist_ok=True)
        rows = []

        for k in tqdm(range(1, 21), desc="Processing k"):
            for model_name, model in MODELS.items():
                pipe = self._create_sfs_pipeline(model, k, balance_per_fold)
                cvres = cross_validate(
                    pipe, X, y,
                    cv=valid_logo_cv,
                    scoring=SCORING,
                    n_jobs=-1,  # Use all cores for the outer LOGO loop
                    return_estimator=True
                )
                row = {
                    "Model": model_name, "k": k,
                    "Mean Acc.": float(np.mean(cvres["test_accuracy"])),
                    "Mean ROC AUC": float(np.mean(cvres["test_roc_auc"])),
                    "Mean PR AUC": float(np.mean(cvres["test_pr_auc"])),
                    "Mean F1": float(np.mean(cvres["test_f1"])),
                    "Prec. (Cracked)": float(np.mean(cvres["test_precision"])),
                    "Recall (Cracked)": float(np.mean(cvres["test_recall"])),
                    "Train Time (s)": float(np.mean(cvres["fit_time"])),
                }
                rows.append(row)

                # Track selected features
                freq = pd.Series(0, index=feat_cols, dtype=int)
                for est in cvres["estimator"]:
                    selector_mask = est.named_steps["selector"].get_support()
                    freq.loc[np.array(feat_cols)[selector_mask]] += 1
                freq_df = freq.sort_values(ascending=False).to_frame("fold_frequency")
                freq_df.to_csv(wl_freq_dir / f"wavelengths_selected_{model_name}_k{k:02d}.csv")

        # Process and save results
        res_df = pd.DataFrame(rows).sort_values(["Model", "k"]).reset_index(drop=True)
        for metric in ["Mean ROC AUC", "Mean PR AUC", "Mean Acc."]:
            res_df[f"delta_{metric.lower().replace(' ', '_')}"] = res_df.groupby("Model")[metric].diff().fillna(0)
        res_df.to_csv(out_dir / "results_k_by_model.csv", index=False)
        self._plot_marginal_gain(res_df, out_dir)
        logging.info(f"✅ Marginal gain analysis saved to {out_dir}")

    def _plot_marginal_gain(self, res_df: pd.DataFrame, out_dir: Path):
        """Generates and saves plots for marginal gain."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plot_configs = [
            ("delta_mean_roc_auc", "ΔROC AUC"),
            ("delta_mean_pr_auc", "ΔPR AUC"),
            ("delta_mean_acc", "ΔAccuracy")
        ]
        for col, title_label in plot_configs:
            plt.figure(figsize=(8, 5))
            for m, sub in res_df.groupby("Model"):
                plt.plot(sub["k"], sub[col], marker="o", label=m)
            plt.axhline(0, linestyle="--", linewidth=1, color="grey")
            plt.title(f"Marginal Gain {title_label} vs. k (SFS, Validated LOGO)")
            plt.xlabel("k (Number of Selected Features)")
            plt.ylabel(f"Marginal Gain ({title_label})")
            plt.xticks(res_df['k'].unique())
            plt.legend();
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(out_dir / f"marginal_gain_{title_label.replace('Δ', '')}.png", dpi=300)
            plt.close()


def main():
    """Main execution function for focused SFS-LOGO analysis with validated folds."""
    env_path = Path(__file__).parent / '.env'
    if not env_path.is_file():
        logging.error(f"❌ FATAL: .env file not found at {env_path}")
        return
    load_dotenv(dotenv_path=env_path)

    base_path_str = os.getenv('BASE_PATH')
    dataset_path_str = os.getenv('DATASET_LDA_PATH')
    if not base_path_str or not dataset_path_str:
        logging.error("❌ FATAL: BASE_PATH or DATASET_LDA_PATH not set in .env file.")
        return

    base_path = Path(base_path_str)
    dataset_path = base_path / dataset_path_str
    if not dataset_path.is_file():
        logging.error(f"❌ FATAL: Dataset file not found at: {dataset_path}")
        return

    runner = AnalysisRunner(dataset_path=dataset_path)

    k_tables_dir = runner.results_dir / "k_tables_logo_balanced"
    logging.info(f"\nRunning per-fold balanced LOGO k-table benchmark (see {k_tables_dir})...")
    runner.run_k_tables_balanced(k_tables_dir)

    marginal_out_dir = runner.results_dir / "logo_k_marginal_analysis"
    logging.info(f"\nRunning LOGO marginal-gain analysis (see {marginal_out_dir})...")
    runner.run_k_marginal_logo(marginal_out_dir, balance_per_fold=True)

    logging.info("\n🎉🎉🎉 All analyses completed successfully! 🎉🎉🎉")


if __name__ == "__main__":
    main()