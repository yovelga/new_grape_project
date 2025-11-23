import os
import joblib
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from itertools import combinations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENTS = [
    {"min_wl": 400, "max_wl": 700, "max_k": 5},   # נסיון קטן - לשנות ל-30 כשמוכנים
    {"min_wl": 700, "max_wl": 1000, "max_k": 5},
]

MODELS = {
    "LDA": LinearDiscriminantAnalysis(solver="svd"),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
    "SVM_rbf": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

class ExhaustiveRunner:
    def __init__(self, base_path: Path, dataset_path: Path):
        self.base_path = base_path
        self.dataset_path = dataset_path
        self.results_dir = Path(__file__).parent / "results_exhaustive"
        self.models_dir = Path(__file__).parent / "best_models_exhaustive"
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    @staticmethod
    def _extract_cluster_id(hs_dir: str) -> str:
        parts = os.path.normpath(hs_dir).split(os.sep)
        return parts[-3] if len(parts) >= 3 else "unknown"

    def _load_data(self, min_wl, max_wl):
        df = pd.read_csv(self.dataset_path)
        feats = [c for c in df.columns if c.endswith("nm")]
        numeric_feats = pd.to_numeric([f.replace("nm", "") for f in feats])
        wl_mask = (numeric_feats >= min_wl) & (numeric_feats <= max_wl)
        filtered_feats = np.array(feats)[wl_mask].tolist()
        if not filtered_feats:
            raise ValueError(f"No features found in {min_wl}-{max_wl}nm")

        df_filtered = df[["hs_dir", "label"] + filtered_feats].copy()
        df_filtered["cluster_id"] = df_filtered["hs_dir"].apply(self._extract_cluster_id)

        cracked = df_filtered[df_filtered.label == 1]
        regular = df_filtered[df_filtered.label == 0].sample(n=len(cracked), random_state=42)
        df_balanced = pd.concat([cracked, regular]).sample(frac=1, random_state=42).reset_index(drop=True)

        X = df_balanced[filtered_feats].values.astype(np.float32)
        y = df_balanced["label"].values
        groups = df_balanced["cluster_id"].values
        return X, y, groups, filtered_feats

    def _evaluate_combo(self, X, y, groups, feats, model, combo):
        """Train + CV on a specific feature combo"""
        cols = [feats[i] for i in combo]
        X_sub = X[:, combo]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
        cv = GroupKFold(n_splits=5)
        accs, rocs = [], []
        for train_idx, test_idx in cv.split(X_sub, y, groups):
            pipe.fit(X_sub[train_idx], y[train_idx])
            y_pred = pipe.predict(X_sub[test_idx])
            y_prob = pipe.predict_proba(X_sub[test_idx])[:, 1] if hasattr(pipe, "predict_proba") else None
            accs.append(accuracy_score(y[test_idx], y_pred))
            if y_prob is not None:
                rocs.append(roc_auc_score(y[test_idx], y_prob))
        return cols, np.mean(accs), np.mean(rocs) if rocs else np.nan

    def run(self):
        for exp in EXPERIMENTS:
            min_wl, max_wl, max_k = exp["min_wl"], exp["max_wl"], exp["max_k"]
            run_id = f"{min_wl}nm_{max_wl}nm"
            X, y, groups, feats = self._load_data(min_wl, max_wl)

            for model_name, model in MODELS.items():
                logging.info(f"Running exhaustive search for {model_name} on {run_id}")
                results = []
                for k in range(1, max_k + 1):
                    for combo in tqdm(combinations(range(len(feats)), k), desc=f"{model_name}-{k}feats", leave=False):
                        cols, acc, roc = self._evaluate_combo(X, y, groups, feats, model, combo)
                        results.append({
                            "k": k,
                            "features": ",".join(cols),
                            "accuracy": acc,
                            "roc_auc": roc
                        })
                df = pd.DataFrame(results)
                save_dir = self.results_dir / model_name / run_id
                save_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_dir / "exhaustive_results.csv", index=False)
                logging.info(f"Saved exhaustive results for {model_name} at {save_dir}")

def main():
    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        logging.error("Missing .env")
        return
    load_dotenv(env_path)
    base_path = Path(os.getenv("BASE_PATH"))
    dataset_path = base_path / os.getenv("DATASET_LDA_PATH")
    runner = ExhaustiveRunner(base_path, dataset_path)
    runner.run()

if __name__ == "__main__":
    main()
