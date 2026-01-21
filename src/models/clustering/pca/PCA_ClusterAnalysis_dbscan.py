import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler  # נירמול ל־[0,1]
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import silhouette_score


def smooth_stamp_moving_average(norm_stamp, kernel_size=50):
    smoothed = uniform_filter(norm_stamp, size=kernel_size)
    return smoothed


def gather_signatures_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    band_cols = [col for col in df.columns if col.startswith("band_")]
    return df[band_cols].values


def main(csv_path_or_dir):
    # 1) איסוף חתימות ספקטרליות מכל הקבצים
    if os.path.isdir(csv_path_or_dir):
        csv_files = glob.glob(os.path.join(csv_path_or_dir, "*.csv"))
    elif os.path.isfile(csv_path_or_dir) and csv_path_or_dir.endswith(".csv"):
        csv_files = [csv_path_or_dir]
    else:
        print("No valid CSV file or directory found:", csv_path_or_dir)
        return

    if not csv_files:
        print("No CSV files found in:", csv_path_or_dir)
        return

    all_signatures = []
    for csv_file in csv_files:
        signatures = gather_signatures_from_csv(csv_file)
        all_signatures.append(signatures)
    all_signatures = np.vstack(all_signatures)
    print("Total signatures gathered:", all_signatures.shape[0])

    # 2) החלקת כל חתימה (Moving Average)
    smoothed_sigs = np.apply_along_axis(
        lambda row: smooth_stamp_moving_average(row, kernel_size=50),
        axis=1,
        arr=all_signatures,
    )

    # 3) טעינת מודל PCA והפעלת טרנספורמציה
    pca_path = "/storage/yovelg/Grape/PCA/PCA_model.joblib"
    pca_model = joblib.load(pca_path)
    pca_result = pca_model.transform(smoothed_sigs)
    print("PCA result shape:", pca_result.shape)

    # 4) נירמול תוצאות ה-PCA באמצעות MinMaxScaler
    scaler = MinMaxScaler()
    pca_scaled = scaler.fit_transform(pca_result)

    # 5) הגדרת פונקציית מטרה לאופטונה - שימוש ב-Silhouette Score להערכת הקלאסטרינג
    def objective(trial):
        eps_value = trial.suggest_float("eps", 0.01, 1.0)
        min_samples_value = trial.suggest_int("min_samples", 1, 30)

        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        labels = dbscan.fit_predict(pca_scaled)

        # מספר הקלאסטרים (לא כולל רעש -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            return -1.0  # אם יש פחות מקלאסטר אחד, מחזירים ערך נמוך

        score = silhouette_score(pca_scaled, labels)
        return score

    # 6) הרצת אופטונה
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value
    print("Best parameters:", best_params)
    print("Best Silhouette Score:", best_score)

    # הפעלת DBSCAN עם הפרמטרים הטובים ביותר
    best_dbscan = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"])
    best_labels = best_dbscan.fit_predict(pca_scaled)

    unique_labels = np.unique(best_labels)
    print(
        f"\nDBSCAN found {len(unique_labels)} cluster labels (including noise labeled as -1)."
    )
    for cluster_label in unique_labels:
        count = np.sum(best_labels == cluster_label)
        print(f"  Cluster {cluster_label}: {count} items")

    # 7) פלוט היסטוריית האופטימיזציה
    plt.figure(figsize=(6, 4))
    trial_numbers = [trial.number + 1 for trial in study.trials]
    trial_scores = [trial.value for trial in study.trials]
    plt.plot(trial_numbers, trial_scores, marker="o", linestyle="-")
    plt.title("Optuna Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("Silhouette Score")
    best_trial_number = study.best_trial.number + 1
    plt.axvline(
        best_trial_number,
        color="r",
        linestyle="--",
        label=f"Best Trial {best_trial_number}",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8) פלוט סופי בתלת-ממד: PC1, PC2, PC3
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pca_scaled[:, 0],
        pca_scaled[:, 1],
        pca_scaled[:, 2],
        c=best_labels,
        cmap="viridis",
        s=5,
    )
    ax.set_title(
        f"3D DBSCAN Clustering\nBest Params: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}\nSilhouette Score: {best_score:.3f}"
    )
    ax.set_xlabel("PC1 (scaled)")
    ax.set_ylabel("PC2 (scaled)")
    ax.set_zlabel("PC3 (scaled)")
    fig.colorbar(sc, ax=ax, label="Cluster Label")
    plt.tight_layout()
    plt.show()

    # 9) פלוט מטריצת פיזור (scatter matrix) לכל זוגי ה-PC הראשונים (PC1 עד PC4)
    pc_names = ["PC1", "PC2", "PC3", "PC4"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (i, j) in enumerate(pairs):
        ax = axes.flat[idx]
        ax.scatter(
            pca_scaled[:, i], pca_scaled[:, j], c=best_labels, cmap="viridis", s=5
        )
        ax.set_xlabel(pc_names[i])
        ax.set_ylabel(pc_names[j])
        ax.set_title(f"{pc_names[i]} vs {pc_names[j]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_input = sys.argv[1]
    else:
        csv_input = (
            "/storage/yovelg/Grape/clastering/data/signatures_4_PCA_merged_all.csv"
        )
        print(f"No directory argument provided, defaulting to: {csv_input}")

    main(csv_input)
