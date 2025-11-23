import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler  # נירמול לתחום [0,1]
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

    # 5) הגדרת פונקציית מטרה לאופטונה עבור KMeans
    def objective(trial):
        n_clusters = trial.suggest_int("n_clusters", 2, 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pca_scaled)
        if len(set(labels)) < 2:
            return -1.0
        score = silhouette_score(pca_scaled, labels)
        return score

    # 6) הרצת אופטונה לאופטימיזציה של מספר הקבוצות
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value
    print("Best parameters:", best_params)
    print("Best Silhouette Score:", best_score)

    # 7) הפעלת KMeans עם הפרמטרים הטובים ביותר
    best_kmeans = KMeans(n_clusters=best_params["n_clusters"], random_state=42)
    best_labels = best_kmeans.fit_predict(pca_scaled)
    unique_labels = np.unique(best_labels)
    print(f"\nKMeans found {len(unique_labels)} clusters.")
    for cluster_label in unique_labels:
        count = np.sum(best_labels == cluster_label)
        print(f"  Cluster {cluster_label}: {count} items")

    # 8) פלוט היסטוריית האופטימיזציה של Optuna
    plt.figure(figsize=(6, 4))
    trial_numbers = [trial.number + 1 for trial in study.trials]
    trial_scores = [trial.value for trial in study.trials]
    plt.plot(trial_numbers, trial_scores, marker="o", linestyle="-")
    plt.title("Optuna Optimization History for KMeans")
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

    # 9) פלוט סופי 2D: תצוגת הקלאסטרינג על PC1 ו-PC2
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(
        pca_scaled[:, 0], pca_scaled[:, 1], c=best_labels, cmap="viridis", s=5
    )
    plt.title(
        f"KMeans Clustering\nBest Params: n_clusters={best_params['n_clusters']}\nSilhouette Score: {best_score:.3f}"
    )
    plt.xlabel("PC1 (scaled)")
    plt.ylabel("PC2 (scaled)")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.show()

    # 10) פלוט תלת-ממדי: תצוגת הקלאסטרינג על PC1, PC2, PC3
    from mpl_toolkits.mplot3d import Axes3D  # ליצירת גרף תלת-ממדי

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
        f"3D KMeans Clustering\nn_clusters={best_params['n_clusters']}\nSilhouette Score: {best_score:.3f}"
    )
    ax.set_xlabel("PC1 (scaled)")
    ax.set_ylabel("PC2 (scaled)")
    ax.set_zlabel("PC3 (scaled)")
    fig.colorbar(sc, ax=ax, label="Cluster Label")
    plt.tight_layout()
    plt.show()

    # 11) פלוט מטריצת פיזור (Scatter Matrix) עבור זוגי PC1 עד PC4
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
            "/storage/yovelg/Grape/clastering/data/signatures_4_PCA_merged_grapes.csv"
        )
        print(f"No directory argument provided, defaulting to: {csv_input}")
    main(csv_input)
