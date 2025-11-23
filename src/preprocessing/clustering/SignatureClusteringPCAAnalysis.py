import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
#  פונקציה לחלקת ספקטרום באמצעות ממוצע נע
# ---------------------------------------------------------
def smooth_stamp_moving_average(norm_stamp, kernel_size=10):
    smoothed = uniform_filter(norm_stamp, size=kernel_size)
    return smoothed


# ---------------------------------------------------------
#  פונקציה לקריאת CSV והחזרת מידע
# ---------------------------------------------------------
def gather_signatures_from_csv(csv_path):
    """
    מניח שיש בקובץ ה-CSV עמודות: id, date, x, y, band_1, band_2, ...
    """
    df = pd.read_csv(csv_path)

    # עמודות ספקטרליות
    band_cols = [col for col in df.columns if col.startswith("band_")]

    # שימוש בעמודות הרפרנס: id, date, x, y
    reference_cols = ["id", "date", "x", "y"]
    for col in reference_cols:
        if col not in df.columns:
            print(
                f"Warning: Column '{col}' not found in {csv_path}. Check your CSV structure."
            )

    # נתוני העמודות הספקטרליות כמערך NumPy
    spectral_data = df[band_cols].values

    # שמירת מידע על הפיקסל/תמונה/קואורדינטות כ-DataFrame
    references_df = df[reference_cols].copy()

    return spectral_data, references_df


# ---------------------------------------------------------
#  פונקציה ראשית
# ---------------------------------------------------------
def main(csv_path_or_dir):
    # 1) איסוף חתימות ספקטרליות מכל קבצי ה-CSV
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

    all_signatures_list = []
    all_references_list = []

    for csv_file in csv_files:
        spectral_data, refs_df = gather_signatures_from_csv(csv_file)
        all_signatures_list.append(spectral_data)
        all_references_list.append(refs_df)

    # מאחדים את כל החתימות הספקטרליות למערך אחד
    all_signatures = np.vstack(all_signatures_list)
    # מאחדים את כל ה-DataFrame של הרפרנסים ל-DataFrame אחד
    all_references = pd.concat(all_references_list, ignore_index=True)

    print("Total signatures gathered:", all_signatures.shape[0])

    # 2) החלקת כל חתימה (Moving Average)
    smoothed_sigs = np.apply_along_axis(
        lambda row: smooth_stamp_moving_average(row, kernel_size=10),
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

    # 5) חישוב Silhouette Score עבור מגוון ערכי K
    k_values = range(2, 11)  # ניתן לשנות את הגבולות לפי הצורך
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pca_scaled)
        # יש צורך בלפחות 2 קבוצות תקפות לחישוב Silhouette
        if len(set(labels)) > 1:
            score = silhouette_score(pca_scaled, labels)
        else:
            score = -1  # אם הכל בקלאסטר אחד
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score:.4f}")

    # 6) פלוט הגרף של K מול ערך ה-Silhouette
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, silhouette_scores, marker="o", linestyle="-")
    plt.title("Silhouette Score vs. Number of Clusters (K)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7) בחירת K הטוב ביותר בעל הציון הגבוה ביותר
    best_score = max(silhouette_scores)
    best_k = k_values[silhouette_scores.index(best_score)]
    print(f"\nBest K according to Silhouette Score: {best_k} (Score={best_score:.4f})")

    # 8) הרצה סופית של KMeans עם K הטוב ביותר
    best_kmeans = KMeans(n_clusters=best_k, random_state=42)
    best_labels = best_kmeans.fit_predict(pca_scaled)
    unique_labels = np.unique(best_labels)
    print(f"\nKMeans found {len(unique_labels)} clusters with K={best_k}.")
    for cluster_label in unique_labels:
        count = np.sum(best_labels == cluster_label)
        print(f"  Cluster {cluster_label}: {count} items")

    # 9) פלוט 2D: תצוגת הקלאסטרינג על PC1 ו-PC2
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(
        pca_scaled[:, 0], pca_scaled[:, 1], c=best_labels, cmap="viridis", s=5
    )
    plt.title(
        f"KMeans Clustering\nn_clusters={best_k}, Silhouette Score={best_score:.3f}"
    )
    plt.xlabel("PC1 (scaled)")
    plt.ylabel("PC2 (scaled)")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.show()

    # 10) פלוט תלת-ממדי: תצוגת הקלאסטרינג על PC1, PC2, PC3
    from mpl_toolkits.mplot3d import Axes3D

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
        f"3D KMeans Clustering\nn_clusters={best_k}, Silhouette Score={best_score:.3f}"
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
        if i >= pca_scaled.shape[1] or j >= pca_scaled.shape[1]:
            continue
        ax = axes.flat[idx]
        ax.scatter(
            pca_scaled[:, i], pca_scaled[:, j], c=best_labels, cmap="viridis", s=5
        )
        ax.set_xlabel(pc_names[i])
        ax.set_ylabel(pc_names[j])
        ax.set_title(f"{pc_names[i]} vs {pc_names[j]}")
    plt.tight_layout()
    plt.show()

    # 12) הצגת דוגמאות (פיקסלים) מכל קלאסטר
    import random

    print("\n=== Sample References from Each Cluster ===")
    for cluster_label in unique_labels:
        # אינדקסים של הדגימות השייכות לאותו קלאסטר
        cluster_indices = np.where(best_labels == cluster_label)[0]

        # אם יש פחות מ-10, ניקח את כולן; אחרת ניקח 10 אקראיות
        sample_count = min(len(cluster_indices), 10)
        chosen_indices = random.sample(list(cluster_indices), sample_count)

        print(f"\nCLUSTER {cluster_label}: showing {sample_count} samples")
        for idx_sample in chosen_indices:
            row_info = all_references.iloc[idx_sample]
            # בניית הנתיב לפי עמודות id ו-date
            img_id = str(row_info.get("id", "no_id"))
            img_date = str(row_info.get("date", "no_date"))
            # הנתיב הבסיסי שאנו בונים:
            img_path = os.path.join("D:", "dest", img_id, img_date)
            x_coord = row_info.get("x", "no_x")
            y_coord = row_info.get("y", "no_y")
            print(f"  Image: {img_path}, x={x_coord}, y={y_coord}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_input = sys.argv[1]
    else:
        csv_input = (
            "/storage/yovelg/Grape/clustering/data/signatures_4_PCA_merged_grapes.csv"
        )
        print(f"No directory argument provided, defaulting to: {csv_input}")
    main(csv_input)
