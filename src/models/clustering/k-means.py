import pandas as pd
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# הגדרת הנתיב שבו שמורים קבצי ה-Parquet
# DATA_PATH = "/storage/yovelg/Grape/spectral_anomaly/checks/output/parquet/2024-09-01/*.parquet"
# print(DATA_PATH)


def load_all_parquet_files(path_pattern):
    """
    טוען את כל קבצי ה-Parquet ל-DataFrame אחד,
    מסיר עמודות לא רלוונטיות (date, id, x, y, mask_file),
    ומחזיר רק את עמודות ה-band.
    """
    all_files = glob.glob(path_pattern)
    dfs = []
    for file in all_files:
        df_temp = pd.read_parquet(file)

        # עמודות שאינן נחוצות
        cols_to_drop = ["date", "id", "x", "y", "mask_file"]
        band_cols = [col for col in df_temp.columns if col not in cols_to_drop]

        df_temp = df_temp[band_cols]
        dfs.append(df_temp)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def evaluate_k_means(data, k_range=range(2, 550)):
    """
    מבצע K-means על טווח k (k_range),
    מחשב את סכום ריבועי המרחקים (SSE) עבור Elbow Method,
    ואת ה-Silhouette Score עבור כל K.
    מחזיר מילון עם התוצאות.
    """
    results = {"k": [], "sse": [], "silhouette": []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        # SSE: Sum of squared distances of samples to their closest cluster center
        sse = kmeans.inertia_

        # silhouette score (רצוי k>=2)
        labels = kmeans.labels_
        sil_score = silhouette_score(data, labels)

        results["k"].append(k)
        results["sse"].append(sse)
        results["silhouette"].append(sil_score)

    return results


def main():
    df = pd.read_csv(
        r"D:\Grape_Project\dataset_builder_grapes\dataset\combined_cleaned_signatures.csv"
    )

    data_array = df.to_numpy(dtype=np.float32)

    # נבדוק K מ-2 ועד 10 (ניתן לשנות)
    k_values = range(2, 20)

    results = evaluate_k_means(data_array, k_values)

    # מציגים גרף Elbow (SSE)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results["k"], results["sse"], marker="o")
    plt.title("Elbow Method (SSE) for K-Means")
    plt.xlabel("K")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.grid(True)

    # מציגים גרף Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(results["k"], results["silhouette"], marker="o", color="red")
    plt.title("Silhouette Score for K-Means")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # הדפסה טבלאית של התוצאות
    for k, sse, sil in zip(results["k"], results["sse"], results["silhouette"]):
        print(f"K={k}, SSE={sse:.2f}, Silhouette={sil:.3f}")


if __name__ == "__main__":
    main()
