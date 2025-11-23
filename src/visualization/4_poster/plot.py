import pandas as pd
import matplotlib.pyplot as plt
from wavelengths import WAVELENGTHS


def normalize_and_plot(file_path, title):
    # טוען את הנתונים
    df = pd.read_csv(file_path)

    # מזהה את עמודות ה-Band
    band_columns = [col for col in df.columns if col.startswith("Band_")]

    # נרמול מינ-מקס לכל שורה בנפרד
    df[band_columns] = df[band_columns].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=1
    )

    # חישוב ממוצע לכל Band
    mean_values = df[band_columns].mean()

    # יצירת גרף עם אורכי גל בציר X
    wavelengths = [
        WAVELENGTHS[i + 1] for i in range(len(band_columns))
    ]  # המרה לאורכי גל

    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, mean_values, marker="o", linestyle="-", color="b")
    plt.xlabel("Wavelength (nm)", fontsize=18)
    plt.ylabel("Mean Reflectance", fontsize=18)
    plt.title(title, fontsize=18)

    plt.grid()
    plt.show()


# דוגמה להפעלת הפונקציה עם שם קובץ
file_path_craks = r"C:\Users\yovel\PycharmProjects\pixel_picker\Crack\detected_pixels.csv"  # עדכן לנתיב הנכון
file_path_no_craks = r"C:\Users\yovel\PycharmProjects\pixel_picker\Not crack\detected_pixels.csv"  # עדכן לנתיב הנכון
normalize_and_plot(
    file_path_craks,
    "Reflectance percentage by channel after MIN MAX Normalization for Cracked Grapes Pixels",
)
normalize_and_plot(
    file_path_no_craks,
    "Reflectance percentage by channel after MIN MAX Normalization for Regular Grapes Pixels",
)
