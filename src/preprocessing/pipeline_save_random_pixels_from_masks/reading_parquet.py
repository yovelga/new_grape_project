import pandas as pd


def main():
    parquet_path = "/storage/yovelg/Grape/spectral_anomaly/checks/output/parquet/2024-09-01/signatures_all_2024-09-01_1_01.parquet"

    try:
        # קביעת תצוגה מלאה
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_rows", 1)

        df = pd.read_parquet(parquet_path)
        print("✅ Loaded Parquet file successfully.")
        print(df[-5:])
    except Exception as e:
        print(f"❌ Failed to load file: {e}")


if __name__ == "__main__":
    main()
