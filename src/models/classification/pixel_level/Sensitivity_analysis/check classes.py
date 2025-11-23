# count_classes.py
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# --- 1) טעינת קובץ ההגדרות והנתיבים (כמו אצלך) ---
load_dotenv(dotenv_path=os.getenv('ENV_PATH', r'/.env'))
BASE_PATH = Path(os.getenv('BASE_PATH', ''))

# ננסה קודם את DATASET_FOR_TRAIN_PATH כמו בקובץ האוטואנקודר, ואם לא קיים נעבור ל-DATASET_LDA_PATH
DATA_PATH = os.getenv('DATASET_FOR_TRAIN_PATH') or os.getenv('DATASET_LDA_PATH')
if not DATA_PATH:
    raise RuntimeError("לא נמצא DATASET_FOR_TRAIN_PATH ולא DATASET_LDA_PATH בקובץ ה-.env")
CSV_PATH = BASE_PATH / DATA_PATH

RESULT_DIR = BASE_PATH / "RESULT_LOGO"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📄 Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# --- 2) חילוץ cluster_id כמו בקוד שלך ---
def extract_cluster_id(hs_dir: str) -> str:
    try:
        # לוקח את התיקייה השלישית מהסוף כפי שעשית
        return Path(hs_dir).parts[-3]
    except Exception:
        return "unknown"

if "cluster_id" not in df.columns:
    if "hs_dir" not in df.columns:
        raise RuntimeError("העמודה 'hs_dir' לא קיימת בקובץ ולא ניתן לחלץ cluster_id")
    df["cluster_id"] = df["hs_dir"].apply(extract_cluster_id)

# --- 3) בדיקת קיום label ---
if "label" not in df.columns:
    raise RuntimeError("לא נמצאה עמודה בשם 'label' (אמור להיות 0=Regular, 1=Cracked)")

# --- 4) ספירה גלובלית של המחלקות ---
label_counts = df["label"].value_counts().sort_index()  # 0 ואז 1
total = int(label_counts.sum())
print("\n=== Global class counts ===")
for cls, cnt in label_counts.items():
    pct = 100.0 * cnt / total if total > 0 else 0.0
    print(f"class {cls}: {cnt} ({pct:.2f}%)")
print(f"TOTAL: {total}")

# --- 5) ספירה לפי Cluster × Class ושמירה ל-CSV ---
counts_by_cluster = (
    df.groupby(["cluster_id", "label"])
      .size()
      .unstack(fill_value=0)
      .rename(columns={0: "count_class_0", 1: "count_class_1"})
      .sort_index()
)
counts_by_cluster["total"] = counts_by_cluster["count_class_0"] + counts_by_cluster["count_class_1"]
out_csv = RESULT_DIR / "cluster_class_counts.csv"
counts_by_cluster.to_csv(out_csv, index=True)
print(f"\n💾 Saved per-cluster class counts to: {out_csv}")

# --- 6) דיאגנוסטיקה: כמה קלסטרים חסרי מחלקה 0/1 ---
no_class_0 = (counts_by_cluster["count_class_0"] == 0).sum()
no_class_1 = (counts_by_cluster["count_class_1"] == 0).sum()
print("\n=== Diagnostics (per cluster) ===")
print(f"Clusters with ZERO class-0 samples: {no_class_0}")
print(f"Clusters with ZERO class-1 samples: {no_class_1}")

# אופציונלי: להציג 5 שורות ראשונות לבדיקה
print("\nHead of per-cluster table:")
print(counts_by_cluster.head())
