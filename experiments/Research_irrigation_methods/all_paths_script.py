from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
import os
dates = [
    "01.07.24",
    "10.07.24",
    "15.07.24",
    # "25.07.24",
    # "25.07.24",
    # "01.08.24",
    # "08.08.24",
    # "16.08.24",
    # "22.08.24",
    # "01.09.24",
    # "05.09.24",
    # "11.09.24",
    # "18.09.24",
    # "25.09.24",
]

base_path = r"/"
# C:\Users\yovel\OneDrive\Desktop\Grape_Project\dest\1_14\01.09.24
paths = []

# CLUSTER IDs 01_01 -> 01_60
for i in range(1, 61):
    cluster_id = f"1_{i:02d}"
    for date in dates:
        # path = rf"{base_path}\{cluster_id}\{date}"
        path = os.path.join(base_path, "dest", cluster_id, date)
        # path = os.path.join(base_path, "dest", cluster_id, date, "HS",',')

        paths.append(path)

# CLUSTER IDs 02_01 -> 02_60
for i in range(1, 61):
    cluster_id = f"2_{i:02d}"
    for date in dates:
        path = os.path.join(base_path, "dest", cluster_id, date)
        paths.append(path)

# שמירה לקובץ
with open("all_paths_10_08.txt", "w", encoding="utf-8") as f:
    for path in paths:
        f.write(path + "\n")

print(f"CREATED {len(paths)} PATHS.")

