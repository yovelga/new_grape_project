
import os

base = r"C:\Users\yovel\Desktop\Grape_Project\data\raw"
date_folder = "25.09.24"

paths = [os.path.join(base, f"2_{i:02d}", date_folder) for i in range(1, 61)]

# Print them
for p in paths:
    print(p)

# Uncomment to create folders
# for p in paths:
#     os.makedirs(p, exist_ok=True)
