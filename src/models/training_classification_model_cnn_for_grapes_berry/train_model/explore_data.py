import matplotlib.pyplot as plt
from dataset_multi import GrapeDataset
from config import TRAIN_DIR

# נגדיר טרנספורם זהות – ללא שינוי
identity = lambda x: x

# טוענים את דאטה סט האימון במצב input_mode="all" (כל בסיס מתורגם ל-3 דוגמאות)
dataset = GrapeDataset(
    TRAIN_DIR, transform=identity, balance_mode=None, input_mode="all"
)

# מספר הדגימות הבסיסיות הוא len(dataset)//3
base_count = len(dataset) // 3
print(f"Number of base samples: {base_count}")

# מילון לאיסוף דוגמאות: עבור כל קלאס, עבור כל אפשרות
samples_by_class = {
    0: {"original": [], "enlarged": [], "segmentation": []},
    1: {"original": [], "enlarged": [], "segmentation": []},
}

# איסוף דוגמאות – 3 לכל אפשרות לכל קלאס
# נניח שכל 3 דגימות עוקבות נוצרו מאותה דגימה בסיסית
for i in range(base_count):
    # קבלת שלושת הדגימות עבור הדגימה הבסיסית i
    img_orig, label = dataset[i * 3]  # modality 0: original crop
    img_enlarged, _ = dataset[i * 3 + 1]  # modality 1: enlarged crop
    img_seg, _ = dataset[i * 3 + 2]  # modality 2: segmentation overlay

    # לאסוף רק אם עדיין לא נאספו 3 דוגמאות עבור אפשרות זו עבור הקלאס
    if len(samples_by_class[label]["original"]) < 3:
        samples_by_class[label]["original"].append(img_orig)
    if len(samples_by_class[label]["enlarged"]) < 3:
        samples_by_class[label]["enlarged"].append(img_enlarged)
    if len(samples_by_class[label]["segmentation"]) < 3:
        samples_by_class[label]["segmentation"].append(img_seg)

    # בדיקה אם נאספו מספיק דוגמאות עבור שני הקלאסים
    if (
        len(samples_by_class[0]["original"]) >= 3
        and len(samples_by_class[0]["enlarged"]) >= 3
        and len(samples_by_class[0]["segmentation"]) >= 3
        and len(samples_by_class[1]["original"]) >= 3
        and len(samples_by_class[1]["enlarged"]) >= 3
        and len(samples_by_class[1]["segmentation"]) >= 3
    ):
        break

# שמות להצגה
possibility_names = {
    "original": "Original Crop",
    "enlarged": "Enlarged Crop",
    "segmentation": "Segmentation Overlay",
}
class_names = {0: "Not Grape", 1: "Grape"}

# ניצור גריד של 6 שורות (2 קלאסים * 3 אפשרויות) ו-3 עמודות (3 דוגמאות לכל אפשרות)
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(12, 18))
row = 0
for label in [0, 1]:
    for poss in ["original", "enlarged", "segmentation"]:
        for col in range(3):
            ax = axes[row, col]
            # נניח שהתמונות הן בפורמט PIL – matplotlib יודעת להציגן ישירות
            ax.imshow(samples_by_class[label][poss][col])
            ax.axis("off")
            if col == 1:
                ax.set_title(
                    f"{class_names[label]} - {possibility_names[poss]}", fontsize=12
                )
        row += 1

plt.tight_layout()
plt.show()
