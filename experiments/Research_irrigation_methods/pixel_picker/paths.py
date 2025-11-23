import pandas as pd

# הגדר את נתיב קובץ האקסל שלך כאן
excel_path = r"C:\Users\yovel\Desktop\Grape_Project\dataset_builder_grapes\Research_irrigation_methods\pixel_picker\first_time_crack.xlsx"
output_txt_path = r"C:\Users\yovel\Desktop\Grape_Project\dataset_builder_grapes\Research_irrigation_methods\pixel_picker\first_time_crack.txt"

# טען את האקסל
df = pd.read_excel(excel_path)

# שלב 1: מצא רק את עמודות התאריכים (אלו שמכילות נקודה בשם)
date_cols = [col for col in df.columns if '.' in str(col) and 'Unnamed' not in str(col)]

lines = []

# שלב 2: עבור כל אשכול, מצא את המיקום שבו יש 30 אחרי 20
for idx, row in df.iterrows():
    grape_id = row['Grape ID']
    values = row[date_cols].tolist()
    for i in range(1, len(values)):
        if values[i] == 20:
            date_30 = date_cols[i+1]
            date_20 = date_cols[i]
            path_30 = fr'C:\Users\yovel\Desktop\Grape_Project\dest\{grape_id}\{date_30}'
            path_20 = fr'C:\Users\yovel\Desktop\Grape_Project\dest\{grape_id}\{date_20}'
            # lines.append(f"{path}\n20 שבוע לפני 30 יש סדק\n")
            lines.append(f"{path_30}\n")
            lines.append(f"{path_20}\n")
            break

# שלב 3: כתוב את הכל לקובץ טקסט
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"נשמר קובץ טקסט ב: {output_txt_path}")
