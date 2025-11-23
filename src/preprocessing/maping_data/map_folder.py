import os


def map_folders(
    root_dir, output_file="/storage/yovelg/Grape/spectral_anomaly/folder_structure.txt"
):
    """
    סורק את תיקיית root_dir באופן רקורסיבי וכותב את מבנה התיקיות לקובץ טקסט.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for current_path, dirs, files in os.walk(root_dir):
            # חשב את העומק בעץ התיקיות
            depth = current_path.replace(root_dir, "").count(os.sep)
            indent = "    " * depth  # רווחים כ-indentation
            folder_name = os.path.basename(current_path)
            f.write(f"{indent}- {folder_name}/\n")
            # כתוב את הקבצים
            for filename in files:
                f.write(f"{indent}    {filename}\n")
    print(f"Folder structure saved to {output_file}")


def main():
    # הנתיב הראשי של הנתונים
    root_dir = "/storage/yovelg/Grape/data"
    # שם הקובץ שבו יישמר מבנה התיקיות
    output_file = "folder_structure.txt"
    map_folders(root_dir, output_file)


if __name__ == "__main__":
    main()
