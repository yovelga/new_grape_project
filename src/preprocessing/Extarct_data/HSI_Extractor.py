import os
import shutil
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm  # Import tqdm for progress tracking


def extract_HS_folders_by_date_and_tag(base_path, dest_path):
    summary_data = {}  # Dictionary to track success/missing status by id and date
    expected_tags = [f"1_{i:02d}" for i in range(1, 61)] + [
        f"2_{i:02d}" for i in range(1, 61)
    ]  # 1_01 to 1_60, 2_01 to 2_60

    # Get all the date folders in the source directory
    date_folders = [
        f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
    ]

    # Create an outer progress bar for dates
    for date_folder in tqdm(date_folders, desc="Processing dates", unit="date"):
        date_path = os.path.join(base_path, date_folder)

        hs_path = os.path.join(date_path, "HS")

        # Check if 'HS' folder exists
        if os.path.exists(hs_path) and os.path.isdir(hs_path):

            present_tags = []  # Track which folders are present

            # Create a progress bar for the folders in each date
            image_folders = os.listdir(hs_path)
            for image_folder in tqdm(
                image_folders,
                desc=f"Processing folders in {date_folder}",
                unit="folder",
                leave=False,
            ):
                image_folder_path = os.path.join(hs_path, image_folder, "metadata")
                xml_file_path = os.path.join(image_folder_path, f"{image_folder}.xml")

                if os.path.isfile(xml_file_path):
                    try:
                        tree = ET.parse(xml_file_path)
                        root = tree.getroot()
                        global_tag_value = None
                        for tag_type in ["global_tag", "material_tag"]:
                            for tag in root.findall(tag_type):
                                for key in tag.findall("key"):
                                    global_tag_value = key.attrib.get("field")
                                    if global_tag_value:
                                        break
                            if global_tag_value:
                                break

                        if global_tag_value and (
                            "0101" <= global_tag_value <= "0160"
                            or "0201" <= global_tag_value <= "0260"
                        ):

                            formatted_date = date_folder.replace("-", ".")
                            tag_prefix = (
                                "1_" if "0101" <= global_tag_value <= "0160" else "2_"
                            )
                            tag_folder_name = tag_prefix + global_tag_value[-2:]
                            present_tags.append(tag_folder_name)

                            destination_folder = os.path.join(
                                dest_path, tag_folder_name, formatted_date, "HS"
                            )
                            src_folder_path = os.path.join(hs_path, image_folder)

                            try:
                                if not os.path.exists(destination_folder):
                                    os.makedirs(destination_folder)
                                for item in os.listdir(src_folder_path):
                                    s = os.path.join(src_folder_path, item)
                                    d = os.path.join(destination_folder, item)
                                    if os.path.isdir(s):
                                        shutil.copytree(s, d, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(s, d)
                                summary_data.setdefault(tag_folder_name, {}).update(
                                    {formatted_date: "succeeded"}
                                )

                            except (shutil.Error, OSError):
                                # We are only recording success or missing, so ignore errors
                                pass

                    except ET.ParseError:
                        # Ignore XML parsing errors, don't record them
                        pass

        # For all missing tags, log as missing
        missing_tags = set(expected_tags) - set(present_tags)
        formatted_date = date_folder.replace("-", ".")
        for missing_tag in missing_tags:
            summary_data.setdefault(missing_tag, {}).update({formatted_date: "missing"})

    return summary_data


def save_summary_to_csv(summary_data, dest_path):
    # Collect all unique dates from the summary data
    all_dates = set()
    for tag, date_info in summary_data.items():
        all_dates.update(date_info.keys())
    all_dates = sorted(all_dates)  # Sort the dates

    # Prepare CSV file path
    csv_file_path = os.path.join(dest_path, "HSI_extraction_summery2.csv")

    # Write the summary data to a CSV file
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header (id/date and all dates)
        header = ["id\\date"] + all_dates
        writer.writerow(header)

        # Write rows for each tag (id/date) and its corresponding success/missing status for each date
        for tag, date_info in summary_data.items():
            row = [tag] + [
                date_info.get(date, "missing") for date in all_dates
            ]  # Use 'missing' if no data for the date
            writer.writerow(row)

    print(f"CSV summary saved to {csv_file_path}")


# Example usage:
source_path = r"C:\Users\yovel\Desktop\row data"
destination_path = r"D:\dest"

# Perform the extraction and copy process
summary_data = extract_HS_folders_by_date_and_tag(source_path, destination_path)

# Save the summary to a CSV file
save_summary_to_csv(summary_data, destination_path)
