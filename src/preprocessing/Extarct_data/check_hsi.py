import os
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
from tqdm import tqdm  # For progress tracking


def extract_tags_by_date(base_path, dest_path):
    tag_data_by_date = defaultdict(
        lambda: defaultdict(int)
    )  # Dictionary to store tag counts by date

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
            # Loop through the folders in the HS directory
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
                        tag_value = None

                        # Search for both <global_tag> and <material_tag>
                        for tag_type in ["global_tag", "material_tag"]:
                            for tag in root.findall(tag_type):
                                for key in tag.findall("key"):
                                    tag_value = key.attrib.get("field")
                                    if tag_value:
                                        break
                            if tag_value:  # Exit the loop if we found a value
                                break

                        # If we found a tag value, count it for the current date
                        if tag_value:
                            formatted_date = date_folder.replace("-", ".")
                            tag_data_by_date[tag_value][formatted_date] += 1

                    except ET.ParseError:
                        print(f"Error parsing XML file: {xml_file_path}")
                        pass

    return tag_data_by_date


def save_transposed_tags_to_csv(tag_data_by_date, dest_path):
    # Collect all unique dates from the tag data
    all_dates = set()
    for tags in tag_data_by_date.values():
        all_dates.update(tags.keys())
    all_dates = sorted(all_dates)  # Sort the dates

    # Prepare CSV file path
    csv_file_path = os.path.join(dest_path, "HSI_tags.csv")

    # Write the transposed tag data to a CSV file
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header (tag/date and all dates)
        header = ["tag\\date"] + all_dates
        writer.writerow(header)

        # Write rows for each tag and its corresponding counts for each date
        for tag, dates in tag_data_by_date.items():
            row = [tag] + [
                dates.get(date, 0) for date in all_dates
            ]  # Write 0 if no count for the tag on the date
            writer.writerow(row)

    print(f"Transposed CSV with tags saved to {csv_file_path}")


# Example usage:
source_path = "D:\\raw_data"
destination_path = "D:\\dest"

# Extract tags by date and handle duplicates
tag_data_by_date = extract_tags_by_date(source_path, destination_path)

# Save the transposed data to a CSV file
save_transposed_tags_to_csv(tag_data_by_date, destination_path)
