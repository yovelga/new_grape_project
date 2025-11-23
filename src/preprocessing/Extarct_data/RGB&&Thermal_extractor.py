import os
import shutil
import pandas as pd
import sys
from tqdm import tqdm  # Import tqdm for progress bar


# Function to get directory path from user input
def choose_path(prompt):
    path = input(f"{prompt}: ")
    if os.path.exists(path):
        # print(f'OK: {path}')
        return path
    else:
        # print('ERROR: Invalid path')
        sys.exit()


# Function to extract folder names (dates)
def extract_folders_name(path):
    # print(f"Extracting folder names (dates) from: {path}")
    folders_name = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):  # Only take directories (dates)
            folders_name.append(item)
    folder_df = pd.DataFrame(folders_name, columns=["Folder Name"])
    # print(f"Found {len(folders_name)} date folders.")
    return folder_df


# Function to read the Excel or CSV file
def read_excel_or_csv(path):
    # print(f"Looking for Excel/CSV file in: {path}")
    for item in os.listdir(path):
        if item.endswith(".csv"):
            # print(f"Found CSV file: {item}")
            return pd.read_csv(
                os.path.join(path, item),
                dtype={"Thermo ID  (4 Digits)": str, "RGB  ID  (4 Digits)": str},
            )
        elif item.endswith(".xlsx"):
            # print(f"Found Excel file: {item}")
            return pd.read_excel(
                os.path.join(path, item),
                dtype={"Thermo ID  (4 Digits)": str, "RGB  ID  (4 Digits)": str},
            )
    # print('No valid CSV or Excel file found.')
    sys.exit()


# Function to iterate over DataFrame and get grape, thermo, and RGB IDs
def extract_ids_from_df(df):
    # print(f"Extracting image IDs from data...")
    grapes_id = df["Grape ID"]
    thermos_id = df["Thermo ID  (4 Digits)"]
    rgbs_id = df["RGB  ID  (4 Digits)"]
    # print(f"Extracted {len(grapes_id)} grape clusters.")
    return grapes_id, thermos_id, rgbs_id


# Function to copy images from source to destination
def copy_image(source_path, destination_path):
    try:
        # Create destination directory if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        # Copy the file to the destination directory
        shutil.copy(source_path, destination_path)
        # print(f"Copied {os.path.basename(source_path)} to {destination_path}")
    except Exception as e:
        pass
        # print(f"Error occurred while copying image: {e}")


# Function to extract and copy images based on the IDs and date
def extract_images(
    date, source, dest, grapes_id, thermos_id, rgbs_id, rgb_summary, thermal_summary
):
    for index in tqdm(range(len(grapes_id)), desc=f"Processing Grape IDs for {date}"):
        grape_id = grapes_id[index]
        thermo_id = str(thermos_id[index])
        rgb_id = str(rgbs_id[index])

        # Reset counters for this grape ID (for each new grape)
        rgb_count = 0
        thermo_count = 0

        # Copy RGB images
        rgb_source_path = os.path.join(source, date, "RGB")
        for img in os.listdir(rgb_source_path):
            if img.endswith(f"{rgb_id}.JPG"):  # Assuming images are .JPG
                destination_folder = os.path.join(dest, grape_id, date)
                copy_image(
                    os.path.join(rgb_source_path, img),
                    os.path.join(destination_folder, "RGB"),
                )
                rgb_count += 1

        # Copy Thermic images
        thermo_source_path = os.path.join(source, date, "Thermic")
        for img in os.listdir(thermo_source_path):
            file_id = img.split("_")[-1].split(".")[0]
            if file_id == thermo_id:
                destination_folder = os.path.join(dest, grape_id, date)
                copy_image(
                    os.path.join(thermo_source_path, img),
                    os.path.join(destination_folder, "Thermal"),
                )
                thermo_count += 1

        # Append results for this grape ID to the RGB summary list
        rgb_summary.append({"Grape ID": grape_id, "Date": date, "Count": rgb_count})

        # Append results for this grape ID to the Thermal summary list
        thermal_summary.append(
            {"Grape ID": grape_id, "Date": date, "Count": thermo_count}
        )


# Main flow of the program
def main():
    # Select base source directory
    base_path = r"C:\Users\yovel\Desktop\row data"
    destination_path = r"D:\dest"

    # Initialize the summary lists for RGB and Thermal
    rgb_summary = []
    thermal_summary = []

    # Extract folder names (dates)
    folders_df = extract_folders_name(base_path)

    # Loop over each date (folder name) with tqdm progress bar
    for date in tqdm(folders_df["Folder Name"], desc="Processing Dates"):
        # Path to the folder of the current date
        date_folder_path = os.path.join(base_path, date)

        # Read the corresponding CSV/Excel file for that date
        data_df = read_excel_or_csv(date_folder_path)

        # Extract grape, thermo, and RGB IDs from the dataframe
        grapes_id, thermos_id, rgbs_id = extract_ids_from_df(data_df)

        # Process and copy images for the current date, and track summary
        extract_images(
            date,
            base_path,
            destination_path,
            grapes_id,
            thermos_id,
            rgbs_id,
            rgb_summary,
            thermal_summary,
        )

    # Convert the summary to dataframes
    rgb_df = pd.DataFrame(rgb_summary)
    thermal_df = pd.DataFrame(thermal_summary)

    # Pivot the data to get the dates as columns
    rgb_pivot = rgb_df.pivot(index="Grape ID", columns="Date", values="Count").fillna(0)
    thermal_pivot = thermal_df.pivot(
        index="Grape ID", columns="Date", values="Count"
    ).fillna(0)

    # Save the pivoted data to separate CSV files
    rgb_pivot.to_csv(os.path.join(destination_path, "summary_rgb.csv"))
    thermal_pivot.to_csv(os.path.join(destination_path, "summary_thermal.csv"))

    print("RGB Summary saved to 'summary_rgb.csv'")
    print("Thermal Summary saved to 'summary_thermal.csv'")


if __name__ == "__main__":
    main()
