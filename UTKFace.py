import os
import pandas as pd

# Mapping for gender
gender_mapping = {0: 'Male', 1: 'Female'}

# Mapping for race
race_mapping = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}


def extract_info_from_filename(filename):
    try:
        # Split the filename using underscores
        parts = filename.split('_')

        # Extract information
        age = int(parts[0])
        gender = gender_mapping[int(parts[1])]
        race = race_mapping[int(parts[2])]

        # Extract date and time information
        date_time = parts[3].split('.')[0]  # Remove the file extension

        return age, gender, race, date_time, None  # No error
    except Exception as e:
        return None, None, None, None, str(e)

def process_dataset(dataset_path):
    # Initialize lists to store extracted information
    paths = []
    image_names = []
    ages = []
    genders = []
    races = []
    date_times = []
    notes = []

    # Iterate over files in the dataset
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            # Construct full file path
            file_path = os.path.join(dataset_path, filename)

            # Extract information from the filename
            age, gender, race, date_time, error_note = extract_info_from_filename(filename)

            # Append information to lists
            if error_note is None:
                # Append information only if no error occurred
                paths.append(file_path)
                image_names.append(filename)
                ages.append(age)
                genders.append(gender)
                races.append(race)
                date_times.append(date_time)
            else:
                # Append None for other fields if an error occurred
                paths.append(file_path)
                image_names.append(filename)
                ages.append(None)
                genders.append(None)
                races.append(None)
                date_times.append(None)

            notes.append(error_note)

    # Create a DataFrame
    df = pd.DataFrame({
        'path': paths,
        'image_name': image_names,
        'age': ages,
        'gender': genders,
        'race': races,
        'date_time': date_times,
        'Notes': notes
    })

    return df


def save_to_excel(dataframe, output_path):
    dataframe.to_excel(output_path, index=False)

if __name__ == "__main__":

    results_folder = "UTKFace"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set the path to your dataset
    dataset_path = "C:/INDRES/DTU/Semester 3/Special course/Datasets/UTKface/part1/"

    # Process the dataset
    dataset_df = process_dataset(dataset_path)

    # Set the output Excel file path
    output_excel_path = "UTKFace/UTKFace_dataset_info.xlsx"

    # Save the DataFrame to Excel
    save_to_excel(dataset_df, output_excel_path)

    print(f"Dataset information saved to {output_excel_path}")
