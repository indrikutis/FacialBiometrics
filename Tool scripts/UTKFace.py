import os
import pandas as pd

# Mapping for gender
gender_mapping = {0: 'Male', 1: 'Female'}

# Mapping for race
race_mapping = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}


def extract_info_from_filename(filename):

    """Extracts information from the filename of an image in the UTKFace dataset.

    Args:
        filename (str): Filename of the image.
    Returns:
        age (int): Age of the person in the image.
    """
    try:

        parts = filename.split('_')

        age = int(parts[0])
        gender = gender_mapping[int(parts[1])]
        race = race_mapping[int(parts[2])]

        # Remove the file extension
        date_time = parts[3].split('.')[0]  

        return age, gender, race, date_time, None  # No error
    except Exception as e:
        return None, None, None, None, str(e)

def process_dataset(dataset_path):

    """Processes the UTKFace dataset and returns a DataFrame containing information about the images.

    Args:
        dataset_path (str): Path to the UTKFace dataset.
    Returns:
        df (DataFrame): DataFrame containing the information about the images in the dataset.
    """

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

            file_path = os.path.join(dataset_path, filename)

            # Extract information from the filename
            age, gender, race, date_time, error_note = extract_info_from_filename(filename)

            # Append information only if no error occurred
            if error_note is None:
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

    """Saves the DataFrame to an Excel file.

    Args:
        dataframe (DataFrame): DataFrame to be saved.
        output_path (str): Path to the output Excel file.
    """

    dataframe.to_excel(output_path, index=False)

if __name__ == "__main__":

    results_folder = "Dataset_info"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set the path to your dataset
    dataset_path = "/zhome/15/a/181503/Indre/Special_course/Datasets/UTKface_part1/"
    output_excel_path = "Dataset_info/UTKFace_dataset_info.xlsx"
        
    # Process the dataset
    dataset_df = process_dataset(dataset_path)

    # Save the DataFrame to Excel
    save_to_excel(dataset_df, output_excel_path)

    print(f"Dataset information saved to {output_excel_path}")
