import os
import pandas as pd
import csv
from itertools import combinations


def extract_age_from_filename(filename):

    """Extracts the age information from the filename of an image in the FGNET dataset.

    Raises:
        ValueError: If no numeric part is found between 'A' and '.jpg' in the filename.

    Args:
        filename (str): Filename of the image.
    Returns:
        age (int): Age of the person in the image.
        error_note (str): Error note if any.
    """

    try:
        # Extract the numeric part between 'A' and '.'
        age_str = ''.join(char for char in filename.split('A')[1].split('.')[0] if char.isdigit())

        if age_str:
            age = int(age_str)
            return age, None 
        else:
            raise ValueError("No numeric part found between 'A' and '.jpg' in the filename")

    except Exception as e:
        return None, str(e)


def process_age_dataset(dataset_path):

    """Processes the FGNET dataset and returns a DataFrame containing information about the images.

    Args:
        dataset_path (str): Path to the FGNET dataset.
    Returns:
        df (DataFrame): DataFrame containing the information about the images in the dataset.
    """

    paths = []
    image_names = []
    ages = []
    notes = []

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(dataset_path, filename)

            # Extract age information from the filename
            age, error_note = extract_age_from_filename(filename)

            paths.append(file_path)
            image_names.append(filename)
            ages.append(age)
            notes.append(error_note)

    # Create a DataFrame
    df = pd.DataFrame({
        'path': paths,
        'image_name': image_names,
        'age': ages,
        'Notes': notes
    })

    return df

def save_to_excel(dataframe, output_path):

    """Saves a DataFrame to an Excel file.

    Args:
        dataframe (DataFrame): DataFrame to be saved.
        output_path (str): Path to save the Excel file.
    """
    dataframe.to_excel(output_path, index=False)


def generate_dataset_info(dataset_path, output_excel_path):
    """Generates information about the FGNET dataset and saves it to an Excel file.

    Args:
        dataset_path (str): Path to the FGNET dataset.
        output_excel_path (str): Path to save the Excel file.  
    """
    results_folder = "Dataset_info"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    age_dataset_df = process_age_dataset(dataset_path)

    # Save the DataFrame to Excel
    save_to_excel(age_dataset_df, output_excel_path)

    print(f"Age dataset information saved to {output_excel_path}")

def get_image_paths(dataset_path):

    """Gets the paths of all the images in the dataset.

    Args:
        dataset_path (str): Path to the dataset.    
    Returns:
        image_paths (list): List of paths of all the images in the dataset.
    """

    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        folder_image_paths = [os.path.join(root, file) for file in image_files]
        image_paths.extend(folder_image_paths)
    return image_paths

def get_image_pairs(image_paths):

    """Generates pairs of image paths from a list of image paths.

    Args:
        image_paths (list): List of image paths.
    Returns:
        image_pairs (list): List of tuples containing pairs of image paths.
    """
    image_pairs = list(combinations(image_paths, 2))
    return image_pairs

def write_to_csv(image_pairs, csv_file):

    """Writes image pairs to a CSV file.

    Args:   
        image_pairs (list): List of tuples containing pairs of image paths.
        csv_file (str): Name of the CSV file.
    """
    folder_name = "Face_recognition_results"
    with open(folder_name + "/" + csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image1', 'image2'])
        csv_writer.writerows(image_pairs)


def generate_image_pairs(dataset_path, output_excel_path):

    """Generates pairs of images from the dataset and writes them to a CSV file.

    Args:
        dataset_path (str): Path to the dataset.
        output_excel_path (str): Path to save the CSV file.

    Returns:
        output_excel_path (str): Path to the CSV file.
    """

    # Get image paths from the dataset
    image_paths = get_image_paths(dataset_path)

    # Extract subject ID from the first 3 characters of the file name
    subject_ids = [os.path.basename(image)[:3] for image in image_paths]

    # Filter images with the same subject ID
    unique_subject_ids = set(subject_ids)
    subject_image_paths = {subject_id: [] for subject_id in unique_subject_ids}

    for subject_id, image_path in zip(subject_ids, image_paths):
        subject_image_paths[subject_id].append(image_path)

    # Generate pairs of image paths for each subject
    all_image_pairs = []
    for subject_paths in subject_image_paths.values():
        image_pairs = get_image_pairs(subject_paths)
        all_image_pairs.extend(image_pairs)

    write_to_csv(all_image_pairs, output_excel_path)
    print(f"Image pairs written to: {output_excel_path}")


if __name__ == "__main__":
    
    # Generate dataset information
    dataset_path = "C:\INDRES\DTU\Semester 3\Special course\Datasets\FGNET\FGNET\images"
    output_excel_path = "Dataset_info/FGNET_dataset_info.xlsx"
    output_excel_path = "FGNET_image_pairs.csv"

    generate_dataset_info(dataset_path, output_excel_path)
    generate_image_pairs(dataset_path, output_excel_path)


