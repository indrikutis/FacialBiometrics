import os
import csv
from itertools import combinations

def get_image_paths(root_folder):
    """Function to get image paths grouped by folder (person)

    Args:
        root_folder (str): Path to the root folder containing images of different persons.

    Returns:
        list: List of lists, where each inner list contains paths to images of a person.
    """
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 1:
            folder_image_paths = [os.path.join(root, file) for file in image_files]
            image_paths.append(folder_image_paths)
    return image_paths

def generate_image_pairs(folder_image_paths):
    """Function to generate pairs of image paths for each folder

    Args:
        folder_image_paths (list): List of paths to images of a person.

    Returns:
        list: List of tuples, where each tuple contains a pair of image paths.
    """

    return list(combinations(folder_image_paths, 2))

def write_to_csv(image_pairs, image_paths, csv_file, folder_name):

    """Function to write image pairs and paths to CSV file and text file

    Args:
        image_pairs (list): List of tuples, where each tuple contains a pair of image paths.
        image_paths (list): List of lists, where each inner list contains paths to images of a person.
        csv_file (str): Path to the CSV file.
        folder_name (str): Name of the folder where the CSV file and text file will be saved.
    """

    with open(folder_name + "/" + csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image1', 'image2'])
        csv_writer.writerows(image_pairs)

    with open(folder_name + '/all_image_paths.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path'])

        for image_path_list in image_paths:
            csv_writer.writerows([[image_path] for image_path in image_path_list])

def generate_and_write_image_pairs(root_folder, csv_file_path, folder_name):

    """Function to generate and write image pairs to a CSV file and text file

    Args:
        root_folder (str): Path to the root folder containing images of different persons.
        csv_file_path (str): Path to the CSV file.
        folder_name (str): Name of the folder where the CSV file and text file will be saved.
    """
    # Get image paths grouped by folder (person)
    folder_image_paths_list = get_image_paths(root_folder)

    # Generate pairs of image paths for each folder
    all_image_pairs = []
    all_image_paths = []
    for folder_image_paths in folder_image_paths_list:
        image_pairs = generate_image_pairs(folder_image_paths)
        all_image_pairs.extend(image_pairs)
        all_image_paths.append(folder_image_paths)

    write_to_csv(all_image_pairs, all_image_paths, csv_file_path, folder_name)
    print(f"Image pairs and paths written to: {csv_file_path} and all_image_paths.txt")


if __name__ == "__main__":
    
    # Specify the root folder of your dataset
    root_folder = "C:/INDRES/DTU/Semester 3/Special course/Datasets/lfw_subset"
    folder_name = "File_paths"
    csv_file_path = "image_pairs_lfw_subset.csv"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Call the function to generate and write image pairs
    generate_and_write_image_pairs(root_folder, csv_file_path, folder_name)
