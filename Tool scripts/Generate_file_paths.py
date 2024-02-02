import os
import csv

def get_image_paths(root_directory):

    """Gets the absolute paths of all images in a directory and its subdirectories

    Args:
        root_directory (str): The root directory to start the search from

    Returns:
        list: List of absolute paths to images
    """

    image_paths = []

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.abspath(os.path.join(root, file))
                image_paths.append(image_path)
    return image_paths

def write_to_csv(file_paths, csv_filename):
    
    """Writes the absolute paths to a csv file

    Args:
        file_paths (list): List of absolute paths to images
        csv_filename (str): The name of the csv file to write to
    """

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Paths'])
        for path in file_paths:
            csv_writer.writerow([path])


if __name__ == "__main__":

    # Specify the results folder, root directory and output file name
    results_folder = "File_paths"
    root_directory_path = 'C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_all_subset_photos'
    csv_filename = results_folder + '/lfw_all_subset_photos.csv'

    # Create a folder to store the results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    absolute_paths = get_image_paths(root_directory_path)
    write_to_csv(absolute_paths, csv_filename)