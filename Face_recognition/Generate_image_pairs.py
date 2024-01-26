import os
import csv
from itertools import combinations

def get_image_paths(root_folder):
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 1:
            folder_image_paths = [os.path.join(root, file) for file in image_files]
            image_paths.append(folder_image_paths)
    return image_paths

def generate_image_pairs(folder_image_paths):
    return list(combinations(folder_image_paths, 2))

def write_to_csv(image_pairs, image_paths, csv_file, folder_name):
    with open(folder_name + "/" + csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image1', 'image2'])
        csv_writer.writerows(image_pairs)

    # Save paths to all images used
    with open(folder_name + '/all_image_paths.txt', 'w') as file:
        for image_path_list in image_paths:
            file.write('\n'.join(image_path_list) + '\n')

def generate_and_write_image_pairs(root_folder, csv_file_path, folder_name):
    # Get image paths grouped by folder (person)
    folder_image_paths_list = get_image_paths(root_folder)

    # Generate pairs of image paths for each folder
    all_image_pairs = []
    all_image_paths = []
    for folder_image_paths in folder_image_paths_list:
        image_pairs = generate_image_pairs(folder_image_paths)
        all_image_pairs.extend(image_pairs)
        all_image_paths.append(folder_image_paths)

    # Write image pairs and paths to CSV file and text file
    write_to_csv(all_image_pairs, all_image_paths, csv_file_path, folder_name)

    # Display the path to the generated CSV file
    print(f"Image pairs and paths written to: {csv_file_path} and all_image_paths.txt")

# Specify the root folder of your dataset
root_folder = "/zhome/15/a/181503/Indre/Special_course/Datasets/lfw"

folder_name = "Face_recognition_results"
csv_file_path = "image_pairs_lfw.csv"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Call the function to generate and write image pairs
generate_and_write_image_pairs(root_folder, csv_file_path, folder_name)
