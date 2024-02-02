import os
import csv

def get_image_paths(root_directory):
    image_paths = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Check if the file is an image (you can add more image extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.abspath(os.path.join(root, file))
                image_paths.append(image_path)
    return image_paths

def write_to_csv(file_paths, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Paths'])
        for path in file_paths:
            csv_writer.writerow([path])

results_folder = "File_paths"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

root_directory_path = 'C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_all_subset_photos'
csv_filename = results_folder + '/lfw_all_subset_photos.csv'

absolute_paths = get_image_paths(root_directory_path)
write_to_csv(absolute_paths, csv_filename)