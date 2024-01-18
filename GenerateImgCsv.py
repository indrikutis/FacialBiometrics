import os
import csv

def get_absolute_paths(directory):
    absolute_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            absolute_path = os.path.abspath(os.path.join(root, file))
            absolute_paths.append(absolute_path)
    return absolute_paths

def write_to_csv(file_paths, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for path in file_paths:
            csv_writer.writerow([path])

# Example usage:
directory_path = 'C:/INDRES\DTU\Semester 3\Special course\Datasets\subjects_0-1999_72_imgs/0'
csv_filename = 'file_paths.csv'

absolute_paths = get_absolute_paths(directory_path)
write_to_csv(absolute_paths, csv_filename)
