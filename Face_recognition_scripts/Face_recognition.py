import os
import shutil
import random
from deepface import DeepFace
import csv
import pandas as pd
from sklearn.metrics import classification_report


def copy_random_folders(src_path, dest_path, max_total_images=200):

    """Function to copy random folders from the source path to the destination path.

    Args:
        src_path (str): Path to the source directory.
        dest_path (str): Path to the destination directory.
        max_total_images (int): Maximum total number of images to copy.
    """

    total_images_copied = 0
    folder_names = os.listdir(src_path)

    # Shuffle the list to select folders randomly
    random.shuffle(folder_names)

    for folder_name in folder_names:
        folder_path = os.path.join(src_path, folder_name)

        if os.path.isdir(folder_path):
            images_in_folder = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            if len(images_in_folder) >= 2:
                # Create the corresponding folder in the destination path
                destination_folder_path = os.path.join(dest_path, folder_name)
                os.makedirs(destination_folder_path, exist_ok=True)

                # Copy each image individually to the destination folder
                for image_file in images_in_folder:
                    source_image_path = os.path.join(folder_path, image_file)
                    destination_image_path = os.path.join(destination_folder_path, image_file)
                    shutil.copyfile(source_image_path, destination_image_path)

                    total_images_copied += 1
                    if total_images_copied >= max_total_images:
                        return

def extract_images_to_one_folder(source_directory, destination_directory):

    """Function to extract images from subfolders to a single destination folder.

    Args:
        source_directory (str): Path to the source directory.
        destination_directory (str): Path to the destination directory.
    """

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterate through each folder and subfolder in the source directory
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_directory, file)

                # Move the image file to the destination directory
                shutil.move(source_path, destination_path)

def perform_face_recognition(db_path, model_name, output_csv_path):

    """Function to perform face recognition and write results to a CSV file.

    Args:
        dataset_path (str): Path to the dataset directory.
        model_name (str): Name of the face recognition model to use.
        output_csv_path (str): Path to the output CSV file.
    """

    # Open a CSV file for writing
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Path', 'Identity', 'Target_X', 'Target_Y', 'Target_W', 'Target_H', 'Source_X', 'Source_Y', 'Source_W', 'Source_H', 'VGG-Face_cosine', 'Notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for image_file in os.listdir(db_path):
            image_path = os.path.normpath(os.path.join(db_path, image_file))

            try:
                # Perform face recognition
                results = DeepFace.find(img_path=image_path, db_path=db_path, model_name=model_name, enforce_detection=False)

                for result_df in results:
                    for index, row in result_df.iterrows():
                        identity = row['identity']
                        target_x = row['target_x']
                        target_y = row['target_y']
                        target_w = row['target_w']
                        target_h = row['target_h']
                        source_x = row['source_x']
                        source_y = row['source_y']
                        source_w = row['source_w']
                        source_h = row['source_h']
                        cosine_similarity = row['VGG-Face_cosine']

                        # Write the result to the CSV file
                        writer.writerow({'Image Path': image_path, 'Identity': os.path.normpath(identity), 'Target_X': target_x, 'Target_Y': target_y, 'Target_W': target_w,
                                            'Target_H': target_h, 'Source_X': source_x, 'Source_Y': source_y, 'Source_W': source_w,
                                            'Source_H': source_h, 'VGG-Face_cosine': cosine_similarity, 'Notes': ''})

                        print(f"Face recognition result for {image_path}: Identity={os.path.normpath(identity)}, Cosine Similarity={cosine_similarity}")
            except Exception as e:
                # If an error occurs, write the error message to the CSV file under 'Notes'
                writer.writerow({'Image Path': image_path, 'Identity': '', 'Target_X': '', 'Target_Y': '', 'Target_W': '',
                                    'Target_H': '', 'Source_X': '', 'Source_Y': '', 'Source_W': '', 'Source_H': '',
                                    'VGG-Face_cosine': '', 'Notes': str(e)})

                print(f"Error for {image_path}: {e}")

def extract_person_lwf(image_path):

    """Extract person's ID from the image path.

    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Extracted person's ID.
    """
    # Assuming the ID is the last part of the path before the file extension
    # Remove the numerical part after the last underscore
    return '_'.join(image_path.split('\\')[-1].split('.')[0].split('_')[:-1])
    
def extract_person_FGNET(image_path):
    
    """Extract person's ID from the image path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted person's ID.
    """
    return os.path.basename(image_path).split('.')[0].split('A')[0]

def extract_age_from_image_name(image_path):

    """Extract age from the image name.

    Args:
        image_path (str): Path to the image file.

    Returns:
        int: Extracted age from the image name.
    """

    filename = os.path.basename(image_path) 
    age_str = ''.join(char for char in filename.split('A')[1].split('.')[0] if char.isdigit())
    return int(age_str) if age_str.isdigit() else None


def calculate_face_recognition_accuracy(csv_file_path, accuracy_csv_path, cosine_threshold=None, age_threshold = 200):

    """Calculate face recognition accuracy based on the input CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing face recognition results.
        accuracy_csv_path (str): Path to save the filtered CSV file with accuracy results.
        cosine_threshold (float): Threshold for cosine similarity. If None, no threshold is applied.
    """

    df = pd.read_csv(csv_file_path)

    # Filter out rows where the "Notes" column is empty
    df_no_notes = df[df['Notes'].isnull()]

    if "FGNET" in accuracy_csv_path:
        df_no_notes['Image Path Image'] = df_no_notes['Image Path'].apply(extract_person_FGNET)
        df_no_notes['Identity Image'] = df_no_notes['Identity'].apply(extract_person_FGNET)

        df_no_notes = df_no_notes[df_no_notes.apply(lambda row: (
            abs(extract_age_from_image_name(row['Image Path']) - extract_age_from_image_name(row['Identity'])) <= age_threshold
        ), axis=1)]
        
    else:
        # Extract person's ID from the "Identity" column
        df_no_notes['Image Path Image'] = df_no_notes['Image Path'].apply(extract_person_lwf)
        df_no_notes['Identity Image'] = df_no_notes['Identity'].apply(extract_person_lwf)

    or_df_length = len(df_no_notes)
    df_no_notes = df_no_notes[df_no_notes['Image Path'] != df_no_notes['Identity']]
    nr_identical_images = or_df_length - len(df_no_notes)
    
    # Filter by threshold
    if cosine_threshold is not None:
        df_no_notes = df_no_notes[df_no_notes['VGG-Face_cosine'] < float(cosine_threshold)]

    df_no_notes['Match'] = df_no_notes['Image Path Image'] == df_no_notes['Identity Image']

    # Calculate accuracy
    accuracy = df_no_notes['Match'].mean() * 100

    # Generate classification report
    classification_rep = classification_report(df_no_notes['Image Path Image'], df_no_notes['Identity Image'])

    # Print the accuracy and the number of rows
    print("Cosine Threshold: ", cosine_threshold)
    print("Age threshold:", age_threshold)
    print(f'Accuracy: {accuracy:.2f}%')
    print("Number of identical recognized images: ", nr_identical_images)
    print(f'Number of rows after filtering: {len(df_no_notes)}\n')

    # print('\nClassification Report:\n', classification_rep)

    df_no_notes.to_csv(accuracy_csv_path, index=False)

if __name__ == "__main__":

    # Step 1: Create a subset of for for the face recognition. If a full subset is used, no need to run this step.
        
    # Set your source and destination paths
    source_path = r"C:/INDRES/DTU/Semester 3/Special course/Datasets/lfw"
    destination_path = r"C:/INDRES/DTU/Semester 3/Special course/Datasets/lfw_subset/"

    # Specify the maximum total number of images to copy
    max_total_images_to_copy = 500

    # Call the function to copy random folders
    copy_random_folders(source_path, destination_path, max_total_images_to_copy)

    print("Folders and images copied successfully.")


    # Step 2: Extract images from subfolders to a single destination folder

    source_directory = r"C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_subset"
    db_path = r"C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_all_subset_photos"

    # extract_images_to_one_folder(source_directory, destination_directory)


    # Step 3: Perform face recognition on the subset of images

    model_name = "VGG-Face"
    output_csv_path = "Face_recognition_results/face_recognition_results_lwf_subset.csv"
    
    # Call the function to perform face recognition and write results to CSV
    # perform_face_recognition(db_path , model_name, output_csv_path)
    # print(f"Face recognition results saved to {output_csv_path}")


    # Step 4: Calculate face recognition accuracy

    # accuracy_csv_path = "Face_recognition_results/face_recognition_accuracy_lfw_th_0.7_subset.csv"
    # calculate_face_recognition_accuracy(output_csv_path, accuracy_csv_path, 0.7)

    # accuracy_csv_path = "Face_recognition_results/face_recognition_accuracy_FGNET_cos_0.7_None_age.csv"
    # calculate_face_recognition_accuracy(output_csv_path, accuracy_csv_path, cosine_threshold=0.7, age_threshold=200)

    