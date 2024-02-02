import os
import shutil
import random
from deepface import DeepFace
import csv
import pandas as pd
from sklearn.metrics import classification_report


def copy_random_folders(src_path, dest_path, max_total_images=200):
    total_images_copied = 0

    # Get a list of folder names in the source path
    folder_names = os.listdir(src_path)

    # Shuffle the list to select folders randomly
    random.shuffle(folder_names)

    # Iterate through each shuffled folder in the source path
    for folder_name in folder_names:
        folder_path = os.path.join(src_path, folder_name)

        # Check if it's a directory and contains 2 or more images
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

                    # Update the total count of copied images
                    total_images_copied += 1

                    # Stop copying when the total reaches the specified limit
                    if total_images_copied >= max_total_images:
                        return

def create_subset_of_folders_for_face_rec():
    # Set your source and destination paths
    source_path = r"C:/INDRES/DTU/Semester 3/Special course/Datasets/lfw"
    destination_path = r"C:/INDRES/DTU/Semester 3/Special course/Datasets/lfw_subset"

    # Specify the maximum total number of images to copy
    max_total_images_to_copy = 500

    # Call the function to copy random folders
    copy_random_folders(source_path, destination_path, max_total_images_to_copy)

    print("Folders and images copied successfully.")

def extract_images_to_one_folder(source_directory, destination_directory):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterate through each folder and subfolder in the source directory
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Check if the file is an image (you can modify the list of extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_directory, file)

                # Move the image file to the destination directory
                shutil.move(source_path, destination_path)

def perform_face_recognition(db_path, model_name, output_csv_path):
    # Open a CSV file for writing
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Path', 'Identity', 'Target_X', 'Target_Y', 'Target_W', 'Target_H', 'Source_X', 'Source_Y', 'Source_W', 'Source_H', 'VGG-Face_cosine', 'Notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV file
        writer.writeheader()

        # Iterate through each folder in the dataset
        # for person_folder in os.listdir(dataset_path):
        #     person_folder_path = os.path.join(dataset_path, person_folder)

            # Check if it's a directory
            # if os.path.isdir(person_folder_path):
                # Iterate through each image in the person's folder
        for image_file in os.listdir(db_path):
            image_path = os.path.normpath(os.path.join(db_path, image_file))
            # image_path = image_file
            print(image_path)

            try:
                # Perform face recognition
                results = DeepFace.find(img_path=image_path, db_path=db_path, model_name=model_name, enforce_detection=False)

                # Iterate through each DataFrame in the list of results
                for result_df in results:
                    # Iterate through each row in the DataFrame and write to CSV
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

                        # Print the result to the console
                        print(f"Face recognition result for {image_path}: Identity={os.path.normpath(identity)}, Cosine Similarity={cosine_similarity}")
            except Exception as e:
                # If an error occurs, write the error message to the CSV file under 'Notes'
                writer.writerow({'Image Path': image_path, 'Identity': '', 'Target_X': '', 'Target_Y': '', 'Target_W': '',
                                    'Target_H': '', 'Source_X': '', 'Source_Y': '', 'Source_W': '', 'Source_H': '',
                                    'VGG-Face_cosine': '', 'Notes': str(e)})

                # Print the error to the console
                print(f"Error for {image_path}: {e}")

# Function to extract person's ID from the image path
def extract_person_lwf(image_path):
    # Assuming the ID is the last part of the path before the file extension
    # Remove the numerical part after the last underscore
    return '_'.join(image_path.split('\\')[-1].split('.')[0].split('_')[:-1])
    
def extract_person_FGNET(image_path):
        
    return os.path.basename(image_path).split('.')[0].split('A')[0]

def extract_age_from_image_name(image_path):

    filename = os.path.basename(image_path) 
    age_str = ''.join(char for char in filename.split('A')[1].split('.')[0] if char.isdigit())
    return int(age_str) if age_str.isdigit() else None


def calculate_face_recognition_accuracy(csv_file_path, accuracy_csv_path, cosine_threshold=None, age_threshold = 200):
    df = pd.read_csv(csv_file_path)

    # Filter out rows where the "Notes" column is empty
    df_no_notes = df[df['Notes'].isnull()]

    if "FGNET" in accuracy_csv_path:
        df_no_notes['Image Path Image'] = df_no_notes['Image Path'].apply(extract_person_FGNET)
        df_no_notes['Identity Image'] = df_no_notes['Identity'].apply(extract_person_FGNET)

        df_no_notes = df_no_notes[df_no_notes.apply(lambda row: (
            # row['verified'] == True and
            abs(extract_age_from_image_name(row['Image Path']) - extract_age_from_image_name(row['Identity'])) <= age_threshold
        ), axis=1)]
        
    else:
        # Extract person's ID from the "Identity" column
        df_no_notes['Image Path Image'] = df_no_notes['Image Path'].apply(extract_person_lwf)
        df_no_notes['Identity Image'] = df_no_notes['Identity'].apply(extract_person_lwf)

    or_df_length = len(df_no_notes)
    # Filter out rows where Image Path is equal to Identity
    df_no_notes = df_no_notes[df_no_notes['Image Path'] != df_no_notes['Identity']]
    nr_identical_images = or_df_length - len(df_no_notes)
    
    # Filter by threshold if provided
    if cosine_threshold is not None:
        df_no_notes = df_no_notes[df_no_notes['VGG-Face_cosine'] < float(cosine_threshold)]

    # Check if the ID in the "Identity" column matches the extracted person's ID
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

    # Print the classification report
    # print('\nClassification Report:\n', classification_rep)

    # Save the filtered DataFrame to a new CSV file
    df_no_notes.to_csv(accuracy_csv_path, index=False)

if __name__ == "__main__":


    # Step 1:
    # create_subset_of_folders_for_face_rec()


    # Step 2:

    source_directory = r"C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_subset"
    db_path = r"C:\INDRES\DTU\Semester 3\Special course\Datasets\lfw_all_subset_photos"

    # Call the function to extract images
    # extract_images_to_one_folder(source_directory, destination_directory)

    # Step 3:

    model_name = "VGG-Face"
    output_csv_path = "Face_recognition_results/face_recognition_results_FGNET.csv"
    
    # Call the function to perform face recognition and write results to CSV
    # perform_face_recognition(db_path , model_name, output_csv_path)
    # print(f"Face recognition results saved to {output_csv_path}")


    # Step 4:
    # accuracy_csv_path = "Face_recognition_results/face_recognition_accuracy_lfw_th_0.7_subset.csv"


    # calculate_face_recognition_accuracy(output_csv_path, accuracy_csv_path, 0.7)

    accuracy_csv_path = "Face_recognition_results/face_recognition_accuracy_FGNET_cos_0.7_None_age.csv"


    calculate_face_recognition_accuracy(output_csv_path, accuracy_csv_path, cosine_threshold=0.7, age_threshold=200)

    