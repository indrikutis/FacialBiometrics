import os
from deepface import DeepFace
import pandas as pd
import random

def iterate_images(img_file_paths, dataset_name, image_sampling_rate, backend = 'opencv'):

    """Iterates through the images and analyzes them using DeepFace framwork

    Args:
        img_file_paths (str): The path to the csv file containing the image paths
        dataset_name (str): The name of the dataset
        image_sampling_rate (float): The rate at which to sample the images
        backend (str): The backend to use for face detection

    Returns:
        List of dictionaries containing the results of the analysis
    """

    results_list = []

    df = pd.read_csv(img_file_paths)
    all_images = df['Image Paths'].tolist()

    # Calculate the number of images to sample based on the rate, and randomly select them
    num_images_to_sample = int(len(all_images) * image_sampling_rate)
    sampled_images = random.sample(all_images, num_images_to_sample)

    # Iterate through the sampled images
    for image_path in sampled_images:

        try:
            # Analyze the image using DeepFace
            objs = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'], detector_backend = backend, enforce_detection=True) #, detector_backend = backend)

            age = objs[0]["age"]
            gender = objs[0]["gender"]
            race = objs[0]["dominant_race"]
            emotion = objs[0]["dominant_emotion"]

            # Extract the dominant gender
            dominant_gender = max(gender, key=gender.get)

            results_list.append({
                'Path': image_path,
                'Dataset name': dataset_name,
                'Image name': os.path.basename(image_path),
                'Age': age,
                'Gender': dominant_gender,
                'Race': race,
                'Emotion': emotion,
                'Backend': backend,
                'Notes': ""
            })

        # In case of the error, append it to the result list
        except ValueError as e:
            results_list.append({
                'Path': image_path,
                'Dataset name': dataset_name,
                'Image name': os.path.basename(image_path),
                'Age': "",
                'Gender': "",
                'Race': "",
                'Emotion': "",
                'Backend': backend,
                'Notes': e
            })

            print(f"Error analyzing {image_path}: {e}")
            continue

    return results_list

def extract_attributes(img_file_paths, output_filename, image_sampling_rate = 1, dataset_name = '', backend = 'opencv'):

    """Extracts attributes from the images and saves the results to an Excel file

    Args:
        img_file_paths (str): The path to the csv file containing the image paths
        output_filename (str): The name of the Excel file to save the results to
        image_sampling_rate (float): The rate at which to sample the images
        dataset_name (str): The name of the dataset
        backend (str): The backend to use for face detection
    """
    results_list = iterate_images(img_file_paths, dataset_name, image_sampling_rate, backend)
    results_df = pd.DataFrame(results_list)

    results_folder = "Tool_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_df.to_excel(os.path.join(results_folder, output_filename), index=False)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":

    # Specify the backend to use
    backends = [
    'opencv',
    'ssd', 
    'dlib', 
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet', 
    'fastmtcnn',
    ]

    img_file_paths = "File_paths/lfw_all_subset_photos.csv"
    output_filename = "DeepFace_analysis_results_lfw_all_subset_photos_1_sampling_new.xlsx"
    dataset_name = 'lfw_all_subset_photos'
    image_sampling_rate = 1

    # Run the analysis and save results to Excel
    extract_attributes(img_file_paths, output_filename, image_sampling_rate, dataset_name, backend = backends[0])
