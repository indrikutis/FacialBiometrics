import os
from deepface import DeepFace
import pandas as pd
import random

def iterate_images(dataset_path, dataset_folder, results_list, image_sampling_rate, backend = 'opencv'):

    all_images = [image_name for image_name in os.listdir(dataset_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Determine the number of images to sample based on the rate
    num_images_to_sample = int(len(all_images) * image_sampling_rate)

    # Randomly select images to analyze
    sampled_images = random.sample(all_images, num_images_to_sample)

    # Iterate through the sampled images
    for image_name in sampled_images:
        image_path = os.path.join(dataset_path, image_name)

        try:

            print(backend)
            # Analyze the image using DeepFace
            objs = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'], detector_backend = backend, enforce_detection=True) #, detector_backend = backend)

            age = objs[0]["age"]
            gender = objs[0]["gender"]
            race = objs[0]["dominant_race"]
            emotion = objs[0]["dominant_emotion"]

            # Extract the dominant gender
            dominant_gender = max(gender, key=gender.get)

            results_list.append({
                'Path': os.path.join(dataset_folder, image_name),
                'Dataset name': dataset_folder,
                'Image name': image_name,
                'Age': age,
                'Gender': dominant_gender,
                'Race': race,
                'Emotion': emotion,
                'Notes': ""
            })

        except ValueError as e:
            results_list.append({
                'Path': os.path.join(dataset_folder, image_name),
                'Dataset name': dataset_folder,
                'Image name': image_name,
                'Age': "",
                'Gender': "",
                'Race': "",
                'Emotion': "",
                'Notes': e
            })

            print(f"Error analyzing {image_path}: {e}")
            continue

    return results_list

def iterate_datasets(dataset_root_folder, output_filename, image_sampling_rate = 1, dataset_name = None, backend = 'opencv'):

    results_list = []

    # Analyze a specific dataset
    if dataset_name is not None:

        dataset_folder = dataset_name
        dataset_path = os.path.join(dataset_root_folder, dataset_folder)

        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            print(f"Dataset '{dataset_name}' not found.")
            return

        # Iterate through all images in the dataset
        results_list = iterate_images(dataset_path, dataset_folder, results_list, image_sampling_rate, backend)

    else:
        # Iterate through all datasets in the root folder
        for dataset_folder in os.listdir(dataset_root_folder):
            dataset_path = os.path.join(dataset_root_folder, dataset_folder)

            if not os.path.isdir(dataset_path):
                print(f"Skipping {dataset_folder} as it is not a folder.")
                continue

            # Iterate through all images in the dataset
            results_list = iterate_images(dataset_path, dataset_folder, results_list, image_sampling_rate, backend = backend)


    results_df = pd.DataFrame(results_list)

    results_folder = "Results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_df.to_excel(os.path.join(results_folder, output_filename), index=False)
    print(f"Results saved to {output_filename}")






backends = [
  'opencv', # Working
  'ssd', 
  'dlib', 
  'mtcnn', # Working
  'retinaface', # Working
  'mediapipe',
  'yolov8',
  'yunet', # Working
  'fastmtcnn',
]

# Root folder containing multiple datasets
dataset_root_folder = "C:/INDRES/DTU/Semester 3/Special course/Datasets/Test/"
output_filename = "DeepFace_analysis_results.xlsx"
image_sampling_rate = 0.7

# Run the analysis and save results to Excel
iterate_datasets(dataset_root_folder, output_filename, image_sampling_rate, 
    dataset_name = '2', backend = backends[8])
