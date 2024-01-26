import os
from deepface import DeepFace
import pandas as pd
import random

def iterate_images(img_file_paths, dataset_name, image_sampling_rate, backend = 'opencv'):

    results_list = []

    # Read image paths from CSV file
    df = pd.read_csv(img_file_paths)
    all_images = df['Image Paths'].tolist()

    # Determine the number of images to sample based on the rate
    num_images_to_sample = int(len(all_images) * image_sampling_rate)

    # Randomly select images to analyze
    sampled_images = random.sample(all_images, num_images_to_sample)

    # Iterate through the sampled images
    for image_path in sampled_images:

        rel_path = image_path[image_path.find(dataset_name):]

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
                'Path': rel_path,
                'Dataset name': dataset_name,
                'Image name': os.path.basename(image_path),
                'Age': age,
                'Gender': dominant_gender,
                'Race': race,
                'Emotion': emotion,
                'Backend': backend,
                'Notes': ""
            })

        except ValueError as e:
            results_list.append({
                'Path': rel_path,
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

    results_list = iterate_images(img_file_paths, dataset_name, image_sampling_rate, backend)
    results_df = pd.DataFrame(results_list)

    results_folder = "Results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_df.to_excel(os.path.join(results_folder, output_filename), index=False)
    print(f"Results saved to {output_filename}")


# Specify the backend to use
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

img_file_paths = "File_paths/file_paths_FGNET.csv"
output_filename = "DeepFace_analysis_results_FGNET_0.3_sampling.xlsx"
dataset_name = 'DeepFace'
image_sampling_rate = 0.3

# Run the analysis and save results to Excel
extract_attributes(img_file_paths, output_filename, image_sampling_rate, dataset_name, backend = backends[0])
