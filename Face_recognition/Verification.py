import pandas as pd
from deepface import DeepFace
import random
import re
import os

def perform_verification(image_pairs_df, output_file, sampling_rate, model):
    results = []
    with open(output_file, 'w') as output:
        csv_headers = ["image1", "image2", "verified", "distance", "threshold", 
                       "model", "detector_backend", "similarity_metric", 
                       "facial_areas_img1", "facial_areas_img2", "time", "error"]
        output.write(','.join(csv_headers) + '\n')

        # Shuffle the image pairs and select a subset based on sampling rate
        image_pairs_df = image_pairs_df.sample(frac=sampling_rate, random_state=42)

        for index, row in image_pairs_df.iterrows():
            img1_path, img2_path = row['image1'], row['image2']
            result = {}
            error_message = None

            try:
                result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name = model)
            except Exception as e:
                print(e)
                error_message = str(e)

            result_data = [img1_path, img2_path] + [result.get(key, None) for key in csv_headers[2:-2]] + [error_message]
            results.append(dict(zip(csv_headers, result_data)))

            output.write(','.join(map(str, result_data)) + '\n')

    return results

# NOTE: is this needed?
def write_results_to_csv(results, excel_file):
    df = pd.DataFrame(results)
    df.to_excel(excel_file, index=False)
    print(f"Results written to: {excel_file}")


def verification():

    # Load image pairs from CSV
    root_folder = "Face_recognition_results/"
    csv_file_path = root_folder + "image_pairs_lfw_subset.csv"  # Replace with your CSV file path
    image_pairs_df = pd.read_csv(csv_file_path)

    sampling_rate = 1

    # Write results to Excel file
    excel_file_path = root_folder + "verification_results_lfw_subset_1_sampling.csv"  # Replace with your desired Excel file path
    # Perform verification for each pair

    models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    ]

    results = perform_verification(image_pairs_df, excel_file_path, sampling_rate, models[0])
    write_results_to_csv(results, excel_file_path)


def extract_age_from_image_name(image_path):

    filename = os.path.basename(image_path) 
    age_str = ''.join(char for char in filename.split('A')[1].split('.')[0] if char.isdigit())
    return int(age_str) if age_str.isdigit() else None

def verification_accuracy_FGNET(csv_file_path, age_threshold=10):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Drop rows where any of the verification-related columns have 'None'
    df_verified = df.dropna(subset=['verified', 'distance', 'threshold', 'model', 'detector_backend', 'similarity_metric'])

    # Calculate overall accuracy
    total_verified = df_verified.shape[0]
    correct_verified = df_verified['verified'].value_counts().get(True, 0)
    accuracy_overall = correct_verified / total_verified if total_verified > 0 else 0

    # print(df_verified)
    # print(df_verified['image1'])

    # Filter out rows based on age difference threshold
    df_age_filtered = df_verified[df_verified.apply(lambda row: (
        # row['verified'] == True and
        abs(extract_age_from_image_name(row['image1']) - extract_age_from_image_name(row['image2'])) <= age_threshold
    ), axis=1)]

    # Calculate accuracy considering age difference threshold
    total_age_filtered = df_age_filtered.shape[0]
    correct_age_filtered = df_age_filtered['verified'].value_counts().get(True, 0)
    accuracy_age_filtered = correct_age_filtered / total_age_filtered if total_age_filtered > 0 else 0

    # Print results
    print("Overall Accuracy:", accuracy_overall)
    print("Number of Images (Overall):", total_verified)
    print("\nAccuracy with Age Difference Threshold (<= {} years):".format(age_threshold), accuracy_age_filtered)
    print("Number of Images (Age Difference <= {} years):".format(age_threshold), total_age_filtered)

def verification_accuracy_lfw(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Drop rows where any of the verification-related columns have 'None'
    df_verified = df.dropna(subset=['verified', 'distance', 'threshold', 'model', 'detector_backend', 'similarity_metric'])

    # Calculate overall accuracy
    total_verified = df_verified.shape[0]
    correct_verified = df_verified['verified'].value_counts().get(True, 0)
    accuracy_overall = correct_verified / total_verified if total_verified > 0 else 0

    # Print results
    print("Overall Accuracy:", accuracy_overall)
    print("Number of Images (Overall):", total_verified)

if __name__ == "__main__":


    verification()

    # csv_file_path = "Face_recognition_results/verification_results_FGNET_1_sampling.csv"
    # verification_accuracy_FGNET(csv_file_path, 0)
    # verification_accuracy_FGNET(csv_file_path, 5)
    # verification_accuracy_FGNET(csv_file_path, 10)
    # verification_accuracy_FGNET(csv_file_path, 15)
    # verification_accuracy_FGNET(csv_file_path, 20)
    # verification_accuracy_FGNET(csv_file_path, 30)
    # verification_accuracy_FGNET(csv_file_path, 40)
    # verification_accuracy_FGNET(csv_file_path, 50)
    # verification_accuracy_FGNET(csv_file_path, 60)
    # verification_accuracy_FGNET(csv_file_path, 200)

    # csv_file_path = "Face_recognition_results/verification_results_lfw_0.3_sampling.csv"
    # verification_accuracy_lfw(csv_file_path)


    


