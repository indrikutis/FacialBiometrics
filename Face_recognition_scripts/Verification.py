import pandas as pd
from deepface import DeepFace
import random
import re
import os

def perform_verification(image_pairs_df, output_file, sampling_rate, model):

    """Perform verification for each pair of images in the input DataFrame and write the results to a CSV file.

    Args:
        image_pairs_df (pd.DataFrame): DataFrame containing image pairs to be verified.
        output_file (str): Path to the output CSV file.
        sampling_rate (float): Fraction of image pairs to be verified.
        model (str): Name of the model to be used for verification.
    Returns:
        list: List of dictionaries containing the results of verification for each pair of images.
    """

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

# # NOTE: is this needed?
# def write_results_to_csv(results, excel_file):
#     df = pd.DataFrame(results)
#     df.to_excel(excel_file, index=False)
#     print(f"Results written to: {excel_file}")


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

def verification_accuracy_FGNET(csv_file_path, age_threshold=10):

    """Calculate verification accuracy for the FGNET dataset based on the input CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing verification results for the FGNET dataset.
        age_threshold (int): Maximum age difference between images for them to be considered for accuracy calculation.
    """
    df = pd.read_csv(csv_file_path)

    # Drop rows where any of the verification-related columns have 'None'
    df_verified = df.dropna(subset=['verified', 'distance', 'threshold', 'model', 'detector_backend', 'similarity_metric'])

    # Calculate overall accuracy
    total_verified = df_verified.shape[0]
    correct_verified = df_verified['verified'].value_counts().get(True, 0)
    accuracy_overall = correct_verified / total_verified if total_verified > 0 else 0

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

    """Calculate verification accuracy for the LFW dataset based on the input CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing verification results for the LFW dataset.
    """

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


    # Verification LFW

    root_folder = "Face_recognition_results/"
    csv_file_path = "File_paths/image_pairs_lfw_subset.csv"  # Replace with your CSV file path
    sampling_rate = 1
    excel_file_path = root_folder + "verification_results_lfw_subset_1_sampling.csv"  # Replace with your desired Excel file path

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

    image_pairs_df = pd.read_csv(csv_file_path)
    # results = perform_verification(image_pairs_df, excel_file_path, sampling_rate, models[0])

    # Verification FGNET

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


    


