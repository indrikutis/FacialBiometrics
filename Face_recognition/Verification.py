import pandas as pd
from deepface import DeepFace

def perform_verification(image_pairs_df):
    results = []
    for index, row in image_pairs_df.iterrows():
        img1_path, img2_path = row['image1'], row['image2']
        result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path)
        print(result)
        results.append({'image1': img1_path, 'image2': img2_path, 'result': result['verified']})
    return results

def write_results_to_excel(results, excel_file):
    df = pd.DataFrame(results)
    df.to_excel(excel_file, index=False)
    print(f"Results written to: {excel_file}")

# Load image pairs from CSV
root_folder = "Face_recognition_results/"
csv_file_path = root_folder + "image_pairs_lfw.csv"  # Replace with your CSV file path
image_pairs_df = pd.read_csv(csv_file_path)

# Perform verification for each pair
results = perform_verification(image_pairs_df)

# Write results to Excel file
excel_file_path = root_folder + "verification_results.xlsx"  # Replace with your desired Excel file path
write_results_to_excel(results, excel_file_path)

