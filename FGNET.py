import os
import pandas as pd

def extract_age_from_filename(filename):
    try:
        # Extract the numeric part between 'A' and '.'
        age_str = ''.join(char for char in filename.split('A')[1].split('.')[0] if char.isdigit())

        if age_str:
            age = int(age_str)
            return age, None 
        else:
            raise ValueError("No numeric part found between 'A' and '.jpg' in the filename")

    except Exception as e:
        return None, str(e)



def process_age_dataset(dataset_path):

    paths = []
    image_names = []
    ages = []
    notes = []

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(dataset_path, filename)

            # Extract age information from the filename
            age, error_note = extract_age_from_filename(filename)

            paths.append(file_path)
            image_names.append(filename)
            ages.append(age)
            notes.append(error_note)

    # Create a DataFrame
    df = pd.DataFrame({
        'path': paths,
        'image_name': image_names,
        'age': ages,
        'Notes': notes
    })

    return df

def save_to_excel(dataframe, output_path):
    dataframe.to_excel(output_path, index=False)

if __name__ == "__main__":
    results_folder = "Dataset_info"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    dataset_path = "/zhome/15/a/181503/Indre/Special_course/Datasets/FGNET/images/"
    age_dataset_df = process_age_dataset(dataset_path)
    output_excel_path = "Dataset_info/FGNET_dataset_info.xlsx"

    # Save the DataFrame to Excel
    save_to_excel(age_dataset_df, output_excel_path)

    print(f"Age dataset information saved to {output_excel_path}")
