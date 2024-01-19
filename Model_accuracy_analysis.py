import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to convert DeepFace race labels to UTKFace race labels
def convert_race_label(race_label):
    if pd.notna(race_label):
        race_mapping = {'asian': 'Asian', 'white': 'White', 'middle eastern': 'Others', 'indian': 'Indian', 'latino': 'Others', 'black': 'Black'}
        return race_mapping.get(race_label.lower(), 'Others')
    else:
        return 'Others'

def convert_gender_label(gender_label):
    gender_mapping = {'Man': 'Male', 'Woman': 'Female'}
    return gender_mapping.get(gender_label, 'Other')

# Function to create a merged dataframe from DeepFace results and UTKFace dataset
def create_merged_df(deepface_results, utkface_dataset):

    # Convert DeepFace race labels to UTKFace race labels
    deepface_results['Race'] = deepface_results['Race'].apply(convert_race_label)

    # Convert DeepFace gender labels to UTKFace gender labels
    deepface_results['Gender'] = deepface_results['Gender'].apply(convert_gender_label)

    # Merge DeepFace results with UTKFace dataset based on image names
    merged_df = pd.merge(deepface_results, utkface_dataset, how='inner', left_on='Image name', right_on='image_name')
    merged_df = merged_df[merged_df['Notes_x'].isna() & merged_df['Notes_y'].isna()]

    return merged_df

def calculate_accuracy(merged_df, age_range, merged_df_name):

    gender_accuracy = (merged_df['Gender'] == merged_df['gender']).mean()
    gender_error_rate = 1 - gender_accuracy

    age_accuracy = (merged_df['Age'] == merged_df['age']).mean()
    age_error_rate = 1 - age_accuracy

    # Age accuracy within a range since predicting the exact age is difficult
    age_within_range = (
        (merged_df['Age'] - age_range <= merged_df['age']) &
        (merged_df['Age'] + age_range >= merged_df['age'])
    )

    age_accuracy_within_range = age_within_range.mean()
    age_error_rate_within_range = 1 - age_accuracy_within_range

    race_accuracy = (merged_df['Race'] == merged_df['race']).mean()
    race_error_rate = 1 - race_accuracy

    # Print results
    print(f"Gender Accuracy: {gender_accuracy * 100:.2f}%")
    print(f"Gender Error Rate: {gender_error_rate * 100:.2f}%\n")
    print(f"Age Accuracy: {age_accuracy * 100:.2f}%")
    print(f"Age Error Rate: {age_error_rate * 100:.2f}%\n")
    print(f"Age Accuracy within {age_range} years: {age_accuracy_within_range * 100:.2f}%")
    print(f"Ages Error Rate within {age_range} years: {age_error_rate_within_range * 100:.2f}%\n")
    print(f"Race Accuracy: {race_accuracy * 100:.2f}%")
    print(f"Race Error Rate: {race_error_rate * 100:.2f}%\n")

    results_folder = "Tool analysis"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    merged_df.to_excel(results_folder + '/' + merged_df_name, index=False)

# Function to calculate the demographic distribution of the dataset
def demographic_distribution(merged_df):
    demographic_distribution = merged_df.groupby('race')['race'].count()
    print("Demographic Distribution:")
    print(demographic_distribution)
    print('\n')


def model_predictions_by_race(merged_df):
    groups = merged_df['race'].unique()

    for group in groups:
        group_data = merged_df[merged_df['race'] == group]
        accuracy = (group_data['Gender'] == group_data['gender']).mean()
        print(f"Accuracy for {group}: {accuracy * 100:.2f}%")
    print('\n')

def analyse_confusion_matrix(merged_df):
    groups = merged_df['race'].unique()

    for group in groups:
        group_data = merged_df[merged_df['race'] == group]
        y_true = group_data['gender']
        y_pred = group_data['Gender']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
        plt.title(f"Confusion Matrix for {group}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # plt.show()
        
        # Classification Report
        report = classification_report(y_true, y_pred)
        print(f"Classification Report for {group}:\n{report}")


# Load the DeepFace results and UTKFace datasets
deepface_results = pd.read_excel("Results/DeepFace_analysis_results_UTKface_part1_0.3_sampling_opencv.xlsx")
utkface_dataset = pd.read_excel("UTKFace/UTKFace_dataset_info.xlsx")
merged_df_name = "UTKface_part1_0.3_sampling_opencv_merged_results.xlsx"
age_range = 10

merged_df = create_merged_df(deepface_results, utkface_dataset)
calculate_accuracy(merged_df, age_range, merged_df_name)
demographic_distribution(merged_df)
model_predictions_by_race(merged_df)
analyse_confusion_matrix(merged_df)