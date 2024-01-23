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

def convert_race4_label(race_label):
    gender_mapping = {'Man': 'Male', 'Woman': 'Female'}
    return gender_mapping.get(race_label, 'Other')


# Function to create a merged dataframe from DeepFace results and UTKFace dataset
def create_merged_df(tool_results, dataset, tool, merged_df_name):

    print(tool_results)
    print(dataset)

    if (tool).lower() == 'deepface':

        if 'Gender' in dataset.columns:
            # Convert DeepFace gender labels to UTKFace gender labels
            tool_results['Gender'] = tool_results['Gender'].apply(convert_gender_label)


        if 'Image name' in tool_results.columns and 'image_name' in dataset.columns:
            merged_df = pd.merge(tool_results, dataset, how='inner', left_on='Image name', right_on='image_name')
        else:
            merged_df = pd.merge(tool_results, dataset, how='inner', on='image_name')


        merged_df = merged_df[merged_df['Notes_x'].isna() & merged_df['Notes_y'].isna()]

    else:
        # Convert DeepFace race labels to UTKFace race labels
        tool_results['race4'] = tool_results['race4'].apply(convert_race4_label)

        merged_df = pd.merge(tool_results, dataset, how='inner', left_on='image_name', right_on='image_name')
        merged_df = merged_df[merged_df['Notes'].isna()]

    results_folder = "Tool analysis"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    merged_df.to_excel(results_folder + '/' + merged_df_name, index=False)

    return merged_df

def extract_age_bounds(age_group, age_range):
    if isinstance(age_group, str) and '-' in age_group:
        lower, upper = map(int, age_group.split('-'))
        return max(0, lower - age_range), upper + age_range  # Ensure lower bound is not negative
    elif isinstance(age_group, str) and '+' in age_group:
        lower = int(age_group[:-1])
        upper = 200  # Set an upper bound of 200 for '70+'
        return max(0, lower - age_range), upper + age_range # Ensure lower bound is not negative
    else:
        return int(age_group), int(age_group)

def calculate_accuracy_FGNET(merged_df, age_range, tool):

    if (tool).lower() == 'deepface':

        # No need to calculate gender accuracy

        age_accuracy = (merged_df['Age'] == merged_df['age']).mean()
        age_error_rate = 1 - age_accuracy

        # Age accuracy within a range since predicting the exact age is difficult
        age_within_range = (
            (merged_df['Age'] - age_range <= merged_df['age']) &
            (merged_df['Age'] + age_range >= merged_df['age'])
        )

        print(age_within_range)

        age_accuracy_within_range = age_within_range.mean()
        age_error_rate_within_range = 1 - age_accuracy_within_range

    else: 

        # No need to calculate gender accuracy
        
        # Extract lower and upper bounds from the predicted age range
        merged_df[['predicted_age_lower', 'predicted_age_upper']] = merged_df['age_group'].apply(lambda x: extract_age_bounds(x, 0)).apply(pd.Series)

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy = age_accuracy_range.mean()
        age_error_rate = 1 - age_accuracy

        merged_df[['predicted_age_lower', 'predicted_age_upper']] = merged_df['age_group'].apply(lambda x: extract_age_bounds(x, int(age_range/2))).apply(pd.Series)

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy_within_range = age_accuracy_range.mean()
        age_error_rate_within_range = 1 - age_accuracy_within_range


    # Print results
    print(f"Age Accuracy: {age_accuracy * 100:.2f}%")
    print(f"Age Error Rate: {age_error_rate * 100:.2f}%\n")
    print(f"Age Accuracy within {age_range} years: {age_accuracy_within_range * 100:.2f}%")
    print(f"Ages Error Rate within {age_range} years: {age_error_rate_within_range * 100:.2f}%\n")


def calculate_accuracy(merged_df, age_range, tool):

    if (tool).lower() == 'deepface':

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

    else: 

        gender_accuracy = (merged_df['gender_x'] == merged_df['gender_y']).mean()
        gender_error_rate = 1 - gender_accuracy
        
        # Extract lower and upper bounds from the predicted age range
        merged_df[['predicted_age_lower', 'predicted_age_upper']] = merged_df['age_group'].apply(lambda x: extract_age_bounds(x, 0)).apply(pd.Series)

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy = age_accuracy_range.mean()
        age_error_rate = 1 - age_accuracy

        merged_df[['predicted_age_lower', 'predicted_age_upper']] = merged_df['age_group'].apply(lambda x: extract_age_bounds(x, int(age_range/2))).apply(pd.Series)

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy_within_range = age_accuracy_range.mean()
        age_error_rate_within_range = 1 - age_accuracy_within_range


    if (tool).lower() == 'deepface':
        race_accuracy = (merged_df['Race'] == merged_df['race']).mean()
    else:
        race_accuracy = (merged_df['race4'] == merged_df['race']).mean()

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


# Function to calculate the demographic distribution of the dataset
def demographic_distribution(merged_df):
    demographic_distribution = merged_df.groupby('race')['race'].count()
    print("Demographic Distribution:")
    print(demographic_distribution)
    print('\n')


def model_predictions_by_race(merged_df, tool):
    groups = merged_df['race'].unique()

    for group in groups:
        group_data = merged_df[merged_df['race'] == group]

        if (tool).lower() == 'deepface':
            accuracy = (group_data['Gender'] == group_data['gender']).mean()

        else:
            accuracy = (group_data['gender_x'] == group_data['gender_y']).mean()


        print(f"Accuracy for {group}: {accuracy * 100:.2f}%")
    print('\n')

def analyse_confusion_matrix(merged_df, tool):
    groups = merged_df['race'].unique()

    for group in groups:
        group_data = merged_df[merged_df['race'] == group]

        if tool.lower() == 'deepface':
            y_true = group_data['gender']
            y_pred = group_data['Gender']

        else:
            y_true = group_data['gender_y']
            y_pred = group_data['gender_x']

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
analysis_results = pd.read_excel("Results/FairFace_analysis_results_FGNET.xlsx")
dataset_info = pd.read_excel("Dataset_info/FGNET_dataset_info.xlsx")
merged_df_name = "FGNET_FairFace_merged_results.xlsx"
age_range = 5
tool = 'FairFace'

merged_df = create_merged_df(analysis_results, dataset_info, tool, merged_df_name)

# calculate_accuracy(merged_df, age_range, tool)
calculate_accuracy_FGNET(merged_df, age_range, tool)

# demographic_distribution(merged_df)
# model_predictions_by_race(merged_df, tool)
# analyse_confusion_matrix(merged_df, tool)