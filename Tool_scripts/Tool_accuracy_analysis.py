import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def convert_race_label(race_label):

    """Converts the race labels from DeepFace to UTKFace format.

    Args:
        race_label (str): Race label from DeepFace.
    Returns:
        race (str): Converted race label.
    """

    if pd.notna(race_label):
        race_mapping = {'asian': 'Asian', 'white': 'White', 'middle eastern': 'Others', 'indian': 'Indian', 'latino': 'Others', 'black': 'Black'}
        return race_mapping.get(race_label.lower(), 'Others')
    else:
        return 'Others'

def convert_gender_label(gender_label):
    
    """Converts the gender labels from DeepFace to UTKFace format.

    Args:
        gender_label (str): Gender label from DeepFace.
    Returns:
        gender (str): Converted gender label.
    """

    gender_mapping = {'Man': 'Male', 'Woman': 'Female'}
    return gender_mapping.get(gender_label, 'Other')


def create_merged_df(tool_results, dataset, tool, merged_df_name):

    """Merges the results from a tool with the dataset and saves the merged DataFrame to an Excel file.

    Args:
        tool_results (pd.DataFrame): Results from a tool.
        dataset (pd.DataFrame): The dataset to merge with the tool results.
        tool (str): The tool used to generate the results.
        merged_df_name (str): The name of the Excel file to save the merged DataFrame.
    Returns:
        pd.DataFrame: The merged DataFrame.
    """

    if (tool).lower() == 'deepface':

        if 'Gender' in tool_results.columns:
            # Convert DeepFace gender labels to UTKFace gender labels
            tool_results['Gender'] = tool_results['Gender'].apply(convert_gender_label)

        if 'Race' in tool_results.columns:
            tool_results['Race'] = tool_results['Race'].apply(convert_race_label)

        if 'Image name' in tool_results.columns and 'image_name' in dataset.columns:
            merged_df = pd.merge(tool_results, dataset, how='inner', left_on='Image name', right_on='image_name')
        else:
            merged_df = pd.merge(tool_results, dataset, how='inner', on='image_name')


        merged_df = merged_df[merged_df['Notes_x'].isna() & merged_df['Notes_y'].isna()]

    else:
        # Convert DeepFace race labels to UTKFace race labels
        bounds_series = tool_results['age_group'].apply(lambda x: extract_age_bounds(x, 0)).apply(pd.Series)

        # Rename the columns
        bounds_series.columns = ['predicted_age_lower', 'predicted_age_upper']
        tool_results = pd.concat([tool_results.iloc[:, :tool_results.columns.get_loc('age_group')],
                            bounds_series,
                            tool_results.iloc[:, tool_results.columns.get_loc('age_group')+1:]],
                        axis=1)

        merged_df = pd.merge(tool_results, dataset, how='inner', left_on='image_name', right_on='image_name')
        merged_df = merged_df[merged_df['Notes'].isna()]

    results_folder = "Tool_analysis_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    merged_df.to_excel(results_folder + '/' + merged_df_name, index=False)

    return merged_df

def extract_age_bounds(age_group, age_range):
    
    """Extracts the lower and upper bounds of an age group.

    Args:
        age_group (str): The age group in the format 'lower-upper' or 'lower+'.
        age_range (int): The range to add/subtract from the lower and upper bounds.
    Returns:
        tuple: A tuple containing the lower and upper bounds of the age group.
    """

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

    """Calculates the accuracy of the tool predictions for the FGNET dataset.
    
    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing the tool predictions and the dataset.
        age_range (int): The range to consider when calculating the age accuracy.
        tool (str): The tool used to generate the predictions.
    """

    if (tool).lower() == 'deepface':

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

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy = age_accuracy_range.mean()
        age_error_rate = 1 - age_accuracy

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower'] - int(age_range/2)) &
            (merged_df['age'] <= merged_df['predicted_age_upper'] + int(age_range/2))
        )

        # Calculate accuracy within the predicted range
        age_accuracy_within_range = age_accuracy_range.mean()
        age_error_rate_within_range = 1 - age_accuracy_within_range

    # Print results
    print(f"Age Accuracy: {age_accuracy * 100:.2f}%")
    print(f"Age Error Rate: {age_error_rate * 100:.2f}%\n")
    print(f"Age Accuracy within {age_range} years: {age_accuracy_within_range * 100:.2f}%")
    print(f"Ages Error Rate within {age_range} years: {age_error_rate_within_range * 100:.2f}%\n")

def calculate_accuracy_UTKFace(merged_df, age_range, tool):

    """Calculates the accuracy of the tool predictions for the UTKFace dataset.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing the tool predictions and the dataset.
        age_range (int): The range to consider when calculating the age accuracy.
        tool (str): The tool used to generate the predictions.
    """

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

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower']) &
            (merged_df['age'] <= merged_df['predicted_age_upper'])
        )

        # Calculate accuracy within the predicted range
        age_accuracy = age_accuracy_range.mean()
        age_error_rate = 1 - age_accuracy

        # Check if the true age falls within the predicted range
        age_accuracy_range = (
            (merged_df['age'] >= merged_df['predicted_age_lower'] - int(age_range/2)) &
            (merged_df['age'] <= merged_df['predicted_age_upper'] + int(age_range/2))
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

    """Calculates the demographic distribution of the dataset.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing the tool predictions and the dataset.
    """

    if 'race' in merged_df.columns:
        demographic_distribution = merged_df.groupby('race')['race'].count()
    else:
        demographic_distribution = merged_df.groupby('race4')['race4'].count()

    print("Demographic Distribution:")
    print(demographic_distribution)
    print('\n')

def model_predictions_by_race(merged_df, tool, age_range):

    """Calculates the accuracy of the tool based on race.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing the tool predictions and the dataset.
        tool (str): The tool used to generate the predictions.
        age_range (int): The range to consider when calculating the age accuracy.
    """

    groups = merged_df['race'].unique()

    for group in groups:
        group_data = merged_df[merged_df['race'] == group]

        if (tool).lower() == 'deepface':
            gender_accuracy = (group_data['Gender'] == group_data['gender']).mean()
            age_accuracy = (group_data['Age'] == group_data['age']).mean()
            age_accuracy_within_range = (abs(group_data['Age'] - group_data['age']) <= age_range).mean()
            ethnicity_accuracy = (group_data['Race'] == group_data['race']).mean()

        else:
            gender_accuracy = (group_data['gender_x'] == group_data['gender_y']).mean()
            age_accuracy_range = (
                (group_data['age'] >= group_data['predicted_age_lower']) &
                (group_data['age'] <= group_data['predicted_age_upper'])
            )

            # Calculate accuracy within the predicted range
            age_accuracy = age_accuracy_range.mean()

            # Check if the true age falls within the predicted range
            age_accuracy_range = (
                (group_data['age'] >= group_data['predicted_age_lower'] - int(age_range/2)) &
                (group_data['age'] <= group_data['predicted_age_upper'] + int(age_range/2))
            )

            # Calculate accuracy within the predicted range
            age_accuracy_within_range = age_accuracy_range.mean()
            

            ethnicity_accuracy = (group_data['race4'] == group_data['race']).mean()

        print("Number of subjects: ", len(group_data))
        print(f"Gender accuracy for {group}: {gender_accuracy * 100:.2f}%")
        print(f"Age accuracy for {group}: {age_accuracy * 100:.2f}%")
        print(f"Age accuracy within {age_range} years for {group}: {age_accuracy_within_range * 100:.2f}%")
        print(f"Race accuracy for {group}: {ethnicity_accuracy * 100:.2f}%")
        print("-----------------")
    print('\n')

def analyse_confusion_matrix(merged_df, tool):

    """Analyzes the confusion matrix for the tool predictions.

    Args:
        merged_df (pd.DataFrame): The merged DataFrame containing the tool predictions and the dataset.
        tool (str): The tool used to generate the predictions.
    """
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

def read_excel_to_dataframe(file_path, sheet_name=0):
    """
    Reads an Excel file and returns a DataFrame.

    Parameters:
    - file_path (str): The path to the Excel file.
    - sheet_name (str or int, default 0): The sheet to read. You can specify the sheet name or index.

    Returns:
    - pd.DataFrame: A DataFrame containing the data from the Excel file.
    """
    try:
        # Read Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


if __name__ == "__main__":

    # analysis_results = pd.read_excel("Tool_results/FairFace_analysis_results_1_FGNET.xlsx")
    # dataset_info = pd.read_excel("Dataset_info/FGNET_dataset_info.xlsx")
    # merged_df_name = "FGNET_FairFace_merged_results_1.xlsx"
    analysis_results = pd.read_excel("Tool_results/FairFace_analysis_results_UTKFace_all.xlsx")
    dataset_info = pd.read_excel("Dataset_info/UTKFace_dataset_info.xlsx")
    merged_df_name = "UTKFace_FairFace_merged_results_1.xlsx"
    age_range = 5
    tool = 'FairFace'

    # For loading the dataset from the file
    # merged_df = read_excel_to_dataframe("Tool analysis/" + merged_df_name)
    
    merged_df = create_merged_df(analysis_results, dataset_info, tool, merged_df_name)


    # Choose the dataset to analyse
    calculate_accuracy_UTKFace(merged_df, age_range, tool)
    # calculate_accuracy_FGNET(merged_df, age_range, tool)

    demographic_distribution(merged_df)

    # NOTE: FGNET dataset does not have model predictions by race since it only encodes the age
    model_predictions_by_race(merged_df, tool, age_range)
    # analyse_confusion_matrix(merged_df, tool)

