import pandas as pd
import os

def calculate_accuracy(merged_df):
    
    """Calculate verification accuracy for the input DataFrame.

    Args:
        merged_df (DataFrame): Merged DataFrame containing verification results and attribute data.
    """

    correct_predictions = merged_df[merged_df['verified'] == True]
    incorrect_predictions = merged_df[merged_df['verified'] == False]

    # Calculate bias for different subgroups depending on the available columns
    if "race4" in merged_df.columns:
        race_bias = calculate_bias('race4', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(race_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

    if "gender" in merged_df.columns:
        gender_bias = calculate_bias('gender', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(gender_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

    if "age_group" in merged_df.columns:
        age_group_bias = calculate_bias('age_group', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(age_group_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

    if "Age" in merged_df.columns:
        age_bias = calculate_bias('Age', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(age_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

    if "Gender" in merged_df.columns:
        gender_bias = calculate_bias('Gender', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(gender_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

    if "Race" in merged_df.columns:
        race_bias = calculate_bias('Race', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(race_bias, 'Bias_analysis_results/bias_results.csv', mode='a')

def calculate_bias(subgroup_column, merged_df, correct_predictions, incorrect_predictions):

    """Calculate bias for the input subgroup column.

    Args:
        subgroup_column (str): Name of the subgroup column.
        merged_df (DataFrame): Merged DataFrame containing verification results and attribute data.
        correct_predictions (DataFrame): DataFrame containing correct verification predictions.
        incorrect_predictions (DataFrame): DataFrame containing incorrect verification predictions.
    Returns:
        DataFrame: Bias results for the input subgroup.
    """

    bias_results = pd.DataFrame(columns=['Subgroup', 'Total Samples', 'Correct Rate', 'Error Rate'])
    unique_subgroups = merged_df[subgroup_column].unique()

    for subgroup in unique_subgroups:
        subgroup_correct = correct_predictions[correct_predictions[subgroup_column] == subgroup]
        subgroup_incorrect = incorrect_predictions[incorrect_predictions[subgroup_column] == subgroup]

        total_samples = len(merged_df[merged_df[subgroup_column] == subgroup])
        correct_rate = len(subgroup_correct) / total_samples
        error_rate = len(subgroup_incorrect) / total_samples

        subgroup_results = pd.DataFrame({
            'Subgroup': [subgroup],
            'Total Samples': [total_samples],
            'Correct Rate': [correct_rate],
            'Error Rate': [error_rate]
        })

        if bias_results.empty:
            bias_results = subgroup_results
        else:
            bias_results = pd.concat([bias_results, subgroup_results], ignore_index=True)

    return bias_results

def save_to_csv(df, filename, mode='w'):

    """Save the input DataFrame to a CSV file.

    Args:
        df (DataFrame): DataFrame to be saved.
        filename (str): Name of the CSV file.
        mode (str): Mode for writing to the CSV file. Default: 'w'.
    """
    df.to_csv(filename, mode=mode, index=False, header=mode=='w')


def investigate_bias(attribute_file_path, verification_path, merged_df_path):

    """Investigate bias based on the input attribute file and verification results.

    Args:
        attribute_file_path (str): Path to the attribute file.
        verification_path (str): Path to the verification results.
        merged_df_path (str): Path to save the merged DataFrame.
    """

    df_attribute = pd.read_excel(attribute_file_path)

    # Drop rows with anything in the 'Notes' column
    if 'Notes' in df_attribute.columns:
        df_attribute = df_attribute[df_attribute['Notes'].isnull()]

    # Load verification DataFrame
    df_verification = pd.read_csv(verification_path)
    df_verification['image1_name'] = df_verification['image1'].apply(os.path.basename)

    # Drop rows with anything in the 'Notes', 'error' column
    if 'time' in df_verification.columns:
        df_verification = df_verification[df_verification['time'].isnull()]

    if 'error' in df_verification.columns:
        df_verification = df_verification[df_verification['error'].isnull()]

    # Merge DataFrames based on 'image_name' and 'image1'
    if 'image_name' in df_attribute.columns:
        merged_df = pd.merge(df_verification, df_attribute, left_on='image1_name', right_on='image_name', how='left')
    else:
        merged_df = pd.merge(df_verification, df_attribute, left_on='image1_name', right_on='Image name', how='left')

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.to_csv(merged_df_path, index=False)

    calculate_accuracy(merged_df)



if __name__ == "__main__":
        
    # lwf

    # Attributes:
    attribute_file_path = "Tool_results/FairFace_analysis_results_1_lfw_all_subset_photos.xlsx"
    # attribute_file_path = "Tool_results/DeepFace_analysis_results_lfw_all_subset_photos_1_sampling.xlsx"

    verification_path = "Face_recognition_results/verification_results_lfw_subset_1_sampling.csv"

    # FGNET

    # Attributes:
    # attribute_file_path = "Tool_results/FairFace_analysis_results_1_FGNET.xlsx"
    # attribute_file_path = "Tool_results/DeepFace_analysis_results_FGNET_1_sampling.xlsx"

    # verification_path = "Face_recognition_results/verification_results_FGNET_1_sampling.csv"

    merged_df_path = "Bias_analysis_results/Fairface_bias_analysis_results_1_lfw_all_subset_photos.csv"

    investigate_bias(attribute_file_path, verification_path, merged_df_path)

