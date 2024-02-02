import pandas as pd
import os


# def extract_unique_and_counts(df, column_names):
#     results = {}
#     for column_name in column_names:
#         unique_values = df[column_name].unique()
#         value_counts = df[column_name].value_counts()
#         results[column_name] = {'unique_values': unique_values, 'value_counts': value_counts}
#     return results

# # Read the Excel file
# df = pd.read_excel("Results/DeepFace_analysis_results_lfw_all_subset_photos_1_sampling.xlsx")


# if 'Notes' in df.columns:
#     # Drop rows where the "Notes" column has anything in it
#     df = df[df['Notes'].isnull()]

# # Specify the column names you want to analyze as a list
# columns_to_analyze = ['Age', 'Gender', 'Race', 'Emotion']

# # Extract information about different columns
# results = extract_unique_and_counts(df, columns_to_analyze)

# # Print the results
# for column_name, result in results.items():
#     print(f"Unique {column_name}s:", result['unique_values'])
#     print(f"{column_name} Counts:\n", result['value_counts'])
#     print()

# def calculate_accuracy(merged_df):

#     # Section 1: Analysis when 'verified' is TRUE
#     # Filter rows where 'verified' is TRUE
#     verified_true_df = merged_df[merged_df['verified'] == True]

#     # Calculate distributions for 'race4', 'gender', and 'age_group'
#     distribution_race4_true = verified_true_df['race4'].value_counts(normalize=True)
#     distribution_gender_true = verified_true_df['gender'].value_counts(normalize=True)
#     distribution_age_group_true = verified_true_df['age_group'].value_counts(normalize=True)

#     # Print the distributions
#     print("Distribution of 'race4' values when 'verified' is TRUE:")
#     print(distribution_race4_true)
#     print("\nDistribution of 'gender' values when 'verified' is TRUE:")
#     print(distribution_gender_true)
#     print("\nDistribution of 'age_group' values when 'verified' is TRUE:")
#     print(distribution_age_group_true)

    # Section 2: Analysis when 'verified' is FALSE
    # Filter rows where 'verified' is FALSE
    # verified_false_df = merged_df[merged_df['verified'] == False]

    # # Calculate distributions for 'race4', 'gender', and 'age_group'
    # distribution_race4_false = verified_false_df['race4'].value_counts(normalize=True)
    # distribution_gender_false = verified_false_df['gender'].value_counts(normalize=True)
    # distribution_age_group_false = verified_false_df['age_group'].value_counts(normalize=True)

    # # Print the distributions
    # print("\nDistribution of 'race4' values when 'verified' is FALSE:")
    # print(distribution_race4_false)
    # print("\nDistribution of 'gender' values when 'verified' is FALSE:")
    # print(distribution_gender_false)
    # print("\nDistribution of 'age_group' values when 'verified' is FALSE:")
    # print(distribution_age_group_false)

# def analyze_verification_distribution(df, column_names, verification_status):
#     # Filter rows based on verification status
#     verification_df = df[df['verified'] == verification_status]

#     for column_name in column_names:
#         # Calculate distribution
#         distribution = verification_df[column_name].value_counts(normalize=True)
        
#         # Print distribution
#         print(f"\nDistribution of '{column_name}' values when 'verified' is {verification_status}:")
#         print(distribution)
        
#         # Print number of rows for each value
#         print(f"\nNumber of rows for each '{column_name}' value:")
#         print(verification_df[column_name].value_counts())


# def calculate_accuracy(merged_df):
        
#     # Define columns to analyze
#     columns_to_analyze = ['race4', 'gender', 'age_group']

#     # # Section 1: Analysis when 'verified' is TRUE
#     # analyze_verification_distribution(merged_df, columns_to_analyze, True)

#     # # Section 2: Analysis when 'verified' is FALSE
#     # analyze_verification_distribution(merged_df, columns_to_analyze, False)

#     # Filter rows where 'verified' is False
#     verification_false_df = merged_df[merged_df['verified'] == False]

    # for column_name in columns_to_analyze:
    #     # Calculate distribution
    #     distribution = verification_false_df[column_name].value_counts(normalize=True)
        
    #     # Print distribution
    #     print(f"\nDistribution of '{column_name}' values when 'verified' is False:")
    #     print(distribution)
        
    #     # Print number of rows for each value
    #     print(f"\nNumber of rows for each '{column_name}' value when 'verified' is False:")
    #     print(verification_false_df[column_name].value_counts())


# def calculate_accuracy(merged_df):
#     correct_predictions = merged_df[merged_df['verified'] == True]
#     incorrect_predictions = merged_df[merged_df['verified'] == False]

#     print(merged_df.columns)

#     # Calculate bias for different subgroups
#     if "race4" in merged_df.columns:
#         race_bias = calculate_bias('race4',merged_df, correct_predictions, incorrect_predictions)
#         print("Race Bias:")
#         print(race_bias)
#     if "gender" in merged_df.columns:
#         gender_bias = calculate_bias('gender', merged_df, correct_predictions, incorrect_predictions)
#         print("\nGender Bias:")
#         print(gender_bias)
#     if "age_group" in merged_df.columns:
#         age_group_bias = calculate_bias('age_group', merged_df, correct_predictions, incorrect_predictions)
#         print("\nAge Group Bias:")
#         print(age_group_bias)
#     if "Age" in merged_df.columns:
#         race_bias = calculate_bias('Age',merged_df, correct_predictions, incorrect_predictions)
#         print("Age Bias:")
#         print(race_bias)
#     if "Gender" in merged_df.columns:
#         gender_bias = calculate_bias('Gender', merged_df, correct_predictions, incorrect_predictions)
#         print("\nGender Bias:")
#         print(gender_bias)
#     if "Race" in merged_df.columns:
#         age_group_bias = calculate_bias('Race', merged_df, correct_predictions, incorrect_predictions)
#         print("\Race Group Bias:")
#         print(age_group_bias)




# def calculate_bias(subgroup_column, merged_df, correct_predictions, incorrect_predictions):
#     bias_results = pd.DataFrame(columns=['Subgroup', 'Total Samples', 'Correct Rate', 'Error Rate'])

    
#     unique_subgroups = merged_df[subgroup_column].unique()
    
#     for subgroup in unique_subgroups:
#         subgroup_correct = correct_predictions[correct_predictions[subgroup_column] == subgroup]
#         subgroup_incorrect = incorrect_predictions[incorrect_predictions[subgroup_column] == subgroup]
        
#         total_samples = len(merged_df[merged_df[subgroup_column] == subgroup])
#         correct_rate = len(subgroup_correct) / total_samples
#         error_rate = len(subgroup_incorrect) / total_samples
        
#         bias_results = pd.concat([bias_results, pd.DataFrame({
#             'Subgroup': [subgroup],
#             'Total Samples': [total_samples],
#             'Correct Rate': [correct_rate],
#             'Error Rate': [error_rate]
#         })], ignore_index=True)

#     return bias_results

def calculate_accuracy(merged_df):
    
    correct_predictions = merged_df[merged_df['verified'] == True]
    incorrect_predictions = merged_df[merged_df['verified'] == False]

    # Calculate bias for different subgroups
    if "race4" in merged_df.columns:
        race_bias = calculate_bias('race4', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(race_bias, 'bias_results.csv', mode='a')

    if "gender" in merged_df.columns:
        gender_bias = calculate_bias('gender', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(gender_bias, 'bias_results.csv', mode='a')

    if "age_group" in merged_df.columns:
        age_group_bias = calculate_bias('age_group', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(age_group_bias, 'bias_results.csv', mode='a')

    if "Age" in merged_df.columns:
        age_bias = calculate_bias('Age', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(age_bias, 'bias_results.csv', mode='a')

    if "Gender" in merged_df.columns:
        gender_bias = calculate_bias('Gender', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(gender_bias, 'bias_results.csv', mode='a')

    if "Race" in merged_df.columns:
        race_bias = calculate_bias('Race', merged_df, correct_predictions, incorrect_predictions)
        save_to_csv(race_bias, 'bias_results.csv', mode='a')

def calculate_bias(subgroup_column, merged_df, correct_predictions, incorrect_predictions):
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
    df.to_csv(filename, mode=mode, index=False, header=mode=='w')


def investigate_bias(attribute_file_path, verification_path, merged_df_path):

    df_attribute = pd.read_excel(attribute_file_path)

    # Drop rows with anything in the 'Notes' column
    if 'Notes' in df_attribute.columns:
        df_attribute = df_attribute[df_attribute['Notes'].isnull()]

    # Load verification DataFrame
    df_verification = pd.read_csv(verification_path)
    print(df_attribute.columns)
    print(df_verification.columns)
    df_verification['image1_name'] = df_verification['image1'].apply(os.path.basename)

    # Drop rows with anything in the 'Notes' column
    if 'time' in df_verification.columns:
        df_verification = df_verification[df_verification['time'].isnull()]

    if 'error' in df_verification.columns:
        df_verification = df_verification[df_verification['error'].isnull()]

    print(df_attribute.columns)
    print(df_verification.columns)

    # Merge DataFrames based on 'image_name' and 'image1'
    if 'image_name' in df_attribute.columns:
        merged_df = pd.merge(df_verification, df_attribute, left_on='image1_name', right_on='image_name', how='left')
    else:
        merged_df = pd.merge(df_verification, df_attribute, left_on='image1_name', right_on='Image name', how='left')


    # Drop duplicate columns (e.g., 'image_name' from df_attribute)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]


    merged_df.to_csv(merged_df_path, index=False)

    calculate_accuracy(merged_df)





# lwf

# Attributes:
# attribute_file_path = "Results/FairFace_analysis_results_1_lfw_all_subset_photos_new.xlsx"
attribute_file_path = "Results/DeepFace_analysis_results_lfw_all_subset_photos_1_sampling_new.xlsx"

# face_recognition_path = "Face_recognition_results/face_recognition_accuracy_lfw_subset_cos_0.7.xlsx"
verification_path = "Face_recognition_results/verification_results_lfw_subset_1_sampling.csv"

# FGNET

# Attributes:
# attribute_file_path = "Results/FairFace_analysis_results_1_FGNET.xlsx"
# attribute_file_path = "Results/DeepFace_analysis_results_FGNET_1_sampling.xlsx"

# face_recognition_path = "Face_recognition_results/face_recognition_accuracy_FGNET_cos_0.7_None_age.csv"
# verification_path = "Face_recognition_results/verification_results_FGNET_1_sampling.csv"

merged_df_path = "Bias_analysis_results/Fairface_bias_analysis_results_1_lfw_all_subset_photos_new.csv"

investigate_bias(attribute_file_path, verification_path, merged_df_path)