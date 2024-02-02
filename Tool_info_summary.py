import pandas as pd

def extract_unique_and_counts(df, column_names):
    results = {}
    for column_name in column_names:
        unique_values = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        results[column_name] = {'unique_values': unique_values, 'value_counts': value_counts}
    return results

# Read the Excel file
df = pd.read_excel("Results/DeepFace_analysis_results_lfw_all_subset_photos_1_sampling.xlsx")


if 'Notes' in df.columns:
    # Drop rows where the "Notes" column has anything in it
    df = df[df['Notes'].isnull()]

# Specify the column names you want to analyze as a list
columns_to_analyze = ['Age', 'Gender', 'Race', 'Emotion']

# Extract information about different columns
results = extract_unique_and_counts(df, columns_to_analyze)

# Print the results
for column_name, result in results.items():
    print(f"Unique {column_name}s:", result['unique_values'])
    print(f"{column_name} Counts:\n", result['value_counts'])
    print()
