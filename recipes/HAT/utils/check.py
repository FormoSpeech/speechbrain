import pandas as pd

# List of file paths
# file_paths = ["/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.csv"]

def id_counts(file_paths):

    # Initialize an empty DataFrame to store concatenated data
    all_data = pd.DataFrame()

    # Loop through the file paths and read each file, then concatenate them
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, data], ignore_index=True)

    # Perform value_counts on the 'ID' column
    id_counts = all_data['ID'].value_counts()
    id_counts_df = id_counts.reset_index()
    id_counts_df.columns = ['ID', 'Count']

    # Save the result to a CSV file
    output_file_path = "./check.csv"
    id_counts_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    id_counts(["/mnt1/user_forbes/datasets/tat_asr_channel/test/metadata_condenser.csv"])
