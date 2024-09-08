import pandas as pd

# Assuming the CSV data is stored in a file named 'data.csv'
# Read the CSV file into a DataFrame

def calculate_duration(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Calculate the sum of the "duration" column
    total_duration = df['duration'].sum()
    row_count = len(df)
    distinct_stem_count = df['stem'].nunique()

    # Print the total duration
    print(f"Total duration of {csv_file_path}", total_duration/3600)
    print(f"Total row_count of {csv_file_path}", row_count)
    print(f"Total distinct 'stem' count: {distinct_stem_count}")
    
def calculate_specific_duration(csv_file_path, specific_spk_ids):
    df = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame to include only rows with speaker IDs in the specific list
    filtered_df = df[df['spk_id'].isin(specific_spk_ids)]
    
    # Calculate the total duration for the filtered rows
    total_duration = filtered_df['duration'].sum()
    
    print(total_duration/3600)
    
    return total_duration

# Example usage:
# total_duration = calculate_specific_duration('path_to_your_file.csv')
# print(total_duration)

def calculate_stems(csv_file_path):
    
    file_path = 'path/to/your/file.csv'  # Replace with your actual file path
    df = pd.read_csv(csv_file_path)

    # Extract the second letter of the speaker
    df['second_letter'] = df['Speaker'].str[1]

    # Group by the second letter and calculate the sum of the number of stems
    grouped_df = df.groupby('second_letter')['number_of_stems'].sum()

    # Display the results
    print(grouped_df)


if __name__ == '__main__':
    calculate_duration('/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.csv')
    calculate_duration('/mnt1/user_forbes/datasets/tat_asr_channel/valid/metadata.csv')
    calculate_duration('/mnt1/user_forbes/datasets/tat_asr_channel/test/metadata.csv')