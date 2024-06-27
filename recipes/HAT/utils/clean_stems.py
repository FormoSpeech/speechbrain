from datasets import load_from_disk
# from pprint import pprint as print
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


'''
Only remove the one doesn't cotain 8 channels, no duration factors are considered.
'''

def rm_incomplete_stems(input_path, output_path, required_channels = 8):
    print("\n============================ Start Cleaning Stems ============================\n")
    # Load dataset
    dataset = load_from_disk(input_path)

    grouped_data = defaultdict(list)
    channel_counts = defaultdict(set)

    print("\nGrouping data by 'stem' and counting channels...\n")
    for d in tqdm(dataset, total=len(dataset)):
        stem = d['stem']
        duration = d['duration']
        channel = d['channel']
        grouped_data[stem].append(duration)
        channel_counts[stem].add(channel)

    # Filter stems that do not have 8 different channels
    print("\nFiltering stems without 8 different channels...")
    incomplete_stems_list = [stem for stem, channels in channel_counts.items() if len(channels) < required_channels]
    incomplete_stems_pair = {stem: len(channels) for stem, channels in channel_counts.items() if len(channels) < required_channels}

    # Save incomplete stems to a separate table
    incomplete_stems_table = [{'stem': stem, 'channel_num': count} for stem, count in incomplete_stems_pair.items()]

    # Convert tables to DataFrames
    incomplete_stems_df = pd.DataFrame(incomplete_stems_table)

    # Save DataFrames to CSV files
    incomplete_stems_df.to_csv(f'{output_path}/incomplete_stems.csv', index=False)

    print(f"\nTable of stems with incomplete channels have been saved as CSV to {output_path}\n")


    # Filter out the incomplete stems from the dataset
    filtered_dataset = dataset.filter(lambda example: example['stem'] not in incomplete_stems_list)

    # Save the filtered dataset back to disk
    filtered_dataset.save_to_disk(output_path)

    print(f"\nDataset now has only data with complete({required_channels}) channels, saved at {output_path}\n")

if __name__ == "__main__":
    rm_incomplete_stems(
        input_path = "./ASR/whisper/train/samples_set",
        output_path = "./ASR/whisper/train/datasets/sixian_reading/processed_data",
        required_channels = 8
    )
