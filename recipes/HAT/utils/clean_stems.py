from datasets import load_from_disk
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def rm_incomplete_stems(input_path, output_path, required_channels=6):
    print("\n============================ Start Cleaning Stems ============================\n")
    # Load dataset
    dataset = load_from_disk(input_path)

    grouped_data = defaultdict(list)
    channel_counts = defaultdict(set)

    print("\nGrouping data by 'stem' and counting channels...\n")
    for d in tqdm(dataset, total=len(dataset)):
        # Normalize the stem ID
        stem = d['stem'].replace('/', '_').replace('.', '_').replace('-', '_')
        duration = d['duration']
        channel = d['channel']
        grouped_data[stem].append(duration)
        channel_counts[stem].add(channel)

    # Filter stems that do not have the required number of channels
    print(f"\nFiltering stems without {required_channels} different channels...\n")
    incomplete_stems_list = [stem for stem, channels in channel_counts.items() if len(channels) < required_channels]
    incomplete_stems_pair = {stem: len(channels) for stem, channels in channel_counts.items() if len(channels) < required_channels}

    # Save incomplete stems to a separate table
    incomplete_stems_table = [{'stem': stem, 'channel_num': count} for stem, count in incomplete_stems_pair.items()]

    # Convert tables to DataFrames
    incomplete_stems_df = pd.DataFrame(incomplete_stems_table)

    # Save DataFrames to CSV files
    incomplete_stems_df.to_csv(f'{output_path}/incomplete_stems.csv', index=False)

    # Filter out the incomplete stems from the dataset
    filtered_dataset = dataset.filter(lambda example: example['stem'].replace('/', '_').replace('.', '_').replace('-', '_') not in incomplete_stems_list)

    # Save the filtered dataset back to disk
    filtered_dataset.save_to_disk(output_path)

    # Calculate statistics
    num_total_stems = len(channel_counts)
    num_incomplete_stems = len(incomplete_stems_list)
    num_clean_stems = num_total_stems - num_incomplete_stems

    percent_incomplete_stems = (num_incomplete_stems / num_total_stems) * 100
    percent_clean_stems = (num_clean_stems / num_total_stems) * 100

    # Print statistics
    print(f"\nTotal stems: {num_total_stems}")
    print(f"Incomplete stems: {num_incomplete_stems} ({percent_incomplete_stems:.2f}%)")
    print(f"Clean stems: {num_clean_stems} ({percent_clean_stems:.2f}%)\n")

    # Save statistics to CSV
    summary_data = {
        'Total Stems': [num_total_stems],
        'Incomplete Stems': [num_incomplete_stems],
        'Incomplete Stems (%)': [f"{percent_incomplete_stems:.2f}%"],
        'Clean Stems': [num_clean_stems],
        'Clean Stems (%)': [f"{percent_clean_stems:.2f}%"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_path}/stem_summary.csv', index=False)

    print(f"\nTable of stems with incomplete channels has been saved as CSV to {output_path}\n")
    print(f"Summary of stem statistics has been saved as CSV to {output_path}/stem_summary.csv\n")
    print(f"\nDataset now has only data with complete ({required_channels}) channels, saved at {output_path}\n")

if __name__ == "__main__":
    # Paths to input and output data
    inputPath = [
        # "/mnt1/user_forbes/datasets/tat_asr_channel/total/temp",
        "/mnt1/user_forbes/datasets/tat_asr_channel/train/temp", 
        "/mnt1/user_forbes/datasets/tat_asr_channel/valid/temp",
        "/mnt1/user_forbes/datasets/tat_asr_channel/test/temp",
    ]
    outputPath = [
        # "/mnt1/user_forbes/datasets/tat_asr_channel/total",
        "/mnt1/user_forbes/datasets/tat_asr_channel/train",
        "/mnt1/user_forbes/datasets/tat_asr_channel/valid",
        "/mnt1/user_forbes/datasets/tat_asr_channel/test",
    ]

    for i in range(len(inputPath)):
        rm_incomplete_stems(
            input_path=inputPath[i],
            output_path=outputPath[i],
            required_channels=6
        )
