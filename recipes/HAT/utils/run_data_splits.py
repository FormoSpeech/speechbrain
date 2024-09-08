import pandas as pd
import os
import shutil

def split_dataset(dataset_path, statistics_path):
    '''
    Split the dataset from train, so called in ./dataset level
    '''
    dataset_folder = dataset_path
    train_path = f"{dataset_folder}/train"
    valid_path = f"{dataset_folder}/valid"
    test_path = f"{dataset_folder}/test"
    
    # Create directories if they do not exist
    os.makedirs(valid_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Read metadata and stems per speaker
    metadata = pd.read_csv(f"{train_path}/metadata.csv")
    stems_per_speaker = pd.read_csv(statistics_path)

    # Separate males and females
    males = stems_per_speaker[stems_per_speaker['Speaker'].str.contains('M')]
    females = stems_per_speaker[stems_per_speaker['Speaker'].str.contains('F')]

    # Get the first three males and females for the test split
    test_speakers = list(females.iloc[:3]['Speaker']) + list(males.iloc[:3]['Speaker'])
    # Get the fourth male and female for the validation split
    valid_speakers = list(females.iloc[3:4]['Speaker']) + list(males.iloc[3:4]['Speaker'])

    # Filter metadata for test and validation splits
    test_split = metadata[metadata['spk_id'].isin(test_speakers)]
    valid_split = metadata[metadata['spk_id'].isin(valid_speakers)]

    # Move corresponding wav files to the test and validation directories
    def move_files(split_df, dest_path):
        for _, row in split_df.iterrows():
            wav_path = row['wav_path']
            if os.path.exists(wav_path):
                dest_file_path = os.path.join(dest_path, os.path.basename(wav_path))
                shutil.move(wav_path, dest_file_path)

    move_files(test_split, test_path)
    move_files(valid_split, valid_path)
    
    # Update wav_path in test and validation metadata
    test_split['wav_path'] = test_split['wav_path'].apply(lambda x: x.replace('/train/', '/test/'))
    valid_split['wav_path'] = valid_split['wav_path'].apply(lambda x: x.replace('/train/', '/valid/'))
    
    test_split.to_csv(f"{test_path}/metadata.csv", index=False)
    valid_split.to_csv(f"{valid_path}/metadata.csv", index=False)

        # Remove test and validation data from the original metadata
    remaining_metadata = metadata[~metadata['spk_id'].isin(test_speakers + valid_speakers)]
    remaining_metadata.to_csv(f"{train_path}/metadata.csv", index=False)

    print("Dataset split completed.")

if __name__ == '__main__':
    split_dataset('/mnt/user_forbes/datasets/sixian_reading', './utils/statistics_result/sixian/stems_per_speaker.csv')
