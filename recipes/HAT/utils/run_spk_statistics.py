import pandas as pd
import csv

def speaker_statistics(file_path, output_dir, gender_check = False):
    df = pd.read_csv(file_path)

    if gender_check == True:
        # Gender value counts
        gender_counts = df['spk_gender'].value_counts()
        gender_counts_df = gender_counts.reset_index()
        gender_counts_df.columns = ['gender', 'count']

    # Number of stems per speaker
    stems_per_speaker = df.groupby('spk_id')['stem'].nunique()
    stems_per_speaker_df = stems_per_speaker.reset_index()
    stems_per_speaker_df.columns = ['Speaker', 'number_of_stems']

    if gender_check == True:
        # Number of speakers per gender
        speakers_per_gender = df.groupby('spk_gender')['spk_id'].nunique()
        speakers_per_gender_df = speakers_per_gender.reset_index()
        speakers_per_gender_df.columns = ['gender', 'number_of_speakers']

        print("Gender counts:")
        print(gender_counts)
        print("\nNumber of speakers per gender:")
        print(speakers_per_gender)
        gender_counts_df.to_csv(f'{output_dir}/gender_counts.csv', index=False)
        speakers_per_gender_df.to_csv(f'{output_dir}/speakers_per_gender.csv', index=False)
        
    
    print("\nNumber of stems per speaker:")
    print(stems_per_speaker)

    stems_per_speaker_df.to_csv(f'{output_dir}/stems_per_speaker.csv', index=False)

if __name__ == '__main__':
    # input metadata.csv
    speaker_statistics('/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.csv', './')
    
