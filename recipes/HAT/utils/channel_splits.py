import pandas as pd

def split_channel(metadata_path, split_dir):
    '''
    Given one metadata.csv which takes the form
    "
        ID,stem,spk_id,spk_gender,channel,text,dialect,duration,wav_path
        0000001,XF0010001A2007_1,XF001,female,android,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_android.wav
        0000002,XF0010001A2007_1,XF001,female,iOS,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_iOS.wav
        ...
    "
    
    Return 9 metadata files:
    - 8 metadata files (metadata_{channel}.csv) each containing data with only one specific channel.
    - 1 metadata file (metadata_no_{channel}.csv) for each channel, containing data with that channel eliminated.
    
    Note that the given metadata should not be modified; new metadata files are derived from it.
    The output files are named accordingly: "metadata_{channel}.csv" and "metadata_no_{channel}.csv".
    '''
    # Read the metadata file
    df = pd.read_csv(metadata_path)
    
    # Get unique channel names
    channels = df['channel'].unique()
    
    # Create separate metadata for each channel
    for channel in channels:
        # Metadata with only one channel
        channel_df = df[df['channel'] == channel]
        channel_df.to_csv(f'{split_dir}/metadata_{channel}.csv', index=False)
        
        # Metadata with this channel eliminated
        no_channel_df = df[df['channel'] != channel]
        no_channel_df.to_csv(f'{split_dir}/metadata_no_{channel}.csv', index=False)
        
    print('Metadatas already generated!')

if __name__ == "__main__":
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_channel(f'/mnt/user_forbes/datasets/sixian_reading/{split}/metadata.csv', f'/mnt/user_forbes/datasets/sixian_reading/{split}')
