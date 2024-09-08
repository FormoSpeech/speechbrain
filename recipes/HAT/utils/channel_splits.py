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
    
    
def channel_with_list(metadata_path, dir, with_channel_list):
    '''
    Given one metadata.csv which takes the form
    "
        ID,stem,spk_id,spk_gender,channel,text,dialect,duration,wav_path
        0000001,XF0010001A2007_1,XF001,female,android,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_android.wav
        0000002,XF0010001A2007_1,XF001,female,iOS,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_iOS.wav
        ...
    "
    with_channel_list is like e.g., ['android', 'iOS']
    Return a metadata file (metadata_with_{x}_{y}.csv, where x,y are the channel names in with_channel_list)
    containing data with only channels in with_channel_list included.
    
    Note that the given metadata should not be modified; new metadata files are derived from it.
    '''
    df = pd.read_csv(metadata_path)
    
    filtered_df = df[df['channel'].isin(with_channel_list)]
    
    # Generate the filename based on the included channels
    with_channel_str = '_'.join(with_channel_list)
    output_filename = f'metadata_with_{with_channel_str}.csv'
    
    filtered_df.to_csv(f'{dir}/{output_filename}', index=False)
    
    print(f'Metadata {output_filename} already generated!')

def split_channel_by_list(metadata_path, split_dir, no_channel_list):
    '''
    Given one metadata.csv which takes the form
    "
        ID,stem,spk_id,spk_gender,channel,text,dialect,duration,wav_path
        0000001,XF0010001A2007_1,XF001,female,android,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_android.wav
        0000002,XF0010001A2007_1,XF001,female,iOS,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_iOS.wav
        ...
    "
    no_channel_list is like e.g., ['condenser', 'webcam']
    Return a metadata files (metadata_no_{x}_{y}.csv, where x,y is the no_channel_name, can be more than 2) containing data with channel in no_channel_list eliminated, 
    i.e., only 8-x channels leave, where the original metadata contains 8 channels
    
    Note that the given metadata should not be modified; new metadata files are derived from it.
    '''
    df = pd.read_csv(metadata_path)
    
    filtered_df = df[~df['channel'].isin(no_channel_list)]
    
    # Generate the filename based on the excluded channels
    no_channel_str = '_'.join(no_channel_list)
    output_filename = f'metadata_no_{no_channel_str}.csv'
    
    filtered_df.to_csv(f'{split_dir}/{output_filename}', index=False)
    
    print(f'Metadatas {output_filename} already generated!')


if __name__ == "__main__":
    # splits = ['train', 'valid', 'test']
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_channel(f'/mnt1/user_forbes/datasets/tat_asr_channel/{split}/metadata.csv', f'/mnt1/user_forbes/datasets/tat_asr_channel/{split}')
    # splits = ['train', 'valid', 'test']
    # for split in splits:
    #     split_channel_by_list(f'/mnt/user_forbes/datasets/sixian_reading/{split}/metadata.csv', f'/mnt/user_forbes/datasets/sixian_reading/{split}', ['condenser', 'webcam'])
    # channel_with_list(f'/mnt/user_forbes/datasets/sixian_reading/train/_metadata.csv', f'/mnt/user_forbes/datasets/sixian_reading/train', ['PCmic', 'android', 'lavalier', 'H8x', 'condenser'])
    # channel_with_list(f'/mnt/user_forbes/datasets/sixian_reading/valid/metadata.csv', f'/mnt/user_forbes/datasets/sixian_reading/valid', ['PCmic', 'android', 'lavalier', 'H8x', 'condenser'])
    # channel_with_list(f'/mnt/user_forbes/datasets/sixian_reading/test/metadata.csv', f'/mnt/user_forbes/datasets/sixian_reading/test', ['PCmic', 'android', 'lavalier', 'H8x', 'condenser'])