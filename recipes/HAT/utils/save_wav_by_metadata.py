'''
    Given one metadata.csv which takes the form
    "
        ID,stem,spk_id,spk_gender,channel,text,dialect,duration,wav_path
        0000001,XF0010001A2007_1,XF001,female,android,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_android.wav
        0000002,XF0010001A2007_1,XF001,female,iOS,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_iOS.wav
        ...
    "
    and a output directory
    save the files in wav_path in this csv to the output directory
'''

import pandas as pd
import shutil
import os

# Load the CSV file
# TODO: here
metadata_path = ''  # replace with the actual path to your metadata.csv
metadata = pd.read_csv(metadata_path)

# TODO: here
# Specify the output directory
output_directory = './sample_set/'  # replace with the actual path to your output directory

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through each row in the metadata and copy the wav files
for index, row in metadata.iterrows():
    source_path = row['wav_path']
    destination_path = os.path.join(output_directory, os.path.basename(source_path))
    
    # Copy the file
    shutil.copyfile(source_path, destination_path)
    print(f"Copied {source_path} to {destination_path}")

print("All files have been copied successfully.")
