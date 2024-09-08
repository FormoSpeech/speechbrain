# - train/metadata_sample_128.csv
# - valid/metadata_sample_128.csv
# - test/metadata_sample_128.csv

'''
    Given one metadata.csv which takes the form
    "
        ID,stem,spk_id,spk_gender,channel,text,dialect,duration,wav_path
        0000001,XF0010001A2007_1,XF001,female,android,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_android.wav
        0000002,XF0010001A2007_1,XF001,female,iOS,人惡人怕天毋怕，人善人欺天不欺,sixian,4.95,/mnt/user_forbes/datasets/sixian_reading/train/XF0010001A2007_1_channel_iOS.wav
        ...
    "
    and a input_path
    only fetch the first n data to generate a new_csv with input_path + f"_sample_{n}"
    e.g., metadata_sample_128.csv
'''

import pandas as pd

def generate_sample_csv(input_csv_path, n):
    # Read the first n rows from the input CSV
    df = pd.read_csv(input_csv_path, nrows=n)
    
    # Generate the output CSV path
    base_name = input_csv_path.rsplit('.', 1)[0]  # Remove the file extension
    output_csv_path = f"{base_name}_sample_{n}.csv"
    
    # Write the first n rows to the new CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Sample CSV generated: {output_csv_path}")

if __name__ == "__main__":
    n = 128               
    splits = ['train', 'valid', 'test']
    for split in splits:
        input_csv_path = f'/mnt/user_forbes/datasets/sixian_reading/{split}/metadata.csv'  # Replace with your input CSV path
        generate_sample_csv(input_csv_path, n)
