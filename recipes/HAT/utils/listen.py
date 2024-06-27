import numpy as np
from datasets import load_from_disk
from scipy.io.wavfile import write
from tqdm import tqdm

# Load the dataset
dataset = load_from_disk('./ASR/whisper/train/datasets/train')
# row_id = 2
# print(dataset[2]['audio']['array'])

for i, d in tqdm(enumerate(dataset), total=len(dataset)):
    print(i)
# Define the sample rate
# sample_rate = 16000

# stem_id = "XF0240001A2036_54"
# count = 0

# # print(dataset[dataset['id'] == '0000004'])

# for i in tqdm(range(842790)):
#     d = dataset[i]
#     if d['stem'] == stem_id:
#         data = d['audio']['array']
#         print(d['duration'])
#         filename = f"{stem_id}_channel_{d['channel']}.wav"
#         write(filename, sample_rate, data)  
#         print(f"Saved {filename}")
#         count += 1
#     if count > 7:
#         print(d['text'])
#         break
# print("All channels saved successfully.")

