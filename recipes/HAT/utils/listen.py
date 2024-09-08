import numpy as np
from datasets import load_from_disk
from scipy.io.wavfile import write
from tqdm import tqdm

# Load the dataset
# dataset = load_from_disk('/mnt/user_forbes/datasets/tat_asr_channel/valid/tat_vol2_eval/train')
dataset_1 = load_from_disk('/mnt1/user_forbes/datasets/tat_asr_channel/test/temp/test_vol1')
dataset_2 = load_from_disk('/mnt1/user_forbes/datasets/tat_asr_channel/test/temp/test_vol2')

# row_id = 2
# print(dataset[2]['audio']['array'])

# {   
#     'id': '0000902', 
#     'stem': 'IU_IUF0001/0011-4.67', 
#     'audio': {
#         'path': '0011-4.67.wav', 
#         'array': array([ 0.00045776,  0.0032959 ,  0.00732422, ..., -0.01922607, -0.01593018, -0.01477051]), 
#         'sampling_rate': 16000
#     }, 
#     'duration': 3.667, 
#     'text': 'na7 bo5 kan1-na1 sai2 hit4 ki1 tshiu2 ki3-a2', 
#     'channel': 'XYH-6-Y', 
#     'speaker': 'IUF001'
# }

# channel = set()
for i, d in tqdm(enumerate(dataset_1['train']), total=len(dataset_1['train'])):
    # print(d['channel'])
    # print(d)
    # channel.add(d['channel'])
    if d['id'] == '0000003':
        print(d)
        
for i, d in tqdm(enumerate(dataset_2['train']), total=len(dataset_2['train'])):
    # print(d['channel'])
    # print(d)
    # channel.add(d['channel'])
    if d['id'] == '0000003':
        print(d)
    
    
# print(channel)
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

