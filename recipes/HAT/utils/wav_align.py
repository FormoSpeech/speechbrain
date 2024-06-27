import numpy as np
from scipy.signal import correlate
from scipy.io.wavfile import write
from datasets import load_from_disk
from tqdm import tqdm
from collections import defaultdict
import os, json
import concurrent.futures
import csv
import threading

def align_audio(audio1, audio2):
    corr = correlate(audio1, audio2, mode='full')
    lag = np.argmax(corr) - len(audio2) + 1
    if lag > 0:
        aligned_audio1 = audio1[lag:]
        aligned_audio2 = audio2
    else:
        aligned_audio1 = audio1
        aligned_audio2 = audio2[-lag:]

    min_length = min(len(aligned_audio1), len(aligned_audio2))
    aligned_audio1 = aligned_audio1[:min_length]
    aligned_audio2 = aligned_audio2[:min_length]

    return aligned_audio1, aligned_audio2

def to_wav(info, output_path, sample_rate=16000):
    channel_mapping = {
        "電容": "condenser",
        "領夾": "lavalier"
    }
    audio_numpy, stem_id, channel_name = info["array"], info['stem'], info['channel']
    if channel_name in channel_mapping:
        channel_name = channel_mapping[channel_name]
    wav_filename = f"{output_path}/{stem_id}_channel_{channel_name}.wav"
    write(wav_filename, sample_rate, audio_numpy)

def find_min_duration_element(dataset, stem):
    min_duration = float('inf')
    min_row_id = None
    to_compare_id_list = []

    for row_id in stem:
        channel = dataset[row_id]
        duration = channel['duration']
        if duration < min_duration:
            min_duration = duration
            min_row_id = row_id

    to_compare_id_list = [s for s in stem if s != min_row_id]
    
    # stem contains the 8 index of df, now append the duration to the back, the len would be 9
    stem.append(min_duration)

    return min_row_id, to_compare_id_list

def process_group_alignment(output_path, dataset, group, sample_rate):
    min_row_id, to_compare_id_list = find_min_duration_element(dataset, group)
    # min_array_size = np.inf
    # cleaned_channel = []

    # for id in to_compare_id_list:
    #     channel = dataset[id]
    #     min_duration_channel = dataset[min_row_id]

    #     aligned_channel_array, min_duration_channel_array = align_audio(
    #         channel['audio']['array'],
    #         min_duration_channel['audio']['array']
    #     )
    #     cleaned_channel.append({
    #         "array": aligned_channel_array,
    #         "stem": channel['stem'],
    #         "channel": channel['channel'],
    #     })
    #     array_size = aligned_channel_array.shape[0]
    #     if array_size < min_array_size:
    #         min_array_size = array_size

    # cleaned_channel.append({
    #     "array": min_duration_channel_array,
    #     "stem": min_duration_channel['stem'],
    #     "channel": min_duration_channel['channel'],
    # })

    # for c in cleaned_channel:
    #     c["array"] = c["array"][:min_array_size]
    #     to_wav(c, output_path, sample_rate)

def process_group(output_path, dataset, group, sample_rate):    
    try:
        process_group_alignment(output_path, dataset, group, sample_rate)
    except Exception as e:
        print(f"Error processing group {group}: {e}")

def wav_alignment(input_path='./ASR/whisper/train/datasets/train', output_path='./temp', sample_rate=16000):
    print("\n============================ Start Waves Alignment ============================\n")
    print(f"\nLoad dataset from {input_path}\n")
    dataset = load_from_disk(input_path)

    print("\nGrouping data by 'stem'...\n")
    grouped_data = defaultdict(list)

    for i, d in tqdm(enumerate(dataset), total=len(dataset)):
        grouped_data[d['stem']].append(i)

    print("\nStart aligning for each stem\n")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_group, output_path, dataset, grouped_data[stem], sample_rate) for stem in grouped_data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
            
            
    # save the grouped data
    import pickle        
    with open('./grouped_data.pickle', 'wb') as handle:
        pickle.dump(grouped_data, handle)   
            
    
    output_csv = f"{output_path}/metadata.csv"
    
    channel_mapping = {
        "電容": "condenser",
        "領夾": "lavalier"
    }
    
    print("\nStart generating metadata\n")
    with open(output_csv, 'w') as fp:
        writer = csv.writer(fp)
        
        # the ID need to be fixed for csv reader
        # TODO: the duration is returned by find_min_duration_element
        writer.writerow(['ID', 'stem', 'spk_id', 'spk_gender', 'channel', 'text', 'dialect', 'duration', 'wav_path'])
        
        for stem in tqdm(grouped_data, total=len(grouped_data)):
            # for common properties to use
            common_info = dataset[grouped_data[stem][0]]
            speaker_id = common_info['speaker']
            speaker_gender = 'female' if speaker_id[1] == 'F' else 'male'
            no_space_text = common_info['text'].replace(" ", "")
            dialect = common_info['dialect']

            # see "find_min_duration" function
            duration = grouped_data[stem][8]
            
            for i in range(8):
                # the only difference are the 'id', 'channel', and 'wav_path';
                channel_info = dataset[grouped_data[stem][i]]
                channel_name = channel_info['channel']
                if channel_name in channel_mapping:
                    channel_name = channel_mapping[channel_name]
                
                wav_path = f"{output_path}/{stem}_channel_{channel_name}.wav"
                writer.writerow(
                    [channel_info['id'], stem, speaker_id, speaker_gender, channel_name, no_space_text, dialect, duration, wav_path])
                
        print(f"processed metadata csv saved at {output_csv}!")  

if __name__ == '__main__':
    # wav_alignment(
    #     input_path = './ASR/whisper/train/raw_samples_set', 
    #     output_path = './ASR/whisper/train/datasets/sixian_reading/sample_train'
    # )
    wav_alignment(
        input_path = './ASR/whisper/run/datasets/sixian_reading/complete_channel_stems', 
        output_path = '/mnt/user_forbes/datasets/sixian_reading/train'
    )