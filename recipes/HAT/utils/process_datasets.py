from clean_stems import rm_incomplete_stems
from utils.wav_align import wav_alignment
import os

dataset_names = ["hailu_reading", "nansixian_reading"]


def data_process(dataset_names):
    for name in dataset_names:
        input_path = f'/mnt/user_forbes/datasets/{name}/raw_data'
        output_path_I = f'/mnt/user_forbes/datasets/{name}/complete_channel_stems'
        output_path_II = f'/mnt/user_forbes/datasets/{name}/train'

        print(f"====================================== Start processing {name} dataset ======================================")
        
        rm_incomplete_stems(input_path, output_path_I, 8)
        wav_alignment(output_path_I, output_path_II)