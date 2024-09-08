# Copyright 2024    Hung-Shin Lee (hungshinlee@gmail.com)
# Apache 2.0

from datasets import load_dataset, Audio

def sixian():
    jsons = [
        ('/mnt/user_forbes/datasets/sixian_reading/train/metadata.json', "train"), 
        ('/mnt/user_forbes/datasets/sixian_reading/valid/metadata.json', "valid"), 
        ('/mnt/user_forbes/datasets/sixian_reading/test/metadata.json', "test"), 
    ]
    for json_path, split_name in jsons:

        dataset = load_dataset("json", data_files=str(json_path))
        dataset = dataset.cast_column("audio_path", Audio()).rename_column(
            "audio_path", "audio"
        )
        
        dataset["train"].push_to_hub("formospeech/hat_asr_aligned", private=True, split = split_name, max_shard_size="500MB" ) 
        
        
def tat():
    jsons = [
        ('/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.json', "train"), 
        ('/mnt1/user_forbes/datasets/tat_asr_channel/valid/metadata.json', "valid"), 
        ('/mnt1/user_forbes/datasets/tat_asr_channel/test/metadata.json', "test"), 
    ]
    for json_path, split_name in jsons:

        dataset = load_dataset("json", data_files=str(json_path))
        dataset = dataset.cast_column("audio_path", Audio()).rename_column(
            "audio_path", "audio"
        )
        
        dataset["train"].push_to_hub("formospeech/tat_asr_aligned", private=True, split = split_name, max_shard_size="500MB") 
        
    

if __name__ == "__main__":
    tat() 
