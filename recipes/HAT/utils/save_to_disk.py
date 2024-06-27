from datasets import load_dataset

dataset = load_dataset("formospeech/hat_asr_channel", "sixian_reading")
dataset.save_to_disk('/mnt/user_forbes/datasets/sixian_reading/raw_data')

dataset = load_dataset("formospeech/hat_asr_channel", "nansixian_reading")
dataset.save_to_disk('/mnt/user_forbes/datasets/nansixian_reading/raw_data')

dataset = load_dataset("formospeech/hat_asr_channel", "hailu_reading")
dataset.save_to_disk('/mnt/user_forbes/datasets/hailu_reading/raw_data')