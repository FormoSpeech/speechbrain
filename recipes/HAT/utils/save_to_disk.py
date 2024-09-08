from datasets import load_dataset, concatenate_datasets

# dataset = load_dataset("formospeech/hat_asr_channel", "sixian_reading")
# dataset.save_to_disk('/mnt/user_forbes/datasets/sixian_reading/raw_data')
'''
save raw data to the temp dir
'''

train_vol1 = load_dataset("formospeech/tat_asr_channel", "tat_vol1_train")
train_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_train")
eval_vol1 = load_dataset("formospeech/tat_asr_channel", "tat_vol1_eval")
eval_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_eval")
test_vol1 = load_dataset("formospeech/tat_asr_channel", "tat_vol1_test")
test_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_test")

# Concatenate total datasets
total_combined = concatenate_datasets(
    [
        train_vol1["train"], 
        train_vol2["train"],
        eval_vol1["train"],
        eval_vol2["train"],
        test_vol1["train"],
        test_vol2["train"],
    ]
)
total_combined.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/total/temp')

# Load train datasets
# train_vol1.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/train/temp/train_vol1')

# train_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_train")
# train_vol2.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/train/temp/train_vol2')

# Concatenate train datasets
# train_combined = concatenate_datasets([train_vol1["train"], train_vol2["train"]])

# # Save combined train dataset to disk
# train_combined.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/train/temp')

# # Load eval datasets
# eval_vol1 = load_dataset("formospeech/tat_asr_channel", "tat_vol1_eval")
# eval_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_eval")

# # Concatenate eval datasets
# eval_combined = concatenate_datasets([eval_vol1["train"], eval_vol2["train"]])

# # Save combined eval dataset to disk
# eval_combined.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/valid/temp')

# # Load test datasets
# test_vol1 = load_dataset("formospeech/tat_asr_channel", "tat_vol1_test")
# test_vol1.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/test/temp/test_vol1')
# test_vol2 = load_dataset("formospeech/tat_asr_channel", "tat_vol2_test")
# test_vol2.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/test/temp/test_vol2')

# # Concatenate test datasets
# test_combined = concatenate_datasets([test_vol1["train"], test_vol2["train"]])

# # Save combined test dataset to disk
# test_combined.save_to_disk('/mnt1/user_forbes/datasets/tat_asr_channel/test/temp')

