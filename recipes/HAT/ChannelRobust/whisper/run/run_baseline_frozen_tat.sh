#!/bin/bash

# Define the list of channels
# channels=("android" "condenser" "XYH-6-X" "XYH-6-Y" "ios" "lavalier")
channels=("condenser" "XYH-6-X" "XYH-6-Y" "ios" "lavalier")

# Loop over each channel and run the python script
for channel in "${channels[@]}"; do
    echo "Running training for channel: $channel"
    python train_vanilla_asr_wer.py hparams/with_one_tat.yaml --project "frozen" --channel "$channel" --number_of_epochs 10 --warmup_steps 682 --freeze_encoder True --sorting "random"
done


