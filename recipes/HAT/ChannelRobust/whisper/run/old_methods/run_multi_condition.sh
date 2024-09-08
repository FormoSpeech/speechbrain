#!/bin/bash

# Define the list of channels
channels=("android" "condenser" "H8x" "H8y" "iOS" "lavalier" "PCmic" "webcam")

# Loop over each channel and run the python script
for channel in "${channels[@]}"; do
    echo "Running training for channel: $channel"
    python train_vanilla_asr.py hparams/leave_one_out.yaml --channel "$channel" --number_of_epochs 5 --warmup_steps 1710 --freeze_encoder False --sorting "random"
done
