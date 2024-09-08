#!/bin/bash

# Define the list of channels
channels=("android" "condenser" "H8x" "H8y" "iOS" "lavalier" "PCmic" "webcam")

# Loop over each channel and run the python script
for channel in "${channels[@]}"; do
    echo "Running training for channel: $channel"
    python train_cr_whisper_pool_all.py hparams/cr_loo_pool_all.yaml --channel "$channel" --number_of_epochs 5 --warmup_steps 1710 --freeze_encoder False --sorting "random" --grl_alpha 0.25
done
