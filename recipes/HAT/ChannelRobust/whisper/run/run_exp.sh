# python train_cm_enc_adpt_pool.py hparams/channel_cm_patch.yaml --number_of_epochs 10 --warmup_steps 18260 --freeze_encoder False --classifier_blocks 4 --channel_classes 6 --sorting "random"
python train_cm_enc_adpt_pool_mse.py hparams/channel_cm_adpt_mse.yaml --number_of_epochs 10 --warmup_steps 18260 --freeze_encoder False --classifier_blocks 4 --channel_classes 6 --sorting "random"
# python train_cm_enc_adpt_pool.py hparams/channel_cm_random.yaml --number_of_epochs 10 --warmup_steps 18260 --freeze_encoder False --classifier_blocks 4 --channel_classes 6 --sorting "random"

