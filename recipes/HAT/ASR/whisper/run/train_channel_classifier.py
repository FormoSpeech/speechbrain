#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on the NLL.

If you want to only use the whisper encoder system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_whisper.py

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

Authors
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
"""

import logging
import os
import sys
from pathlib import Path

import torch
from pytorch_revgrad import RevGrad

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main

import wandb

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        channel_id_encoded, _ = batch.channel_id_encoded 
         
        # add another classifier here of 8 channels
        # logits = self.modules.channel_classifier(enc_out[-1])
        # temperary only for encoder, so the second term: logit is ignored as "_" 
        enc_out, dec_logits, _ = self.modules.whisper(wavs, channel_id_encoded)
        pooled_embedding = self.modules.adaptor(enc_out[1:])
        
        cls_logits = self.modules.channel_classifier(pooled_embedding)     

        return cls_logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        # (logits, dec_logits, wav_lens) = predictions
        (logits, wav_lens) = predictions
        batch = batch.to(self.device)
        ids = batch.id        
        channel_id, _ = batch.channel_id_encoded

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            channel_id = self.hparams.wav_augment.replicate_labels(channel_id)
            
        # ref: https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py
        loss = self.hparams.compute_cost(
            logits, channel_id, length = wav_lens
        )
        
        # valid and testing
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(ids, logits, channel_id, wav_lens)       

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            
    def on_fit_batch_end(self, batch, outputs, loss, should_step = True):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        # lr = self.optimizer.param_groups[-1]["lr"]
        # self.hparams.lr_annealing.step()
        lr = self.hparams.lr_annealing_whisper.current_lr
        if should_step:
            self.hparams.lr_annealing_whisper(self.optimizer)

        if sb.utils.distributed.if_main_process():
            stage_stats = {"loss": loss}
            
            self.hparams.train_logger.log_stats(
                stats_meta={"(batch)learning rate": lr},
                train_stats=stage_stats,
            )
            
            wandb.log({"train/loss": loss, "learning rate": lr})
            

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            stage_stats["Valid/ErrorRate"] = self.error_metrics.summarize("average")
            lr = self.hparams.lr_annealing_whisper.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"Valid/ErrorRate": stage_stats["Valid/ErrorRate"]},
                min_keys=["Valid/ErrorRate"],
            )
            
            wandb.log({"valid/error rate": stage_stats["Valid/ErrorRate"]})
            wandb.log({"valid/loss": stage_stats["loss"]})
            
            
        elif stage == sb.Stage.TEST:
            stage_stats["Test/ErrorRate"] = self.error_metrics.summarize("average")
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_channel_file, "w") as w:
                    self.error_metrics.write_stats(w)
                    
            wandb.log({"test/error rate": stage_stats["Test/ErrorRate"]})
            wandb.log({"test/loss": stage_stats["loss"]})

def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:

    # TODO: how to get a slice
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )
    
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # label token need to call add_unk for the unseen words
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.add_unk()

    # 2. Define audio pipeline(read wav path to generate signal):   
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        sig = sb.dataio.dataio.read_audio(wav_path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define classifier pipeline:
    @sb.utils.data_pipeline.takes("channel")
    @sb.utils.data_pipeline.provides("channel", "channel_id_encoded")
    def label_pipeline(channel):
        yield channel
        channel_id_encoded = label_encoder.encode_sequence_torch([channel])
        yield channel_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "channel_id_encoded"])

    # here output 3 splits, there maybe a lot of datasets in testing part
    return train_data, valid_data, test_datasets

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
        
    # wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="whisper-tiny-channel-classifier",

        # track hyperparameters and run metadata
        config = {
            "seed": hparams["seed"],
            "output_folder": hparams["output_folder"],
            "output_channel_folder": hparams["output_channel_folder"],
            "save_folder": hparams["save_folder"],
            "train_log": hparams["train_log"],
            "whisper_hub": hparams["whisper_hub"],
            "whisper_folder": hparams["whisper_folder"],
            "normalized_transcripts": hparams["normalized_transcripts"],
            "data_folder": hparams["data_folder"],
            "train_csv": hparams["train_csv"],
            "valid_csv": hparams["valid_csv"],
            "test_csv": hparams["test_csv"],
            "ckpt_interval_minutes": hparams["ckpt_interval_minutes"],
            "channel_classes": hparams["channel_classes"],
            "freeze_encoder": hparams["freeze_encoder"],
            "number_of_epochs": hparams["number_of_epochs"],
            "weight_decay": hparams["weight_decay"],
            "lr_whisper": hparams["lr_whisper"],
            "warmup_steps": hparams["warmup_steps"],
            "max_grad_norm": hparams["max_grad_norm"],
            "sorting": hparams["sorting"],
            "batch_size": hparams["batch_size"],
            "test_batch_size": hparams["test_batch_size"],
            "grad_accumulation_factor": hparams["grad_accumulation_factor"],
        }
    )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    logger.info("========================= Start Training =========================")
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing: since there maybe a lot of testing datasets(for experiments, we iterate the keys)
    os.makedirs(hparams["output_channel_folder"], exist_ok=True)

    
    logger.info("========================= Start Testing =========================")
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_channel_file = os.path.join(
            hparams["output_channel_folder"], f"channel_preds_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            progressbar = True,
            min_key="Test/ErrorRate",
        )