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
from speechbrain.core import AMPConfig

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main

import wandb


logger = logging.getLogger(__name__)


# Define training procedure
import torch
import speechbrain as sb

class ASR(sb.Brain):
    def __init__(self, modules, opt_class, hparams, run_opts, checkpointer):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.lambda_grl = 1.0  # Initial lambda
        self.gradient_norm = 0.0  # Initial gradient norm

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        
        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)
            bos_tokens_lens = self.hparams.wav_augment.replicate_labels(
                bos_tokens_lens
            )

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        enc_out, dec_logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(dec_logits)
        
        pooled_embeddings = self.modules.adaptor(enc_out[1:])
        cls_logits = self.modules.channel_classifier(pooled_embeddings)     
        
        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out[-1].detach(), wav_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out[-1].detach(), wav_lens)

        return log_probs, hyps, cls_logits, wav_lens, pooled_embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (log_probs, hyps, cls_logits, wav_lens, pooled_embeddings) = predictions
        batch = batch.to(self.device)
        
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        channel_ids, _ = batch.channel_id_encoded

        # decoder loss
        L_y = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens
        )
        
        # discriminator loss
        L_d = self.hparams.compute_cost(
            cls_logits, channel_ids, length = wav_lens
        )
        loss = L_y + L_d
        
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(ids, cls_logits, channel_ids, wav_lens)       
            
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [
                    self.tokenizer.normalize(text).split(" ")
                    for text in predicted_words
                ]

                target_words = [
                    self.tokenizer.normalize(text).split(" ")
                    for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]
            
            # self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return L_y, L_d, loss
    
    def fit_batch(self, batch):
        """Fit a single batch (with an optional gradient accumulation), dynamic-weighted GRL is applied"""

        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        
        # gradient accumulation can't eliminate gradients each batch
        if self.step % self.grad_accumulation_factor == 0:
            self.optimizer.zero_grad()
            
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                L_y, L_d, loss = self.compute_objectives(
                    outputs, batch, sb.Stage.TRAIN
                )
                
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            L_y, L_d, loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
                
        # Compute gradients for L_y losses
        self.scaler.scale(L_y).backward(retain_graph=True)  

        # Update the gradient norms and lambda
        self.gradient_norm = 0.0  # Initialize gradient norm

        " Calculate the gradient norm of the decoder parameters "
        decoder_params = list(self.modules.whisper.model.decoder.parameters())
                
        # partial L_y/ partial theta_y
        decoder_grads = [p.grad.detach().cpu() for p in decoder_params if p.grad is not None]

        " grl_lambda calculation "
        # Flatten all gradients into a single tensor
        all_grads = torch.cat([g.view(-1) for g in decoder_grads])
        # gradient clipping
        all_grads = torch.clamp(all_grads, min=-2, max=2)

        # Calculate the L2-norm
        self.gradient_norm = torch.sqrt(torch.sum(all_grads ** 2)).item()
                 
        # Compute lambda based on the gradient norm
        self.lambda_grl = 1.0 / (1.0 + self.gradient_norm)

        " Save (partial L_y/ partial theta_f) for dynamic GRL "
        L_y_enc_gradients = {name: param.grad.clone() for name, param in self.modules.whisper.model.encoder.named_parameters() if param.grad is not None}
        
        " Update discrminator & calculate (partial L_d/ partial theta_f) "
        self.scaler.scale(L_d).backward()  

        " Update Encoder with the gradient: (partial L_y/ partial theta_f) - lambda * (partial L_d/ partial theta_f)"
        L_d_enc_gradients = {name: param.grad.clone() for name, param in self.modules.whisper.model.encoder.named_parameters() if param.grad is not None}
        
        # update L_d_enc_gradients to the final gradients
        for name, grad in L_d_enc_gradients.items():
            
            # (partial L_y/ partial theta_f) - lambda * (partial L_d/ partial theta_f)
            L_d_enc_gradients[name] = L_y_enc_gradients[name]  - self.lambda_grl * grad  # Example modification
                
        # Reassign modified gradients
        for name, param in self.modules.whisper.model.encoder.named_parameters():
            if param.grad is not None:
                param.grad = L_d_enc_gradients[name]
        
        if should_step:
            # self.optimizers_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Calculate the loss for monitoring
        loss = L_y.detach().cpu() + L_d.detach().cpu()
        self.on_fit_batch_end(loss.detach().cpu(), self.lambda_grl, self.gradient_norm, should_step)

        return loss.detach().cpu()

    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                dtype=amp.dtype, device_type=torch.device(self.device).type
            ):
                out = self.compute_forward(batch, stage=stage)
                L_y, L_d, loss = self.compute_objectives(out, batch, stage=stage)
        else:
            out = self.compute_forward(batch, stage=stage)
            L_y, L_d, loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()


    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.gradient_norm = 0.0  # Reset gradient norm at the start of each epoch
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.error_metrics = self.hparams.error_stats()
        return super().on_stage_start(stage, epoch)
            
    def on_fit_batch_end(self, loss, lambda_grl, gradient_norm, should_step):
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
        lr = self.hparams.lr_annealing_whisper.current_lr
        
        if should_step == True:
            self.hparams.lr_annealing_whisper(self.optimizer)

        if sb.utils.distributed.if_main_process():
            stage_stats = {"loss": loss, "lambda_grl": lambda_grl, "gradient_norm": gradient_norm}
            
            self.hparams.train_logger.log_stats(
                stats_meta={"(batch)learning rate": lr},
                train_stats=stage_stats,
            )
            
            wandb.log({"train/loss": loss, "learning rate": lr, "lambda_grl": lambda_grl, "gradient_norm": gradient_norm})

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            stage_stats["VALID/CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["Valid/ErrorRate"] = self.error_metrics.summarize("average")
            
            # stage_stats["VALID/WER"] = self.wer_metric.summarize("error_rate")
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={
                    "Valid/ErrorRate": stage_stats["Valid/ErrorRate"],
                    "VALID/CER": stage_stats["VALID/CER"]
                },
                min_keys=["VALID/CER"],
            )
            
            # wandb.log({"valid/wer": stage_stats["VALID/WER"]})
            wandb.log({"valid/cer": stage_stats["VALID/CER"]})
            wandb.log({"valid/error rate": stage_stats["Valid/ErrorRate"]})
            wandb.log({"valid/loss": stage_stats["loss"]})
            
            
        elif stage == sb.Stage.TEST:
            
            stage_stats["Test/ErrorRate"] = self.error_metrics.summarize("average")
            stage_stats["TEST/CER"] = self.cer_metric.summarize("error_rate")
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_channel_file, "w") as w:
                    self.error_metrics.write_stats(w)
                with open(self.hparams.test_cer_file, "w") as w:
                    self.cer_metric.write_stats(w)
            # wandb.log({"test/wer": stage_stats["TEST/WER"]})
            wandb.log({"test/cer": stage_stats["TEST/CER"]})
            wandb.log({"test/loss": stage_stats["loss"]})


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

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
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )
        test_dataset_representation = test_datasets[name]
        

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    # 3. Define classifier pipeline:
    # label token need to call add_unk for the unseen words
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    @sb.utils.data_pipeline.takes("channel")
    @sb.utils.data_pipeline.provides("channel", "channel_id_encoded")
    def label_pipeline(channel):
        yield channel
        channel_id_encoded = label_encoder.encode_sequence_torch([channel])
        yield channel_id_encoded
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    print('lab_enc_file', lab_enc_file)
    
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="channel",
    )
    
    
    # 4. Define text pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides(
        "text", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(text):
        if hasattr(hparams, "normalized_transcripts"):
            text = tokenizer.normalize(text)
        yield text
        tokens_list = tokenizer.encode(text, add_special_tokens=False)
        yield tokens_list
        
        # special tokens build here
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "channel_id_encoded"],
    )

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
        project=f"whisper-tiny-channel-robust-asr-{hparams['project']}{hparams['channel']}-dwgrl",

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

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

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

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    
    logger.info("========================= Start Training =========================")
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing
    os.makedirs(hparams["output_cer_folder"], exist_ok=True)
    
    logger.info("========================= Start Testing =========================")
    for k in test_datasets.keys():  # keys are test_clean, test_other etc

        asr_brain.hparams.test_channel_file = os.path.join(
            hparams["output_channel_folder"], f"channel_preds_{k}.txt"
        )
        asr_brain.hparams.test_cer_file = os.path.join(
            hparams["output_cer_folder"], f"cer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="Valid/CER",
        )