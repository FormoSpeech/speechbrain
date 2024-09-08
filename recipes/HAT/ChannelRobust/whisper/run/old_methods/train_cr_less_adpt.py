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
from hyperpyyaml import load_hyperpyyaml
from speechbrain.core import AMPConfig
from ASR.whisper.run.module.scatteredness_utils import cal_scatteredness, plotTSNE



import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from sklearn.metrics import multilabel_confusion_matrix


import wandb


logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        # channel_id_encoded, _ = batch.channel_id_encoded 
        
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
        
        # TODO: now the classifier only use the last layer for attentive pooling
        pooled_embeddings = self.modules.adaptor(enc_out[-1])
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
        stem_ids, _ = batch.stem_encoded

        loss_dec = self.hparams.nll_loss(
            log_probs, tokens_eos, length = tokens_eos_lens
        )
        
        loss_clf = self.hparams.compute_cost(
            cls_logits, channel_ids, length = wav_lens
        )
        
        loss = loss_dec + loss_clf

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
            
        if stage == sb.Stage.TEST:
            self.y_true.append(channel_ids)
            self.y_pred.append(torch.argmax(cls_logits, dim=-1))

            # TODO: only at test stage
            for i in range(len(batch)):
                self.pooled_embedding_list.append(
                    {
                        "id": int(ids[i]),
                        "stem_id": stem_ids[i].item(),
                        "channel_id": channel_ids[i].item(),
                        "embedding": pooled_embeddings[i]
                    }
                ) 

        return loss
    
    
    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        
        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()
        
        # Initialize gradient norm
        self.gradient_norm = 0.0  
        
        # Calculate the gradient norm of the decoder loss
        decoder_params = list(self.modules.whisper.model.decoder.parameters())
        decoder_grads = [p.grad.detach().cpu() for p in decoder_params if p.grad is not None]
        
        # 2-norm
        self.gradient_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in decoder_grads)).item()
        
        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(self.gradient_norm, loss, should_step)
        return loss.detach().cpu()
    

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.error_metrics = self.hparams.error_stats()
        
        if stage == sb.Stage.TEST:
            # Show at on_stage_end of test
            # a list to accomondate dictionaries
            self.pooled_embedding_list = []
            self.y_true = [] 
            self.y_pred = []
            
            
    def on_fit_batch_end(self, gradient_norm, loss, should_step = True):
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
        if should_step:
            self.hparams.lr_annealing_whisper(self.optimizer)

        if sb.utils.distributed.if_main_process():
            stage_stats = {"loss": loss, "gradient_norm": gradient_norm}
            
            self.hparams.train_logger.log_stats(
                stats_meta={"(batch)learning rate": lr},
                train_stats=stage_stats,
            )
            
            wandb.log({"train/loss": loss, "learning rate": lr, "gradient_norm": gradient_norm})


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
            scatteredness = cal_scatteredness(self.pooled_embedding_list, self.hparams.output_folder)
            
            stage_stats["Test/ErrorRate"] = self.error_metrics.summarize("average")
            stage_stats["Test/scatteredness"] = scatteredness
            stage_stats["TEST/CER"] = self.cer_metric.summarize("error_rate")
            
            y_true = torch.cat(self.y_true).cpu().numpy()
            y_pred = torch.cat(self.y_pred).cpu().numpy()
            report = multilabel_confusion_matrix(y_true, y_pred)    
            
            logger.info(report)
            logger.info(f"scatteredness: {scatteredness}")
            
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_channel_file, "w") as w:
                    self.error_metrics.write_stats(w)
                with open(self.hparams.test_cer_file, "w") as w:
                    self.cer_metric.write_stats(w)
                    
            wandb.log({"test/cer": stage_stats["TEST/CER"]})
            wandb.log({"test/loss": stage_stats["loss"]})
            wandb.log({"scatteredness": scatteredness})
            
            
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
    
    
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_stem = sb.dataio.encoder.CategoricalEncoder()
    
    # 3. Define classifier pipeline:
    @sb.utils.data_pipeline.takes("channel")
    @sb.utils.data_pipeline.provides("channel", "channel_id_encoded")
    def label_pipeline(channel):
        yield channel
        channel_id_encoded = label_encoder.encode_sequence_torch([channel])
        yield channel_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    
    @sb.utils.data_pipeline.takes("stem")
    @sb.utils.data_pipeline.provides("stem", "stem_encoded")
    def stem_pipeline(stem):
        yield stem
        stem_encoded = label_encoder_stem.encode_sequence_torch([stem])
        yield stem_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, stem_pipeline)
    
    
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
        
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="channel",
    )
    
    lab_enc_file_stem = os.path.join(hparams["save_folder"], "label_encoder_stem.txt")
    
    label_encoder_stem.load_or_create(
        path=lab_enc_file_stem,
        from_didatasets=[train_data, valid_data, test_dataset_representation],
        output_key="stem",
    )
    

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "channel_id_encoded", "stem_encoded"],
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
        project=f"{hparams["project"]}{hparams["channel"]}_grl0.5_epo_{hparams["number_of_epochs"]}_{hparams["seed"]}_record_grad_norm_772123",

        # track hyperparameters and run metadata
        config = {
            "channel": hparams["channel"],
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
            "adaptor": hparams["adaptor"],
            "adaptor-grl-weight": hparams["grl_weight"]
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
            min_key="VALID/CER",
        )