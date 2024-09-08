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

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from module.utils import compute_distances


import wandb

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.Brain):
    
    # add label encoder as input
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None, label_encoder=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.label_encoder = label_encoder
        
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos
        stem_ids, _ = batch.stem_encoded

        # Forward encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        enc_out = enc_out[-1].detach().cpu()
        
        # print(enc_out[0])
        # assert 1==2
        hyps = None

        return hyps, wav_lens, enc_out

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""
        (hyps, _, enc_out) = predictions
        batch = batch.to(self.device)
        loss = 0
        if stage == sb.Stage.TEST:
            channel_ids, _ = batch.channel_id_encoded
            # stem_ids, _ = batch.stem_encoded  
            decoded_channel_ids = self.label_encoder.decode_ndim(channel_ids)
            
            # Calculate distances for the current batch
            # batch_dist_vector = compute_distances(enc_out, channel_ids, label_encoder)
            batch_dist_matrix = compute_distances(enc_out, channel_ids, label_encoder, matrix=True)
            # self.dist_vector += batch_dist_vector
            self.dist_matrix += batch_dist_matrix
            self.num_stems += len(enc_out) // 8  # Each batch has 4 stems

            # Print current average distance vector
            # print("current average dist vector:", self.dist_vector / self.num_stems)
            print("current average dist matrix:\n", self.dist_matrix / self.num_stems)
                
        return loss

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
                loss = self.compute_objectives(out, batch, stage=stage)
        else:
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
        return torch.tensor(loss)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TEST:
            self.dist_vector = torch.zeros(8)
            self.dist_matrix = torch.zeros((8, 8))
            self.num_stems = 0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""            
        if stage == sb.Stage.TEST:
            # dist_vec = self.dist_vector/self.num_stems
            dist_mat = self.dist_matrix/self.num_stems
            # logger.info(dist_vec)
            logger.info(dist_mat)
            
            variances = []
            for row in dist_mat:
                filtered_row = row[row != 0]
                variance = filtered_row.var(unbiased=True) 
                variances.append(variance.item()) 
                       
            var_vec = torch.tensor(variances).round(4)
            logger.info(var_vec)
            # self.hparams.train_logger.log_stats(dist_vec)
            self.hparams.train_logger.log_stats(dist_mat)
            self.hparams.train_logger.log_stats(var_vec)


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

    # label_encoder_stem = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        sig = sb.dataio.dataio.read_audio(wav_path)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    # 3. Define classifier pipeline:
    # label token need to call add_unk for the unseen words
    label_encoder = sb.dataio.encoder.CategoricalEncoder() 
    label_encoder_stem = sb.dataio.encoder.CategoricalEncoder()
    
    # label_encoder.add_unk()  
    @sb.utils.data_pipeline.takes("channel")
    @sb.utils.data_pipeline.provides("channel", "channel_id_encoded")
    def label_pipeline(channel):
        yield channel
        channel_id_encoded = label_encoder.encode_sequence_torch([channel])
        yield channel_id_encoded
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path='label_encoder.txt',
        from_didatasets=[train_data],
        output_key="channel",
    )

    @sb.utils.data_pipeline.takes("stem")
    @sb.utils.data_pipeline.provides("stem", "stem_encoded")
    def stem_pipeline(stem):
        yield stem
        stem_encoded = label_encoder_stem.encode_sequence_torch([stem])
        yield stem_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, stem_pipeline)
    lab_enc_file_stem = os.path.join(hparams["save_folder"], "label_encoder_stem.txt")
    
    label_encoder_stem.load_or_create(
        path=lab_enc_file_stem,
        from_didatasets=[train_data, valid_data, test_dataset_representation],
        output_key="stem",
    )

    # 3. Define text pipeline:
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
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "channel_id_encoded", "stem_encoded"],
    )

    return train_data, valid_data, test_datasets, label_encoder


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
        project=f"whisper-tiny-asr-{hparams['project']}-{hparams['channel']}-dec",

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
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
        label_encoder=label_encoder
    )
    
    if hparams['crisper_ckpt'] != False:
        asr_brain.modules.whisper.load_state_dict(torch.load(hparams['crisper_ckpt']))
    

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    # Testing
    os.makedirs(hparams["output_cer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        # asr_brain.hparams.test_wer_file = os.path.join(
        #     hparams["output_wer_folder"], f"wer_{k}.txt"
        # )
        asr_brain.hparams.test_cer_file = os.path.join(
            hparams["output_cer_folder"], f"cer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="VALID/CER",
        )